//! The working region lent to the expert pool for decode, taken back for a prompt; standing loans keep their experts.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use mistralrs_core::paged_attention::StepKind;
use oscar_offload::region::Region;

use super::{carrier_of, ExpertPool};

/// Fixed at the load's tail; a lend refused in a decode is not asked again before the next prompt.
pub(super) struct Lending {
    region: Arc<Region>,
    decode_bound: usize,
    refused: AtomicBool,
}

/// Lends and take-backs landed, moves refused, and the µs the landed ones took.
static CHANGES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// `(lends, take-backs, refused, µs)` so far.
pub fn phase_changes() -> (u64, u64, u64, u64) {
    let at = |i: usize| CHANGES[i].load(Ordering::Relaxed);
    (at(0), at(1), at(2), at(3))
}

impl ExpertPool {
    /// After the pool's launch and the region's take, at the live plan's shape; decode's then, until a prompt.
    pub fn lend_at_load(
        &self, shape: (usize, usize), decode_bound: u64, decode: Option<(usize, u64)>, recording: bool,
    ) {
        let laid = || self.lay_loans(shape, decode_bound, decode, recording);
        let Some(lending) = self.lending.get_or_init(laid).as_ref() else { return };
        match self.lend(&lending.region, lending.decode_bound) {
            Ok(_) => CHANGES[0].fetch_add(1, Ordering::Relaxed),
            Err(e) => {
                tracing::info!(target: "oscar::experts", "nothing lent at load's tail - {e}");
                CHANGES[2].fetch_add(1, Ordering::Relaxed)
            }
        };
    }

    /// Decode's bound in the device queue's region and the loans above it; `None` before a record or under a recording.
    fn lay_loans(
        &self, (hot_budget, carriers): (usize, usize), decode_bound: u64, decode: Option<(usize, u64)>, recording: bool,
    ) -> Option<Lending> {
        let decode_bound = Some(decode_bound as usize).filter(|&b| b > 0 && !recording)?;
        let device_queue = |r: &Arc<Region>| r.queue() < candle_metal_kernels::metal::STREAM_QUEUE_KEYS;
        let region = oscar_offload::pools::pools()?
            .regions()
            .into_iter()
            .find(device_queue)
            .filter(|r| decode_bound < r.size())?;
        if let Err(e) = region.wrap_first(decode_bound) {
            tracing::info!(target: "oscar::experts", "decode steps cut from whole region's object - {e}");
        }
        if let Some((decode_hot_budget, lent_bytes)) = decode {
            let carrier_of = carrier_of(carriers, self.layer_count());
            let laid = self.plan_loans(hot_budget, decode_hot_budget, &carrier_of, region.size(), decode_bound);
            let bytes: usize = self.loans().iter().map(|l| l.len).sum();
            crate::detail!(target: "oscar::experts",
                "{laid} segments, {:.3} GB of {:.3} GB working region above {:.3} GB decode bound, take paged layer \
                 from {} slots to {}{}",
                bytes as f64 / crate::machine::GB,
                region.size() as f64 / crate::machine::GB,
                decode_bound as f64 / crate::machine::GB,
                hot_budget,
                decode_hot_budget,
                if bytes as u64 == lent_bytes { "" } else { " - NOT the plan's figure" },
            );
        }
        Some(Lending { region, decode_bound, refused: Default::default() })
    }

    /// A completion step lends down to decode's bound; a prompt left short fails here, not later in an allocation.
    /// `sweep` releases the engine's pins first, as a park does: a pin outliving its request would hold its loan.
    pub fn phase_change(
        &self, step: &StepKind<'_>, quiesce: impl FnOnce(),
        sweep: impl FnOnce(&candle_core::MetalDevice) -> Result<usize, String>,
    ) -> Result<(), String> {
        let Some(lending) = self.lending.get().and_then(Option::as_ref) else { return Ok(()) };
        let candle_core::Device::Metal(md) = step.device else { return Ok(()) };
        let region = &lending.region;
        debug_assert_eq!(region.queue(), md.queue_key(), "the engine's steps cut from the region that lends");
        let want = if step.prompt {
            crate::load::census::region_at(step.tokens)
                .map_or(region.size(), |b| (b as usize).min(region.size()))
                .max(lending.decode_bound)
        } else {
            lending.decode_bound
        };
        let bound = region.bound();
        if step.prompt {
            lending.refused.store(false, Ordering::Relaxed);
        }
        if bound == want || (step.prompt && bound > want) || lending.refused.load(Ordering::Relaxed) {
            return Ok(());
        }
        let began = std::time::Instant::now();
        if bound < want {
            quiesce();
        }
        (step.drop_state_indices)();
        let moved = step.device.synchronize().map_err(|e| e.to_string()).and_then(|()| {
            if bound > want {
                return self.lend(region, want).map_err(|e| e.to_string());
            }
            sweep(md)?;
            let committed = md.completion_counter().map_err(|e| e.to_string())?.load(Ordering::SeqCst);
            self.take_back(region, want, self.tick.load(Ordering::Relaxed), committed).map_err(|e| e.to_string())
        });
        let micros = began.elapsed().as_micros() as u64;
        match moved {
            Ok(segments) => {
                CHANGES[usize::from(bound < want)].fetch_add(1, Ordering::Relaxed);
                CHANGES[3].fetch_add(micros, Ordering::Relaxed);
                if crate::cli::console::traced() {
                    let (standing, residents) = self.loans_standing();
                    tracing::info!(target: "oscar::experts",
                        "{} {segments} segments in {} - region bound {:.3} -> {:.3} GB, {standing} loans standing, \
                         {residents} experts resident{}",
                        if bound > want { "lent" } else { "took back" },
                        mistralrs_core::live::duration_text(micros as f64 / 1e6),
                        bound as f64 / crate::machine::GB,
                        want as f64 / crate::machine::GB,
                        crate::cli::console::stamp()
                    );
                }
            }
            Err(e) => {
                CHANGES[2].fetch_add(1, Ordering::Relaxed);
                lending.refused.store(!step.prompt, Ordering::Relaxed);
                let stays = format!(
                    "the region's bound stays at {:.3} GB - {e} (the pool's committed count {})",
                    bound as f64 / crate::machine::GB,
                    md.completion_counter().map_or(0, |c| c.load(Ordering::SeqCst))
                );
                tracing::error!(target: "oscar::experts", "{stays}");
                if step.prompt {
                    return Err(stays);
                }
            }
        }
        Ok(())
    }
}
