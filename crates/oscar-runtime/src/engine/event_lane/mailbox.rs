//! The event lane mailbox: one shared buffer, a region per layer of generation-tagged blocks.

use super::*;

/// The fill value: the router produces only ids `< num_experts`.
pub(crate) const MAILBOX_SENTINEL: u32 = u32::MAX;

/// Under 0xFF, so a gen byte is never the sentinel, and no gen aliases within a pre-wait blit's one-token staleness.
const GEN_MOD: u64 = 251;

pub(crate) fn gen_byte(wait_value: u64) -> u8 {
    (wait_value % GEN_MOD) as u8
}

/// As the publish kernel writes it: the gen in byte 0 and, for a double block, byte 1; the rest 0xFF, as refilled.
pub(crate) fn gen_word(wait_value: u64, double: bool) -> u32 {
    let g = gen_byte(wait_value) as u32;
    if double {
        g | (g << 8) | 0xFFFF_0000
    } else {
        g | 0xFFFF_FF00
    }
}

pub(crate) const POLL_TIMEOUT: Duration = Duration::from_secs(5);

/// The load's GPU backlog and first-use pipeline builds delay the first tokens' blits by seconds.
pub(crate) const POLL_TIMEOUT_AFTER_LOAD: Duration = Duration::from_secs(120);

/// For the first [`WARMUP_TOKENS`] decode tokens after a prefill, while the pool churns off the prefill's working set.
pub(crate) const WARMUP_TIMEOUT: Duration = Duration::from_millis(30_000);
pub(crate) const WARMUP_TOKENS: u64 = 8;

/// Past this the routing blit is not µs away, so the poll sleeps in 50 µs quanta rather than hold a core.
pub(crate) const POLL_SPIN: Duration = Duration::from_micros(300);

/// The blocks of a layer's region, in order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Block {
    Ids,
    /// The GPU prediction of the next layer: its top-B ids, then their logits.
    Pred,
    Tap(Tap),
    /// The router logits, then the top-k weights.
    Theta,
}

const ORDER: [Block; 5] = [Block::Ids, Block::Pred, Block::Tap(Tap::Early), Block::Tap(Tap::Late), Block::Theta];

const WORD: usize = std::mem::size_of::<u32>();

/// A double block brackets its payload with two gen bytes: the ids and θ's weights cannot trust a partial payload.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Span {
    pub(crate) word: usize,
    pub(crate) words: usize,
    pub(crate) double: bool,
}

impl Span {
    pub(crate) fn offset(&self) -> usize {
        self.word * WORD
    }

    pub(crate) fn bytes(&self) -> usize {
        self.words * WORD
    }

    /// The gen byte, filled after the payload: the one the reader polls.
    pub(crate) fn gen_offset(&self) -> usize {
        self.offset() + self.bytes()
    }

    /// Byte 1 of the same word, filled before a double block's payload.
    pub(crate) fn gen_begin_offset(&self) -> usize {
        self.gen_offset() + 1
    }
}

/// One value the GPU encoder and the resolver both address every block from, so they cannot disagree.
#[derive(Clone, Copy)]
pub(crate) struct MailboxLayout {
    pub(crate) topk: usize,
    pub(crate) num_experts: usize,
    pub(crate) gpu_pred_width: usize,
    pub(crate) early_tap_words: usize,
    pub(crate) late_tap_words: usize,
    pub(crate) theta_words: usize,
}

impl MailboxLayout {
    /// A zero-width block is never blitted or polled; the late tap, where offered, replaces the GPU prediction.
    pub(crate) fn plan(topk: usize, num_experts: usize, hidden_dim: usize) -> Self {
        let params = crate::config::params();
        let host_chain = params.ep.lookahead == crate::config::params::Lookahead::Host;
        // An architecture without router tables can only run the GPU plane, or the host chain would starve silently.
        if host_chain {
            if let Some(m) = oscar_models::loaded_model() {
                assert!(
                    m.router_tables,
                    "OSCAR_LOOKAHEAD=host needs the host router tables, and \
                     this model's port provisions none - select the GPU \
                     plane (OSCAR_LOOKAHEAD=gpu) or none"
                );
            }
        }
        let late_tap = host_chain && oscar_models::loaded_model().is_some_and(|m| m.late_tap);
        Self {
            topk,
            num_experts,
            // Width is not a lever: wider predictions stage candidates the state would never rank highly.
            gpu_pred_width: if late_tap {
                0
            } else {
                GPU_PRED_WIDTH.max(topk).min(num_experts).min(oscar_kernels::EP_PRED_MAX_B)
            },
            early_tap_words: if !late_tap { hidden_dim } else { 0 },
            late_tap_words: if late_tap { hidden_dim } else { 0 },
            // Off, the layout is identical to one without the feature.
            theta_words: if params.approx_theta.is_some() { num_experts + topk } else { 0 },
        }
    }

    /// `block`'s payload words; 0 is a block this run never writes or polls.
    pub(crate) fn words(&self, block: Block) -> usize {
        match block {
            Block::Ids => self.topk,
            Block::Pred => 2 * self.gpu_pred_width,
            Block::Tap(Tap::Early) => self.early_tap_words,
            Block::Tap(Tap::Late) => self.late_tap_words,
            Block::Theta => self.theta_words,
        }
    }

    fn footprint(&self, block: Block) -> usize {
        match self.words(block) {
            0 => 0,
            w => w + 1,
        }
    }

    /// The present blocks in [`ORDER`]: arming a plane moves only the blocks after it, and θ stays the tail.
    pub(crate) fn region_words(&self) -> usize {
        ORDER.iter().map(|&b| self.footprint(b)).sum()
    }

    pub(crate) fn span(&self, layer: usize, block: Block) -> Span {
        let before: usize = ORDER.iter().take_while(|&&b| b != block).map(|&b| self.footprint(b)).sum();
        Span {
            word: layer * self.region_words() + before,
            words: self.words(block),
            double: matches!(block, Block::Ids | Block::Theta),
        }
    }
}

// SAFETY: MTLSharedEvent is documented thread-safe; the Retained keeps it alive.
#[derive(Clone)]
pub(crate) struct SharedEventHandle(pub(crate) Retained<ProtocolObject<dyn MTLSharedEvent>>);
unsafe impl Send for SharedEventHandle {}
unsafe impl Sync for SharedEventHandle {}

/// Sentinel-filled (0xFF bytes), so no gen byte can match before its blit.
pub(crate) fn alloc_mailbox(
    device: &ProtocolObject<dyn MTLDevice>, mailbox_bytes: usize,
) -> std::result::Result<crate::gpu::Words<u8>, String> {
    crate::gpu::words(device, mailbox_bytes, 0xFF, "event lane mailbox")
}

pub(crate) fn alloc_shared_event(
    device: &ProtocolObject<dyn MTLDevice>,
) -> std::result::Result<SharedEventHandle, String> {
    Ok(SharedEventHandle(device.newSharedEvent().ok_or_else(|| "event lane: newSharedEvent returned nil".to_string())?))
}

/// Until `span`'s gen carries `expect` (both bytes of a double block) and `ready` accepts the payload.
pub(crate) fn poll(
    mailbox: &KernelBuffer, span: Span, expect: u8, timeout: Duration, knee: Option<Duration>,
    ready: impl Fn(*const u32) -> bool,
) -> Option<*mut u32> {
    // SAFETY: the layout sized the mailbox to hold every layer's region.
    let at = |off: usize| unsafe { (mailbox.contents() as *mut u8).add(off) };
    let (base, gen, begin) = (at(span.offset()) as *mut u32, at(span.gen_offset()), at(span.gen_begin_offset()));
    let t0 = Instant::now();
    loop {
        // SAFETY: as above; single-byte volatile reads of the shared buffer.
        let landed =
            unsafe { gen.read_volatile() } == expect && (!span.double || unsafe { begin.read_volatile() } == expect);
        if landed {
            // Volatile alone does not order the payload's loads after the gen's on ARM.
            std::sync::atomic::fence(Ordering::Acquire);
            if ready(base) {
                return Some(base);
            }
        }
        let waited = t0.elapsed();
        if waited > timeout {
            return None;
        }
        if knee.is_some_and(|k| waited > k) {
            std::thread::sleep(Duration::from_micros(50));
        } else {
            std::hint::spin_loop();
        }
    }
}

/// `layer`'s routed ids for the dispatch at `wait_value`, re-polled while a sentinel is left in the payload.
pub(crate) fn poll_ids(
    mailbox: &KernelBuffer, layout: &MailboxLayout, layer: usize, wait_value: u64, timeout: Duration,
) -> Option<Vec<u32>> {
    let span = layout.span(layer, Block::Ids);
    // SAFETY: inside the span `poll` hands over, after its fence.
    let ids = |p: *const u32| (0..span.words).map(move |i| unsafe { p.add(i).read_volatile() });
    let base =
        poll(mailbox, span, gen_byte(wait_value), timeout, Some(POLL_SPIN), |p| ids(p).all(|v| v != MAILBOX_SENTINEL))?;
    Some(ids(base).collect())
}

/// Sentinel-refill a consumed block, or only its gen word where the payload is read whole before the next blit.
pub(crate) fn refill(mailbox: &KernelBuffer, span: Span, payload: bool) {
    let from = if payload { span.word } else { span.word + span.words };
    for w in from..=span.word + span.words {
        // SAFETY: inside the live mailbox; the block's dispatch was consumed, so no GPU writer is in flight on it.
        unsafe { (mailbox.contents() as *mut u32).add(w).write_volatile(MAILBOX_SENTINEL) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Written out independently: at runtime a wrong gen-begin byte only shows as the reader waiting out its deadline.
    #[test]
    fn every_block_and_gen_byte_sits_where_both_ends_expect_it() {
        let w = WORD;
        let layout = MailboxLayout {
            topk: 8,
            num_experts: 128,
            gpu_pred_width: 8,
            early_tap_words: 4096,
            late_tap_words: 0,
            theta_words: 128 + 8,
        };
        let stride = 8 + 1 + (16 + 1) + (4096 + 1) + (128 + 8 + 1);
        assert_eq!(layout.region_words(), stride);
        for layer in [0usize, 1, 93] {
            let region = layer * stride * w;
            let ids = layout.span(layer, Block::Ids);
            assert_eq!((ids.offset(), ids.bytes(), ids.double), (region, 8 * w, true), "ids start the region");
            assert_eq!(ids.gen_offset(), region + 8 * w, "byte 0 of the gen word");
            assert_eq!(ids.gen_begin_offset(), region + 8 * w + 1, "byte 1, inside the word the refill covers");
            let pred = layout.span(layer, Block::Pred);
            assert_eq!((pred.offset(), pred.bytes(), pred.double), (region + 9 * w, 16 * w, false));
            let tap = layout.span(layer, Block::Tap(Tap::Early));
            assert_eq!((tap.offset(), tap.gen_offset()), (region + 26 * w, region + (26 + 4096) * w));
            assert_eq!(layout.span(layer, Block::Tap(Tap::Late)).words, 0, "an absent block has no payload");
            // θ is the region's tail: arming another plane moves it and nothing earlier.
            let theta = layout.span(layer, Block::Theta);
            let tail = region + (stride - (128 + 8 + 1)) * w;
            assert_eq!((theta.offset(), theta.bytes(), theta.double), (tail, (128 + 8) * w, true));
            assert_eq!(theta.gen_offset(), tail + (128 + 8) * w);
            assert!(theta.gen_begin_offset() < region + stride * w, "inside its own layer's region");
        }
        // The publish kernel's gen word: the gen in byte 0, again in byte 1 for a double block, the rest refilled 0xFF.
        let g = gen_byte(1000) as u32;
        assert_eq!((gen_word(1000, true), gen_word(1000, false)), (g | (g << 8) | 0xFFFF_0000, g | 0xFFFF_FF00));
    }
}
