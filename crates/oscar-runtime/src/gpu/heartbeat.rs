//! A trivial GPU dispatch holding the clock up while oscar serves: the GPU idles between a paged decode's buffers.
//! It holds a clock only if already beating when the heavy work stops, so it never makes way for a drafter.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use candle_core::DType;
use candle_metal_kernels::metal::{CommandBuffer, CommandSemaphore};
use candle_metal_kernels::BufferOffset;
use objc2_metal::MTLCommandQueue;

/// Small enough that the dispatch is pure occupancy.
const HEARTBEAT_ELEMS: usize = 64;

/// Beating past the last step: longer than the gaps within a request.
const IDLE_AFTER: Duration = Duration::from_millis(500);

struct Steps {
    running: usize,
    last_end: Option<Instant>,
}

impl Steps {
    fn beating(&self, now: Instant) -> bool {
        self.running > 0 || self.last_end.is_some_and(|t| now.duration_since(t) < IDLE_AFTER)
    }
}

static STEPS: Mutex<Steps> = Mutex::new(Steps { running: 0, last_end: None });
static WAKE: Condvar = Condvar::new();

/// A step in flight, prompt passes and prefill chunks included.
pub struct Step(());

pub fn step() -> Step {
    STEPS.lock().unwrap().running += 1;
    WAKE.notify_one();
    Step(())
}

impl Drop for Step {
    fn drop(&mut self) {
        let mut steps = STEPS.lock().unwrap();
        (steps.running, steps.last_end) = (steps.running - 1, Some(Instant::now()));
    }
}

/// Once, at the load, where the config parameter asks; the thread parks while oscar is idle.
pub fn start() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    let us = crate::config::params().heartbeat_us;
    if us == 0 {
        return;
    }
    STARTED.call_once(|| {
        std::thread::Builder::new()
            .name("oscar-gpu-heartbeat".into())
            .spawn(move || {
                let tick = Tick::new().unwrap_or_else(|e| panic!("GPU heartbeat start failed ({e})"));
                crate::detail!(target: "oscar::heartbeat", "heartbeat every {us} us while request runs");
                loop {
                    drop(WAKE.wait_while(STEPS.lock().unwrap(), |s| !s.beating(Instant::now())).unwrap());
                    // Without a pool per tick the GPU driver keeps every buffer and encoder.
                    objc2::rc::autoreleasepool(|_| tick.beat())
                        .unwrap_or_else(|e| panic!("GPU heartbeat failed ({e})"));
                    std::thread::sleep(Duration::from_micros(us));
                }
            })
            .expect("spawn oscar-gpu-heartbeat");
    });
}

/// On its own queue: in the engine's pool a tick would consume a wave's armed wait.
struct Tick {
    device: candle_core::MetalDevice,
    queue: candle_metal_kernels::metal::CommandQueue,
    buffer: Arc<candle_metal_kernels::metal::Buffer>,
    semaphore: Arc<CommandSemaphore>,
}

impl Tick {
    fn new() -> candle_core::Result<Self> {
        // The load installed the pools first, so every buffer comes from one.
        let candle_core::Device::Metal(device) = candle_core::Device::new_metal(0)? else {
            candle_core::bail!("no Metal device");
        };
        let queue = device.device().new_command_queue().map_err(candle_core::Error::wrap)?;
        // The floor's: held for the process's life, and read from this queue.
        let buffer = device.new_fixed_buffer(HEARTBEAT_ELEMS, DType::F32, "heartbeat")?;
        oscar_offload::fixed::fixed_pool(device.device().as_ref()).map_err(candle_core::Error::wrap)?.attach(&queue);
        Ok(Self { device, queue, buffer, semaphore: Arc::new(CommandSemaphore::new()) })
    }

    /// One command buffer, run to completion.
    fn beat(&self) -> candle_core::Result<()> {
        let raw = self.queue.commandBuffer().ok_or_else(|| candle_core::Error::Msg("no command buffer".into()))?;
        let commands = CommandBuffer::new(raw, Arc::clone(&self.semaphore));
        let encoder = commands.compute_command_encoder();
        candle_metal_kernels::call_affine(
            self.device.metal_device(),
            &encoder,
            self.device.kernels(),
            "affine_f32",
            DType::F32.size_in_bytes(),
            HEARTBEAT_ELEMS,
            BufferOffset::zero_offset(&self.buffer),
            &self.buffer,
            1.0001,
            0.0,
        )
        .map_err(candle_core::Error::wrap)?;
        encoder.end_encoding();
        commands.commit();
        commands.wait_until_completed();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_heartbeat_beats_while_a_request_runs_and_parks_when_idle() {
        let (t0, ms, later) = (Instant::now(), Duration::from_millis(1), Instant::now() + 60 * IDLE_AFTER);
        assert!(!Steps { running: 0, last_end: None }.beating(t0), "parked before the first step");
        assert!(Steps { running: 1, last_end: None }.beating(later), "a step of seconds keeps it beating");
        let ended = Steps { running: 0, last_end: Some(t0) };
        assert!(ended.beating(t0 + IDLE_AFTER - ms), "beating between one request's steps");
        assert!(!ended.beating(t0 + IDLE_AFTER), "parked once idle");
        assert!(Steps { running: 1, ..ended }.beating(later), "the next step wakes it");
    }
}
