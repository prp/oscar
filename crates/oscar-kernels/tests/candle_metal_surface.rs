//! The candle fork's behaviour oscar stands on, checked here so the fork's patches carry no oscar code.

use candle_core::{Device, Result, Tensor};
use candle_metal_kernels::metal::Commands;

/// The resolver's retire check, with no synchronise: each commit's buffer signals its generation on the GPU.
#[test]
fn completion_event_signals_on_commit() {
    use objc2_metal::MTLSharedEvent;
    use std::sync::atomic::Ordering;

    let device = candle_metal_kernels::metal::Device::system_default().unwrap();
    let queue = device.new_command_queue().unwrap();
    let commands = Commands::new(queue, 0, None).unwrap();

    assert_eq!(commands.completion_counter().load(Ordering::SeqCst), 0);
    assert_eq!(commands.next_completion_generation(), 1);

    let (_flush, encoder) = commands.command_encoder().unwrap();
    drop(encoder);
    commands.flush().unwrap();

    assert_eq!(commands.completion_counter().load(Ordering::SeqCst), 1);
    assert_eq!(commands.next_completion_generation(), 2);

    let event = commands.completion_event();
    assert!(event.waitUntilSignaledValue_timeoutMS(1, 2_000), "completion event never reached generation 1");
}

/// Hazard tracking orders a segment's reuse by the next kernel to write it; CPU writes are the staged class's to fence.
/// `#[track_caller]` carries the caller's location down to the seam; a hop without it labels candle's line instead.
#[test]
fn pooled_buffers_are_shared_and_tracked_and_a_host_built_tensor_names_the_line_that_built_it() {
    use objc2_metal::{MTLHazardTrackingMode, MTLResource, MTLStorageMode};

    oscar_offload::pools::install_for_tests();
    let dev = Device::new_metal(0).unwrap();
    let Device::Metal(metal) = &dev else { unreachable!("a Metal device") };
    let buf = metal.new_buffer(4096, candle_core::DType::U8, "surface").unwrap();
    let res: &candle_metal_kernels::metal::MetalResource = (&*buf).into();
    assert_eq!((res.hazardTrackingMode(), res.storageMode()), (MTLHazardTrackingMode::Tracked, MTLStorageMode::Shared));
    let built = std::panic::Location::caller().file();
    let _t = Tensor::new(&[1u32, 2, 3, 4], &dev).unwrap();
    let here = line!() - 1;
    let file = built.rsplit('/').next().unwrap_or(built);
    let layout: String = oscar_offload::pools::pools().expect("the pools are installed").describe();
    assert!(layout.contains(&format!("{file}:{here}")), "the region names the line that built the tensor: {layout}");
}

/// The parallel `QuantizedType for Vec<T>` against one serial call, for every block type a model can carry.
#[test]
fn host_quantiser_runs_are_one_serial_call() {
    use candle_core::quantized::k_quants::*;
    use candle_core::quantized::QuantizedType;
    use candle_core::CpuStorage;
    fn bytes<T>(v: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }
    fn check<T: GgmlType + Send + Sync>() {
        let blocks = 8192;
        let xs: Vec<f32> = (0..blocks * T::BLCK_SIZE).map(|i| (i as f32 + 0.37).sin() * 0.5).collect();
        let mut serial: Vec<T> = (0..blocks).map(|_| T::zeros()).collect();
        T::from_float(&xs, &mut serial);
        let mut split: Vec<T> = (0..blocks).map(|_| T::zeros()).collect();
        QuantizedType::from_float(&mut split, &xs);
        assert!(bytes(&serial) == bytes(&split), "{:?}: the split quantises differently", T::DTYPE);
        let mut back = vec![0f32; xs.len()];
        T::to_float(&serial, &mut back);
        let CpuStorage::F32(par) = QuantizedType::dequantize(&split, xs.len()).unwrap() else {
            panic!("{:?}: dequantise storage", T::DTYPE)
        };
        assert!(back == par, "{:?}: the split dequantises differently", T::DTYPE);
    }
    macro_rules! each {
        ($($t:ty),*) => { $(check::<$t>();)* };
    }
    each!(
        BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K,
        BlockQ8K
    );
}

/// The floor check is `>=`: the counters are process-wide and tests run in parallel, so another test can only add.
#[test]
fn a_kept_tensor_comes_from_the_floor_and_carries_the_source_bytes() -> Result<()> {
    let device = oscar_offload::pools::test_device()?;
    // Not a power of two, so the class rounds up.
    let (rows, cols) = (1301usize, 97usize);
    let src: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.0011 + 0.31).sin() * 0.5).collect();
    let src = Tensor::from_vec(src, (rows, cols), &Device::Cpu)?;
    let floor = || oscar_offload::fixed::fixed_pool_if_built().map_or(0, |p| p.bytes());
    let before = floor();
    let kept = src.to_device_kept(&device)?;
    let taken = floor() - before;
    assert_eq!(kept.dims(), [rows, cols]);
    assert_eq!(kept.to_vec2::<f32>()?, src.to_vec2::<f32>()?);
    let own = (rows * cols * 4).next_multiple_of(oscar_offload::segments::PAGE);
    assert!(taken >= own, "the floor gave {taken} for {} bytes, under the page it hands out in", rows * cols * 4);
    // A source already on a device is refused: build what you keep on the host.
    assert!(kept.to_device_kept(&device).is_err());
    Ok(())
}
