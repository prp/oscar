//! K-quants through ggml's reference quantisers, as `llama-quantize` runs them: candle's is worse and takes no imatrix.

use candle_core::quantized::GgmlDType;

extern "C" {
    fn quantize_q3_K(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, qw: *const f32) -> usize;
    fn quantize_q4_K(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, qw: *const f32) -> usize;
    fn quantize_q5_K(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, qw: *const f32) -> usize;
    fn quantize_q6_K(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, qw: *const f32) -> usize;
    fn quantize_q5_0(src: *const f32, dst: *mut u8, nrow: i64, n_per_row: i64, qw: *const f32) -> usize;
}

/// Q5_0 is for rows off the 256-weight k-quant superblock.
pub fn is_writable(dtype: GgmlDType) -> bool {
    matches!(dtype, GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::Q5_0)
}

pub fn quant_bytes(dtype: GgmlDType, elems: usize) -> usize {
    elems / dtype.block_size() * dtype.type_size()
}

/// Imatrix-weighted per column if given; split by whole rows across `threads`, byte-identical to one call.
pub fn quantize(
    dtype: GgmlDType, data: &[f32], n_per_row: usize, imatrix: Option<&[f32]>, threads: usize,
) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(is_writable(dtype), "no ggml quantiser linked for {dtype:?}");
    anyhow::ensure!(
        n_per_row % dtype.block_size() == 0,
        "row length {n_per_row} is not a multiple of the {dtype:?} block ({})",
        dtype.block_size()
    );
    anyhow::ensure!(
        n_per_row > 0 && data.len() % n_per_row == 0,
        "{} floats do not divide into rows of {n_per_row}",
        data.len()
    );
    if let Some(im) = imatrix {
        anyhow::ensure!(im.len() == n_per_row, "imatrix has {} weights for row length {n_per_row}", im.len());
    }

    let rows = data.len() / n_per_row;
    let row_bytes = quant_bytes(dtype, n_per_row);
    let chunk = rows.div_ceil(threads.max(1));
    let mut out = vec![0u8; rows * row_bytes];

    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(chunk * n_per_row)
            .zip(out.chunks_mut(chunk * row_bytes))
            .map(|(src, dst)| {
                s.spawn(move || {
                    let n = src.len() / n_per_row;
                    let qw = imatrix.map_or(std::ptr::null(), <[f32]>::as_ptr);
                    // SAFETY: src holds n whole rows and dst exactly the bytes the quantiser writes, all it touches.
                    let written = unsafe {
                        (match dtype {
                            GgmlDType::Q3K => quantize_q3_K,
                            GgmlDType::Q4K => quantize_q4_K,
                            GgmlDType::Q5K => quantize_q5_K,
                            GgmlDType::Q6K => quantize_q6_K,
                            GgmlDType::Q5_0 => quantize_q5_0,
                            _ => unreachable!("is_writable admitted {dtype:?}"),
                        })(src.as_ptr(), dst.as_mut_ptr(), n as i64, n_per_row as i64, qw)
                    };
                    written == dst.len()
                })
            })
            .collect();
        anyhow::ensure!(
            handles.into_iter().all(|h| h.join().expect("quantise worker panicked")),
            "ggml quantiser wrote an unexpected byte count"
        );
        Ok(())
    })?;
    Ok(out)
}
