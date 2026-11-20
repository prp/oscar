use candle_core::quantized::{metal::QMetalStorage, GgmlDType};
use candle_core::{DType, Result, Tensor};

use crate::loader::{launch, out, pipeline, set_params, Family, Metal};

/// The rows at `ids` (U32, any shape), dequantised bit-identically to the whole table's: F32 `ids.dims() ++ [hidden]`.
pub fn metal_embedding_gather(table: &QMetalStorage, vocab: usize, hidden: usize, ids: &Tensor) -> Result<Tensor> {
    let (kernel_name, block) = match table.dtype() {
        GgmlDType::Q4K => ("oscar_embed_gather_q4_K_f32", 256),
        GgmlDType::Q8_0 => ("oscar_embed_gather_q8_0_f32", 32),
        other => candle_core::bail!("metal_embedding_gather: unsupported table dtype {other:?} (only Q4_K / Q8_0)"),
    };
    refuse_if!(
        hidden == 0 || hidden % block != 0,
        "metal_embedding_gather: hidden ({hidden}) must be a non-zero multiple of {block}"
    );
    refuse_if!(ids.dtype() != DType::U32, "metal_embedding_gather: ids must be U32, got {:?}", ids.dtype());
    let n_ids = ids.elem_count();
    refuse_if!(n_ids == 0, "metal_embedding_gather: empty id set");

    let ids_c = ids.flatten_all()?.contiguous()?;
    let idm = Metal::of(&ids_c, "metal_embedding_gather: ids")?;
    let p = pipeline(&idm.dev, Family::Moe, kernel_name)?;
    let mut dims = ids.dims().to_vec();
    dims.push(hidden);
    let (y, o) = out(&idm.dev, dims, DType::F32, "embedding-gather-output")?;
    let tg = 256.min(p.max_total_threads_per_threadgroup() as usize).min(hidden);
    launch(&idm.dev, &p, "embedding-gather", [n_ids, 1, 1], [tg, 1, 1], |e| {
        set_params!(e, (table.buffer(), &idm, &o, hidden as i64, vocab as i64));
        Ok(())
    })?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_bitwise, host, sine};
    use candle_core::quantized::QTensor;
    use candle_core::Device;

    /// The reference is candle's `to_float` on the same bytes, so an `fma` contraction fails it.
    #[test]
    fn embedding_gather_is_bit_identical_to_a_whole_table_dequantisation() -> Result<()> {
        let device = oscar_offload::pools::test_device()?;
        let (vocab, hidden) = (97usize, 512usize);
        let ids: Vec<u32> = (0..vocab * 2).map(|i| ((i * 37) % vocab) as u32).collect();
        let ids_t = Tensor::from_vec(ids.clone(), (2, vocab), &device)?;
        for ggml in [GgmlDType::Q4K, GgmlDType::Q8_0] {
            let floats = Tensor::from_vec(sine(vocab * hidden, 0.0013, 0.7), (vocab, hidden), &Device::Cpu)?;
            let table = QTensor::quantize_onto(&floats, ggml, &device)?;
            let whole = table.dequantize(&Device::Cpu)?.to_vec2::<f32>()?;
            let gathered = metal_embedding_gather(table.metal_storage().expect("on Metal"), vocab, hidden, &ids_t)?;
            assert_eq!(gathered.dims(), [2, vocab, hidden]);
            let want: Vec<f32> = ids.iter().flat_map(|&id| whole[id as usize].iter().copied()).collect();
            assert_bitwise(&host(&gathered)?, &want, &format!("{ggml:?} gather"));
        }
        Ok(())
    }
}
