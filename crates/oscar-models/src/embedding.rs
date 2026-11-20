//! The token embedding table, held at its GGUF size.

use candle_core::quantized::QTensor;
use candle_core::{DType, Result, Tensor};

/// A `(vocab, hidden)` table kept quantized on the device, dequantizing only the rows a batch gathers.
pub struct QuantizedEmbedding {
    table: QTensor,
    vocab: usize,
    pub hidden: usize,
    dtype: DType,
}

impl QuantizedEmbedding {
    /// A width other than `hidden` is refused, not taken as a row stride the layers read past.
    pub fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut mistralrs_core::gguf::Content<'_, R>, device: &candle_core::Device, hidden: usize, dtype: DType,
    ) -> Result<Self> {
        let table = ct.tensor("token_embd.weight", device)?;
        let vocab = match table.shape().dims() {
            &[v, h] if h == hidden => v,
            d => candle_core::bail!("token embedding must be (vocab, {hidden}), got {d:?}"),
        };
        Ok(Self { table, vocab, hidden, dtype })
    }

    /// `ids` U32, any shape → `ids.dims() ++ [hidden]` in the stream's dtype.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let Some(storage) = self.table.metal_storage() else {
            candle_core::bail!("token embedding must be on Metal");
        };
        oscar_kernels::metal_embedding_gather(storage, self.vocab, self.hidden, ids)?.to_dtype(self.dtype)
    }
}
