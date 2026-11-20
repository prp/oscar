//! K3's Block Attention Residuals: each sub-layer's input a retrieval over the embedding and the blocks' output sums.

use candle_core::{DType, Result, Tensor};

/// Threaded through the stack in place of one hidden tensor.
pub struct AttnResState {
    /// `blocks[0]` the embedding, then one sum of sub-layer outputs per completed block, each `(tokens, hidden)` BF16.
    blocks: Vec<Tensor>,
    /// The block in progress, `(tokens, hidden)` F32; `None` only between a promotion and the next output.
    prefix_sum: Option<Tensor>,
    /// `attn_res.block_size` counts transformer layers: the paper's halving gives a wrong graph.
    block_size: usize,
    eps: f64,
}

impl AttnResState {
    /// Seed the stream with the token embedding, `(tokens, hidden)` F32.
    pub fn new(embed: Tensor, block_size: usize, eps: f64) -> Result<Self> {
        if block_size == 0 {
            candle_core::bail!("kimi-k3: attn_res.block_size must be non-zero");
        }
        Ok(Self { blocks: Vec::new(), prefix_sum: Some(embed), block_size, eps })
    }

    /// `score` is the pseudo-query, the key norm's gain folded in by the writer.
    pub fn retrieve(&self, score: &Tensor) -> Result<Tensor> {
        use oscar_kernels::metal_kernel::RowKernel;
        let sources = self.sources()?;
        let (rows, hidden) = sources[0].dims2()?;
        let n = sources.len();
        // candle's `mean` (sum × 1/hidden) and `+ eps` in F32, the constants cast as candle casts them.
        let inv_d = (1f64 / hidden as f64) as f32;
        let eps = self.eps as f32;
        let mut body = format!("  float logit[{n}];\n");
        for i in 0..n {
            body.push_str(&format!(
                "  {{\n    float ss = 0.0f;\n    for (int j = tid; j < D; j += block_dim) {{ const float v = (float)src{i}[j]; const float q = v * v; ss = ss + q; }}\n    \
                 const float denom = sqrt(row_sum(ss, shm, tid, block_dim) * {inv_d:?}f + {eps:?}f);\n    \
                 float dot = 0.0f;\n    for (int j = tid; j < D; j += block_dim) {{ const float k = (float)src{i}[j] / denom; const float t = k * score[j]; dot = dot + t; }}\n    \
                 logit[{i}] = row_sum(dot, shm, tid, block_dim);\n  }}\n"
            ));
        }
        body.push_str(&format!(
            "  float m = logit[0];\n  for (int i = 1; i < {n}; i++) m = max(m, logit[i]);\n  \
             float p[{n}];\n  float sum = 0.0f;\n  for (int i = 0; i < {n}; i++) {{ p[i] = exp(logit[i] - m); sum = sum + p[i]; }}\n  \
             for (int i = 0; i < {n}; i++) p[i] = p[i] / sum;\n  \
             for (int j = tid; j < D; j += block_dim) {{\n    float acc = (float)src0[j] * p[0];\n"
        ));
        for i in 1..n {
            body.push_str(&format!("    {{ const float t = (float)src{i}[j] * p[{i}]; acc = acc + t; }}\n"));
        }
        body.push_str("    out[j] = acc;\n  }\n");
        const NAMES: [&str; 9] = ["src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8"];
        if n > NAMES.len() {
            candle_core::bail!("kimi-k3: {n} retrieval sources, kernel names at most {}", NAMES.len());
        }
        let mut names: Vec<&'static str> = NAMES[..n].to_vec();
        names.push("score");
        let mut inputs: Vec<&Tensor> = sources.iter().collect();
        inputs.push(score);
        let kernel = RowKernel { name: "attn_res_retrieve", inputs: &names, outputs: &["out"], body: &body };
        Ok(kernel.run(&inputs, rows, hidden)?.pop().expect("one output"))
    }

    /// The completed blocks, then the running sum unless just promoted.
    fn sources(&self) -> Result<Vec<Tensor>> {
        let mut out = self.blocks.clone();
        if let Some(p) = self.prefix_sum.as_ref() {
            out.push(p.clone());
        }
        if out.is_empty() {
            candle_core::bail!("kimi-k3: attention-residual retrieval with no sources");
        }
        Ok(out)
    }

    /// False only at layer 0's first sub-layer, which has one source.
    pub fn has_completed_blocks(&self) -> bool {
        !self.blocks.is_empty()
    }

    /// After the layer's pre-attention retrieval, before its attention; the frozen block is stored BF16, rounded once.
    pub fn maybe_close_block(&mut self, layer_idx: usize) -> Result<()> {
        if layer_idx % self.block_size != 0 {
            return Ok(());
        }
        let Some(p) = self.prefix_sum.take() else {
            candle_core::bail!("kimi-k3: block boundary at layer {layer_idx} with no partial sum to promote");
        };
        self.blocks.push(p.to_dtype(DType::BF16)?);
        Ok(())
    }

    pub fn accumulate(&mut self, sublayer_out: &Tensor) -> Result<()> {
        self.prefix_sum = Some(match self.prefix_sum.take() {
            Some(p) => (p + sublayer_out)?,
            None => sublayer_out.clone(),
        });
        Ok(())
    }

    /// The stream into a sub-layer with no retrieval (layer 0's attention).
    pub fn passthrough(&self) -> Result<Tensor> {
        self.prefix_sum.clone().ok_or_else(|| candle_core::Error::Msg("kimi-k3: no partial sum to pass through".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::host;
    use candle_core::Device;

    /// Per token, the raw sources weighed by the softmax over them of `RMSNorm(source)·score`: the definition, in f64.
    fn retrieval(sources: &[Vec<f32>], score: &[f32], eps: f64) -> Vec<f64> {
        let hidden = score.len();
        let mut out = vec![0f64; sources[0].len()];
        for (t, row) in out.chunks_mut(hidden).enumerate() {
            let source = |i: usize| sources[i][t * hidden..][..hidden].iter().map(|&v| f64::from(v));
            let logits: Vec<f64> = (0..sources.len())
                .map(|i| {
                    let rms = (source(i).map(|v| v * v).sum::<f64>() / hidden as f64 + eps).sqrt();
                    source(i).zip(score).map(|(v, &q)| v / rms * f64::from(q)).sum()
                })
                .collect();
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
            let z: f64 = p.iter().sum();
            for (i, p) in p.iter().enumerate() {
                for (o, v) in row.iter_mut().zip(source(i)) {
                    *o += v * p / z;
                }
            }
        }
        out
    }

    /// Eight BF16 blocks and the F32 partial at K3's width, a block a layer, within rounding of the definition. At the
    /// narrowest width, four columns: one source is the identity; a zero query averages the raw sources, which a
    /// softmax over the wrong axis would not; scaling a source moves its value, not its weight, as the key is normed.
    #[test]
    fn retrieval_matches_its_definition_and_weighs_the_raw_sources_by_their_normed_keys() -> Result<()> {
        let dev = oscar_offload::pools::test_device()?;
        let (tokens, hidden, eps) = (4usize, 7168usize, 1e-6);
        let fill = |seed: usize| -> Result<Tensor> {
            let v: Vec<f32> = (0..tokens * hidden)
                .map(|i| (((i * 7919 + seed * 104729) % 10007) as f32 - 5003.0) / (300.0 + seed as f32 * 50.0))
                .collect();
            Tensor::from_vec(v, (tokens, hidden), &dev)
        };
        let mut st = AttnResState::new(fill(0)?, 1, eps)?;
        for layer in 1..=8 {
            st.maybe_close_block(layer)?;
            st.accumulate(&fill(layer)?)?;
        }
        let score: Vec<f32> = (0..hidden).map(|i| ((i * 31 % 97) as f32 - 48.0) / 200.0).collect();
        let got = host(&st.retrieve(&Tensor::new(score.as_slice(), &dev)?)?)?;
        let sources = st.sources()?.iter().map(|s| host(&s.to_dtype(DType::F32)?)).collect::<Result<Vec<_>>>()?;
        let want = retrieval(&sources, &score, eps);
        let scale = want.iter().fold(0f64, |m, w| m.max(w.abs()));
        assert!(got.iter().all(|g| g.is_finite()), "the kernel produced a non-finite output");
        let worst = got.iter().zip(&want).fold(0f64, |m, (&g, w)| m.max((f64::from(g) - w).abs()));
        assert!(worst / scale < 1e-5, "the kernel departs from the definition by {worst:e} on a scale of {scale:e}");
        let hidden = 128;
        let row = |f: &dyn Fn(usize) -> f32| -> Result<Tensor> {
            Tensor::from_vec((0..hidden).map(f).collect::<Vec<f32>>(), (1, hidden), &dev)
        };
        // `first` the embedding; with `then`, the embedding promoted and `then` the partial.
        let retrieve = |first: Tensor, then: Option<Tensor>, score: Tensor| -> Result<Vec<f32>> {
            let mut st = AttnResState::new(first, 2, 1e-5)?;
            if let Some(then) = then {
                st.maybe_close_block(0)?;
                st.accumulate(&then)?;
            }
            host(&st.retrieve(&score.flatten_all()?)?)
        };
        let got = retrieve(row(&|i| i as f32 * 3.0)?, None, row(&|i| i as f32)?)?;
        assert!(got.iter().enumerate().all(|(i, x)| (x - i as f32 * 3.0).abs() < 1e-5), "one source: {got:?}");
        // mean(1, 100) = 50.5, not the normalised keys' ~1.0.
        let got = retrieve(row(&|_| 1.0)?, Some(row(&|_| 100.0)?), row(&|_| 0.0)?)?;
        assert!(got.iter().all(|x| (x - 50.5).abs() < 1e-3), "a zero query: {got:?}");
        // The score at 1/32, so the logits are the four columns'.
        let (a, b, q) = ([1f32, 2.0, -1.0, 0.5], [-0.5f32, 1.0, 2.0, 1.5], [0.7f32, -0.2, 0.4, 0.1]);
        let alphas = |scale: f32| -> Result<Vec<f32>> {
            let got = retrieve(row(&|i| a[i % 4])?, Some(row(&|i| b[i % 4] * scale)?), row(&|i| q[i % 4] / 32.0)?)?;
            // got = α·a + (1 − α)·b: solved per component, all agree.
            Ok(got.iter().enumerate().map(|(i, g)| (g - b[i % 4] * scale) / (a[i % 4] - b[i % 4] * scale)).collect())
        };
        for (p, q) in alphas(1.0)?.iter().zip(alphas(1000.0)?) {
            assert!((p - q).abs() < 1e-4, "alpha moved with source scale: {p} vs {q}");
        }
        Ok(())
    }

    /// The report's 9 blocks: 93 layers at block size 12 promote 8 times, never the tail; a boundary promotes between
    /// its layer's two sub-layers.
    #[test]
    fn ninety_three_layers_reach_nine_sources() -> Result<()> {
        let dev = Device::Cpu;
        let mut st = AttnResState::new(Tensor::ones((1, 4), DType::F32, &dev)?, 12, 1e-5)?;
        let out = Tensor::ones((1, 4), DType::F32, &dev)?;
        let mut most = 0;
        for layer in 0..93 {
            assert_eq!(st.has_completed_blocks(), layer > 0, "only layer 0's first sub-layer has one source");
            let before = st.sources()?.len();
            st.maybe_close_block(layer)?;
            st.accumulate(&out)?; // the attention sub-layer
            let after = st.sources()?.len();
            if layer == 12 {
                assert_eq!((before, after), (2, 3), "layer 12: the embedding block and the partial, then two blocks");
            }
            most = most.max(after);
            st.accumulate(&out)?; // the FFN sub-layer
        }
        assert_eq!(st.blocks.len(), 8, "expected 8 promotions");
        assert_eq!(most.max(st.sources()?.len()), 9);
        Ok(())
    }
}
