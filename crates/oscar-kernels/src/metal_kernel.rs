//! Row kernels written as their Metal body at the call site: a chain of candle ops in one body agrees to the byte.

use candle_core::{DType, MetalDevice, Result, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{OnceLock, RwLock};

use crate::loader::{launch, out, Metal};

/// The body sees each named input and output at its row, `D`, `tid`, `block_dim`, `shm`, `row_sum` and `row_sum_tree`.
pub struct RowKernel<'a> {
    pub name: &'static str,
    pub inputs: &'a [&'static str],
    pub outputs: &'a [&'static str],
    pub body: &'a str,
}

/// How an input lies against the output's `(rows, D)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Lay {
    Full,
    Row,
    Col,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Input {
    lay: Lay,
    bf16: bool,
}

impl Input {
    fn of(t: &Tensor, rows: usize, d: usize, name: &str) -> Result<Self> {
        let bf16 = match t.dtype() {
            DType::F32 => false,
            DType::BF16 => true,
            other => candle_core::bail!("row kernel: {name} must be F32 or BF16, got {other:?}"),
        };
        let n = t.elem_count();
        let last = t.dims().last().copied().unwrap_or(1);
        let lay = if n == rows * d && last == d {
            Lay::Full
        } else if n == d && last == d {
            Lay::Row
        } else if n == rows && last == 1 {
            Lay::Col
        } else {
            candle_core::bail!(
                "row kernel: {name} has shape {:?}, expected ({rows}, {d}), ({d},) or ({rows}, 1)",
                t.dims()
            )
        };
        Ok(Self { lay, bf16 })
    }
    fn ty(self) -> &'static str {
        if self.bf16 {
            "bfloat"
        } else {
            "float"
        }
    }
    fn offset(self) -> &'static str {
        match self.lay {
            Lay::Full => "(size_t)row * D",
            Lay::Row => "0",
            Lay::Col => "row",
        }
    }
}

const PRELUDE: &str = r#"#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)
#pragma clang fp reassociate(off)

// candle's fast_sum over one row, the sum returned to every thread; block_dim is a power of two of at least 64.
// The last barrier lets a body call it again on the same scratch: else its first write races a lagging thread's read.
static inline float row_sum(float v, threadgroup float *shm, uint tid, uint block_dim) {
  shm[tid] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = block_dim / 2; s >= 64; s >>= 1) {
    if (tid < s) shm[tid] = shm[tid] + shm[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float r = 0.0f;
  if (tid < 32) {
    r = shm[tid] + shm[tid + 32];
    simdgroup_barrier(mem_flags::mem_none);
    r = simd_sum(r);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) shm[0] = r;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = shm[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

// candle's rmsnorm reduction over one row: the shared-memory tree all the way down, the sum to every thread.
static inline float row_sum_tree(float v, threadgroup float *shm, uint tid, uint block_dim) {
  shm[tid] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = block_dim / 2; s > 0; s >>= 1) {
    if (tid < s) shm[tid] = shm[tid] + shm[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = shm[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}
"#;

fn source(k: &RowKernel<'_>, ins: &[Input]) -> String {
    let mut s = String::from(PRELUDE);
    s.push_str(&format!("\n[[kernel]] void {}(\n", k.name));
    let mut slot = 0;
    for (name, i) in k.inputs.iter().zip(ins) {
        s.push_str(&format!("    const device {} *{name}_rows [[buffer({slot})]],\n", i.ty()));
        slot += 1;
    }
    for name in k.outputs {
        s.push_str(&format!("    device float *{name}_rows [[buffer({slot})]],\n"));
        slot += 1;
    }
    s.push_str(&format!(
        "    constant int &D [[buffer({slot})]],\n    threadgroup float *shm [[threadgroup(0)]],\n    \
         uint row [[threadgroup_position_in_grid]],\n    uint tid [[thread_position_in_threadgroup]],\n    \
         uint block_dim [[threads_per_threadgroup]]) {{\n"
    ));
    for (name, i) in k.inputs.iter().zip(ins) {
        s.push_str(&format!("  const device {} *{name} = {name}_rows + {};\n", i.ty(), i.offset()));
    }
    for name in k.outputs {
        s.push_str(&format!("  device float *{name} = {name}_rows + (size_t)row * D;\n"));
    }
    s.push_str(k.body);
    s.push_str("\n}\n");
    s
}

fn pipeline(dev: &MetalDevice, k: &RowKernel<'_>, ins: &[Input]) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<RwLock<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| RwLock::new(HashMap::new()));
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (k.name, k.body, ins).hash(&mut h);
    let key = h.finish();
    if let Some(p) = cache.read().map_err(|e| candle_core::Error::Msg(e.to_string()))?.get(&key) {
        return Ok(p.clone());
    }
    // The pragmas in `source` keep contraction and reassociation off.
    let named = |e: candle_core::Error| candle_core::Error::Msg(format!("row kernel {}: {e}", k.name));
    let lib = crate::loader::compile(dev.device(), &source(k, ins)).map_err(named)?;
    let p = crate::loader::make_pipeline(dev.device(), &lib, k.name, &[]).map_err(named)?;
    cache.write().map_err(|e| candle_core::Error::Msg(e.to_string()))?.insert(key, p.clone());
    Ok(p)
}

impl RowKernel<'_> {
    /// One `(rows, d)` F32 tensor per output, in `outputs`' order.
    pub fn run(&self, inputs: &[&Tensor], rows: usize, d: usize) -> Result<Vec<Tensor>> {
        refuse_if!(
            inputs.len() != self.inputs.len(),
            "row kernel {}: {} inputs named, {} given",
            self.name,
            self.inputs.len(),
            inputs.len()
        );
        let Some(first) = inputs.first() else {
            candle_core::bail!("row kernel {}: no inputs", self.name);
        };
        let dev = Metal::of(first, self.name)?.dev;
        let mut owned = Vec::with_capacity(inputs.len());
        let mut ins = Vec::with_capacity(inputs.len());
        for (t, name) in inputs.iter().zip(self.inputs) {
            ins.push(Input::of(t, rows, d, name)?);
            owned.push(t.contiguous()?);
        }
        let pipeline = pipeline(dev, self, &ins)?;
        // candle's reduce block width, so `row_sum`'s tree is candle's: a power of two under any pipeline limit.
        let block_dim = 1usize << pipeline.max_total_threads_per_threadgroup().min((d / 2).next_power_of_two()).ilog2();
        refuse_if!(block_dim < 64, "row kernel {}: D = {d} is below the 128 the row sum needs", self.name);
        let outputs =
            self.outputs.iter().map(|name| Ok(out(dev, (rows, d), DType::F32, name)?.0)).collect::<Result<Vec<_>>>()?;
        let bound = owned
            .iter()
            .zip(self.inputs)
            .chain(outputs.iter().zip(self.outputs))
            .map(|(t, name)| Metal::of(t, name))
            .collect::<Result<Vec<_>>>()?;
        launch(dev, &pipeline, self.name, [rows, 1, 1], [block_dim, 1, 1], |e| {
            for (slot, m) in bound.iter().enumerate() {
                e.set_buffer(slot, Some(&m.buf), m.off);
            }
            e.set_bytes(bound.len(), &(d as i32));
            e.set_threadgroup_memory_length(0, block_dim * std::mem::size_of::<f32>());
            Ok(())
        })?;
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_bitwise, fill};

    /// The `fma` grouping candle's F32 `rope` compiles to, which a row kernel that ropes in place must use; the row
    /// sum at every block width (64-1024) and the models' elements-per-thread counts.
    #[test]
    fn rope_fma_grouping_and_the_row_sum_are_candles_to_the_bit() -> Result<()> {
        let dev = oscar_offload::pools::test_device()?;
        let (rows, d) = (4096usize, 128usize);
        let x = Tensor::from_vec(fill(rows * d, 1), (1, rows, 1, d), &dev)?;
        let cos = Tensor::from_vec(fill(d / 2, 2), (1, d / 2), &dev)?;
        let sin = Tensor::from_vec(fill(d / 2, 3), (1, d / 2), &dev)?;
        let want = candle_nn::rotary_emb::rope(&x, &cos, &sin)?.flatten_all()?.to_vec1::<f32>()?;
        let k = RowKernel {
            name: "rope_grouping",
            inputs: &["x", "cs", "sn"],
            outputs: &["out"],
            body: "  const int hr = D / 2;\n  for (int j = tid; j < D; j += block_dim) {\n    const int i = j < hr ? j : j - hr;\n    \
                   const float x1 = x[i]; const float x2 = x[i + hr]; const float c = cs[i]; const float s = sn[i];\n    \
                   out[j] = j < hr ? fma(x1, c, -(x2 * s)) : fma(x2, c, x1 * s);\n  }\n",
        };
        // The facility wants (D,) rows; the body reads only the first D/2.
        let pad = |t: &Tensor| -> Result<Tensor> {
            Tensor::cat(&[t.flatten_all()?, Tensor::zeros(d / 2, DType::F32, &dev)?], 0)
        };
        let got = k.run(&[&x.reshape((rows, d))?, &pad(&cos)?, &pad(&sin)?], rows, d)?.remove(0);
        assert_bitwise(&got.flatten_all()?.to_vec1::<f32>()?, &want, "rope");
        let k = RowKernel {
            name: "row_sum_probe",
            inputs: &["x"],
            outputs: &["out"],
            body: "  float acc = 0.0f;\n  for (int i = tid; i < D; i += block_dim) acc = acc + x[i];\n  \
                   const float s = row_sum(acc, shm, tid, block_dim);\n  \
                   for (int i = tid; i < D; i += block_dim) out[i] = s;\n",
        };
        let rows = 2048;
        for d in [128usize, 200, 2048, 3584, 4096, 6144, 7168] {
            let v: Vec<f32> = (0..rows * d).map(|i| ((i * 7919 % 10007) as f32 - 5003.0) / 977.0).collect();
            let x = Tensor::from_vec(v, (rows, d), &dev)?;
            let want = x.sum_keepdim(1)?.flatten_all()?.to_vec1::<f32>()?;
            let got: Vec<f32> = k.run(&[&x], rows, d)?.remove(0).to_vec2::<f32>()?.iter().map(|row| row[0]).collect();
            assert_bitwise(&got, &want, &format!("d={d}"));
        }
        Ok(())
    }
}
