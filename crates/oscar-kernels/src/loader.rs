//! The kernels' launch vocabulary: the shader families and their pipelines, a tensor's binding, outputs, the dispatch.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use candle_core::{DType, MetalDevice, MetalStorage, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{
    Buffer, ComputeCommandEncoder, ComputePipeline, ConstantValues, Device, Library, Value,
};
use candle_metal_kernels::source::Source;
use candle_metal_kernels::utils::EncoderParam;
use objc2_metal::{MTLCompileOptions, MTLDevice as _, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};

pub(crate) use candle_metal_kernels::set_params;

/// A metallib from `build.rs`, or source compiled at first use, both under the fast math the byte-exact gates assume.
enum Shader {
    Metallib(&'static [u8]),
    Source(&'static str),
}

#[derive(Clone, Copy)]
pub(crate) enum Family {
    Moe,
    Gdn,
    Mla,
    Sampler,
    Paged,
}

fn error(what: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(what.to_string())
}

/// Candle's options, Metal's defaults set explicitly: a new default moves no bytes, a row kernel lowers as candle's.
pub(crate) fn compile(device: &Device, source: &str) -> Result<Library> {
    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Fast);
    opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
    device.new_library_with_source(source, Some(&opts)).map_err(|e| error(format!("runtime compile: {e}")))
}

/// `constants` the kernel's boolean function constants by index.
pub(crate) fn make_pipeline(
    device: &Device, lib: &Library, name: &str, constants: &[(usize, bool)],
) -> Result<ComputePipeline> {
    let values = ConstantValues::new(constants.iter().map(|&(i, b)| (i, Value::Bool(b))).collect());
    let values = (!constants.is_empty()).then_some(&values);
    let func = lib.get_function(name, values).map_err(|e| error(format!("pipeline {name}: {e}")))?;
    let p =
        device.new_compute_pipeline_state_with_function(&func).map_err(|e| error(format!("pipeline {name}: {e}")))?;
    Ok(p.with_name(name))
}

struct Kernels {
    shader: Shader,
    library: OnceLock<Library>,
    pipelines: RwLock<HashMap<(String, Vec<(usize, bool)>), ComputePipeline>>,
}

impl Kernels {
    fn library(&self, device: &Device) -> Result<Library> {
        if let Some(lib) = self.library.get() {
            return Ok(lib.clone());
        }
        let lib = match self.shader {
            Shader::Metallib(bytes) => {
                let data = dispatch2::DispatchData::from_static_bytes(bytes);
                let raw_lib = device
                    .as_ref()
                    .newLibraryWithData_error(&data)
                    .map_err(|e| error(format!("Failed to load precompiled metallib: {e}")))?;
                Library::new(raw_lib)
            }
            Shader::Source(source) => compile(device, source)?,
        };
        Ok(self.library.get_or_init(|| lib).clone())
    }

    fn load_pipeline(&self, device: &Device, name: &str, constants: &[(usize, bool)]) -> Result<ComputePipeline> {
        let key = (name.to_string(), constants.to_vec());
        if let Some(p) = self.pipelines.read().map_err(error)?.get(&key) {
            return Ok(p.clone());
        }
        let mut pipelines = self.pipelines.write().map_err(error)?;
        if let Some(p) = pipelines.get(&key) {
            return Ok(p.clone());
        }
        let pipeline = make_pipeline(device, &self.library(device)?, name, constants)?;
        pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}

pub(crate) fn pipeline(dev: &MetalDevice, family: Family, name: &str) -> Result<ComputePipeline> {
    pipeline_with(dev, family, name, &[])
}

/// A pipeline under `constants`, the kernel's boolean function constants by index.
pub(crate) fn pipeline_with(
    dev: &MetalDevice, family: Family, name: &str, constants: &[(usize, bool)],
) -> Result<ComputePipeline> {
    static FAMILIES: [OnceLock<Kernels>; 5] = [const { OnceLock::new() }; 5];
    let shader = match family {
        Family::Moe => Shader::Metallib(include_bytes!(concat!(env!("OUT_DIR"), "/oscar_kernels.metallib"))),
        // Source-only: precompiling would change the compile path its byte-exact gate assumes.
        Family::Gdn => Shader::Source(include_str!("gdn.metal")),
        // Compiled from source as `mistralrs-quant` compiled it, which its byte-exact results assume.
        Family::Mla => Shader::Source(include_str!("mla.metal")),
        Family::Sampler => Shader::Source(include_str!("sampler.metal")),
        Family::Paged => Shader::Metallib(include_bytes!(concat!(env!("OUT_DIR"), "/oscar_paged.metallib"))),
    };
    let kernels = || Kernels { shader, library: OnceLock::new(), pipelines: RwLock::new(HashMap::new()) };
    FAMILIES[family as usize].get_or_init(kernels).load_pipeline(dev.device(), name, constants)
}

pub(crate) fn candle_pipeline(dev: &MetalDevice, name: &str) -> Result<ComputePipeline> {
    dev.kernels()
        .load_pipeline(dev.device(), Source::Quantized, name.to_string())
        .map_err(|e| candle_core::Error::Msg(format!("load_pipeline({name}): {e:?}")))
}

/// A tensor as a launch binds it.
pub struct Metal<'a> {
    pub dev: &'a MetalDevice,
    pub buf: Buffer,
    /// The tensor's start in `buf`, in bytes.
    pub off: usize,
}

impl<'a> Metal<'a> {
    pub fn of(t: &'a Tensor, what: &str) -> Result<Self> {
        let candle_core::Device::Metal(dev) = t.device() else { candle_core::bail!("{what} must be on Metal") };
        let (storage, layout) = t.storage_and_layout();
        let Storage::Metal(ms) = &*storage else { candle_core::bail!("{what} must be on Metal") };
        Ok(Self { dev, buf: ms.buffer().clone(), off: layout.start_offset() * t.dtype().size_in_bytes() })
    }
}

impl EncoderParam for &Metal<'_> {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, m: Self) {
        encoder.set_buffer(position, Some(&m.buf), m.off);
    }
}

/// A tensor over a Metal buffer, no copy.
pub fn tensor_over(buffer: impl Into<Arc<Buffer>>, dev: &MetalDevice, shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let shape = shape.into();
    Tensor::from((Storage::Metal(MetalStorage::new(buffer.into(), dev.clone(), shape.elem_count(), dtype)), shape))
}

/// A new output and its binding, unfilled: a zeroed kernel output is a host fill in the working region.
pub(crate) fn out<'a>(
    dev: &'a MetalDevice, shape: impl Into<Shape>, dtype: DType, label: &'static str,
) -> Result<(Tensor, Metal<'a>)> {
    let shape = shape.into();
    let buf = dev.new_buffer(shape.elem_count(), dtype, label)?;
    let m = Metal { dev, buf: (*buf).clone(), off: 0 };
    Ok((tensor_over(buf, dev, shape, dtype), m))
}

pub(crate) fn size([width, height, depth]: [usize; 3]) -> MTLSize {
    MTLSize { width, height, depth }
}

/// A dispatch of `grid` threadgroups of `tg` threads, on the calling thread's stream so no buffer crosses queues.
pub(crate) fn launch(
    dev: &MetalDevice, p: &ComputePipeline, label: &str, grid: [usize; 3], tg: [usize; 3],
    bind: impl FnOnce(&ComputeCommandEncoder) -> Result<()>,
) -> Result<()> {
    let e = dev.command_encoder()?;
    let e: &ComputeCommandEncoder = &e;
    e.set_label(label);
    e.set_compute_pipeline_state(p);
    bind(e)?;
    e.dispatch_thread_groups(size(grid), size(tg));
    Ok(())
}
