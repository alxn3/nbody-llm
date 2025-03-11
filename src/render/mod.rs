mod camera;
mod pipeline;
mod renderer;

pub use camera::*;
pub use pipeline::*;
pub use renderer::*;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Shader<'a> {
    pub name: &'static str,
    pub desc: wgpu::ShaderModuleDescriptor<'a>,
    // pub entry_points: &'a [(&'a str, ShaderType)],
}
