mod renderer;
mod scene;

pub use renderer::*;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
