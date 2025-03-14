mod camera;
mod pipeline;
mod renderer;

pub use camera::*;
pub use pipeline::*;
pub use renderer::*;
use wgpu::util::DeviceExt;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Shader<'a> {
    pub name: &'static str,
    pub desc: wgpu::ShaderModuleDescriptor<'a>,
    // pub entry_points: &'a [(&'a str, ShaderType)],
}

#[derive(Debug, Clone)]
pub struct BufferWrapper {
    pub buffer: wgpu::Buffer,
    pub label: Option<&'static str>,
    pub usage: wgpu::BufferUsages,
}

impl BufferWrapper {
    pub fn new<T: bytemuck::Pod>(
        device: &wgpu::Device,
        label: Option<&'static str>,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: bytemuck::cast_slice(data),
            usage,
        });

        Self {
            buffer,
            label,
            usage,
        }
    }

    pub fn update<T: bytemuck::Pod>(&mut self, context: &Context, data: &[T]) {
        let data: &[u8] = bytemuck::cast_slice(data);
        if self.buffer.size() < data.len() as wgpu::BufferAddress {
            self.buffer = context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: self.label,
                    contents: data,
                    usage: self.usage,
                });
        } else {
            context.queue.write_buffer(&self.buffer, 0, data);
        }
    }
}
