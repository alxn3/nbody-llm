mod camera;
mod pipeline;
mod renderer;

pub use camera::*;
pub use pipeline::*;
pub use renderer::*;
use wgpu::RenderPass;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Shader<'a> {
    pub name: &'static str,
    pub desc: wgpu::ShaderModuleDescriptor<'a>,
    // pub entry_points: &'a [(&'a str, ShaderType)],
}

pub trait Drawable: std::fmt::Debug {
    fn init(&mut self, context: &mut Context);
    fn get_pipeline_type(&self) -> PipelineType;
    fn get_vertex_buffer(&self) -> &wgpu::Buffer;
    fn get_instance_buffer(&self) -> Option<&wgpu::Buffer> {
        None
    }
    fn get_num_instances(&self) -> u32 {
        1
    }
    fn get_index_buffer(&self) -> &wgpu::Buffer;
    fn get_num_indices(&self) -> u32;
    fn buffer_needs_update(&self) -> bool;
    fn update_buffers(&mut self, queue: &mut wgpu::Queue);
    fn draw(&mut self, render_pass: &mut RenderPass, queue: &mut wgpu::Queue) {
        if self.buffer_needs_update() {
            self.update_buffers(queue);
        }
        render_pass.set_vertex_buffer(0, self.get_vertex_buffer().slice(..));
        if let Some(instance_buffer) = self.get_instance_buffer() {
            render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        }
        render_pass.set_index_buffer(self.get_index_buffer().slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.get_num_indices(), 0, 0..self.get_num_instances());
    }
}
