use wgpu::include_wgsl;

use super::{Context, Shader};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineType {
    Points,
    PointsSized,
}

pub fn add_required_shaders(pipeline_type: PipelineType, context: &mut Context) {
    match pipeline_type {
        PipelineType::Points => {
            context.add_shader(Shader {
                name: "points",
                desc: include_wgsl!("../out/points.wgsl"),
            });
        }
        _ => {
            context.add_shader(Shader {
                name: "points",
                desc: include_wgsl!("../out/points.wgsl"),
            });
        }
    }
}

pub fn create_render_pipeline(
    pipeline_type: PipelineType,
    context: &Context,
    render_pipeline_layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    match pipeline_type {
        PipelineType::Points => create_points_pipeline(context, render_pipeline_layout),
        _ => create_points_pipeline(context, render_pipeline_layout),
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}

fn create_points_pipeline(
    context: &Context,
    render_pipeline_layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    let shader = context.get_shader("points").expect("Shader not found");
    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vertex_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        })
}
