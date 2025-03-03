use std::{collections::HashMap, sync::Arc};

use wgpu::{ShaderModule, util::DeviceExt};
use winit::{event::WindowEvent, window::Window};

use super::{
    Camera, CameraController, Drawable, OrbitCameraController, add_required_shaders,
    create_render_pipeline, pipeline::PipelineType,
};

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct WorldUniform {
    resolution: [f32; 2],
    _padding: [f32; 2],
}

#[derive(Debug)]
pub struct RenderObject {
    pub pipeline_type: PipelineType,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

#[derive(Debug)]
pub struct Renderer<CameraController = OrbitCameraController> {
    pub context: Context,
    depth_texture: wgpu::TextureView,
    pipelines: HashMap<PipelineType, wgpu::RenderPipeline>,
    camera: Camera,
    camera_controller: CameraController,
    resolution_uniform: WorldUniform,
    resolution_buffer: wgpu::Buffer,
    world_bind_group: wgpu::BindGroup,
    world_bind_group_layout: wgpu::BindGroupLayout,
}

impl<C: CameraController> Renderer<C> {
    pub async fn init_async(window: Arc<Window>) -> Self {
        let context = Context::init_async(window).await;

        let width = context.surface_config.width;
        let height = context.surface_config.height;

        let depth_texture = context.create_depth_texture(width, height);

        log::info!("Renderer initialized");

        let camera = Camera::new(
            &context.device,
            (0.0, 1.0, 2.0).into(),
            (0.0, 0.0, 0.0).into(),
            (0.0, 1.0, 0.0).into(),
            width as f32 / height as f32,
            45.0,
            0.1,
            100.0,
        );

        let camera_controller = C::new();

        let resolution_uniform = WorldUniform {
            resolution: [width as f32, height as f32],
            _padding: [0.0; 2],
        };

        let resolution_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("World Buffer"),
                contents: bytemuck::cast_slice(&[resolution_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let world_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("World Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let world_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("World Bind Group"),
                layout: &world_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: resolution_buffer.as_entire_binding(),
                    },
                ],
            });

        Self {
            context,
            depth_texture,
            pipelines: HashMap::new(),
            camera,
            camera_controller,
            resolution_uniform,
            resolution_buffer,
            world_bind_group,
            world_bind_group_layout,
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.context.resize(size);
        self.depth_texture = self.context.create_depth_texture(size.width, size.height);
        self.camera
            .set_aspect(size.width as f32 / size.height as f32);
        self.resolution_uniform.resolution = [size.width as f32, size.height as f32];
        self.context.queue.write_buffer(
            &self.resolution_buffer,
            0,
            bytemuck::cast_slice(&[self.resolution_uniform]),
        );
    }

    pub fn process_input(&mut self, event: &WindowEvent) {
        self.camera_controller.process_input(event);
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.context.queue.write_buffer(
            &self.camera.buffer,
            0,
            bytemuck::cast_slice(&[self.camera.uniform]),
        );
    }

    pub fn render(&mut self, objects: Vec<&mut dyn Drawable>) {
        let Ok(surface_texture) = self.context.surface.get_current_texture() else {
            return;
        };
        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let mut render_groups: HashMap<PipelineType, Vec<&mut dyn Drawable>> = HashMap::new();

            for obj in objects {
                render_groups
                    .entry(obj.get_pipeline_type())
                    .or_insert_with(Vec::new)
                    .push(obj);
            }

            for (pipeline_type, objects) in render_groups {
                let pipeline = self.get_pipeline(pipeline_type);

                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &self.world_bind_group, &[]);

                for obj in objects {
                    obj.draw(&mut render_pass, &mut self.context.queue);
                }
            }
        }
        self.context.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }

    fn get_pipeline(&mut self, pipeline_type: PipelineType) -> &wgpu::RenderPipeline {
        if self.pipelines.contains_key(&pipeline_type) {
            return self.pipelines.get(&pipeline_type).unwrap();
        }

        add_required_shaders(pipeline_type, &mut self.context);
        let pipeline = create_render_pipeline(
            pipeline_type,
            &self.context,
            &self
                .context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &self.world_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                }),
        );

        self.pipelines.insert(pipeline_type, pipeline);
        self.pipelines.get(&pipeline_type).unwrap()
    }
}

#[derive(Debug)]
pub struct Context {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    shaders: HashMap<&'static str, ShaderModule>,
}

impl Context {
    pub async fn init_async(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Unable to find a suitable GPU adapter!");

        #[cfg(target_arch = "wasm32")]
        let limits = if let Some(win) = web_sys::window() {
            if win.navigator().gpu().is_object() {
                wgpu::Limits::default().using_resolution(adapter.limits())
            } else {
                wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits())
            }
        } else {
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits())
        };
        #[cfg(not(target_arch = "wasm32"))]
        let limits = wgpu::Limits::default().using_resolution(adapter.limits());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    memory_hints: wgpu::MemoryHints::default(),
                    required_features: wgpu::Features::default(),
                    required_limits: limits,
                },
                None,
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        let size = window.inner_size();
        let mut surface_config = surface
            .get_default_config(&adapter, size.width, size.height)
            .expect("Surface configuration not supported by adapter");
        surface_config
            .view_formats
            .push(surface_config.format.remove_srgb_suffix());

        surface.configure(&device, &surface_config);

        log::info!("Context initialized");

        Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            surface,
            surface_config,
            shaders: HashMap::new(),
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.surface_config.width = size.width;
        self.surface_config.height = size.height;
        self.surface.configure(&self.device, &self.surface_config);
        log::info!("Resized to {:?}", size);
    }

    pub fn create_depth_texture(&self, width: u32, height: u32) -> wgpu::TextureView {
        let texture = self.device.create_texture(
            &(wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: super::DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }),
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn add_shader(&mut self, shader: super::Shader) -> &ShaderModule {
        if self.shaders.contains_key(shader.name) {
            return self.shaders.get(shader.name).unwrap();
        }
        self.shaders
            .insert(shader.name, self.device.create_shader_module(shader.desc));
        self.shaders.get(shader.name).unwrap()
    }

    pub fn get_shader(&self, name: &'static str) -> Option<&ShaderModule> {
        self.shaders.get(name)
    }
}
