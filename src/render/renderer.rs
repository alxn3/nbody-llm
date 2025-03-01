use std::sync::Arc;

use winit::window::Window;

use super::scene::Scene;

#[derive(Debug)]
pub struct Renderer {
    context: Context,
    depth_texture: wgpu::TextureView,
    scenes: Vec<Scene>,
}

impl Renderer {
    pub async fn init_async(window: Arc<Window>) -> Self {
        let context = Context::init_async(window).await;

        let depth_texture = context
            .create_depth_texture(context.surface_config.width, context.surface_config.height);

        log::info!("Renderer initialized");

        let format = context.surface_config.format;
        let device = context.device.clone();

        Self {
            context,
            depth_texture,
            scenes: vec![Scene::new(&device, format)],
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.context.resize(size);
        self.depth_texture = self.context.create_depth_texture(size.width, size.height);
    }

    pub fn add_scene(&mut self, scene: Scene) {
        self.scenes.push(scene);
    }

    pub fn render(&mut self) {
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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
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

            for scene in &self.scenes {
                scene.render(&mut render_pass);
            }
        }
        self.context.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }
}

#[derive(Debug)]
struct Context {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
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
}
