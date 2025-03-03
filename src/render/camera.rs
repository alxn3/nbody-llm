use wgpu::util::DeviceExt;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[derive(Debug)]
struct CameraState {
    pub eye: glam::Vec3,
    pub target: glam::Vec3,
    pub up: glam::Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl CameraState {
    pub fn new(
        eye: glam::Vec3,
        target: glam::Vec3,
        up: glam::Vec3,
        aspect: f32,
        fovy: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            eye,
            target,
            up,
            aspect,
            fovy,
            znear,
            zfar,
        }
    }

    pub fn get_view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    pub fn get_projection_matrix(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh_gl(self.fovy, self.aspect, self.znear, self.zfar)
    }

    pub fn get_view_projection_matrix(&self) -> glam::Mat4 {
        self.get_projection_matrix() * self.get_view_matrix()
    }
}

#[derive(Debug)]
pub struct Camera {
    state: CameraState,
    pub uniform: CameraUniform,
    pub buffer: wgpu::Buffer,
}

impl Camera {
    pub fn new(
        device: &wgpu::Device,
        eye: glam::Vec3,
        target: glam::Vec3,
        up: glam::Vec3,
        aspect: f32,
        fovy: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        let state = CameraState::new(eye, target, up, aspect, fovy, znear, zfar);

        let uniform = CameraUniform {
            view_proj: state.get_view_projection_matrix().to_cols_array_2d(),
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            state,
            uniform,
            buffer,
        }
    }

    pub fn update_uniform(&mut self) {
        self.uniform.view_proj = self.state.get_view_projection_matrix().to_cols_array_2d();
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.state.aspect = aspect;
    }
}

pub trait CameraController: std::fmt::Debug {
    fn new() -> Self;
    fn process_input(&mut self, event: &WindowEvent);
    fn update_camera(&mut self, camera: &mut Camera);
}

#[derive(Debug)]
pub struct OrbitCameraController {
    yaw: f32,
    pitch: f32,
    last_cursor_pos: Option<winit::dpi::PhysicalPosition<f64>>,
    zoom: f32,
    pressed: bool,
}

impl CameraController for OrbitCameraController {
    fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            last_cursor_pos: None,
            zoom: 1.0,
            pressed: false,
        }
    }

    fn process_input(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                self.pressed = *state == winit::event::ElementState::Pressed
                    && *button == winit::event::MouseButton::Left;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if !self.pressed {
                    self.last_cursor_pos = None;
                    return;
                }
                if let Some(last_cursor_pos) = self.last_cursor_pos {
                    let delta = (
                        position.x - last_cursor_pos.x,
                        position.y - last_cursor_pos.y,
                    );
                    self.yaw -= delta.0 as f32 * 0.005;
                    self.pitch -= delta.1 as f32 * 0.005;
                    self.pitch = self.pitch.clamp(-1.5, 1.5);
                }
                self.last_cursor_pos = Some(*position);
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => {
                    self.zoom -= y * 0.001;
                    self.zoom = self.zoom.clamp(0.1, 10.0);
                }
                winit::event::MouseScrollDelta::PixelDelta(d) => {
                    self.zoom -= d.y as f32 * 0.001;
                    self.zoom = self.zoom.clamp(0.1, 10.0);
                }
            },
            
            _ => {}
        }
    }

    fn update_camera(&mut self, camera: &mut Camera) {
        let eye = glam::Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        ) * 5.0;
        camera.state.eye = eye * self.zoom;
        camera.uniform.view_proj = camera.state.get_view_projection_matrix().to_cols_array_2d();
    }
}
