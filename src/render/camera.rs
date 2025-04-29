use wgpu::util::DeviceExt;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
}

#[derive(Debug, Clone)]
pub struct CameraState {
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
    init_state: CameraState,
    pub uniform: CameraUniform,
    pub buffer: wgpu::Buffer,
}

impl Camera {
    #[allow(clippy::too_many_arguments)]
    pub fn new(device: &wgpu::Device, state: CameraState) -> Self {
        let uniform = CameraUniform {
            view: state.get_view_matrix().to_cols_array_2d(),
            proj: state.get_projection_matrix().to_cols_array_2d(),
            view_proj: state.get_view_projection_matrix().to_cols_array_2d(),
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            init_state: state.clone(),
            state,
            uniform,
            buffer,
        }
    }

    fn update_uniform(&mut self) {
        self.uniform.view = self.state.get_view_matrix().to_cols_array_2d();
        self.uniform.proj = self.state.get_projection_matrix().to_cols_array_2d();
        self.uniform.view_proj = self.state.get_view_projection_matrix().to_cols_array_2d();
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.state.aspect = aspect;
    }

    pub fn reset(&mut self) {
        self.init_state.aspect = self.state.aspect;
        self.state = self.init_state.clone();
    }
}

pub trait CameraController: std::fmt::Debug {
    fn new() -> Self;
    fn process_input(&mut self, event: &WindowEvent) -> bool;
    fn update_camera(&mut self, camera: &mut Camera);
    fn reset(&mut self);
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

    fn process_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                self.pressed = *state == winit::event::ElementState::Pressed
                    && *button == winit::event::MouseButton::Left;
                false
            }
            WindowEvent::CursorMoved { position, .. } => {
                if !self.pressed {
                    self.last_cursor_pos = None;
                    return false;
                }
                if let Some(last_cursor_pos) = self.last_cursor_pos {
                    let delta = (
                        position.x - last_cursor_pos.x,
                        position.y - last_cursor_pos.y,
                    );
                    self.yaw -= delta.0 as f32 * 0.005;
                    self.pitch += delta.1 as f32 * 0.005;
                    self.pitch = self.pitch.clamp(-1.5, 1.5);
                }
                self.last_cursor_pos = Some(*position);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => {
                    self.zoom -= y * 0.001;
                    self.zoom = self.zoom.clamp(0.1, 10.0);
                    true
                }
                winit::event::MouseScrollDelta::PixelDelta(d) => {
                    self.zoom -= d.y as f32 * 0.001;
                    self.zoom = self.zoom.clamp(0.1, 10.0);
                    true
                }
            },
            WindowEvent::Touch(winit::event::Touch {
                phase, location, ..
            }) => match phase {
                winit::event::TouchPhase::Started => {
                    self.pressed = true;
                    false
                }
                winit::event::TouchPhase::Ended => {
                    self.pressed = false;
                    self.last_cursor_pos = None;
                    false
                }
                winit::event::TouchPhase::Moved => {
                    if let Some(last_cursor_pos) = self.last_cursor_pos {
                        let delta = (
                            location.x - last_cursor_pos.x,
                            location.y - last_cursor_pos.y,
                        );
                        self.yaw -= delta.0 as f32 * 0.005;
                        self.pitch += delta.1 as f32 * 0.005;
                        self.pitch = self.pitch.clamp(-1.5, 1.5);
                    }
                    self.last_cursor_pos = Some(*location);
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn update_camera(&mut self, camera: &mut Camera) {
        let eye = glam::Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        ) * 5.0;
        camera.state.eye = eye * self.zoom;
        camera.update_uniform();
    }

    fn reset(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.zoom = 1.0;
    }
}
