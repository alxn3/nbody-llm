use std::sync::Arc;
use web_time::{Duration, Instant};

use nlib::{
    render::Renderer,
    shared::{Float, Integrator, Particle, Simulation},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    window::{Theme, Window, WindowId},
};

#[derive(Debug)]
enum UserEvent {
    #[cfg(target_arch = "wasm32")]
    WebInitialized(Renderer),
}

const FRAME_SAMPLES: usize = 60;

#[derive(Debug)]
struct SimulationState<F: Float, const D: usize, P, I, S>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
    S: Simulation<F, D, P, I>,
{
    pub start_time: Instant,
    pub step_count: u64,
    pub max_fps: f64,
    pub frame_count: u64,
    pub frame_list: [Duration; FRAME_SAMPLES],
    pub frame_index: usize,
    pub max_steps_per_frame: u64,
    pub starting_dt: F,
    pub starting_g: F,
    pub starting_g_soft: F,
    pub paused: bool,
    pub step_by: u64,
    _phantom: std::marker::PhantomData<(F, P, I, S)>,
}

impl<F: Float, const D: usize, P, I, S> SimulationState<F, D, P, I, S>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
    S: Simulation<F, D, P, I>,
{
    pub fn set_starting_values(&mut self, sim: &S)
    where
        F: Float,
        P: Particle<F, D>,
        I: Integrator<F, D, P>,
    {
        self.starting_dt = sim.dt();
        self.starting_g = sim.g();
        self.starting_g_soft = sim.g_soft();
    }

    pub fn update_frame_time(&mut self, frame_time: Duration) {
        self.frame_list[self.frame_index] = frame_time;
        self.frame_index = (self.frame_index + 1) % FRAME_SAMPLES;
    }

    pub fn get_frame_time(&self) -> Duration {
        self.frame_list.iter().sum()
    }

    pub fn get_fps(&self) -> f64 {
        let frame_time = self.get_frame_time();
        if frame_time == Duration::ZERO {
            return 0.0;
        }
        1.0 / (frame_time.as_secs_f64() / FRAME_SAMPLES as f64)
    }

    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.step_count = 0;
        self.frame_count = 0;
    }
}

struct App<F: Float, const D: usize, P, I, S>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
    S: Simulation<F, D, P, I>,
{
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    event_proxy: Arc<EventLoopProxy<UserEvent>>,
    gui_state: Option<egui_winit::State>,
    state: SimulationState<F, D, P, I, S>,
    simulation: S,
    _phantom: std::marker::PhantomData<(F, P, I)>,
}

impl<F: Float, const D: usize, P, I, S> App<F, D, P, I, S>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
    S: Simulation<F, D, P, I>,
{
    fn new(event_proxy: Arc<EventLoopProxy<UserEvent>>, simulation: S) -> Self {
        let mut state = SimulationState {
            start_time: Instant::now(),
            step_count: 0,
            max_fps: 60.0,
            frame_count: 0,
            frame_list: [Duration::ZERO; FRAME_SAMPLES],
            frame_index: 0,
            max_steps_per_frame: 100,
            starting_dt: F::from(0.0).unwrap(),
            starting_g: F::from(0.0).unwrap(),
            starting_g_soft: F::from(0.0).unwrap(),
            paused: false,
            step_by: 100,
            _phantom: std::marker::PhantomData,
        };

        state.set_starting_values(&simulation);

        Self {
            window: None,
            renderer: None,
            event_proxy,
            gui_state: None,
            state,
            simulation,
            _phantom: std::marker::PhantomData,
        }
    }

    fn init(&mut self) {
        self.simulation.init();
        self.simulation
            .init_drawables(&mut self.renderer.as_mut().unwrap().context);
        self.state.start_time = Instant::now();
    }
}

// There are 1_000_000_000 nanoseconds in a second.
const NANOS_PER_SECOND: u64 = 1_000_000_000;

impl<F: Float, const D: usize, P, I, S> ApplicationHandler<UserEvent> for App<F, D, P, I, S>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
    S: Simulation<F, D, P, I>,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let mut attributes = Window::default_attributes();
        let gui_context = egui::Context::default();

        #[cfg(not(target_arch = "wasm32"))]
        {
            attributes = attributes.with_title("N-body simulation");
        }

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            let canvas = wgpu::web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("canvas")
                .unwrap()
                .dyn_into::<wgpu::web_sys::HtmlCanvasElement>()
                .unwrap();
            attributes = attributes.with_canvas(Some(canvas));
        }

        self.window = Some(Arc::new(event_loop.create_window(attributes).unwrap()));
        let window = self.window.as_ref().unwrap().clone();

        let viewport_id = gui_context.viewport_id();
        self.gui_state = Some(egui_winit::State::new(
            gui_context,
            viewport_id,
            &window,
            Some(window.scale_factor() as _),
            Some(Theme::Dark),
            None,
        ));

        #[cfg(target_arch = "wasm32")]
        {
            // window.request_redraw();
            let proxy = self.event_proxy.clone();
            wasm_bindgen_futures::spawn_local(async move {
                proxy
                    .send_event(UserEvent::WebInitialized(
                        Renderer::init_async(window).await,
                    ))
                    .unwrap();
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.renderer = Some(pollster::block_on(Renderer::init_async(window)));
            self.init();
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            #[cfg(target_arch = "wasm32")]
            UserEvent::WebInitialized(renderer) => {
                self.renderer = Some(renderer);
                self.window.as_ref().unwrap().request_redraw();
                self.init();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(window), Some(renderer), Some(gui_state)) = (
            self.window.as_mut(),
            self.renderer.as_mut(),
            self.gui_state.as_mut(),
        ) else {
            log::error!("Window or renderer not initialized");
            return;
        };

        if gui_state.on_window_event(window, &event).consumed {
            match event {
                WindowEvent::MouseInput {
                    device_id: _,
                    state: winit::event::ElementState::Released,
                    button: winit::event::MouseButton::Left,
                } => {}
                _ => {
                    return;
                }
            }
        }

        log::trace!("Window event: {:?}", event);

        renderer.process_input(&event);

        match event {
            WindowEvent::Resized(size) => {
                log::info!("Resized to {:?}", size);
                renderer.resize(size);
            }
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                window.request_redraw();
                let start = Instant::now();
                let gui_input = gui_state.take_egui_input(window);
                gui_state.egui_ctx().begin_pass(gui_input);

                egui::Window::new("Simulation")
                    .max_width(100.0)
                    .show(gui_state.egui_ctx(), |ui| {
                        egui::Grid::new("stats")
                            .min_col_width(70.0)
                            .spacing([20.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("FPS");
                                ui.with_layout(
                                    egui::Layout::top_down_justified(egui::Align::Max),
                                    |ui| {
                                        ui.label(format!("{:.2}", self.state.get_fps()));
                                    },
                                );
                                ui.end_row();
                                ui.label("Last draw time");
                                ui.with_layout(
                                    egui::Layout::top_down_justified(egui::Align::Max),
                                    |ui| {
                                        ui.label(format!("{:.05}", self.state.frame_list[(FRAME_SAMPLES + self.state.frame_index - 1) % FRAME_SAMPLES].as_secs_f64()));
                                    },
                                );
                                ui.end_row();
                                ui.label("Time");
                                ui.with_layout(
                                    egui::Layout::top_down_justified(egui::Align::Max),
                                    |ui| {
                                        ui.label(format!("{:.05}", self.simulation.elapsed()));
                                    },
                                );
                                ui.end_row();
                                ui.label("Steps");
                                ui.with_layout(
                                    egui::Layout::top_down_justified(egui::Align::Max),
                                    |ui| {
                                        ui.label(format!("{}", self.state.step_count));
                                    },
                                );
                                ui.end_row();
                            });

                        ui.add_space(10.0);

                        ui.collapsing("Settings", |ui| {
                            egui::Grid::new("grid").show(ui, |ui| {
                                ui.label("Max FPS");
                                ui.add(
                                    egui::Slider::new(&mut self.state.max_fps, 1.0..=240.0)
                                        .trailing_fill(true)
                                        .clamping(egui::SliderClamping::Never),
                                );
                                ui.end_row();
                                ui.label("Max Steps/F");
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.state.max_steps_per_frame,
                                        1..=1000,
                                    )
                                    .trailing_fill(true)
                                    .clamping(egui::SliderClamping::Never),
                                );
                                ui.end_row();
                                ui.separator();
                                ui.end_row();
                                ui.label("dt");
                                ui.add(
                                    egui::Slider::new(
                                        self.simulation.dt_mut(),
                                        F::from(-0.005).unwrap()..=F::from(0.005).unwrap(),
                                    )
                                    .logarithmic(true)
                                    .handle_shape(egui::style::HandleShape::Rect {
                                        aspect_ratio: 0.50,
                                    })
                                    .clamping(egui::SliderClamping::Never),
                                );
                                ui.end_row();
                                ui.label("G");
                                ui.add(
                                    egui::Slider::new(
                                        self.simulation.g_mut(),
                                        F::from(-1.0).unwrap()..=F::from(2.0).unwrap(),
                                    )
                                    .handle_shape(egui::style::HandleShape::Rect {
                                        aspect_ratio: 0.50,
                                    })
                                    .clamping(egui::SliderClamping::Never),
                                );
                                ui.end_row();
                                ui.label("Softening");
                                ui.add(
                                    egui::Slider::new(
                                        self.simulation.g_soft_mut(),
                                        F::from(0.0).unwrap()..=F::from(0.5).unwrap(),
                                    )
                                    .trailing_fill(true)
                                    .clamping(egui::SliderClamping::Never),
                                );
                            });
                            ui.add_space(4.0);
                            if ui
                                .button("Reset Settings")
                                .on_hover_ui(|ui| {
                                    ui.label("Reset the settings to its initial state");
                                })
                                .clicked()
                            {
                                *self.simulation.dt_mut() = self.state.starting_dt;
                                *self.simulation.g_mut() = self.state.starting_g;
                                *self.simulation.g_soft_mut() = self.state.starting_g_soft;
                            }
                        });
                        ui.separator();
                        ui.add_space(4.0);

                        ui.horizontal(|ui| {
                            if ui
                                .button(if self.state.paused { "⏵" } else { "⏸" })
                                .on_hover_text("Pauses and unpauses the simulation")
                                .clicked()
                            {
                                self.state.paused = !self.state.paused;
                            }

                            if ui
                                .button("Reset")
                                .on_hover_text("This button resets the simulation")
                                .clicked()
                            {
                                self.simulation.reset();
                                self.simulation.init();
                                self.state.reset();
                            }
                        });

                        if self.state.paused {
                            ui.add_space(6.0);
                            ui.label("Step Controller");
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                if ui
                                    .button("⏪")
                                    .on_hover_text(
                                        "Press and hold to continously rewind the simulation by the step size",
                                    )
                                    .is_pointer_button_down_on()
                                {
                                    let dt = -self.simulation.dt();
                                    for _ in 0..self.state.step_by {
                                        self.simulation.step_by(dt);
                                        self.state.step_count -= 1;
                                    }
                                }
                                if ui
                                    .button("⏴")
                                    .on_hover_text(
                                        "Press to rewind the simulation by the step size",
                                    )
                                    .clicked()
                                {
                                    let dt = -self.simulation.dt();
                                    for _ in 0..self.state.step_by {
                                        self.simulation.step_by(dt);
                                        self.state.step_count -= 1;
                                    }
                                }

                                ui.add(egui::DragValue::new(&mut self.state.step_by).speed(1))
                                    .on_hover_text("Drag or edit to change the step size");
                                if ui.button("⏵")
                                    .on_hover_text(
                                        "Press and hold to continously forward the simulation by the step size"
                                    )
                                    .clicked()
                                {
                                    for _ in 0..self.state.step_by {
                                        self.simulation.step();
                                        self.state.step_count += 1;
                                    }
                                }
                                if ui
                                    .button("⏩")
                                    .on_hover_text(
                                        "Press to forward the simulation by the step size",
                                    )
                                    .is_pointer_button_down_on()
                                {
                                    for _ in 0..self.state.step_by {
                                        self.simulation.step();
                                        self.state.step_count += 1;
                                    }
                                }
                            });
                        }
                    });

                let egui_winit::egui::FullOutput {
                    textures_delta,
                    shapes,
                    pixels_per_point,
                    platform_output,
                    ..
                } = gui_state.egui_ctx().end_pass();

                gui_state.handle_platform_output(window, platform_output);

                let paint_jobs = gui_state.egui_ctx().tessellate(shapes, pixels_per_point);

                let screen_descriptor = {
                    let size = window.inner_size();
                    egui_wgpu::ScreenDescriptor {
                        size_in_pixels: [size.width, size.height],
                        pixels_per_point: window.scale_factor() as f32,
                    }
                };

                renderer.render(
                    screen_descriptor,
                    paint_jobs,
                    textures_delta,
                    self.simulation.get_drawables(),
                );

                self.state.frame_count += 1;

                let mut i = 0;
                if self.state.paused {
                } else {
                    while start.elapsed()
                        < Duration::from_nanos(NANOS_PER_SECOND / self.state.max_fps as u64)
                    {
                        if i < self.state.max_steps_per_frame {
                            self.simulation.step();
                            self.state.step_count += 1;
                            i += 1;
                        }
                    }
                }

                self.state.update_frame_time(start.elapsed());
            }
            _ => {}
        }
    }
}

/**
*
*     let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
   let event_proxy = Arc::new(event_loop.create_proxy());

   event_loop.set_control_flow(ControlFlow::Poll);

   #[cfg(not(target_arch = "wasm32"))]
   {
       let mut app: App = App::new(event_proxy);
       event_loop.run_app(&mut app);
   }
   #[cfg(target_arch = "wasm32")]
   {
       use winit::platform::web::EventLoopExtWebSys;

       let app: App = App::new(event_proxy);
       event_loop.spawn_app(app);
   }
*/

pub fn run<F: Float, const D: usize, P, I, S>(simulation: S)
where
    P: Particle<F, D> + 'static,
    I: Integrator<F, D, P> + 'static,
    S: Simulation<F, D, P, I> + 'static,
{
    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    let event_proxy = Arc::new(event_loop.create_proxy());

    event_loop.set_control_flow(ControlFlow::Poll);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = App::new(event_proxy, simulation);
        event_loop.run_app(&mut app);
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;

        let app = App::new(event_proxy, simulation);
        event_loop.spawn_app(app);
    }
}
