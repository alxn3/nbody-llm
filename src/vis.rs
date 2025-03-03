use std::sync::Arc;
use web_time::{Duration, Instant};

use nlib::{
    render::{Drawable, Renderer},
    shared::{Float, Integrator, Particle, Simulation},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    window::{Window, WindowId},
};

#[derive(Debug)]
enum UserEvent {
    #[cfg(target_arch = "wasm32")]
    WebInitialized(Renderer),
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
        Self {
            window: None,
            renderer: None,
            event_proxy,
            simulation,
            _phantom: std::marker::PhantomData,
        }
    }

    fn init(&mut self) {
        self.simulation.init();
        self.simulation
            .init_drawables(&mut self.renderer.as_mut().unwrap().context);
    }

    fn update(&mut self) {
        self.simulation.step();
    }
}

// There are 1_000_000_000 nanoseconds in a second.
const NANOS_PER_FRAME: u64 = 1_000_000_000 / 60;

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
        #[cfg(target_arch = "wasm32")]
        {
            // window.request_redraw();
            let proxy = self.event_proxy.clone();
            wasm_bindgen_futures::spawn_local(async move {
                proxy
                    .send_event(UserEvent::WebInitialized(
                        Renderer::init_async(window).await,
                    ))
                    .expect("Initialize web");
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
        let (Some(window), Some(renderer)) = (self.window.as_mut(), self.renderer.as_mut()) else {
            log::error!("Window or renderer not initialized");
            return;
        };

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

                renderer.update();
                renderer.render(self.simulation.get_drawables());

                let mut i = 0;

                while start.elapsed() < Duration::from_nanos(NANOS_PER_FRAME) {
                    if i < 100 {
                        self.update();
                        i += 1;
                    }
                }
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
