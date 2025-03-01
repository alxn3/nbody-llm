use std::sync::Arc;
use web_time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

use nlib::render::Renderer;

#[derive(Debug)]
enum UserEvent {
    #[cfg(target_arch = "wasm32")]
    WebInitialized(Renderer),
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    event_proxy: Arc<EventLoopProxy<UserEvent>>,
}

impl App {
    fn new(event_proxy: Arc<EventLoopProxy<UserEvent>>) -> Self {
        Self {
            window: None,
            renderer: None,
            event_proxy,
        }
    }
}

// There are 1_000_000_000 nanoseconds in a second.
const NANOS_PER_FRAME: u64 = 1_000_000_000 / 60;

impl ApplicationHandler<UserEvent> for App {
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
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            #[cfg(target_arch = "wasm32")]
            UserEvent::WebInitialized(renderer) => {
                self.renderer = Some(renderer);
                self.window.as_ref().unwrap().request_redraw();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(window), Some(renderer)) = (self.window.as_mut(), self.renderer.as_mut()) else {
            log::error!("Window or renderer not initialized");
            return;
        };

        log::trace!("Window event: {:?}", event);

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
                let start = Instant::now();
                renderer.render();
                window.request_redraw();

                while start.elapsed() < Duration::from_nanos(NANOS_PER_FRAME) {
                    // Do stuff.
                }
            }
            _ => {}
        }
    }
}

fn init_logger() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        fern::Dispatch::new()
            .level(log::LevelFilter::Info)
            .chain(fern::Output::call(console_log::log))
            .apply()
            .unwrap();
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        log::info!("Logger initialized");
    }
}

fn main() {
    init_logger();

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
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
}
