use clap::Parser;

use nlib::{
    llm, manual,
    shared::{self, Bounds, Integrator, LeapFrogIntegrator, Particle, Simulation},
};

#[cfg(feature = "render")]
mod vis;

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

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "0")]
    threads: usize,

    #[arg(short, long, default_value = "10000")]
    num_points: usize,
}

fn main() {
    init_logger();

    let args = Args::parse();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();

    // let mut points: Vec<shared::PointParticle<f64, 3>> = vec![
    //     shared::PointParticle::new(
    //         [-1.0, 0.0, 0.0].into(),
    //         [0.1983865989, 0.1226004003, 0.0].into(),
    //         1.0,
    //         0.0,
    //     ),
    //     shared::PointParticle::new(
    //         [1.0, 0.0, 0.0].into(),
    //         [0.1983865989, 0.1226004003, 0.0].into(),
    //         1.0,
    //         0.0,
    //     ),
    //     shared::PointParticle::new(
    //         [0.0, 0.0, 0.0].into(),
    //         [-0.7935463956, -0.4904016012, 0.0].into(),
    //         0.5,
    //         0.0,
    //     ),
    // ];

    // for _ in 0..1000 {
    //     let p = shared::PointParticle::new(
    //         [
    //             rand::random::<f64>() * 2.0 - 1.0,
    //             rand::random::<f64>() * 2.0 - 1.0,
    //             rand::random::<f64>() * 2.0 - 1.0,
    //         ]
    //         .into(),
    //         [
    //             rand::random::<f64>() * 2.0 - 1.0,
    //             rand::random::<f64>() * 2.0 - 1.0,
    //             rand::random::<f64>() * 2.0 - 1.0,
    //         ]
    //         .into(),
    //         rand::random::<f64>(),
    //         0.0,
    //     );
    //     points.push(p);
    // }

    let mut points = vec![shared::PointParticle::new(
        [0.0, 0.0, 0.0].into(),
        [0.0, 0.0, 0.0].into(),
        1.0,
        0.0,
    )];

    let box_width = 10.0;

    let disc_mass = 2e-1;
    let disc_max: f64 = box_width / 2.0 / 1.2;
    let disc_min: f64 = box_width / 10.0;
    let disc_points = args.num_points;

    // Same setup as rebound's Self-gravitating disc example
    for _ in 0..disc_points {
        let a: f64 = ((disc_max.powf(-0.5) - disc_min.powf(-0.5)) * rand::random::<f64>()
            + disc_min.powf(-0.5))
        .powf(-2.0);
        let phi = rand::random::<f64>() * 2.0 * std::f64::consts::PI;
        let x = a * phi.cos();
        let y = a * phi.sin();
        let z = a * rand::random::<f64>() * 0.001 - 0.0005;
        let mu = 1.0
            + disc_mass * (a.powf(-1.5) - disc_min.powf(-1.5))
                / (disc_max.powf(-1.5) - disc_min.powf(-1.5));
        let vkep = (mu * 1.0 / a).sqrt();
        let vx = vkep * phi.sin();
        let vy = -vkep * phi.cos();
        let vz = 0.0;
        let m = disc_mass / disc_points as f64;
        points.push(shared::PointParticle::new(
            [x, y, z].into(),
            [vx, vy, vz].into(),
            m,
            0.0,
        ));
    }

    let mut sim = llm::create_barnes_hut_3d(
        points,
        LeapFrogIntegrator::new(),
        Bounds::new([0.0, 0.0, 0.0].into(), box_width),
    );
    sim.settings_mut().dt = 3e-2;
    sim.settings_mut().g_soft = 0.02;
    sim.settings_mut().theta2 = 1.0;

    #[cfg(feature = "render")]
    vis::run(sim);
    #[cfg(not(feature = "render"))]
    {
        println!("Running simulation without rendering...");
        sim.init();
        let start = std::time::Instant::now();

        let mut i = 0;
        let steps = 1000;

        // Print progress every 100 steps
        while i < steps {
            sim.step();
            i += 1;
        }

        let elapsed = start.elapsed();
        println!("Elapsed: {:?}", elapsed);

        let steps_per_second = steps as f64 / elapsed.as_secs_f64();
        println!("Performance: {:.2} steps/second", steps_per_second);
    }
}
