use nlib::{manual, shared::{self, Integrator, LeapFrogIntegrator, Particle, Simulation}};

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

fn main() {
    init_logger();

    let mut points: Vec<shared::PointParticle<f64, 3>> = vec![
        shared::PointParticle::new(
            nalgebra::SVector::from_vec(vec![-1.0, 0.0, 0.0]),
            nalgebra::SVector::from_vec(vec![0.1983865989, 0.1226004003, 0.0]),
            1.0,
            0.0,
        ),
        shared::PointParticle::new(
            nalgebra::SVector::from_vec(vec![1.0, 0.0, 0.0]),
            nalgebra::SVector::from_vec(vec![0.1983865989, 0.1226004003, 0.0]),
            1.0,
            0.0,
        ),
        shared::PointParticle::new(
            nalgebra::SVector::from_vec(vec![0.0, 0.0, 0.0]),
            nalgebra::SVector::from_vec(vec![-0.7935463956, -0.4904016012, 0.0]),
            0.5,
            0.0,
        ),
    ];

    // for _ in 0..1000 {
    //     let p = shared::PointParticle::new(
    //         nalgebra::SVector::from_vec(vec![
    //             rand::random::<f64>() * 2.0 - 1.0,
    //             rand::random::<f64>() * 2.0 - 1.0,
    //             rand::random::<f64>() * 2.0 - 1.0,
    //         ]),
    //         nalgebra::SVector::from_vec(vec![
    //             rand::random::<f64>() * 2.0 - 1.0,
    //             rand::random::<f64>() * 2.0 - 1.0,
    //             rand::random::<f64>() * 2.0 - 1.0,
    //         ]),
    //         rand::random(),
    //         0.0,
    //     );
    //     points.push(p);
    // }

    let mut sim = manual::BruteForceSimulation::new(points, LeapFrogIntegrator::new());
    *sim.dt_mut() = 0.0001;
    *sim.g_soft_mut() = 0.02;

    #[cfg(feature = "render")]
    vis::run(sim);
    #[cfg(not(feature = "render"))]
    {
        let start = std::time::Instant::now();

        let mut i = 0;
        while i < 1000 {
            sim.step();
            i += 1;
        }

        let elapsed = start.elapsed();
        println!("Elapsed: {:?}", elapsed);

        // println!("{:?}", sim.get_points());
    }
}
