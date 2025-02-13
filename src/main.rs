// Make sure you add these dependencies in Cargo.toml:
// [dependencies]
// piston_window = "0.127.0"
// rand = "0.8"
// (Barnes–Hut implementation has been moved to barnes_hut_3d.rs)
pub mod barnes_hut_3d;
use barnes_hut_3d::{BHOctree, Body3D, Oct, Vec3};
use rand::Rng;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn main() {
    // Simulation parameters
    const NUM_BODIES: usize = 20;
    const BOUND: f64 = 400.0; // The simulation domain is from -BOUND to +BOUND
    const G: f64 = 100000.0; // gravitational constant unchanged (adjust if needed)
    const THETA: f64 = 0.5;
    const SOFTENING: f64 = 3.0;

    // Initialize bodies with random positions and velocities.
    let mut rng = rand::thread_rng();
    let mut bodies: Vec<Body3D> = (0..NUM_BODIES)
        .map(|_| {
            let pos = Vec3 {
                x: rng.gen_range(-BOUND..BOUND),
                y: rng.gen_range(-BOUND..BOUND),
                z: rng.gen_range(-BOUND..BOUND),
            };
            Body3D {
                position: pos,
                _prev_position: pos,
                velocity: Vec3::zero(),
                _acceleration: Vec3::zero(),
                mass: rng.gen_range(100.0..500.0),
            }
        })
        .collect();

    // Instead of rendering, we now run a simulation loop for 1000 frames.
    let fixed_dt = 1.0 / 240.0;
    let simulation_frames = 1000;
    let start_time = Instant::now();

    let file = File::create("positions_3d.csv").unwrap();
    let mut writer = BufWriter::new(file);
    writeln!(writer, "frame,body,x,y,z").unwrap();

    for frame in 0..simulation_frames {
        // Build the Barnes-Hut tree for the current state.
        let mut tree = BHOctree::new(Oct {
            center: Vec3::zero(),
            half_dimension: BOUND,
        });
        for body in &bodies {
            tree.insert(body.clone());
        }

        // Update simulation state at the fixed timestep.
        for body in &mut bodies {
            let force = tree.calc_force(body, THETA, G, SOFTENING);
            let acceleration = force / body.mass;
            body.velocity = body.velocity + acceleration * fixed_dt;
            body.position = body.position + body.velocity * fixed_dt;
        }

        // Log positions of each body for this frame
        for (i, body) in bodies.iter().enumerate() {
            writeln!(
                writer,
                "{},{},{},{},{}",
                frame, i, body.position.x, body.position.y, body.position.z
            )
            .unwrap();
        }
    }

    writer.flush().unwrap();
    let elapsed = start_time.elapsed();
    println!("Simulated {} frames in {:?}", simulation_frames, elapsed);
}
