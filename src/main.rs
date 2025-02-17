// Make sure you add these dependencies in Cargo.toml:
// [dependencies]
// piston_window = "0.127.0"
// rand = "0.8"
// (Barnes–Hut implementation has been moved to barnes_hut_3d.rs)
pub mod barnes_hut;
use barnes_hut::{BHOctree, Body3D, Oct, Vec3};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn main() {
    // Simulation parameters
    const NUM_PARTICLES_PER_AXIS: usize = 10;
    const BOUND: f64 = 500.0; // The simulation domain is from -BOUND to +BOUND
    const G: f64 = 1.0; // gravitational constant unchanged (adjust if needed)
    const THETA: f64 = 0.5;
    const SOFTENING: f64 = 0.02;

    // Calculate spacing between particles
    let spacing = BOUND / NUM_PARTICLES_PER_AXIS as f64;

    // Initialize bodies in a 3D grid
    let mut bodies: Vec<Body3D> = Vec::new();

    for x in 0..NUM_PARTICLES_PER_AXIS {
        for y in 0..NUM_PARTICLES_PER_AXIS {
            for z in 0..NUM_PARTICLES_PER_AXIS {
                let pos = Vec3 {
                    x: spacing * (x as f64) - BOUND / 2.0,
                    y: spacing * (y as f64) - BOUND / 2.0,
                    z: spacing * (z as f64) - BOUND / 2.0,
                };
                bodies.push(Body3D {
                    position: pos,
                    _prev_position: pos,
                    velocity: Vec3::zero(),
                    _acceleration: Vec3::zero(),
                    mass: 500.0,
                });
            }
        }
    }

    // Instead of rendering, we now run a simulation loop for 1000 frames.
    let fixed_dt = 0.03;
    let simulation_frames = 1000;

    let file = File::create("positions_3d.csv").unwrap();
    let mut writer = BufWriter::new(file);
    writeln!(writer, "frame,body,x,y,z").unwrap();

    let mut frame_times = Vec::with_capacity(simulation_frames);

    for frame in 0..simulation_frames {
        let frame_start = Instant::now();

        // Build the Barnes-Hut tree for the current state.
        let mut tree = BHOctree::new(Oct {
            center: Vec3::zero(),
            half_dimension: BOUND / 2.0,
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
        frame_times.push(frame_start.elapsed());

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

    // Calculate statistics
    let total_time = frame_times.iter().sum::<std::time::Duration>();
    let mean_time = total_time / simulation_frames as u32;

    // Calculate standard deviation
    let mean_nanos = mean_time.as_nanos() as f64;
    let variance: f64 = frame_times
        .iter()
        .map(|t| {
            let diff = t.as_nanos() as f64 - mean_nanos;
            diff * diff
        })
        .sum::<f64>()
        / simulation_frames as f64;
    let std_dev = std::time::Duration::from_nanos(variance.sqrt() as u64);
    println!("Simulated {} frames in {:?}", simulation_frames, total_time);
    println!("Mean frame time: {:?}", mean_time);
    println!("Standard deviation: {:?}", std_dev);
}
