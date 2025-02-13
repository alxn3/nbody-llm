use piston_window::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug)]
struct BodyPosition {
    _body: usize,
    x: f64,
    y: f64,
}

/// Reads the CSV file at `file_path` and groups body positions by simulation frame.
///
/// Assumes the CSV file has a header line ("frame,body,x,y") followed by one line per body.
fn load_positions(file_path: &str) -> BTreeMap<usize, Vec<BodyPosition>> {
    let file = File::open(file_path).expect("Failed to open positions.csv");
    let reader = BufReader::new(file);
    let mut frames = BTreeMap::new();

    // Skip the header and then parse each line.
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.expect("Failed to read line");
        if line_idx == 0 {
            // Skip header line.
            continue;
        }
        let parts: Vec<&str> = line.trim().split(',').collect();
        if parts.len() < 4 {
            continue;
        }
        // Parse frame number, body id, and coordinates.
        let frame: usize = parts[0].parse().expect("Invalid frame number in CSV");
        let body: usize = parts[1].parse().expect("Invalid body id in CSV");
        let x: f64 = parts[2].parse().expect("Invalid x coordinate");
        let y: f64 = parts[3].parse().expect("Invalid y coordinate");
        let pos = BodyPosition { _body: body, x, y };

        frames.entry(frame).or_insert_with(Vec::new).push(pos);
    }
    frames
}

fn main() {
    // Load positions from the CSV file.
    let frames_map = load_positions("positions.csv");
    // Convert the map (sorted by frame number) to a vector:
    let mut frames: Vec<(usize, Vec<BodyPosition>)> = frames_map.into_iter().collect();
    frames.sort_by_key(|entry| entry.0);

    if frames.is_empty() {
        eprintln!("No frames found in positions.csv");
        return;
    }

    // Create a Piston window. We use an 800x800 window.
    let (width, height) = (800, 800);
    let mut window: PistonWindow = WindowSettings::new("N-Body Simulation", [width, height])
        .exit_on_esc(true)
        .build()
        .expect("Failed to build PistonWindow");

    // For animating the simulation, we use a frame timer.
    let total_frames = frames.len();
    let mut current_frame_idx: usize = 0;
    let mut accumulator = 0.0;
    // Playback rate: advance one simulation frame every 1/24 second.
    let frame_rate = 1.0 / 30.0;

    while let Some(event) = window.next() {
        // Update the current frame based on update events.
        if let Some(update_args) = event.update_args() {
            accumulator += update_args.dt;
            while accumulator >= frame_rate {
                accumulator -= frame_rate;
                current_frame_idx = (current_frame_idx + 1) % total_frames;
            }
        }

        // Render the current simulation frame.
        println!("current_frame_idx: {}", current_frame_idx);
        window.draw_2d(&event, |context, graphics, _| {
            // Clear to a white background.
            clear([1.0, 1.0, 1.0, 1.0], graphics);

            // In our simulation, positions range over (approximately) -BOUND to +BOUND.
            // Setting the transformation: translate to the window's center and flip the y-axis so that
            // simulation "up" is on the screen top.
            let transform = context
                .transform
                .trans((width / 2) as f64, (height / 2) as f64)
                .scale(1.0, -1.0);

            // Get the positions for the current frame.
            let (_frame_number, positions) = &frames[current_frame_idx];

            // Optionally, you might want to display the frame number.
            // For example, using a text-rendering library (omitted here for brevity).

            // Draw each body as a blue circle (radius 3.0).
            for pos in positions.iter() {
                let radius = 3.0;
                let circle = ellipse::circle(pos.x, pos.y, radius);
                ellipse([0.0, 0.0, 1.0, 1.0], circle, transform, graphics);
            }
        });
    }
}
