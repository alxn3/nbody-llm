use bevy::prelude::*;
use nbody::barnes_hut_3d;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

type Vec3 = barnes_hut_3d::Vec3;

#[derive(Component)]
struct Body;

#[derive(Resource)]
struct SimulationState {
    frames: Vec<Vec<Vec3>>,
    current_frame_idx: usize,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(ClearColor(Color::WHITE))
        .insert_resource(SimulationState {
            frames: load_positions("positions_3d.csv"),
            current_frame_idx: 0,
        })
        .add_systems(Startup, setup_scene)
        .add_systems(FixedUpdate, update_positions)
        .run();
}

fn load_positions(file_path: &str) -> Vec<Vec<Vec3>> {
    let file = File::open(file_path).expect("Failed to open positions file");
    let reader = BufReader::new(file);
    let mut frames = BTreeMap::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.expect("Failed to read line");
        if line_idx == 0 {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 5 {
            continue;
        }

        let frame: usize = parts[0].parse().expect("Invalid frame number");
        let x: f64 = parts[2].parse().expect("Invalid x");
        let y: f64 = parts[3].parse().expect("Invalid y");
        let z: f64 = parts[4].parse().expect("Invalid z");

        frames
            .entry(frame)
            .or_insert_with(Vec::new)
            .push(Vec3 { x, y, z });
    }

    frames.into_values().collect()
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim_state: Res<SimulationState>,
) {
    // Lighting
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 100.0, 0.0)
            .looking_at(bevy::math::Vec3::ZERO, bevy::math::Vec3::Z),
        ..default()
    });

    // Camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 1000.0)
            .looking_at(bevy::math::Vec3::ZERO, bevy::math::Vec3::Y),
        ..default()
    });

    // Spawn initial bodies
    let mesh = meshes.add(Sphere::new(3.0));
    let material = materials.add(StandardMaterial {
        base_color: Color::BLUE,
        metallic: 0.0,
        perceptual_roughness: 0.5,
        ..default()
    });

    for pos in &sim_state.frames[0] {
        commands.spawn((
            PbrBundle {
                mesh: mesh.clone(),
                material: material.clone(),
                transform: Transform::from_xyz(pos.x as f32, pos.y as f32, pos.z as f32),
                ..default()
            },
            Body,
        ));
    }
}

fn update_positions(
    mut query: Query<&mut Transform, With<Body>>,
    mut sim_state: ResMut<SimulationState>,
    time: Res<Time>,
) {
    let frame_duration = 0.1; // Seconds per simulation frame
    let current_frame = sim_state.current_frame_idx;

    // Update positions for all bodies
    for (mut transform, pos) in query.iter_mut().zip(sim_state.frames[current_frame].iter()) {
        transform.translation = bevy::math::Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32);
    }

    // Advance frame based on real time
    sim_state.current_frame_idx =
        ((time.elapsed_seconds() / frame_duration) as usize) % sim_state.frames.len();
}
