use nalgebra::{SimdComplexField, SVector};
use rayon::prelude::*;

#[cfg(feature = "render")]
use {
    crate::render::PipelineType,
    wgpu::util::DeviceExt,
};

#[cfg(feature = "render")]
use crate::render::Renderable;

use crate::shared::{Bounds, Float, Integrator, LeapFrogIntegrator, Particle, Simulation, SimulationSettings, AABB};
use std::fmt::Debug;
use std::marker::PhantomData;

/// Arena for storing octree nodes contiguously in memory
#[derive(Clone, Debug)]
pub struct NodeArena<F: Float, const D: usize> {
    nodes: Vec<ArenaNode<F, D>>,
}

// Node type stored in the arena
#[derive(Debug, Clone)]
pub enum ArenaNode<F: Float, const D: usize> {
    Internal {
        bounds: Bounds<F, D>,
        center_of_mass: SVector<F, D>,
        total_mass: F,
        children: [Option<usize>; 8], // Indices into the arena instead of Box pointers
    },
    Leaf {
        bounds: Bounds<F, D>,
        particle_idx: usize,
        particle_position: SVector<F, D>,
        particle_mass: F,
    },
    Empty {
        bounds: Bounds<F, D>,
    },
}

impl<F: Float, const D: usize> NodeArena<F, D> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(1024), // Pre-allocate space for nodes
        }
    }
    
    pub fn add_node(&mut self, node: ArenaNode<F, D>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }
    
    pub fn get(&self, idx: usize) -> &ArenaNode<F, D> {
        &self.nodes[idx]
    }
    
    pub fn get_mut(&mut self, idx: usize) -> &mut ArenaNode<F, D> {
        &mut self.nodes[idx]
    }
    
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn clear(&mut self) {
        self.nodes.clear();
    }
}

// Add a new struct to use in the iterative version of calculate_force
#[derive(Clone)]
struct OctreeTraversalItem {
    node_idx: usize,
    depth: usize,
}

impl<F: Float, const D: usize> ArenaNode<F, D> {
    // Define constants at the impl level so they're available to all methods
    const MAX_FORCE_DEPTH: usize = 2048; // Maximum depth for force calculation
    
    pub fn new_empty(bounds: Bounds<F, D>) -> Self {
        Self::Empty { bounds }
    }

    pub fn new_leaf<P: Particle<F, D>>(bounds: Bounds<F, D>, particle_idx: usize, particle: &P) -> Self {
        Self::Leaf {
            bounds,
            particle_idx,
            particle_position: *particle.position(),
            particle_mass: particle.get_mass(),
        }
    }

    pub fn new_internal(bounds: Bounds<F, D>) -> Self {
        Self::Internal {
            bounds,
            center_of_mass: SVector::zeros(),
            total_mass: F::from(0.0).unwrap(),
            children: Default::default(),
        }
    }

    // Get octant index based on position - uses Bounds::get_orthant
    pub fn get_octant_idx(bounds: &Bounds<F, D>, position: &SVector<F, D>) -> usize {
        // Use the manual implementation's get_orthant method directly
        bounds.get_orthant(position)
    }

    // Get bounds for a specific octant - uses Bounds::create_orthant
    pub fn get_octant_bounds(bounds: &Bounds<F, D>, octant_idx: usize) -> Bounds<F, D> {
        // Use the manual implementation's create_orthant method directly
        bounds.create_orthant(octant_idx)
    }
}

// Modified simulation struct to use lifetimes
#[derive(Clone)]
pub struct BarnesHutSimulation<F: Float, const D: usize, P, I = LeapFrogIntegrator<F, D, P>>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    points: Vec<P>,
    bounds: Bounds<F, D>,
    integrator: I,
    settings: SimulationSettings<F>,
    elapsed: F,
    theta: F, // Barnes-Hut opening angle parameter
    arena: NodeArena<F, D>, // Arena for storing octree nodes
    root: Option<usize>, // Index to root node in the arena
    rebuild_counter: usize, // Counter to determine when to rebuild the tree
    #[cfg(feature = "render")]
    points_vertex_buffer: Option<wgpu::Buffer>,
    #[cfg(feature = "render")]
    bounds_vertex_buffer: Option<wgpu::Buffer>,
    #[cfg(feature = "render")]
    num_bounds: u32, // Track the number of octree nodes to display
    _marker: PhantomData<P>, // Marker to handle lifetimes
}

impl<F: Float, const D: usize, P, I> Simulation<F, D, P, I> for BarnesHutSimulation<F, D, P, I>
where
    P: Particle<F, D> + Send + Sync,
    I: Integrator<F, D, P> + Send + Sync,
{
    fn new(points: Vec<P>, integrator: I, bounds: Bounds<F, D>) -> Self {
        Self {
            points,
            bounds,
            integrator,
            settings: SimulationSettings::default(),
            elapsed: F::from(0.0).unwrap(),
            theta: F::from(0.5).unwrap(), // Default opening angle
            arena: NodeArena::new(),
            root: None,
            rebuild_counter: 0,
            #[cfg(feature = "render")]
            points_vertex_buffer: None,
            #[cfg(feature = "render")]
            bounds_vertex_buffer: None,
            #[cfg(feature = "render")]
            num_bounds: 0,
            _marker: PhantomData,
        }
    }

    fn init(&mut self) {
        self.integrator.init();
        self.elapsed = F::from(0.0).unwrap();
        self.build_octree();
    }

    fn settings(&self) -> &SimulationSettings<F> {
        &self.settings
    }

    fn settings_mut(&mut self) -> &mut SimulationSettings<F> {
        &mut self.settings
    }

    fn elapsed(&self) -> F {
        self.elapsed
    }

    fn update_forces(&mut self) {
        // Reset accelerations - can be done in parallel for large numbers of particles
        if self.points.len() > 10000 {
            self.points.par_iter_mut().for_each(|point| {
                point.acceleration_mut().fill(F::from(0.0).unwrap());
            });
        } else {
            for point in self.points.iter_mut() {
                point.acceleration_mut().fill(F::from(0.0).unwrap());
            }
        }

        // Manual implementation rebuilds tree every step - let's do the same
        self.build_octree();

        // Store these values to avoid borrowing conflicts
        let g = self.settings.g;
        let g_soft = self.settings.g_soft;
        let theta = self.settings.theta2.sqrt();
        
        // Calculate forces using Barnes-Hut approximation
        if let Some(root_idx) = self.root {
            // Create a slice of points to allow parallel iteration
            let arena = &self.arena;
            let points = &mut self.points;
            
            // Parallelize force calculation for improved performance
            points.par_iter_mut().enumerate().for_each(|(i, point)| {
                let force = calculate_force(arena, root_idx, point, i, theta, g, g_soft);
                *point.acceleration_mut() += force;
            });
        }
    }

    fn step_by(&mut self, dt: F) {
        // Pre-integrate
        self.integrator.integrate_pre_force(&mut self.points, dt);
        
        // Filter particles outside the bounds - do this right after pre-integration
        // to match the manual implementation's behavior
        self.points.retain(|p| self.bounds.contains(p.position()));
        
        // Update forces
        self.update_forces();
        
        // Post-integrate
        self.integrator.integrate_after_force(&mut self.points, dt);
        
        self.elapsed += dt;
    }

    fn add_point(&mut self, point: P) {
        self.points.push(point);
        // We'll rebuild the tree during the next force update
        self.rebuild_counter = 5; 
    }

    fn remove_point(&mut self, index: usize) {
        self.points.swap_remove(index);
        // We'll rebuild the tree during the next force update
        self.rebuild_counter = 5;
    }

    fn get_points(&self) -> &Vec<P> {
        &self.points
    }
}

#[cfg(feature = "render")]
impl<F: Float, P, I> Renderable for BarnesHutSimulation<F, 3, P, I>
where
    P: Particle<F, 3> + Send + Sync,
    I: Integrator<F, 3, P> + Send + Sync,
{
    fn render(&mut self, renderer: &mut crate::render::Renderer) {
        // Prepare all the data we need before borrowing the renderer mutably
        
        // Prepare points data
        let position_data: Vec<f32> = self
            .points
            .iter()
            .flat_map(|p| {
                p.position()
                    .iter()
                    .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                    .collect::<Vec<f32>>()
            })
            .collect();
            
        // Prepare bounds data
        let bounds_data = if let Some(root_idx) = self.root {
            let mut bounds_data = Vec::new();
            self.num_bounds = 0;
            
            // Pre-collect all nodes for rendering to avoid borrowing issues
            let nodes = {
                let mut result = Vec::new();
                let mut queue = vec![(root_idx, 0)]; // (node_idx, depth)
                
                while let Some((node_idx, depth)) = queue.pop() {
                    result.push((depth, node_idx));
                    
                    // Add children to queue
                    if let ArenaNode::Internal { children, .. } = self.arena.get(node_idx) {
                        for &child_opt in children.iter() {
                            if let Some(child_idx) = child_opt {
                                queue.push((child_idx, depth + 1));
                            }
                        }
                    }
                }
                
                // Sort by depth for rendering
                result.sort_by_key(|&(depth, _)| depth);
                result
            };
            
            // If we have nodes, calculate the maximum depth for coloring
            if !nodes.is_empty() {
                let max_depth = nodes.last().unwrap().0;
                
                // Limit to MAX_NODES to avoid buffer overflow
                const MAX_NODES: usize = 20000;
                let nodes_to_render = if nodes.len() > MAX_NODES {
                    // If we have too many nodes, prioritize displaying higher-level nodes
                    // as they give a better overview of the structure
                    &nodes[0..MAX_NODES]
                } else {
                    &nodes[..]
                };
                
                for (depth, node_idx) in nodes_to_render.iter() {
                    // Color based on depth (similar to manual implementation)
                    let depth_f32 = *depth as f32;
                    let max_depth_f32 = max_depth as f32;
                    let s = depth_f32 / max_depth_f32 * 0.7 + 0.3;
                    
                    // Get min/max bounds and add color
                    let (min, max) = match &self.arena.get(*node_idx) {
                        ArenaNode::Empty { bounds } => (bounds.min(), bounds.max()),
                        ArenaNode::Leaf { bounds, .. } => (bounds.min(), bounds.max()),
                        ArenaNode::Internal { bounds, .. } => (bounds.min(), bounds.max()),
                    };
                    
                    bounds_data.extend(
                        min.iter()
                            .chain(max.iter())
                            .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                            .chain::<[f32; 4]>([(1. - s * s) * 0.5, s * s, (1. - s) * 0.5, s])
                            .collect::<Vec<f32>>(),
                    );
                    self.num_bounds += 1;
                }
            } else {
                self.num_bounds = 1;
                bounds_data = self.bounds
                    .min()
                    .iter()
                    .chain(self.bounds.max().iter())
                    .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                    .chain::<[f32; 4]>([1.0, 1.0, 0.0, 1.0])
                    .collect::<Vec<f32>>();
            }
            
            bounds_data
        } else {
            // If no octree, just show the overall bounds
            self.num_bounds = 1;
            self.bounds
                .min()
                .iter()
                .chain(self.bounds.max().iter())
                .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                .chain::<[f32; 4]>([1.0, 1.0, 0.0, 1.0])
                .collect::<Vec<f32>>()
        };

        // Now update buffers and render
        {
            // Update particles buffer
            let queue = &renderer.context.queue;
            let bodies_vertex_buffer = self.points_vertex_buffer.as_ref().unwrap();
            queue.write_buffer(
                bodies_vertex_buffer,
                0,
                bytemuck::cast_slice(&position_data),
            );
            
            // Update bounds buffer
            let bounds_vertex_buffer = self.bounds_vertex_buffer.as_ref().unwrap();
            queue.write_buffer(
                bounds_vertex_buffer,
                0,
                bytemuck::cast_slice(&bounds_data),
            );
        }
        
        // Now do the rendering with mutable borrows
        {
            // Render particles
            renderer.set_pipeline(PipelineType::Points);
            let render_pass = renderer.get_render_pass();
            render_pass.set_vertex_buffer(0, self.points_vertex_buffer.as_ref().unwrap().slice(..));
            render_pass.draw(0..4, 0..self.points.len() as u32);
        }

        {
            // Render bounds
            renderer.set_pipeline(PipelineType::AABB);
            let render_pass = renderer.get_render_pass();
            render_pass.set_vertex_buffer(0, self.bounds_vertex_buffer.as_ref().unwrap().slice(..));
            render_pass.draw(0..16, 0..self.num_bounds);
        }
    }

    fn render_init(&mut self, context: &crate::render::Context) {
        use crate::shared::AABB;

        let device = &context.device;
        let queue = &context.queue;

        // Create initial points buffer
        let point_position_data: Vec<f32> = self
            .points
            .iter()
            .flat_map(|p| {
                p.position()
                    .iter()
                    .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                    .collect::<Vec<f32>>()
            })
            .collect();

        let points_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&point_position_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        queue.write_buffer(
            &points_vertex_buffer,
            0,
            bytemuck::cast_slice(&point_position_data),
        );

        // Create bounds buffer with sufficient size for many octree nodes
        // Each node needs 3 (min) + 3 (max) + 4 (color) = 10 floats * 4 bytes = 40 bytes per node
        // Allocate space for up to 20,000 nodes
        const MAX_NODES: usize = 20000;
        const FLOATS_PER_NODE: usize = 10; // 3 for min, 3 for max, 4 for color
        const BUFFER_SIZE: usize = MAX_NODES * FLOATS_PER_NODE * std::mem::size_of::<f32>();
        
        // Initially just fill with the overall bounds
        let mut bounds_data = [self.bounds.min(), self.bounds.max()]
            .iter()
            .flat_map(|p| {
                p.iter()
                    .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<f32>>();
        let color = [0.0, 1.0, 0.0, 1.0];
        bounds_data.extend(color);

        let bounds_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bounds Vertex Buffer"),
            size: BUFFER_SIZE as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Write initial data
        queue.write_buffer(
            &bounds_vertex_buffer,
            0,
            bytemuck::cast_slice(&bounds_data),
        );

        self.points_vertex_buffer = Some(points_vertex_buffer);
        self.bounds_vertex_buffer = Some(bounds_vertex_buffer);
    }
}

impl<F: Float, const D: usize, P, I> BarnesHutSimulation<F, D, P, I>
where
    P: Particle<F, D> + Send + Sync,
    I: Integrator<F, D, P> + Send + Sync,
{
    // Build octree from current particle positions using the arena
    fn build_octree(&mut self) {
        // Clear the arena and reset the root
        self.arena.clear();
        self.root = None;
        
        // Extract bounds to avoid borrowing issues
        let bounds = self.bounds.clone();
        
        // First collect all the data we need from particles to avoid borrowing issues
        let points_data: Vec<(usize, SVector<F, D>, F)> = self.points
            .iter()
            .enumerate()
            .filter(|(_, p)| bounds.contains(p.position()))
            .map(|(idx, p)| (idx, *p.position(), p.get_mass()))
            .collect();
        
        // Then process each particle's data
        for (idx, position, mass) in points_data {
            // Handle the root case separately first time
            if self.root.is_none() {
                let particle_proxy = ParticleProxy { position, mass };
                let node = ArenaNode::new_leaf(bounds.clone(), idx, &particle_proxy);
                let node_idx = self.arena.add_node(node);
                self.root = Some(node_idx);
                continue;
            }
            
            // Otherwise insert the particle into the existing tree
            if let Some(root_idx) = self.root {
                let particle_proxy = ParticleProxy { position, mass };
                self.insert_particle(root_idx, idx, &particle_proxy);
            }
        }
    }
    
    // Insert a particle into an existing node
    fn insert_particle<T: Particle<F, D>>(&mut self, node_idx: usize, point_index: usize, particle: &T) {
        match self.arena.get(node_idx).clone() {
            ArenaNode::Empty { bounds } => {
                // Convert empty to leaf
                let leaf = ArenaNode::new_leaf(bounds, point_index, particle);
                *self.arena.get_mut(node_idx) = leaf;
            },
            ArenaNode::Leaf { 
                bounds, 
                particle_idx: existing_idx, 
                particle_position: existing_position, 
                particle_mass: existing_mass 
            } => {
                // Convert leaf to internal
                // Calculate octants for both particles
                let o1 = bounds.get_orthant(&existing_position);
                let o2 = bounds.get_orthant(particle.position());
                
                // Create a new internal node
                let internal = ArenaNode::Internal {
                    bounds: bounds.clone(),
                    center_of_mass: SVector::zeros(),
                    total_mass: F::from(0.0).unwrap(),
                    children: Default::default(),
                };
                
                // Update the node in place
                *self.arena.get_mut(node_idx) = internal;
                
                // Create children for each particle
                let child1_bounds = bounds.create_orthant(o1);
                let child1 = ArenaNode::new_leaf(child1_bounds, existing_idx, &ParticleProxy {
                    position: existing_position,
                    mass: existing_mass,
                });
                let child1_idx = self.arena.add_node(child1);
                
                let child2_bounds = bounds.create_orthant(o2);
                let child2 = ArenaNode::new_leaf(child2_bounds, point_index, particle);
                let child2_idx = self.arena.add_node(child2);
                
                // Update the internal node's children and mass properties
                if let ArenaNode::Internal { ref mut children, ref mut center_of_mass, ref mut total_mass, .. } = *self.arena.get_mut(node_idx) {
                    children[o1] = Some(child1_idx);
                    children[o2] = Some(child2_idx);
                    
                    *total_mass = existing_mass + particle.get_mass();
                    *center_of_mass = (existing_position * existing_mass + *particle.position() * particle.get_mass()) / *total_mass;
                }
            },
            ArenaNode::Internal { 
                bounds, 
                center_of_mass, 
                total_mass, 
                children 
            } => {
                // Update center of mass and total mass
                let particle_mass = particle.get_mass();
                let new_total_mass = total_mass + particle_mass;
                let new_center_of_mass = if new_total_mass > F::from(0.0).unwrap() {
                    (center_of_mass * total_mass + *particle.position() * particle_mass) / new_total_mass
                } else {
                    center_of_mass
                };
                
                // Update the internal node
                if let ArenaNode::Internal { ref mut center_of_mass, ref mut total_mass, .. } = *self.arena.get_mut(node_idx) {
                    *center_of_mass = new_center_of_mass;
                    *total_mass = new_total_mass;
                }
                
                // Get orthant for this particle
                let orthant = bounds.get_orthant(particle.position());
                
                match children[orthant] {
                    Some(child_idx) => {
                        // Add to existing child
                        self.insert_particle(child_idx, point_index, particle);
                    },
                    None => {
                        // Create new child for this orthant
                        let child_bounds = bounds.create_orthant(orthant);
                        let child = ArenaNode::new_leaf(child_bounds, point_index, particle);
                        let child_idx = self.arena.add_node(child);
                        
                        // Update the internal node's children
                        if let ArenaNode::Internal { ref mut children, .. } = *self.arena.get_mut(node_idx) {
                            children[orthant] = Some(child_idx);
                        }
                    }
                }
            }
        }
    }
    
    // Set the opening angle parameter for the Barnes-Hut algorithm
    pub fn set_theta(&mut self, theta: F) {
        self.theta = theta;
    }
    
    // Get the opening angle parameter
    pub fn theta(&self) -> F {
        self.theta
    }
}

// Simple proxy for particles
#[derive(Clone)]
struct ParticleProxy<F: Float, const D: usize> {
    position: SVector<F, D>,
    mass: F,
}

impl<F: Float, const D: usize> Particle<F, D> for ParticleProxy<F, D> {
    fn new(_position: SVector<F, D>, _velocity: SVector<F, D>, _mass: F, _radius: F) -> Self {
        panic!("ParticleProxy should not be created with new")
    }

    fn position(&self) -> &SVector<F, D> {
        &self.position
    }

    fn velocity(&self) -> &SVector<F, D> {
        panic!("ParticleProxy does not have velocity")
    }

    fn acceleration(&self) -> &SVector<F, D> {
        panic!("ParticleProxy does not have acceleration")
    }

    fn position_mut(&mut self) -> &mut SVector<F, D> {
        panic!("ParticleProxy does not have mutable position")
    }

    fn velocity_mut(&mut self) -> &mut SVector<F, D> {
        panic!("ParticleProxy does not have mutable velocity")
    }

    fn acceleration_mut(&mut self) -> &mut SVector<F, D> {
        panic!("ParticleProxy does not have mutable acceleration")
    }

    fn get_mass(&self) -> F {
        self.mass
    }
}

// Define a standalone force calculation function to avoid borrowing the entire simulation
fn calculate_force<F: Float, const D: usize, T: Particle<F, D>>(
    arena: &NodeArena<F, D>,
    node_idx: usize,
    particle: &T,
    particle_idx: usize,
    theta: F,
    g: F,
    g_soft: F
) -> SVector<F, D> {
    let mut total_force = SVector::<F, D>::zeros();
    // Pre-allocate stack with capacity to reduce reallocations
    let mut stack: Vec<OctreeTraversalItem> = Vec::with_capacity(64); 
    
    // Start with the root node
    stack.push(OctreeTraversalItem { node_idx, depth: 0 });
    
    // Process the stack iteratively
    while let Some(item) = stack.pop() {
        let node_idx = item.node_idx;
        let depth = item.depth;
        
        match arena.get(node_idx) {
            ArenaNode::Empty { .. } => {
                // Empty nodes don't contribute to force
                continue;
            },
            ArenaNode::Leaf { particle_idx: other_idx, particle_position, particle_mass, .. } => {
                // Skip self-interaction
                if *other_idx == particle_idx {
                    continue;
                }
                
                // Calculate force contribution
                let r = particle_position - particle.position();
                let r2 = r.norm_squared();
                let r_dist = SimdComplexField::simd_sqrt(r2 + g_soft * g_soft);
                let r_cubed = r_dist * r_dist * r_dist;
                
                if r_cubed > F::from(0.0).unwrap() {
                    let force = g * (*particle_mass) / r_cubed;
                    total_force += r * force;
                }
            },
            ArenaNode::Internal { bounds, center_of_mass, total_mass, children } => {
                // Calculate position difference and distance once
                let r = center_of_mass - particle.position();
                let r2 = r.norm_squared();
                
                // Determine if we should use approximation
                // Either because of depth or distance criterion
                let use_approximation = depth >= ArenaNode::<F, D>::MAX_FORCE_DEPTH || {
                    let node_width = bounds.max()[0] - bounds.min()[0];
                    let theta2 = theta * theta;
                    node_width * node_width < theta2 * r2
                };
                
                if use_approximation {
                    // Use approximation for force calculation
                    let r_dist = SimdComplexField::simd_sqrt(r2 + g_soft * g_soft);
                    let r_cubed = r_dist * r_dist * r_dist;
                    
                    if r_cubed > F::from(0.0).unwrap() {
                        let force = g * (*total_mass) / r_cubed;
                        total_force += r * force;
                    }
                } else {
                    // Process children individually
                    for &child_opt in children.iter() {
                        if let Some(child_idx) = child_opt {
                            stack.push(OctreeTraversalItem { node_idx: child_idx, depth: depth + 1 });
                        }
                    }
                }
            }
        }
    }
    
    total_force
}
