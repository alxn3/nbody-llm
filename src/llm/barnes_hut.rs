use nalgebra::{SimdComplexField, SVector};
use rayon::prelude::*;

#[cfg(feature = "render")]
use {
    crate::render::{PipelineType, Renderer, Context},
    wgpu::util::DeviceExt,
};

#[cfg(feature = "render")]
use crate::render::Renderable;

use crate::shared::{Bounds, Float, Integrator, LeapFrogIntegrator, Particle, Simulation, SimulationSettings, AABB};
use std::fmt::Debug;
use std::marker::PhantomData;

// Octree node used for Barnes-Hut algorithm
#[derive(Debug, Clone)]
pub enum OctreeNode<F: Float, const D: usize> {
    Internal {
        bounds: Bounds<F, D>,
        center_of_mass: SVector<F, D>,
        total_mass: F,
        children: [Option<Box<OctreeNode<F, D>>>; 8],
    },
    Leaf {
        bounds: Bounds<F, D>,
        particle_idx: usize,
        particle_position: SVector<F, D>, // Store only position instead of full particle
        particle_mass: F,                 // Store only mass instead of full particle
    },
    Empty {
        bounds: Bounds<F, D>,
    },
}

// Add a new struct to use in the iterative version of calculate_force
#[derive(Clone)]
struct OctreeTraversalItem<'a, F: Float, const D: usize> {
    node: &'a OctreeNode<F, D>,
    depth: usize,
}

impl<F: Float, const D: usize> OctreeNode<F, D> {
    // Define constants at the impl level so they're available to all methods
    const MAX_TREE_DEPTH: usize = 16; // Limit tree depth to prevent stack overflow
    const MAX_FORCE_DEPTH: usize = 32; // Maximum depth for force calculation
    
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

    pub fn insert<P: Particle<F, D>>(&mut self, particle_idx: usize, particle: &P) {
        self.insert_with_depth(particle_idx, particle, 0);
    }

    // Modified insert method that tracks depth and uses references
    fn insert_with_depth<P: Particle<F, D>>(&mut self, particle_idx: usize, particle: &P, depth: usize) {
        // Prevent excessive tree depth
        if depth >= Self::MAX_TREE_DEPTH {
            // If we've reached max depth, just update the current node
            match self {
                Self::Empty { bounds } => {
                    let bounds_clone = bounds.clone();
                    *self = Self::Leaf {
                        bounds: bounds_clone,
                        particle_idx,
                        particle_position: *particle.position(),
                        particle_mass: particle.get_mass(),
                    };
                }
                Self::Leaf { bounds, particle_idx: _existing_idx, particle_position, particle_mass } => {
                    // Convert leaf to internal, but don't split further
                    let bounds_clone = bounds.clone();
                    let new_mass = *particle_mass + particle.get_mass();
                    let center_of_mass = (*particle_position * *particle_mass + 
                                        *particle.position() * particle.get_mass()) / new_mass;
                    
                    *self = Self::Internal {
                        bounds: bounds_clone.clone(),
                        center_of_mass,
                        total_mass: new_mass,
                        children: Default::default(),
                    };
                }
                Self::Internal { center_of_mass, total_mass, .. } => {
                    // Just update the center of mass and total mass for the current node
                    let particle_mass = particle.get_mass();
                    let new_total_mass = *total_mass + particle_mass;
                    
                    if new_total_mass > F::from(0.0).unwrap() {
                        *center_of_mass = (*center_of_mass * *total_mass + *particle.position() * particle_mass) / new_total_mass;
                        *total_mass = new_total_mass;
                    }
                }
            }
            return;
        }

        match self {
            Self::Empty { bounds } => {
                let bounds_clone = bounds.clone();
                *self = Self::Leaf {
                    bounds: bounds_clone,
                    particle_idx,
                    particle_position: *particle.position(),
                    particle_mass: particle.get_mass(),
                };
            }
            Self::Leaf {
                bounds,
                particle_idx: existing_idx,
                particle_position: existing_position,
                particle_mass: existing_mass,
            } => {
                // Convert leaf to internal node and insert both particles
                let bounds_clone = bounds.clone();
                let existing_idx_clone = *existing_idx;
                let existing_position_clone = *existing_position;
                let existing_mass_clone = *existing_mass;

                *self = Self::new_internal(bounds_clone.clone());
                
                // Create temporary particle-like object for existing particle
                let existing_particle = ParticleProxy {
                    position: existing_position_clone,
                    mass: existing_mass_clone,
                };
                
                // Insert existing particle data at next depth
                if let Self::Internal { children, bounds, .. } = self {
                    // Find which octant the existing particle goes into
                    let octant_idx = Self::get_octant_idx(bounds, &existing_position_clone);
                    
                    if children[octant_idx].is_none() {
                        let child_bounds = Self::get_octant_bounds(bounds, octant_idx);
                        let mut child = OctreeNode::new_empty(child_bounds);
                        child.insert_with_depth(existing_idx_clone, &existing_particle, depth + 1);
                        children[octant_idx] = Some(Box::new(child));
                    } else if let Some(child) = &mut children[octant_idx] {
                        child.insert_with_depth(existing_idx_clone, &existing_particle, depth + 1);
                    }
                    
                    // Now insert the new particle
                    let octant_idx = Self::get_octant_idx(bounds, particle.position());
                    
                    if children[octant_idx].is_none() {
                        let child_bounds = Self::get_octant_bounds(bounds, octant_idx);
                        let mut child = OctreeNode::new_empty(child_bounds);
                        child.insert_with_depth(particle_idx, particle, depth + 1);
                        children[octant_idx] = Some(Box::new(child));
                    } else if let Some(child) = &mut children[octant_idx] {
                        child.insert_with_depth(particle_idx, particle, depth + 1);
                    }
                }
            }
            Self::Internal {
                bounds,
                center_of_mass,
                total_mass,
                children,
            } => {
                // Update center of mass and total mass
                let particle_mass = particle.get_mass();
                let new_total_mass = *total_mass + particle_mass;
                
                if new_total_mass > F::from(0.0).unwrap() {
                    // Calculate new center of mass
                    *center_of_mass = (*center_of_mass * *total_mass + *particle.position() * particle_mass) / new_total_mass;
                    *total_mass = new_total_mass;
                }

                // Find the octant this particle belongs to and insert it there
                let octant_idx = Self::get_octant_idx(bounds, particle.position());
                
                if children[octant_idx].is_none() {
                    let child_bounds = Self::get_octant_bounds(bounds, octant_idx);
                    let mut child = OctreeNode::new_empty(child_bounds);
                    child.insert_with_depth(particle_idx, particle, depth + 1);
                    children[octant_idx] = Some(Box::new(child));
                } else if let Some(child) = &mut children[octant_idx] {
                    child.insert_with_depth(particle_idx, particle, depth + 1);
                }
            }
        }
    }

    pub fn get_octant_idx(bounds: &Bounds<F, D>, position: &SVector<F, D>) -> usize {
        let center = bounds.center();
        let mut idx = 0;
        
        if position[0] >= center[0] {
            idx |= 1;
        }
        if position[1] >= center[1] {
            idx |= 2;
        }
        if position[2] >= center[2] {
            idx |= 4;
        }
        
        idx
    }

    pub fn get_octant_bounds(bounds: &Bounds<F, D>, octant_idx: usize) -> Bounds<F, D> {
        let center = bounds.center();
        let min = bounds.min();
        let max = bounds.max();
        
        let mut new_center = SVector::<F, D>::zeros();
        let half_width = bounds.half_width * F::from(0.5).unwrap();
        
        // x dimension
        if octant_idx & 1 == 0 {
            new_center[0] = (min[0] + center[0]) * F::from(0.5).unwrap();
        } else {
            new_center[0] = (center[0] + max[0]) * F::from(0.5).unwrap();
        }
        
        // y dimension
        if octant_idx & 2 == 0 {
            new_center[1] = (min[1] + center[1]) * F::from(0.5).unwrap();
        } else {
            new_center[1] = (center[1] + max[1]) * F::from(0.5).unwrap();
        }
        
        // z dimension
        if octant_idx & 4 == 0 {
            new_center[2] = (min[2] + center[2]) * F::from(0.5).unwrap();
        } else {
            new_center[2] = (center[2] + max[2]) * F::from(0.5).unwrap();
        }
        
        Bounds::new(new_center, half_width * F::from(2.0).unwrap())
    }

    pub fn calculate_force<T: Particle<F, D>>(&self, particle: &T, particle_idx: usize, theta: F, g: F, g_soft2: F) -> SVector<F, D> {
        let mut total_force = SVector::<F, D>::zeros();
        // Pre-allocate stack with capacity to reduce reallocations
        let mut stack: Vec<OctreeTraversalItem<F, D>> = Vec::with_capacity(64); 
        
        // Start with the root node
        stack.push(OctreeTraversalItem { node: self, depth: 0 });
        
        // Process the stack iteratively instead of recursively
        while let Some(item) = stack.pop() {
            let node = item.node;
            let depth = item.depth;
            
            match node {
                Self::Empty { .. } => {
                    // Empty nodes don't contribute to force
                    continue;
                },
                Self::Leaf { particle_idx: other_idx, particle_position, particle_mass, .. } => {
                    // Skip self-interaction
                    if *other_idx == particle_idx {
                        continue;
                    }
                    
                    // Direct calculation for leaf nodes using stored position and mass
                    let r = particle.position() - particle_position;
                    let r2 = r.norm_squared() + g_soft2;
                    
                    // Check for extremely small distances to prevent numerical instability
                    if r2 < F::from(1e-10).unwrap() {
                        continue;
                    }
                    
                    let r_dist = SimdComplexField::simd_sqrt(r2);
                    let r_cubed = r_dist * r2;
                    
                    if r_cubed > F::from(0.0).unwrap() {
                        let force = g * (*particle_mass) / r_cubed;
                        total_force += -r * force;
                    }
                },
                Self::Internal { bounds, center_of_mass, total_mass, children } => {
                    // Skip if we've reached maximum depth to prevent stack overflow
                    if depth >= Self::MAX_FORCE_DEPTH {
                        // Just use approximation at max depth
                        let r = particle.position() - center_of_mass;
                        let r2 = r.norm_squared() + g_soft2;
                        
                        if r2 < F::from(1e-10).unwrap() {
                            continue;
                        }
                        
                        let r_dist = SimdComplexField::simd_sqrt(r2);
                        let r_cubed = r_dist * r2;
                        
                        if r_cubed > F::from(0.0).unwrap() {
                            let force = g * (*total_mass) / r_cubed;
                            total_force += -r * force;
                        }
                        continue;
                    }
                    
                    // Check if we can use approximation
                    let s = bounds.max()[0] - bounds.min()[0];  // Size of the region
                    let d = (center_of_mass - particle.position()).norm();
                    
                    // Prevent division by zero or extremely small values
                    if d < F::from(1e-10).unwrap() {
                        continue;
                    }
                    
                    if d > F::from(0.0).unwrap() && s / d < theta {
                        // Use approximation - treat internal node as a single body
                        let r = particle.position() - center_of_mass;
                        let r2 = r.norm_squared() + g_soft2;
                        
                        if r2 < F::from(1e-10).unwrap() {
                            continue;
                        }
                        
                        let r_dist = SimdComplexField::simd_sqrt(r2);
                        let r_cubed = r_dist * r2;
                        
                        if r_cubed > F::from(0.0).unwrap() {
                            let force = g * (*total_mass) / r_cubed;
                            total_force += -r * force;
                        }
                    } else {
                        // Add non-empty children to the stack
                        for child in children.iter().filter_map(|c| c.as_ref()) {
                            stack.push(OctreeTraversalItem { node: child, depth: depth + 1 });
                        }
                    }
                }
            }
        }
        
        total_force
    }
}

// A simple proxy for storing just the position and mass of a particle
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
    octree: Option<OctreeNode<F, D>>, // Removed the 'static lifetime
    rebuild_counter: usize, // Counter to determine when to rebuild the tree
    #[cfg(feature = "render")]
    points_vertex_buffer: Option<wgpu::Buffer>,
    #[cfg(feature = "render")]
    bounds_vertex_buffer: Option<wgpu::Buffer>,
    _marker: PhantomData<P>, // Marker to handle lifetimes
}

impl<F: Float, const D: usize, P, I> Simulation<F, D, P, I> for BarnesHutSimulation<F, D, P, I>
where
    P: Particle<F, D> + Send + Sync,
    I: Integrator<F, D, P>,
{
    fn new(points: Vec<P>, integrator: I, bounds: Bounds<F, D>) -> Self {
        Self {
            points,
            bounds,
            integrator,
            settings: SimulationSettings::default(),
            elapsed: F::from(0.0).unwrap(),
            theta: F::from(0.5).unwrap(), // Default opening angle
            octree: None,
            rebuild_counter: 0,
            #[cfg(feature = "render")]
            points_vertex_buffer: None,
            #[cfg(feature = "render")]
            bounds_vertex_buffer: None,
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

        // Only rebuild the tree periodically to save computation
        // The rebuild_counter determines how often we rebuild
        self.rebuild_counter += 1;
        if self.rebuild_counter >= 5 || self.octree.is_none() {
            self.build_octree();
            self.rebuild_counter = 0;
        }

        // Store these values to avoid borrowing conflicts
        let theta = self.theta;
        let g = self.settings.g;
        let g_soft2 = self.settings.g_soft * self.settings.g_soft;
        
        // Calculate forces using Barnes-Hut approximation in parallel
        if let Some(octree) = &self.octree {
            // Parallel force calculation - only use for larger particle counts
            // as parallelization has overhead
            if self.points.len() > 1000 {
                let accelerations: Vec<_> = self.points.par_iter().enumerate().map(|(i, point)| {
                    octree.calculate_force::<P>(point, i, theta, g, g_soft2)
                }).collect();
                
                // Apply the calculated accelerations
                for (i, accel) in accelerations.into_iter().enumerate() {
                    if i < self.points.len() {
                        *self.points[i].acceleration_mut() = accel;
                    }
                }
            } else {
                // Sequential processing for small particle counts
                for (i, point) in self.points.iter_mut().enumerate() {
                    *point.acceleration_mut() = octree.calculate_force::<P>(point, i, theta, g, g_soft2);
                }
            }
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
        self.points.remove(index);
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
    I: Integrator<F, 3, P>,
{
    fn render(&mut self, renderer: &mut crate::render::Renderer) {
        let queue = &renderer.context.queue;

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
        let bodies_vertex_buffer = self.points_vertex_buffer.as_ref().unwrap();

        queue.write_buffer(
            bodies_vertex_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );

        {
            renderer.set_pipeline(PipelineType::Points);
            let render_pass = renderer.get_render_pass();
            render_pass.set_vertex_buffer(0, bodies_vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..self.points.len() as u32);
        }

        {
            renderer.set_pipeline(PipelineType::AABB);
            let render_pass: &mut wgpu::RenderPass<'_> = renderer.get_render_pass();
            render_pass.set_vertex_buffer(0, self.bounds_vertex_buffer.as_ref().unwrap().slice(..));
            render_pass.draw(0..16, 0..1);
        }
    }

    fn render_init(&mut self, context: &crate::render::Context) {
        use crate::shared::AABB;

        let device = &context.device;
        let queue = &context.queue;

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

        // Use AABB trait methods to get min and max
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

        let bounds_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bounds Vertex Buffer"),
            contents: bytemuck::cast_slice(&bounds_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        self.points_vertex_buffer = Some(points_vertex_buffer);
        self.bounds_vertex_buffer = Some(bounds_vertex_buffer);
    }
}

impl<F: Float, const D: usize, P, I> BarnesHutSimulation<F, D, P, I>
where
    P: Particle<F, D> + Send + Sync,
    I: Integrator<F, D, P>,
{
    // Build octree from current particle positions using references
    fn build_octree(&mut self) {
        // Create root node
        let mut root: OctreeNode<F, D> = OctreeNode::new_empty(self.bounds.clone());
        
        // Insert all particles that are within bounds
        for (idx, particle) in self.points.iter().enumerate() {
            if self.bounds.contains(particle.position()) {
                root.insert(idx, particle);
            }
        }
        
        // Store the octree - now without 'static lifetime issues
        self.octree = Some(root);
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
