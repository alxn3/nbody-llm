use nalgebra::{SimdComplexField, SVector};
use rayon::prelude::*;
use std::collections::HashSet;

#[cfg(feature = "render")]
use {
    crate::render::PipelineType,
    wgpu::util::DeviceExt,
};

#[cfg(feature = "render")]
use crate::render::Renderable;

use crate::shared::{Bounds, Float, Integrator, LeapFrogIntegrator, Particle, Simulation, SimulationSettings, AABB};

// Node type for the Barnes-Hut tree
#[derive(Clone, Copy)]
enum NodeData {
    PointIndex(usize),
    PointCount(usize),
}

#[derive(Clone)]
pub struct Node<F: Float, const D: usize> {
    center_of_mass: SVector<F, D>,
    bounds: Bounds<F, D>,
    mass: F,
    node_data: NodeData,
    // Array of child nodes (indices into arena)
    children: Vec<Option<usize>>,
}

impl<F: Float, const D: usize> Node<F, D> {
    #[allow(dead_code)]
    fn new(bounds: Bounds<F, D>, node_data: NodeData) -> Self {
        Self {
            bounds,
            node_data,
            // fill with none - 2^D children for D dimensions
            children: vec![None; 1 << D],
            center_of_mass: SVector::<F, D>::zeros(),
            mass: F::from(0.0).unwrap(),
        }
    }
}

// Arena to store nodes compactly
#[derive(Clone)]
pub struct NodeArena<F: Float, const D: usize> {
    nodes: Vec<Node<F, D>>,
    free_indices: Vec<usize>,
    capacity_increment: usize,
}

impl<F: Float, const D: usize> NodeArena<F, D> {
    #[allow(dead_code)]
    fn new() -> Self {
        // Start with a reasonable capacity for most simulations
        let initial_capacity = 1024;
        Self { 
            nodes: Vec::with_capacity(initial_capacity),
            free_indices: Vec::with_capacity(initial_capacity / 4),
            capacity_increment: initial_capacity,
        }
    }
    
    fn with_capacity(capacity: usize) -> Self {
        Self { 
            nodes: Vec::with_capacity(capacity),
            free_indices: Vec::with_capacity(capacity / 4),
            capacity_increment: capacity,
        }
    }

    fn add_node(&mut self, node: Node<F, D>) -> usize {
        // Reuse a free slot if available
        if let Some(idx) = self.free_indices.pop() {
            self.nodes[idx] = node;
            idx
        } else {
            // Otherwise add to the end
            let idx = self.nodes.len();
            self.nodes.push(node);
            idx
        }
    }
    
    fn reserve(&mut self, additional: usize) {
        // Reserve space for more nodes if needed
        if self.nodes.len() + additional > self.nodes.capacity() {
            // Grow by capacity_increment or additional, whichever is larger
            let grow_by = std::cmp::max(self.capacity_increment, additional);
            self.nodes.reserve(grow_by);
        }
    }

    fn get(&self, idx: usize) -> &Node<F, D> {
        &self.nodes[idx]
    }

    fn get_mut(&mut self, idx: usize) -> &mut Node<F, D> {
        &mut self.nodes[idx]
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.free_indices.clear();
    }
    
    // Release a node back to the arena without removing it
    #[allow(dead_code)]
    fn release_node(&mut self, idx: usize) {
        self.free_indices.push(idx);
    }
    
    // Get current active node count
    #[allow(dead_code)]
    fn active_count(&self) -> usize {
        self.nodes.len() - self.free_indices.len()
    }
}

pub struct NodeIterator<'a, F: Float, const D: usize> {
    arena: &'a NodeArena<F, D>,
    current: Vec<(usize, usize)>, // (node_index, depth)
    next: Vec<(usize, usize)>,
    current_index: usize,
}

impl<'a, F: Float, const D: usize> Iterator for NodeIterator<'a, F, D> {
    type Item = (usize, &'a Node<F, D>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.current.len() {
            let (node_idx, depth) = self.current[self.current_index];
            let node = self.arena.get(node_idx);
            
            // Add all children to the next level
            for &child_idx in node.children.iter().flatten() {
                self.next.push((child_idx, depth + 1));
            }
            
            self.current_index += 1;
            Some((depth, node))
        } else if self.next.is_empty() {
            None
        } else {
            self.current = self.next.drain(..).collect();
            self.current_index = 0;
            self.next()
        }
    }
}

#[derive(Clone)]
pub struct BarnesHutSimulation<F: Float, const D: usize, P, I = LeapFrogIntegrator<F, D, P>>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    points: Vec<P>,
    arena: NodeArena<F, D>,
    root_idx: Option<usize>,
    bounds: Bounds<F, D>,
    integrator: I,
    settings: SimulationSettings<F>,
    elapsed: F,
    previous_positions: Vec<SVector<F, D>>,
    rebuild_threshold_squared: F,
    #[cfg(feature = "render")]
    points_vertex_buffer: Option<wgpu::Buffer>,
    #[cfg(feature = "render")]
    bounds_vertex_buffer: Option<wgpu::Buffer>,
    #[cfg(feature = "render")]
    num_bounds: u32,
}

impl<F: Float, const D: usize, P, I> BarnesHutSimulation<F, D, P, I>
where
    P: Particle<F, D> + Send + Sync,
    I: Integrator<F, D, P> + Sync,
    F: Send + Sync,
{
    fn estimate_node_count(&self) -> usize {
        // Each particle typically needs log(N) nodes, with some overhead
        let n = self.points.len();
        if n == 0 {
            return 0;
        }
        // Estimate based on tree depth - typically log(N) * N for N particles
        // with some additional overhead factor
        let log_n = (n as f64).log2().ceil() as usize;
        n * log_n * 2
    }

    fn add_point_to_tree(
        &mut self,
        point_index: usize,
        node_idx: Option<usize>,
        parent_idx: Option<usize>,
        orthant: usize,
    ) -> usize {
        match node_idx {
            Some(node_idx) => {
                assert!(self.arena.get(node_idx).bounds.contains(self.points[point_index].position()));
                
                match self.arena.get(node_idx).node_data {
                    NodeData::PointIndex(index) => {
                        let o1 = self.arena.get(node_idx).bounds.get_orthant(self.points[index].position());
                        let o2 = self.arena.get(node_idx).bounds.get_orthant(self.points[point_index].position());
                        
                        // Mark as internal node with 2 points
                        self.arena.get_mut(node_idx).node_data = NodeData::PointCount(2);
                        
                        // Create children for existing and new point
                        let child1_idx = self.add_point_to_tree(
                            index,
                            self.arena.get(node_idx).children[o1],
                            Some(node_idx),
                            o1,
                        );
                        
                        self.arena.get_mut(node_idx).children[o1] = Some(child1_idx);
                        
                        let child2_idx = self.add_point_to_tree(
                            point_index,
                            self.arena.get(node_idx).children[o2],
                            Some(node_idx),
                            o2,
                        );
                        
                        self.arena.get_mut(node_idx).children[o2] = Some(child2_idx);
                        
                        // Update center of mass and total mass
                        let node = self.arena.get_mut(node_idx);
                        let m1 = self.points[index].get_mass();
                        let m2 = self.points[point_index].get_mass();
                        let total_mass = m1 + m2;
                        
                        if total_mass > F::from(0.0).unwrap() {
                            node.mass = total_mass;
                            node.center_of_mass = (self.points[index].position().scale(m1) + 
                                                 self.points[point_index].position().scale(m2)) / total_mass;
                        }
                        
                        node_idx
                    },
                    NodeData::PointCount(count) => {
                        // Increment point count
                        let count = count + 1;
                        self.arena.get_mut(node_idx).node_data = NodeData::PointCount(count);
                        
                        // Add new point to appropriate child
                        let orthant = self.arena.get(node_idx).bounds.get_orthant(self.points[point_index].position());
                        let child_idx = self.add_point_to_tree(
                            point_index,
                            self.arena.get(node_idx).children[orthant],
                            Some(node_idx),
                            orthant,
                        );
                        
                        self.arena.get_mut(node_idx).children[orthant] = Some(child_idx);
                        
                        // Update center of mass and total mass
                        let node = self.arena.get_mut(node_idx);
                        let m1 = node.mass;
                        let m2 = self.points[point_index].get_mass();
                        let com1 = node.center_of_mass;
                        let com2 = *self.points[point_index].position();
                        let total_mass = m1 + m2;
                        
                        if total_mass > F::from(0.0).unwrap() {
                            node.mass = total_mass;
                            node.center_of_mass = (com1.scale(m1) + com2.scale(m2)) / total_mass;
                        }
                        
                        node_idx
                    }
                }
            },
            None => {
                let point = &self.points[point_index];
                let bounds = match parent_idx {
                    Some(parent_idx) => self.arena.get(parent_idx).bounds.create_orthant(orthant),
                    None => self.bounds.clone(),
                };
                
                // Prepare the node, reusing memory from the arena
                let new_node = Node {
                    bounds,
                    center_of_mass: *point.position(),
                    mass: point.get_mass(),
                    node_data: NodeData::PointIndex(point_index),
                    children: vec![None; 1 << D],
                };
                
                self.arena.add_node(new_node)
            }
        }
    }

    fn build_tree(&mut self) {
        // Estimate required capacity and ensure we have enough space
        let estimated_nodes = self.estimate_node_count();
        self.arena.reserve(estimated_nodes);
        
        // Clear existing tree data
        self.arena.clear();
        self.root_idx = None;
        
        // Save current positions for future movement tracking
        self.previous_positions.clear();
        self.previous_positions.reserve(self.points.len());
        self.previous_positions.extend(self.points.iter().map(|p| *p.position()));
        
        // If we have very few points, just build sequentially
        if self.points.len() < 1000 {
            for i in 0..self.points.len() {
                if let Some(root_idx) = self.root_idx {
                    let new_root_idx = self.add_point_to_tree(i, Some(root_idx), None, usize::MAX);
                    self.root_idx = Some(new_root_idx);
                } else {
                    self.root_idx = Some(self.add_point_to_tree(i, None, None, usize::MAX));
                }
            }
            return;
        }
        
        // Create the root node first
        let root_node = Node {
            bounds: self.bounds.clone(),
            center_of_mass: SVector::<F, D>::zeros(),
            mass: F::from(0.0).unwrap(),
            node_data: NodeData::PointCount(0),
            children: vec![None; 1 << D],
        };
        let root_idx = self.arena.add_node(root_node);
        self.root_idx = Some(root_idx);
        
        // Partition points by octant at the root level
        let mut octant_points: Vec<Vec<usize>> = vec![Vec::new(); 1 << D];
        
        for i in 0..self.points.len() {
            let octant = self.bounds.get_orthant(self.points[i].position());
            octant_points[octant].push(i);
        }
        
        // Process each octant in parallel
        let octant_trees: Vec<_> = octant_points.into_par_iter()
            .enumerate()
            .filter_map(|(octant, points)| {
                if points.is_empty() {
                    return None;
                }
                
                // Create a sequential tree for each octant
                let octant_bounds = self.bounds.create_orthant(octant);
                let mut com = SVector::<F, D>::zeros();
                let mut total_mass = F::from(0.0).unwrap();
                
                // Build a temporary tree sequentially for this octant
                let mut local_arena = NodeArena::<F, D>::new();
                let mut local_root = None;
                
                for &point_idx in &points {
                    if let Some(root) = local_root {
                        let new_root = self.add_point_to_local_tree(&mut local_arena, 
                                                                    point_idx, 
                                                                    Some(root), 
                                                                    None, 
                                                                    usize::MAX, 
                                                                    &octant_bounds);
                        local_root = Some(new_root);
                    } else {
                        local_root = Some(self.add_point_to_local_tree(&mut local_arena, 
                                                                      point_idx, 
                                                                      None, 
                                                                      None, 
                                                                      usize::MAX, 
                                                                      &octant_bounds));
                    }
                    
                    // Update center of mass contribution
                    let point_mass = self.points[point_idx].get_mass();
                    com += self.points[point_idx].position().scale(point_mass);
                    total_mass += point_mass;
                }
                
                // Calculate final center of mass for this octant
                if total_mass > F::from(0.0).unwrap() {
                    com = com / total_mass;
                }
                
                Some((octant, local_arena, local_root.unwrap(), total_mass, com, points.len()))
            })
            .collect();
        
        // Combine the results into the main tree
        let mut total_mass = F::from(0.0).unwrap();
        let mut weighted_com = SVector::<F, D>::zeros();
        let mut total_points = 0;
        
        for (octant, local_arena, local_root_idx, octant_mass, octant_com, point_count) in octant_trees {
            // Merge this octant's tree into the main arena
            let new_idx = self.merge_trees(&local_arena, local_root_idx);
            self.arena.get_mut(root_idx).children[octant] = Some(new_idx);
            
            // Update total mass and center of mass
            weighted_com += octant_com.scale(octant_mass);
            total_mass += octant_mass;
            total_points += point_count;
        }
        
        // Update root node properties
        if total_mass > F::from(0.0).unwrap() {
            let root = self.arena.get_mut(root_idx);
            root.mass = total_mass;
            root.center_of_mass = weighted_com / total_mass;
            root.node_data = NodeData::PointCount(total_points);
        }
    }
    
    // Helper method to add a point to a local tree (within a parallel section)
    fn add_point_to_local_tree(
        &self,
        arena: &mut NodeArena<F, D>,
        point_index: usize,
        node_idx: Option<usize>,
        parent_idx: Option<usize>,
        orthant: usize,
        bounds: &Bounds<F, D>,
    ) -> usize {
        match node_idx {
            Some(node_idx) => {
                let node_bounds = &arena.get(node_idx).bounds;
                assert!(node_bounds.contains(self.points[point_index].position()));
                
                match arena.get(node_idx).node_data {
                    NodeData::PointIndex(index) => {
                        let o1 = arena.get(node_idx).bounds.get_orthant(self.points[index].position());
                        let o2 = arena.get(node_idx).bounds.get_orthant(self.points[point_index].position());
                        
                        // Mark as internal node with 2 points
                        arena.get_mut(node_idx).node_data = NodeData::PointCount(2);
                        
                        // Create children for existing and new point
                        let child1_idx = self.add_point_to_local_tree(
                            arena,
                            index,
                            arena.get(node_idx).children[o1],
                            Some(node_idx),
                            o1,
                            bounds,
                        );
                        
                        arena.get_mut(node_idx).children[o1] = Some(child1_idx);
                        
                        let child2_idx = self.add_point_to_local_tree(
                            arena,
                            point_index,
                            arena.get(node_idx).children[o2],
                            Some(node_idx),
                            o2,
                            bounds,
                        );
                        
                        arena.get_mut(node_idx).children[o2] = Some(child2_idx);
                        
                        // Update center of mass and total mass
                        let node = arena.get_mut(node_idx);
                        let m1 = self.points[index].get_mass();
                        let m2 = self.points[point_index].get_mass();
                        let total_mass = m1 + m2;
                        
                        if total_mass > F::from(0.0).unwrap() {
                            node.mass = total_mass;
                            node.center_of_mass = (self.points[index].position().scale(m1) + 
                                                 self.points[point_index].position().scale(m2)) / total_mass;
                        }
                        
                        node_idx
                    },
                    NodeData::PointCount(count) => {
                        // Increment point count
                        let count = count + 1;
                        arena.get_mut(node_idx).node_data = NodeData::PointCount(count);
                        
                        // Add new point to appropriate child
                        let orthant = arena.get(node_idx).bounds.get_orthant(self.points[point_index].position());
                        let child_idx = self.add_point_to_local_tree(
                            arena,
                            point_index,
                            arena.get(node_idx).children[orthant],
                            Some(node_idx),
                            orthant,
                            bounds,
                        );
                        
                        arena.get_mut(node_idx).children[orthant] = Some(child_idx);
                        
                        // Update center of mass and total mass
                        let node = arena.get_mut(node_idx);
                        let m1 = node.mass;
                        let m2 = self.points[point_index].get_mass();
                        let com1 = node.center_of_mass;
                        let com2 = *self.points[point_index].position();
                        let total_mass = m1 + m2;
                        
                        if total_mass > F::from(0.0).unwrap() {
                            node.mass = total_mass;
                            node.center_of_mass = (com1.scale(m1) + com2.scale(m2)) / total_mass;
                        }
                        
                        node_idx
                    }
                }
            },
            None => {
                let point = &self.points[point_index];
                let node_bounds = match parent_idx {
                    Some(parent_idx) => arena.get(parent_idx).bounds.create_orthant(orthant),
                    None => bounds.clone(),
                };
                
                // Prepare the node, reusing memory from the arena
                let new_node = Node {
                    bounds: node_bounds,
                    center_of_mass: *point.position(),
                    mass: point.get_mass(),
                    node_data: NodeData::PointIndex(point_index),
                    children: vec![None; 1 << D],
                };
                
                arena.add_node(new_node)
            }
        }
    }
    
    // Merge a local tree into the main arena
    fn merge_trees(&mut self, local_arena: &NodeArena<F, D>, local_root_idx: usize) -> usize {
        // Create a mapping from local to main arena indices
        let mut index_map = std::collections::HashMap::new();
        
        // Recursive helper to copy nodes
        fn copy_node<F: Float, const D: usize>(
            local_arena: &NodeArena<F, D>,
            main_arena: &mut NodeArena<F, D>,
            local_idx: usize,
            index_map: &mut std::collections::HashMap<usize, usize>
        ) -> usize {
            // If we've already copied this node, return the mapped index
            if let Some(&main_idx) = index_map.get(&local_idx) {
                return main_idx;
            }
            
            // Get the local node to copy
            let local_node = local_arena.get(local_idx);
            
            // Create a copy of the node with empty children
            let mut main_node = Node {
                bounds: local_node.bounds.clone(),
                center_of_mass: local_node.center_of_mass,
                mass: local_node.mass,
                node_data: local_node.node_data,
                children: vec![None; 1 << D],
            };
            
            // Add the node to the main arena
            let main_idx = main_arena.add_node(main_node.clone());
            index_map.insert(local_idx, main_idx);
            
            // Recursively process children
            for (i, child_opt) in local_node.children.iter().enumerate() {
                if let Some(child_idx) = child_opt {
                    let main_child_idx = copy_node(local_arena, main_arena, *child_idx, index_map);
                    main_arena.get_mut(main_idx).children[i] = Some(main_child_idx);
                }
            }
            
            main_idx
        }
        
        // Copy the entire tree
        copy_node(local_arena, &mut self.arena, local_root_idx, &mut index_map)
    }

    fn update_tree(&mut self) {
        // If we have no previous positions or different number of points, do a full rebuild
        if self.previous_positions.len() != self.points.len() {
            self.build_tree();
            return;
        }

        // Identify particles that have moved significantly
        let moved_indices: HashSet<usize> = (0..self.points.len())
            .into_par_iter()
            .filter(|&i| {
                let current_pos = self.points[i].position();
                let prev_pos = &self.previous_positions[i];
                let dist_squared = (current_pos - prev_pos).norm_squared();
                dist_squared > self.rebuild_threshold_squared
            })
            .collect();

        // If too many particles moved, just rebuild the whole tree
        let rebuild_fraction = F::from(0.3).unwrap(); // Threshold for full rebuild
        if F::from(moved_indices.len()).unwrap() > rebuild_fraction * F::from(self.points.len()).unwrap() {
            self.build_tree();
            return;
        }

        // If we need to selectively update
        if !moved_indices.is_empty() {
            // First, remove moved particles from tree
            self.remove_particles_from_tree(&moved_indices);
            
            // Update previous positions and re-add moved particles
            for &i in &moved_indices {
                self.previous_positions[i] = *self.points[i].position();
                
                if let Some(root_idx) = self.root_idx {
                    let new_root_idx = self.add_point_to_tree(i, Some(root_idx), None, usize::MAX);
                    self.root_idx = Some(new_root_idx);
                } else {
                    self.root_idx = Some(self.add_point_to_tree(i, None, None, usize::MAX));
                }
            }
        }
    }

    fn remove_particles_from_tree(&mut self, indices: &HashSet<usize>) {
        // We can't easily remove individual particles, so rebuild with only non-moved particles
        if indices.is_empty() {
            return;
        }

        // Prepare a new arena and save the old one
        let estimated_nodes = self.estimate_node_count();
        let new_arena = NodeArena::with_capacity(estimated_nodes);
        let _old_arena = std::mem::replace(&mut self.arena, new_arena);
        let _old_root_idx = self.root_idx;
        self.root_idx = None;
        
        // Re-add only the particles that haven't moved
        for i in 0..self.points.len() {
            if !indices.contains(&i) {
                if let Some(root_idx) = self.root_idx {
                    let new_root_idx = self.add_point_to_tree(i, Some(root_idx), None, usize::MAX);
                    self.root_idx = Some(new_root_idx);
                } else {
                    self.root_idx = Some(self.add_point_to_tree(i, None, None, usize::MAX));
                }
            }
        }
    }

    fn calc_force(&self, node_idx: usize, point: &P) -> SVector<F, D> {
        // Use non-recursive traversal for better performance
        let mut stack = Vec::with_capacity(32); // Preallocate stack for typical tree depth
        stack.push(node_idx);
        let mut force = SVector::<F, D>::zeros();
        
        // Cache point position and settings once
        let point_pos = point.position();
        let theta2 = self.settings().theta2;
        let g_soft = self.settings().g_soft;
        let g_soft2 = g_soft * g_soft; // Precompute squared softening
        let g = self.settings().g;
        
        // Avoid pointer checks for most common case
        let point_idx = match point {
            p if std::ptr::eq(p as *const P, &self.points[0] as *const P) => Some(0),
            p if self.points.len() > 1 && std::ptr::eq(p as *const P, &self.points[1] as *const P) => Some(1),
            _ => None,
        };
        
        while let Some(current_idx) = stack.pop() {
            let node = self.arena.get(current_idx);
            
            // Calculate displacement vector once
            let r = node.center_of_mass - *point_pos;
            let r2 = r.norm_squared();
            
            // Skip extremely close nodes or self interactions
            if r2 < F::from(1e-10).unwrap() {
                continue;
            }
            
            // Fast path: if node is far enough away, use multipole approximation
            let s2 = node.bounds.width * node.bounds.width;
            if s2 < theta2 * r2 {
                // Avoid expensive multiple divisions by computing inverse distance factors
                let r_soft2 = r2 + g_soft2;
                
                // Use fast inverse square root approximation when possible
                #[cfg(feature = "fast_math")]
                let inv_r = F::from(1.0).unwrap() / SimdComplexField::simd_sqrt(r_soft2);
                #[cfg(not(feature = "fast_math"))]
                let inv_r = F::from(1.0).unwrap() / SimdComplexField::simd_sqrt(r_soft2);
                
                let inv_r3 = inv_r * inv_r * inv_r;
                
                // Apply force with minimal operations
                let force_magnitude = g * node.mass * inv_r3;
                force += r.scale(force_magnitude);
                continue;
            }
            
            // Handle special case for leaf nodes
            match node.node_data {
                NodeData::PointIndex(idx) => {
                    // Avoid calculating forces with self
                    if let Some(p_idx) = point_idx {
                        if p_idx == idx {
                            continue;
                        }
                    } else if std::ptr::eq(point as *const P, &self.points[idx] as *const P) {
                        continue;
                    }
                    
                    // Direct calculation for leaf nodes
                    let r_soft2 = r2 + g_soft2;
                    let inv_r = F::from(1.0).unwrap() / SimdComplexField::simd_sqrt(r_soft2);
                    let inv_r3 = inv_r * inv_r * inv_r;
                    let force_magnitude = g * node.mass * inv_r3;
                    force += r.scale(force_magnitude);
                },
                NodeData::PointCount(_) => {
                    // For internal nodes, add children to stack in reverse order
                    // This improves cache locality by processing nearby nodes together
                    for i in (0..node.children.len()).rev() {
                        if let Some(child_idx) = node.children[i] {
                            stack.push(child_idx);
                        }
                    }
                }
            }
        }
        
        force
    }
    
    fn get_nodes_for_rendering(&self) -> Box<dyn Iterator<Item = (usize, &Node<F, D>)> + '_> {
        match self.root_idx {
            Some(root_idx) => Box::new(NodeIterator {
                arena: &self.arena,
                current: vec![(root_idx, 0)],
                next: Vec::new(),
                current_index: 0,
            }),
            None => Box::new(std::iter::empty())
        }
    }
}

impl<F: Float, const D: usize, P, I> Simulation<F, D, P, I> for BarnesHutSimulation<F, D, P, I>
where
    P: Particle<F, D> + Send + Sync,
    I: Integrator<F, D, P> + Sync,
    F: Send + Sync,
{
    fn new(points: Vec<P>, integrator: I, bounds: Bounds<F, D>) -> Self {
        // Set a reasonable threshold - particles that move more than 5% of the domain size
        // will trigger a tree update
        let domain_size = bounds.width;
        let rebuild_threshold = domain_size * F::from(0.05).unwrap();
        
        // Estimate initial node capacity
        let n = points.len();
        let initial_capacity = if n > 0 {
            // Estimate based on tree depth - typically log(N) * N for N particles
            let log_n = (n as f64).log2().ceil() as usize;
            n * log_n * 2
        } else {
            1024 // Default initial capacity
        };
        
        Self {
            points,
            arena: NodeArena::with_capacity(initial_capacity),
            root_idx: None,
            bounds,
            integrator,
            settings: SimulationSettings::default(),
            elapsed: F::from(0.0).unwrap(),
            previous_positions: Vec::new(),
            rebuild_threshold_squared: rebuild_threshold * rebuild_threshold,
            #[cfg(feature = "render")]
            points_vertex_buffer: None,
            #[cfg(feature = "render")]
            bounds_vertex_buffer: None,
            #[cfg(feature = "render")]
            num_bounds: 0,
        }
    }

    fn init(&mut self) {
        self.integrator.init();
        self.elapsed = F::from(0.0).unwrap();
        self.build_tree();
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
        // Update tree selectively instead of rebuilding
        self.update_tree();

        // Reset accelerations
        for point in self.points.iter_mut() {
            point.acceleration_mut().fill(F::from(0.0).unwrap());
        }

        if let Some(root_idx) = self.root_idx {
            // Calculate forces in parallel
            let forces: Vec<SVector<F, D>> = (0..self.points.len())
                .into_par_iter()
                .map(|i| self.calc_force(root_idx, &self.points[i]))
                .collect();
            
            // Apply calculated forces
            for (i, force) in forces.into_iter().enumerate() {
                *self.points[i].acceleration_mut() += force;
            }
        }
    }

    fn step_by(&mut self, dt: F) {
        self.integrator.integrate_pre_force(&mut self.points, dt);
        self.points.retain(|p| self.bounds.contains(p.position()));
        self.update_forces();
        self.integrator.integrate_after_force(&mut self.points, dt);
        self.elapsed += dt;
    }

    fn add_point(&mut self, point: P) {
        self.points.push(point);
        // Force a rebuild on next update since positions don't match
    }

    fn remove_point(&mut self, index: usize) {
        self.points.swap_remove(index);
    }

    fn get_points(&self) -> &Vec<P> {
        &self.points
    }
}

#[cfg(feature = "render")]
impl<F, P, I> Renderable for BarnesHutSimulation<F, 3, P, I>
where
    F: Float + Send + Sync,
    P: Particle<F, 3> + Send + Sync,
    I: Integrator<F, 3, P> + Sync,
{
    fn render(&mut self, renderer: &mut crate::render::Renderer) {
        use crate::shared::AABB;

        if let Some(ref points_buffer) = self.points_vertex_buffer {
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

            renderer.context.queue.write_buffer(
                points_buffer,
                0,
                bytemuck::cast_slice(&position_data),
            );

            renderer.set_pipeline(PipelineType::Points);
            let render_pass = renderer.get_render_pass();
            render_pass.set_vertex_buffer(0, points_buffer.slice(..));
            render_pass.draw(0..4, 0..self.points.len() as u32);
        }

        if let Some(ref bounds_buffer) = self.bounds_vertex_buffer {
            let bounds_data = match self.root_idx {
                Some(_) => {
                    let mut bounds_data = Vec::new();
                    let mut num_bounds = 0;
                    let bounds = self.get_nodes_for_rendering().collect::<Vec<_>>();
                    let max_depth = bounds.last().map_or(0, |&(depth, _)| depth);
                    for (depth, node) in bounds {
                        let s = (depth as f32) / (max_depth as f32 + 0.001) * 0.7 + 0.3;
                        bounds_data.extend(
                            node.bounds
                                .min()
                                .iter()
                                .chain(node.bounds.max().iter())
                                .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                                .chain::<[f32; 4]>([(1. - s * s) * 0.5, s * s, (1. - s) * 0.5, s])
                                .collect::<Vec<f32>>(),
                        );
                        num_bounds += 1;
                    }
                    self.num_bounds = num_bounds;
                    bounds_data
                }
                None => {
                    self.num_bounds = 1;
                    self.bounds
                        .min()
                        .iter()
                        .chain(self.bounds.max().iter())
                        .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                        .chain::<[f32; 4]>([1.0, 1.0, 0.0, 1.0])
                        .collect::<Vec<f32>>()
                }
            };

            renderer.context.queue.write_buffer(
                bounds_buffer,
                0,
                bytemuck::cast_slice(&bounds_data),
            );

            renderer.set_pipeline(PipelineType::AABB);
            let render_pass = renderer.get_render_pass();
            render_pass.set_vertex_buffer(0, bounds_buffer.slice(..));
            render_pass.draw(0..16, 0..self.num_bounds);
        }
    }

    fn render_init(&mut self, context: &crate::render::Context) {
        // Create points buffer with initial data
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

        let points_vertex_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Points Vertex Buffer"),
            contents: bytemuck::cast_slice(&point_position_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Create bounds buffer with sufficient size
        // Each node needs 3 (min) + 3 (max) + 4 (color) = 10 floats * 4 bytes = 40 bytes per node
        const MAX_NODES: usize = 20000;
        const FLOATS_PER_NODE: usize = 10; // 3 for min, 3 for max, 4 for color
        const BUFFER_SIZE: usize = MAX_NODES * FLOATS_PER_NODE * std::mem::size_of::<f32>();
        
        // Create with initial bounds data
        let initial_bounds_data: Vec<f32> = self.bounds
            .min()
            .iter()
            .chain(self.bounds.max().iter())
            .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
            .chain([1.0, 1.0, 0.0, 1.0].iter().copied())
            .collect();

        let bounds_vertex_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bounds Vertex Buffer"),
            size: BUFFER_SIZE as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Write initial data
        context.queue.write_buffer(
            &bounds_vertex_buffer,
            0,
            bytemuck::cast_slice(&initial_bounds_data),
        );

        self.points_vertex_buffer = Some(points_vertex_buffer);
        self.bounds_vertex_buffer = Some(bounds_vertex_buffer);
    }
}
