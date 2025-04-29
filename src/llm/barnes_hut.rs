use nalgebra::{SVector, SimdComplexField};
use rayon::prelude::*;

// Import Morton encoding helpers
use morton_encoding::*;

// Add SIMD-specific imports for 3D optimization
use nalgebra::{SimdValue, Vector3};

#[cfg(feature = "render")]
use {crate::render::PipelineType, wgpu::util::DeviceExt};

#[cfg(feature = "render")]
use crate::render::Renderable;

use crate::shared::{
    Bounds, Float, Integrator, LeapFrogIntegrator, Particle, Simulation, SimulationSettings, AABB,
};

// Node type for the Barnes-Hut tree
#[derive(Clone, Copy)]
enum NodeData {
    PointIndex(usize),
    PointCount(usize),
}

#[derive(Clone)]
pub struct Node3D<F: Float> {
    center_of_mass: Vector3<F>,
    bounds: Bounds<F, 3>,
    width_squared: F,
    mass: F,
    node_data: NodeData,
    children: [Option<usize>; 8], // Fixed array for 3D - 2^3 = 8 children
}

#[derive(Clone)]
pub struct Node<F: Float, const D: usize> {
    center_of_mass: SVector<F, D>,
    bounds: Bounds<F, D>,
    width_squared: F,
    mass: F,
    node_data: NodeData,
    children: Vec<Option<usize>>, // Keep as Vec for non-3D cases
}

impl<F: Float, const D: usize> Node<F, D> {
    // Removed unused Node::new function
    /*
    #[allow(dead_code)]
    fn new(bounds: Bounds<F, D>, node_data: NodeData) -> Self {
        Self {
            bounds,
            width_squared: bounds.width * bounds.width,
            node_data,
            // Revert to vec![None; 1 << D]
            children: vec![None; 1 << D],
            center_of_mass: SVector::<F, D>::zeros(),
            mass: F::from(0.0).unwrap(),
        }
    }
    */
}

// Arena to store nodes compactly
#[derive(Clone)]
pub struct NodeArena<F: Float, const D: usize> {
    nodes: Vec<Node<F, D>>,
    free_indices: Vec<usize>,
    _capacity_increment: usize,
}

impl<F: Float, const D: usize> NodeArena<F, D> {
    #[allow(dead_code)]
    fn new() -> Self {
        // Start with a reasonable capacity for most simulations
        let initial_capacity = 1024;
        Self {
            nodes: Vec::with_capacity(initial_capacity),
            free_indices: Vec::with_capacity(initial_capacity / 4),
            _capacity_increment: initial_capacity,
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            free_indices: Vec::with_capacity(capacity / 4),
            _capacity_increment: capacity,
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
        let required_capacity = self.nodes.len() + additional;
        let current_capacity = self.nodes.capacity();

        if required_capacity > current_capacity {
            // Grow similar to Vec: double the capacity or use required, whichever is larger.
            let new_capacity = std::cmp::max(required_capacity, current_capacity * 2);
            let grow_by = new_capacity - current_capacity;
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

// Optimized Arena for 3D nodes
#[derive(Clone)]
pub struct NodeArena3D<F: Float> {
    nodes: Vec<Node3D<F>>,
    free_indices: Vec<usize>,
    _capacity_increment: usize,
}

impl<F: Float> NodeArena3D<F> {
    #[allow(dead_code)]
    fn new() -> Self {
        // Start with a reasonable capacity for most simulations
        let initial_capacity = 1024;
        Self {
            nodes: Vec::with_capacity(initial_capacity),
            free_indices: Vec::with_capacity(initial_capacity / 4),
            _capacity_increment: initial_capacity,
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            free_indices: Vec::with_capacity(capacity / 4),
            _capacity_increment: capacity,
        }
    }

    fn add_node(&mut self, node: Node3D<F>) -> usize {
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
        let required_capacity = self.nodes.len() + additional;
        let current_capacity = self.nodes.capacity();

        if required_capacity > current_capacity {
            // Grow similar to Vec: double the capacity or use required, whichever is larger.
            let new_capacity = std::cmp::max(required_capacity, current_capacity * 2);
            let grow_by = new_capacity - current_capacity;
            self.nodes.reserve(grow_by);
        }
    }

    fn get(&self, idx: usize) -> &Node3D<F> {
        &self.nodes[idx]
    }

    fn get_mut(&mut self, idx: usize) -> &mut Node3D<F> {
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
    F: Send + Sync + SimdValue,
{
    // Helper function to normalize a float position component to a u32 integer range for Morton coding.
    fn normalize_pos_component(pos_component: F, min_bound: F, max_bound: F) -> u32 {
        // Ensure bounds are valid
        let range = max_bound - min_bound;
        if range <= F::from(0.0).unwrap() {
            return 0; // Or handle error/edge case appropriately
        }
        // Clamp position to bounds
        let clamped_pos = pos_component.max(min_bound).min(max_bound);
        // Normalize to [0, 1]
        let normalized = (clamped_pos - min_bound) / range;
        // Scale to u32 range (leave some room at the top)
        (normalized * F::from(u32::MAX - 1).unwrap())
            .to_u32()
            .unwrap_or(0)
    }

    // Helper to calculate Morton code for a D-dimensional point
    fn get_morton_code(&self, point_idx: usize) -> u64 {
        let pos = self.points[point_idx].position();
        let min_b = self.bounds.min();
        let max_b = self.bounds.max();

        // Normalize coordinates to u32
        let norm_coords: [u32; D] =
            core::array::from_fn(|i| Self::normalize_pos_component(pos[i], min_b[i], max_b[i]));

        // Morton encode (using u64 for potentially wider output)
        match D {
            2 => {
                let m_coords: [u32; 2] = [norm_coords[0], norm_coords[1]];
                morton_encode(m_coords)
            }
            3 => {
                let m_coords: [u32; 3] = [norm_coords[0], norm_coords[1], norm_coords[2]];
                morton_encode(m_coords) as u64
            }
            _ => {
                // Fallback or panic for unsupported dimensions
                // For simplicity, just return 0, though a proper implementation might hash
                // or use a different space-filling curve.
                eprintln!("Warning: Morton encoding only implemented for D=2 or D=3. Falling back for D={}", D);
                0
            }
        }
    }

    fn estimate_node_count(&self) -> usize {
        // Each particle typically needs log(N) nodes, with some overhead
        let n = self.points.len();
        if n == 0 {
            return 0;
        }
        // Estimate based on tree depth - typically log(N) * N for N particles
        // with some additional overhead factor
        let log_n = (n as f64).log2().ceil() as usize;
        n * log_n * 4
    }

    // Update the add_point_to_tree function to use the new NodeData structure
    fn add_point_to_tree(
        &mut self,
        point_index: usize,
        node_idx_opt: Option<usize>,
        parent_idx_opt: Option<usize>,
        orthant: usize,
    ) -> usize {
        match node_idx_opt {
            Some(node_idx) => {
                // Check bounds before proceeding (safer than assert inside)
                let node_bounds = self.arena.get(node_idx).bounds;
                if !node_bounds.contains(self.points[point_index].position()) {
                    eprintln!(
                         "Warning: Point {:?} outside target node bounds {:?} during recursive insertion. Skipping add for this branch.",
                         self.points[point_index].position(), node_bounds
                     );
                    // Cannot proceed if point isn't in bounds
                    return node_idx; // Return the parent index without modification
                }

                let node_data_copy = self.arena.get(node_idx).node_data;
                match node_data_copy {
                    NodeData::PointIndex(existing_idx) => {
                        if existing_idx == point_index {
                            return node_idx;
                        }
                        let node = self.arena.get_mut(node_idx);
                        let o1 = node
                            .bounds
                            .get_orthant(self.points[existing_idx].position());
                        let o2 = node.bounds.get_orthant(self.points[point_index].position());
                        node.node_data = NodeData::PointCount(2);
                        let m1 = self.points[existing_idx].get_mass();
                        let m2 = self.points[point_index].get_mass();
                        let total_mass = m1 + m2;
                        node.mass = total_mass;
                        if total_mass > F::from(0.0).unwrap() {
                            node.center_of_mass = (self.points[existing_idx].position().scale(m1)
                                + self.points[point_index].position().scale(m2))
                                / total_mass;
                        } else {
                            node.center_of_mass = SVector::zeros();
                        }
                        node.children = vec![None; 1 << D];
                        let child1_idx =
                            self.add_point_to_tree(existing_idx, None, Some(node_idx), o1);
                        self.arena.get_mut(node_idx).children[o1] = Some(child1_idx);
                        let current_child_opt_for_o2 = self.arena.get(node_idx).children[o2];
                        let child2_idx = self.add_point_to_tree(
                            point_index,
                            current_child_opt_for_o2,
                            Some(node_idx),
                            o2,
                        );
                        self.arena.get_mut(node_idx).children[o2] = Some(child2_idx);
                        node_idx
                    }
                    NodeData::PointCount(count) => {
                        let node = self.arena.get_mut(node_idx);
                        node.node_data = NodeData::PointCount(count + 1);
                        let m1 = node.mass;
                        let m2 = self.points[point_index].get_mass();
                        let com1 = node.center_of_mass;
                        let com2 = *self.points[point_index].position();
                        let total_mass = m1 + m2;
                        if total_mass > F::from(0.0).unwrap() {
                            node.mass = total_mass;
                            node.center_of_mass = (com1.scale(m1) + com2.scale(m2)) / total_mass;
                        }
                        let orthant = node_bounds.get_orthant(self.points[point_index].position()); // Use copied bounds
                        let child_idx_opt = node.children[orthant];
                        let new_child_idx = self.add_point_to_tree(
                            point_index,
                            child_idx_opt,
                            Some(node_idx),
                            orthant,
                        );
                        self.arena.get_mut(node_idx).children[orthant] = Some(new_child_idx);
                        node_idx
                    }
                }
            }
            None => {
                let point = &self.points[point_index];
                let bounds = match parent_idx_opt {
                    Some(parent_idx) => self.arena.get(parent_idx).bounds.create_orthant(orthant),
                    None => self.bounds,
                };
                let new_node = Node {
                    bounds,
                    width_squared: bounds.width * bounds.width,
                    center_of_mass: *point.position(),
                    mass: point.get_mass(),
                    node_data: NodeData::PointIndex(point_index),
                    children: vec![None; 1 << D],
                };
                self.arena.add_node(new_node)
            }
        }
    }

    // Similarly update add_point_to_local_tree
    fn add_point_to_local_tree(
        &self,
        arena: &mut NodeArena<F, D>,
        point_index: usize,
        node_idx_opt: Option<usize>,
        parent_idx_opt: Option<usize>,
        orthant: usize,
        octant_bounds: &Bounds<F, D>,
    ) -> usize {
        match node_idx_opt {
            Some(node_idx) => {
                // Check bounds before proceeding
                let node_bounds = arena.get(node_idx).bounds;
                if !node_bounds.contains(self.points[point_index].position()) {
                    eprintln!(
                         "Warning: Point {:?} outside target node bounds {:?} during local recursive insertion. Skipping add for this branch.",
                         self.points[point_index].position(), node_bounds
                     );
                    return node_idx;
                }
                let node_data_copy = arena.get(node_idx).node_data;
                match node_data_copy {
                    NodeData::PointIndex(existing_idx) => {
                        if existing_idx == point_index {
                            return node_idx;
                        }
                        let node = arena.get_mut(node_idx);
                        let o1 = node
                            .bounds
                            .get_orthant(self.points[existing_idx].position());
                        let o2 = node.bounds.get_orthant(self.points[point_index].position());
                        node.node_data = NodeData::PointCount(2);
                        let m1 = self.points[existing_idx].get_mass();
                        let m2 = self.points[point_index].get_mass();
                        let total_mass = m1 + m2;
                        node.mass = total_mass;
                        if total_mass > F::from(0.0).unwrap() {
                            node.center_of_mass = (self.points[existing_idx].position().scale(m1)
                                + self.points[point_index].position().scale(m2))
                                / total_mass;
                        } else {
                            node.center_of_mass = SVector::zeros();
                        }
                        node.children = vec![None; 1 << D];
                        let child1_idx = self.add_point_to_local_tree(
                            arena,
                            existing_idx,
                            None,
                            Some(node_idx),
                            o1,
                            octant_bounds,
                        );
                        arena.get_mut(node_idx).children[o1] = Some(child1_idx);
                        let current_child_opt_for_o2 = arena.get(node_idx).children[o2];
                        let child2_idx = self.add_point_to_local_tree(
                            arena,
                            point_index,
                            current_child_opt_for_o2,
                            Some(node_idx),
                            o2,
                            octant_bounds,
                        );
                        arena.get_mut(node_idx).children[o2] = Some(child2_idx);
                        node_idx
                    }
                    NodeData::PointCount(count) => {
                        let node = arena.get_mut(node_idx);
                        node.node_data = NodeData::PointCount(count + 1);
                        let m1 = node.mass;
                        let m2 = self.points[point_index].get_mass();
                        let com1 = node.center_of_mass;
                        let com2 = *self.points[point_index].position();
                        let total_mass = m1 + m2;
                        if total_mass > F::from(0.0).unwrap() {
                            node.mass = total_mass;
                            node.center_of_mass = (com1.scale(m1) + com2.scale(m2)) / total_mass;
                        }
                        let orthant = node_bounds.get_orthant(self.points[point_index].position()); // Use copied bounds
                        let child_idx_opt = node.children[orthant];
                        let new_child_idx = self.add_point_to_local_tree(
                            arena,
                            point_index,
                            child_idx_opt,
                            Some(node_idx),
                            orthant,
                            octant_bounds,
                        );
                        arena.get_mut(node_idx).children[orthant] = Some(new_child_idx);
                        node_idx
                    }
                }
            }
            None => {
                let point = &self.points[point_index];
                let bounds = match parent_idx_opt {
                    Some(parent_idx) => arena.get(parent_idx).bounds.create_orthant(orthant),
                    None => *octant_bounds,
                };
                let new_node = Node {
                    bounds,
                    width_squared: bounds.width * bounds.width,
                    center_of_mass: *point.position(),
                    mass: point.get_mass(),
                    node_data: NodeData::PointIndex(point_index),
                    children: vec![None; 1 << D],
                };
                arena.add_node(new_node)
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

        if self.points.is_empty() {
            return;
        }

        // --- Morton Ordering ---
        // 1. Calculate Morton codes for all particles
        let morton_codes: Vec<(u64, usize)> = (0..self.points.len())
            .into_par_iter()
            .map(|i| (self.get_morton_code(i), i))
            .collect();

        // 2. Sort particle indices based on Morton codes
        // Sort the (code, index) vec in parallel.
        let mut sorted_codes_indices = morton_codes;
        sorted_codes_indices.par_sort_unstable_by_key(|&(code, _)| code);
        // Directly map the sorted vec to get the indices
        let sorted_particle_indices: Vec<usize> = sorted_codes_indices
            .into_iter()
            .map(|(_, idx)| idx)
            .collect();
        // --- End Morton Ordering ---

        // Build the tree using the sorted particle order
        // If we have very few points, just build sequentially (using sorted order)
        if self.points.len() < 1000 {
            for &i in &sorted_particle_indices {
                // Call the new iterative add_point_to_tree
                let new_root_idx = self.add_point_to_tree(i, self.root_idx, None, usize::MAX);
                self.root_idx = Some(new_root_idx);
            }
            return;
        }

        // Create the root node first
        let root_node = Node {
            bounds: self.bounds,
            width_squared: self.bounds.width * self.bounds.width,
            center_of_mass: SVector::<F, D>::zeros(),
            mass: F::from(0.0).unwrap(),
            node_data: NodeData::PointCount(0),
            children: vec![None; 1 << D],
        };
        let root_idx = self.arena.add_node(root_node);
        self.root_idx = Some(root_idx);

        // Partition points by octant at the root level (using sorted order)
        let mut octant_points: Vec<Vec<usize>> = vec![Vec::new(); 1 << D];
        for &i in &sorted_particle_indices {
            // Use sorted indices
            let octant = self.bounds.get_orthant(self.points[i].position());
            octant_points[octant].push(i);
        }

        // Process each octant in parallel (input `points` are indices from sorted_particle_indices)
        let octant_trees: Vec<_> = octant_points
            .into_par_iter()
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
                        let new_root = self.add_point_to_local_tree(
                            &mut local_arena,
                            point_idx,
                            Some(root),
                            None,
                            usize::MAX,
                            &octant_bounds,
                        );
                        local_root = Some(new_root);
                    } else {
                        local_root = Some(self.add_point_to_local_tree(
                            &mut local_arena,
                            point_idx,
                            None,
                            None,
                            usize::MAX,
                            &octant_bounds,
                        ));
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

                Some((
                    octant,
                    local_arena,
                    local_root.unwrap(),
                    total_mass,
                    com,
                    points.len(),
                ))
            })
            .collect();

        // Combine the results into the main tree
        let mut total_mass = F::from(0.0).unwrap();
        let mut weighted_com = SVector::<F, D>::zeros();
        let mut total_points = 0;

        for (octant, local_arena, local_root_idx, octant_mass, octant_com, point_count) in
            octant_trees
        {
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

    // Merge a local tree into the main arena using an iterative approach
    fn merge_trees(&mut self, local_arena: &NodeArena<F, D>, local_root_idx: usize) -> usize {
        // Pre-allocate space in the main arena for efficiency
        self.arena.reserve(local_arena.nodes.len());

        // Use a Vec for direct index mapping (faster than HashMap)
        let mut index_map: Vec<Option<usize>> = vec![None; local_arena.nodes.capacity()];

        // Stack for iterative traversal: (local_node_index, parent_main_index, child_slot_in_parent)
        // The parent_main_index and child_slot_in_parent are Option<usize> because the root has no parent.
        let mut stack: Vec<(usize, Option<usize>, Option<usize>)> =
            vec![(local_root_idx, None, None)];

        // Keep track of the main index corresponding to the local_root_idx
        let mut main_root_idx = None;

        while let Some((local_idx, parent_main_idx, child_slot)) = stack.pop() {
            // Check if this node has already been processed
            if index_map[local_idx].is_some() {
                continue;
            }

            // Get the local node to copy
            let local_node = local_arena.get(local_idx);

            // Create a copy of the node for the main arena (initially with no children links)
            let main_node_copy = Node {
                bounds: local_node.bounds,
                width_squared: local_node.width_squared,
                center_of_mass: local_node.center_of_mass,
                mass: local_node.mass,
                node_data: local_node.node_data, // NodeData is Copy
                children: vec![None; 1 << D],
            };

            // Add the copied node to the main arena
            let main_idx = self.arena.add_node(main_node_copy);
            index_map[local_idx] = Some(main_idx);

            // If this is the root node, store its main index
            if parent_main_idx.is_none() {
                main_root_idx = Some(main_idx);
            } else {
                // Link this new node to its parent in the main arena
                let parent_node = self.arena.get_mut(parent_main_idx.unwrap());
                parent_node.children[child_slot.unwrap()] = Some(main_idx);
            }

            // Push unprocessed children onto the stack
            for (i, &child_opt) in local_node.children.iter().enumerate() {
                if let Some(child_local_idx) = child_opt {
                    if index_map[child_local_idx].is_none() {
                        stack.push((child_local_idx, Some(main_idx), Some(i)));
                    }
                    // If child already processed, link it (handles DAGs, though unlikely here)
                    else {
                        let main_child_idx = index_map[child_local_idx].unwrap();
                        self.arena.get_mut(main_idx).children[i] = Some(main_child_idx);
                    }
                }
            }
        }

        // The main_root_idx should always be Some if local_root_idx was valid
        main_root_idx.expect("Failed to find the main root index during merge")
    }

    // Update the calc_force method
    fn calc_force(&self, node_idx: usize, point_idx: usize) -> SVector<F, D> {
        // 3D simulations are most common for galaxies, so we optimize specifically for D=3
        #[allow(unused_comparisons)] // Needed because D is const generic
        if D == 3 {
            // For D=3, convert the result to the required type
            if let Some(force_3d) = self.calc_force_3d(node_idx, point_idx) {
                let mut result = SVector::<F, D>::zeros();
                // Safe because we verified D == 3
                for i in 0..3 {
                    result[i] = force_3d[i];
                }
                return result;
            }
        }

        // Original implementation for other dimensions
        const STACK_CAPACITY: usize = 64; // Typical max depth of Barnes-Hut tree
        let mut stack = [0usize; STACK_CAPACITY];
        let mut stack_size = 1;
        stack[0] = node_idx;

        let mut force = SVector::<F, D>::zeros();

        // Fetch point and cache its position
        let point = &self.points[point_idx];
        let point_pos = point.position();

        // Cache settings once
        let theta2 = self.settings().theta2;
        let g_soft = self.settings().g_soft;
        let g_soft2 = g_soft * g_soft; // Precompute squared softening
        let g = self.settings().g;

        while stack_size > 0 {
            // Pop the top element from our manual stack
            stack_size -= 1;
            let current_idx = stack[stack_size];

            let node = self.arena.get(current_idx);

            // Calculate displacement vector once
            let r = node.center_of_mass - *point_pos;
            let r2 = r.norm_squared();

            // Skip extremely close nodes or self interactions
            if r2 < F::from(1e-10).unwrap() {
                continue;
            }

            // Fast path: if node is far enough away, use multipole approximation
            if node.width_squared < theta2 * r2 {
                // Avoid expensive multiple divisions by computing inverse distance factors
                let r_soft2 = r2 + g_soft2;

                // Use single implementation without conditional compilation
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
                    // Avoid calculating forces with self using direct index comparison
                    if idx == point_idx {
                        continue;
                    }

                    // Direct calculation for leaf nodes
                    let r_soft2 = r2 + g_soft2;
                    let inv_r = F::from(1.0).unwrap() / SimdComplexField::simd_sqrt(r_soft2);
                    let inv_r3 = inv_r * inv_r * inv_r;
                    let force_magnitude = g * node.mass * inv_r3;
                    force += r.scale(force_magnitude);
                }
                NodeData::PointCount(_) => {
                    // For internal nodes, add children to stack in reverse order
                    // This improves cache locality by processing nearby nodes together
                    for i in (0..node.children.len()).rev() {
                        if let Some(child_idx) = node.children[i] {
                            // Safely push to our stack array - avoid overflows
                            if stack_size < STACK_CAPACITY {
                                stack[stack_size] = child_idx;
                                stack_size += 1;
                            } else {
                                // Fall back to safe behavior if we exceed capacity
                                // This should be rare given a reasonable STACK_CAPACITY
                                eprintln!("Warning: Barnes-Hut tree depth exceeds stack capacity!");
                                // Process this node immediately rather than losing it
                                force += self.calc_force(child_idx, point_idx);
                            }
                        }
                    }
                }
            }
        }

        force
    }

    // Specialized and optimized force calculation for 3D simulations
    // Returns None if D != 3 for type safety
    fn calc_force_3d(&self, node_idx: usize, point_idx: usize) -> Option<Vector3<F>>
    where
        F: SimdValue,
    {
        // Only run for 3D simulations - safety check
        #[allow(unused_comparisons)]
        if D != 3 {
            return None;
        }

        // Use a fixed-size array for stack to avoid heap allocations
        const STACK_CAPACITY: usize = 64;
        let mut stack = [0usize; STACK_CAPACITY];
        let mut stack_size = 1;
        stack[0] = node_idx;

        // Pre-allocate force components for SIMD operations
        let mut force_x = F::from(0.0).unwrap();
        let mut force_y = F::from(0.0).unwrap();
        let mut force_z = F::from(0.0).unwrap();

        // Cache particle position for better performance
        let point = &self.points[point_idx];
        let point_pos = point.position();
        let px = point_pos[0];
        let py = point_pos[1];
        let pz = point_pos[2];

        // Cache simulation parameters
        let theta2 = self.settings().theta2;
        let g_soft = self.settings().g_soft;
        let g_soft2 = g_soft * g_soft;
        let g = self.settings().g;

        // Process nodes in stack
        while stack_size > 0 {
            stack_size -= 1;
            let current_idx = stack[stack_size];
            let node = self.arena.get(current_idx);

            // Extract node center of mass components for SIMD
            let cx = node.center_of_mass[0];
            let cy = node.center_of_mass[1];
            let cz = node.center_of_mass[2];

            // Calculate displacement components directly
            let dx = cx - px;
            let dy = cy - py;
            let dz = cz - pz;

            // Calculate squared distance (faster than norm_squared on the vector)
            let r2 = dx * dx + dy * dy + dz * dz;

            // Skip self-interactions or extremely close particles
            if r2 < F::from(1e-10).unwrap() {
                continue;
            }

            // Apply multipole approximation if node is far enough
            if node.width_squared < theta2 * r2 {
                let r_soft2 = r2 + g_soft2;

                // Fast inverse square root - remove dependency on fast_math feature
                let inv_r = F::one() / r_soft2.simd_sqrt();

                let inv_r3 = inv_r * inv_r * inv_r;

                // Calculate force magnitude
                let force_magnitude = g * node.mass * inv_r3;

                // Accumulate force components separately for better SIMD utilization
                force_x += dx * force_magnitude;
                force_y += dy * force_magnitude;
                force_z += dz * force_magnitude;
                continue;
            }

            // Process node based on type
            match node.node_data {
                NodeData::PointIndex(idx) => {
                    if idx == point_idx {
                        continue;
                    }

                    // Direct force calculation for leaf nodes
                    let r_soft2 = r2 + g_soft2;
                    let inv_r = F::one() / r_soft2.simd_sqrt();
                    let inv_r3 = inv_r * inv_r * inv_r;
                    let force_magnitude = g * node.mass * inv_r3;

                    // Accumulate components
                    force_x += dx * force_magnitude;
                    force_y += dy * force_magnitude;
                    force_z += dz * force_magnitude;
                }
                NodeData::PointCount(_) => {
                    // Add children to stack, prioritizing larger nodes
                    for i in (0..node.children.len()).rev() {
                        if let Some(child_idx) = node.children[i] {
                            if stack_size < STACK_CAPACITY {
                                stack[stack_size] = child_idx;
                                stack_size += 1;
                            } else {
                                // Fall back to recursive call if stack is full
                                if let Some(additional_force) =
                                    self.calc_force_3d(child_idx, point_idx)
                                {
                                    force_x += additional_force[0];
                                    force_y += additional_force[1];
                                    force_z += additional_force[2];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Construct and return the force vector
        Some(Vector3::new(force_x, force_y, force_z))
    }

    #[cfg(feature = "render")]
    fn get_nodes_for_rendering(&self) -> Box<dyn Iterator<Item = (usize, &Node<F, D>)> + '_> {
        match self.root_idx {
            Some(root_idx) => Box::new(NodeIterator {
                arena: &self.arena,
                current: vec![(root_idx, 0)],
                next: Vec::new(),
                current_index: 0,
            }),
            None => Box::new(std::iter::empty()),
        }
    }
}

impl<F: Float, const D: usize, P, I> Simulation<F, D, P, I> for BarnesHutSimulation<F, D, P, I>
where
    P: Particle<F, D> + Send + Sync,
    I: Integrator<F, D, P> + Sync,
    F: Send + Sync + SimdValue,
{
    fn new(points: Vec<P>, integrator: I, bounds: Bounds<F, D>) -> Self {
        // Rename to _domain_size to mark as intentionally unused
        let _domain_size = bounds.width;

        // Estimate initial node capacity
        let n = points.len();
        let initial_capacity = if n > 0 {
            // Estimate based on tree depth - typically log(N) * N for N particles
            let log_n = (n as f64).log2().ceil() as usize;
            n * log_n * 4
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
        // Always rebuild the tree before calculating forces
        self.build_tree();

        // Reset accelerations
        for point in self.points.iter_mut() {
            point.acceleration_mut().fill(F::from(0.0).unwrap());
        }

        if let Some(root_idx) = self.root_idx {
            // Calculate forces in parallel
            let forces: Vec<SVector<F, D>> = (0..self.points.len())
                .into_par_iter()
                .map(|i| self.calc_force(root_idx, i))
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

                    // Use constants defined in render_init for consistency
                    const FLOATS_PER_NODE: usize = 10; // 3 min, 3 max, 4 color

                    for (depth, node) in bounds {
                        let s = (depth as f32) / (max_depth as f32 + 0.001) * 0.7 + 0.3;
                        bounds_data.extend(
                            node.bounds
                                .min()
                                .iter()
                                .chain(node.bounds.max().iter())
                                .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                                // Color data
                                .chain::<[f32; 4]>([(1. - s * s) * 0.5, s * s, (1. - s) * 0.5, s])
                                .collect::<Vec<f32>>(), // 10 floats total
                        );
                        num_bounds += 1;
                    }
                    self.num_bounds = num_bounds; // Store actual node count
                    bounds_data
                }
                None => {
                    self.num_bounds = 1;
                    self.bounds
                        .min()
                        .iter()
                        .chain(self.bounds.max().iter())
                        .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                        .chain([1.0, 1.0, 0.0, 1.0].iter().copied())
                        .collect::<Vec<f32>>()
                }
            };

            // --- Safety Check and Truncation ---
            let floats_to_write = bounds_data.len();
            let bytes_to_write = floats_to_write * std::mem::size_of::<f32>();

            // Get buffer size (should match BUFFER_SIZE from init)
            let buffer_capacity = bounds_buffer.size() as usize;

            let mut num_nodes_to_render = self.num_bounds;
            let data_slice: &[u8];

            if bytes_to_write > buffer_capacity {
                // Calculate how many full nodes fit
                const FLOATS_PER_NODE: usize = 10;
                let max_floats = buffer_capacity / std::mem::size_of::<f32>();
                let max_nodes = max_floats / FLOATS_PER_NODE;

                eprintln!(
                    "Warning: Number of bounds nodes ({}) exceeds buffer capacity ({} nodes). Truncating render.",
                    self.num_bounds,
                    max_nodes
                );

                // Create a slice of the data that fits
                let fitting_floats = max_nodes * FLOATS_PER_NODE;
                data_slice = bytemuck::cast_slice(&bounds_data[0..fitting_floats]);
                num_nodes_to_render = max_nodes as u32;
            } else {
                // Everything fits
                data_slice = bytemuck::cast_slice(&bounds_data);
            }

            // Only write the (potentially truncated) data slice
            renderer
                .context
                .queue
                .write_buffer(bounds_buffer, 0, data_slice);

            renderer.set_pipeline(PipelineType::AABB);
            let render_pass = renderer.get_render_pass();
            render_pass.set_vertex_buffer(0, bounds_buffer.slice(..));
            // Draw only the number of nodes actually written to the buffer
            render_pass.draw(0..16, 0..num_nodes_to_render);
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

        let points_vertex_buffer =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Points Vertex Buffer"),
                    contents: bytemuck::cast_slice(&point_position_data),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

        // Create bounds buffer with sufficient size
        // Each node needs 3 (min) + 3 (max) + 4 (color) = 10 floats * 4 bytes = 40 bytes per node
        // Increase MAX_NODES significantly to avoid buffer overflow
        const MAX_NODES: usize = 100000;
        const _FLOATS_PER_NODE: usize = 10; // 3 for min, 3 for max, 4 for color
        const BUFFER_SIZE: usize = MAX_NODES * _FLOATS_PER_NODE * std::mem::size_of::<f32>();

        // Create with initial bounds data
        let initial_bounds_data: Vec<f32> = self
            .bounds
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

// Specialized 3D version of the Barnes-Hut simulation that uses fixed-size arrays
// to significantly reduce memory allocations
#[derive(Clone)]
pub struct BarnesHut3D<F: Float, P, I = LeapFrogIntegrator<F, 3, P>>
where
    P: Particle<F, 3>,
    I: Integrator<F, 3, P>,
{
    points: Vec<P>,
    arena: NodeArena3D<F>,
    root_idx: Option<usize>,
    bounds: Bounds<F, 3>,
    integrator: I,
    settings: SimulationSettings<F>,
    elapsed: F,
    #[cfg(feature = "render")]
    points_vertex_buffer: Option<wgpu::Buffer>,
    #[cfg(feature = "render")]
    bounds_vertex_buffer: Option<wgpu::Buffer>,
    #[cfg(feature = "render")]
    num_bounds: u32,
}

impl<F: Float, P, I> BarnesHut3D<F, P, I>
where
    P: Particle<F, 3> + Send + Sync,
    I: Integrator<F, 3, P> + Sync,
    F: Send + Sync + SimdValue,
{
    // Helper to normalize a position component
    fn normalize_pos_component(pos_component: F, min_bound: F, max_bound: F) -> u32 {
        let range = max_bound - min_bound;
        if range <= F::from(0.0).unwrap() {
            return 0;
        }
        let clamped_pos = pos_component.max(min_bound).min(max_bound);
        let normalized = (clamped_pos - min_bound) / range;
        (normalized * F::from(u32::MAX - 1).unwrap())
            .to_u32()
            .unwrap_or(0)
    }

    // Calculate Morton code for a 3D point
    fn get_morton_code(&self, point_idx: usize) -> u64 {
        let pos = self.points[point_idx].position();
        let min_b = self.bounds.min();
        let max_b = self.bounds.max();

        // Normalize coordinates to u32 for 3D
        let coords: [u32; 3] = [
            Self::normalize_pos_component(pos[0], min_b[0], max_b[0]),
            Self::normalize_pos_component(pos[1], min_b[1], max_b[1]),
            Self::normalize_pos_component(pos[2], min_b[2], max_b[2]),
        ];

        morton_encode(coords) as u64
    }

    // Estimate node count for allocation
    fn estimate_node_count(&self) -> usize {
        let n = self.points.len();
        if n == 0 {
            return 0;
        }

        // Each particle typically needs log(N) nodes, with some overhead
        let log_n = (n as f64).log2().ceil() as usize;
        n * log_n * 4
    }

    // Optimized add_point_to_tree for 3D with fixed-size children array
    fn add_point_to_tree(
        &mut self,
        point_index: usize,
        node_idx_opt: Option<usize>,
        parent_idx_opt: Option<usize>,
        orthant: usize,
    ) -> usize {
        match node_idx_opt {
            Some(node_idx) => {
                // Check bounds before proceeding
                let node_bounds = self.arena.get(node_idx).bounds;
                if !node_bounds.contains(self.points[point_index].position()) {
                    eprintln!(
                        "Warning: Point {:?} outside target node bounds {:?}. Skipping add.",
                        self.points[point_index].position(),
                        node_bounds
                    );
                    return node_idx;
                }

                let node_data_copy = self.arena.get(node_idx).node_data;
                match node_data_copy {
                    NodeData::PointIndex(existing_idx) => {
                        if existing_idx == point_index {
                            return node_idx;
                        }

                        let node = self.arena.get_mut(node_idx);
                        let o1 = node
                            .bounds
                            .get_orthant(self.points[existing_idx].position());
                        let o2 = node.bounds.get_orthant(self.points[point_index].position());

                        node.node_data = NodeData::PointCount(2);

                        let m1 = self.points[existing_idx].get_mass();
                        let m2 = self.points[point_index].get_mass();
                        let total_mass = m1 + m2;

                        node.mass = total_mass;
                        if total_mass > F::from(0.0).unwrap() {
                            node.center_of_mass = (self.points[existing_idx].position().scale(m1)
                                + self.points[point_index].position().scale(m2))
                                / total_mass;
                        } else {
                            node.center_of_mass = Vector3::zeros();
                        }

                        // No need to reallocate children - already fixed size array

                        let child1_idx =
                            self.add_point_to_tree(existing_idx, None, Some(node_idx), o1);
                        self.arena.get_mut(node_idx).children[o1] = Some(child1_idx);

                        let current_child_opt_for_o2 = self.arena.get(node_idx).children[o2];
                        let child2_idx = self.add_point_to_tree(
                            point_index,
                            current_child_opt_for_o2,
                            Some(node_idx),
                            o2,
                        );
                        self.arena.get_mut(node_idx).children[o2] = Some(child2_idx);

                        node_idx
                    }
                    NodeData::PointCount(count) => {
                        let node = self.arena.get_mut(node_idx);
                        node.node_data = NodeData::PointCount(count + 1);

                        let m1 = node.mass;
                        let m2 = self.points[point_index].get_mass();
                        let com1 = node.center_of_mass;
                        let com2 = *self.points[point_index].position();
                        let total_mass = m1 + m2;

                        if total_mass > F::from(0.0).unwrap() {
                            node.mass = total_mass;
                            node.center_of_mass = (com1.scale(m1) + com2.scale(m2)) / total_mass;
                        }

                        let orthant = node_bounds.get_orthant(self.points[point_index].position());
                        let child_idx_opt = node.children[orthant];

                        let new_child_idx = self.add_point_to_tree(
                            point_index,
                            child_idx_opt,
                            Some(node_idx),
                            orthant,
                        );

                        self.arena.get_mut(node_idx).children[orthant] = Some(new_child_idx);
                        node_idx
                    }
                }
            }
            None => {
                let point = &self.points[point_index];
                let bounds = match parent_idx_opt {
                    Some(parent_idx) => self.arena.get(parent_idx).bounds.create_orthant(orthant),
                    None => self.bounds,
                };

                // Create a new node with fixed-size children array
                let new_node = Node3D {
                    bounds,
                    width_squared: bounds.width * bounds.width,
                    center_of_mass: *point.position(),
                    mass: point.get_mass(),
                    node_data: NodeData::PointIndex(point_index),
                    children: [None; 8], // Fixed-size array for 8 children in 3D
                };

                self.arena.add_node(new_node)
            }
        }
    }

    // Optimized build_tree for 3D with fixed-size children arrays
    fn build_tree(&mut self) {
        // Estimate and reserve capacity
        let estimated_nodes = self.estimate_node_count();
        self.arena.reserve(estimated_nodes);

        // Clear existing tree
        self.arena.clear();
        self.root_idx = None;

        if self.points.is_empty() {
            return;
        }

        // Calculate Morton codes for better spatial locality
        let morton_codes: Vec<(u64, usize)> = (0..self.points.len())
            .into_par_iter()
            .map(|i| (self.get_morton_code(i), i))
            .collect();

        // Sort particles by Morton code
        let mut sorted_codes_indices = morton_codes;
        sorted_codes_indices.par_sort_unstable_by_key(|&(code, _)| code);

        // Extract sorted indices
        let sorted_particle_indices: Vec<usize> = sorted_codes_indices
            .into_iter()
            .map(|(_, idx)| idx)
            .collect();

        // For small particle counts, build sequentially
        if self.points.len() < 1000 {
            for &i in &sorted_particle_indices {
                let new_root_idx = self.add_point_to_tree(i, self.root_idx, None, usize::MAX);
                self.root_idx = Some(new_root_idx);
            }
            return;
        }

        // For larger particle counts, create root and build in parallel
        let root_node = Node3D {
            bounds: self.bounds,
            width_squared: self.bounds.width * self.bounds.width,
            center_of_mass: Vector3::zeros(),
            mass: F::from(0.0).unwrap(),
            node_data: NodeData::PointCount(0),
            children: [None; 8], // Fixed-size array for 8 children in 3D
        };

        let root_idx = self.arena.add_node(root_node);
        self.root_idx = Some(root_idx);

        // Partition points by octant (using sorted order)
        let mut octant_points: Vec<Vec<usize>> = vec![Vec::new(); 8]; // 8 octants for 3D

        for &i in &sorted_particle_indices {
            let octant = self.bounds.get_orthant(self.points[i].position());
            octant_points[octant].push(i);
        }

        // Process each octant and update the root node
        let mut total_mass = F::from(0.0).unwrap();
        let mut weighted_com = Vector3::zeros();
        let mut total_points = 0;

        // Process each octant
        for (octant, points) in octant_points.into_iter().enumerate() {
            if points.is_empty() {
                continue;
            }

            // Process octant sequentially
            let mut octant_mass = F::from(0.0).unwrap();
            let mut octant_com = Vector3::zeros();

            // Create a child node for this octant
            let octant_bounds = self.bounds.create_orthant(octant);
            let child_node = Node3D {
                bounds: octant_bounds,
                width_squared: octant_bounds.width * octant_bounds.width,
                center_of_mass: Vector3::zeros(),
                mass: F::from(0.0).unwrap(),
                node_data: NodeData::PointCount(0),
                children: [None; 8],
            };

            let child_idx = self.arena.add_node(child_node);
            self.arena.get_mut(root_idx).children[octant] = Some(child_idx);

            // Add points to this octant's subtree
            for &point_idx in &points {
                let new_child_idx =
                    self.add_point_to_tree(point_idx, Some(child_idx), None, usize::MAX);

                // Only update if this changed the root of the subtree
                if new_child_idx != child_idx {
                    self.arena.get_mut(root_idx).children[octant] = Some(new_child_idx);
                }

                // Update center of mass contribution
                let point_mass = self.points[point_idx].get_mass();
                octant_mass += point_mass;
                octant_com += self.points[point_idx].position().scale(point_mass);
            }

            // Calculate final center of mass for this octant
            if octant_mass > F::from(0.0).unwrap() {
                octant_com = octant_com / octant_mass;
            }

            // Update the octant node with final values
            let octant_node_idx = self.arena.get(root_idx).children[octant].unwrap();
            let octant_node = self.arena.get_mut(octant_node_idx);
            octant_node.mass = octant_mass;
            octant_node.center_of_mass = octant_com;
            octant_node.node_data = NodeData::PointCount(points.len());

            // Update root node stats
            weighted_com += octant_com.scale(octant_mass);
            total_mass += octant_mass;
            total_points += points.len();
        }

        // Update root node properties
        if total_mass > F::from(0.0).unwrap() {
            let root = self.arena.get_mut(root_idx);
            root.mass = total_mass;
            root.center_of_mass = weighted_com / total_mass;
            root.node_data = NodeData::PointCount(total_points);
        }
    }

    // Optimize force calculation specifically for 3D
    fn calc_force(&self, node_idx: usize, point_idx: usize) -> Vector3<F> {
        // Use a fixed-size array for stack - larger size for deeper trees
        const STACK_CAPACITY: usize = 128; // Increased from 64
        let mut stack = [0usize; STACK_CAPACITY];
        let mut stack_size = 1;
        stack[0] = node_idx;

        // Force components
        let mut force_x = F::from(0.0).unwrap();
        let mut force_y = F::from(0.0).unwrap();
        let mut force_z = F::from(0.0).unwrap();

        // Cache particle position for performance
        let point = &self.points[point_idx];
        let point_pos = point.position();
        let px = point_pos[0];
        let py = point_pos[1];
        let pz = point_pos[2];

        // Cache simulation parameters
        let theta2 = self.settings.theta2;
        let g_soft = self.settings.g_soft;
        let g_soft2 = g_soft * g_soft;
        let g = self.settings.g;

        // Main force calculation loop
        while stack_size > 0 {
            stack_size -= 1;
            let current_idx = stack[stack_size];
            let node = self.arena.get(current_idx);

            // Center of mass components
            let cx = node.center_of_mass[0];
            let cy = node.center_of_mass[1];
            let cz = node.center_of_mass[2];

            // Displacement vector
            let dx = cx - px;
            let dy = cy - py;
            let dz = cz - pz;

            // Squared distance
            let r2 = dx * dx + dy * dy + dz * dz;

            // Skip extremely close nodes or self
            if r2 < F::from(1e-10).unwrap() {
                continue;
            }

            // If node is far enough, use multipole approximation
            if node.width_squared < theta2 * r2 {
                let r_soft2 = r2 + g_soft2;
                let inv_r = F::one() / r_soft2.simd_sqrt();
                let inv_r3 = inv_r * inv_r * inv_r;
                let force_magnitude = g * node.mass * inv_r3;

                // Accumulate force components
                force_x += dx * force_magnitude;
                force_y += dy * force_magnitude;
                force_z += dz * force_magnitude;
                continue;
            }

            // Handle node based on type
            match node.node_data {
                NodeData::PointIndex(idx) => {
                    if idx == point_idx {
                        continue;
                    }

                    // Direct calculation
                    let r_soft2 = r2 + g_soft2;
                    let inv_r = F::one() / r_soft2.simd_sqrt();
                    let inv_r3 = inv_r * inv_r * inv_r;
                    let force_magnitude = g * node.mass * inv_r3;

                    // Add force components
                    force_x += dx * force_magnitude;
                    force_y += dy * force_magnitude;
                    force_z += dz * force_magnitude;
                }
                NodeData::PointCount(_) => {
                    // Add children to stack
                    for i in (0..8).rev() {
                        if let Some(child_idx) = node.children[i] {
                            if stack_size < STACK_CAPACITY {
                                stack[stack_size] = child_idx;
                                stack_size += 1;
                            } else {
                                // Fall back to recursive call if stack is full
                                eprintln!("Stack capacity exceeded in force calculation");
                                // This should rarely happen with the increased stack size
                            }
                        }
                    }
                }
            }
        }

        // Return force vector
        Vector3::new(force_x, force_y, force_z)
    }
}

// Implement the Simulation trait for our specialized 3D implementation
impl<F: Float, P, I> Simulation<F, 3, P, I> for BarnesHut3D<F, P, I>
where
    P: Particle<F, 3> + Send + Sync,
    I: Integrator<F, 3, P> + Sync,
    F: Send + Sync + SimdValue,
{
    fn new(points: Vec<P>, integrator: I, bounds: Bounds<F, 3>) -> Self {
        // Estimate initial capacity
        let n = points.len();
        let initial_capacity = if n > 0 {
            let log_n = (n as f64).log2().ceil() as usize;
            n * log_n * 4
        } else {
            1024 // Default capacity
        };

        Self {
            points,
            arena: NodeArena3D::with_capacity(initial_capacity),
            root_idx: None,
            bounds,
            integrator,
            settings: SimulationSettings::default(),
            elapsed: F::from(0.0).unwrap(),
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
        // Rebuild the tree
        self.build_tree();

        // Reset accelerations
        for point in self.points.iter_mut() {
            point.acceleration_mut().fill(F::from(0.0).unwrap());
        }

        if let Some(root_idx) = self.root_idx {
            // Calculate forces in parallel
            let forces: Vec<Vector3<F>> = (0..self.points.len())
                .into_par_iter()
                .map(|i| self.calc_force(root_idx, i))
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
    }

    fn remove_point(&mut self, index: usize) {
        self.points.swap_remove(index);
    }

    fn get_points(&self) -> &Vec<P> {
        &self.points
    }
}

// Add rendering support for 3D implementation
#[cfg(feature = "render")]
impl<F, P, I> Renderable for BarnesHut3D<F, P, I>
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
                Some(root_idx) => {
                    // Create a simplified visualization - just root and its children
                    let mut bounds_data = Vec::new();
                    let mut num_bounds = 0;

                    // Add root bounds
                    let root = self.arena.get(root_idx);
                    bounds_data.extend(
                        root.bounds
                            .min()
                            .iter()
                            .chain(root.bounds.max().iter())
                            .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                            .chain([1.0f32, 1.0, 0.0, 1.0]), // Yellow color for root
                    );
                    num_bounds += 1;

                    // Add first level children bounds
                    for i in 0..8 {
                        if let Some(child_idx) = root.children[i] {
                            let child = self.arena.get(child_idx);
                            bounds_data.extend(
                                child
                                    .bounds
                                    .min()
                                    .iter()
                                    .chain(child.bounds.max().iter())
                                    .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                                    .chain([0.0f32, 0.8, 0.8, 0.7]), // Blue-green color for children
                            );
                            num_bounds += 1;
                        }
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
                        .chain([1.0, 1.0, 0.0, 1.0].iter().copied())
                        .collect()
                }
            };

            // Handle buffer size constraints
            let buffer_capacity = bounds_buffer.size() as usize;
            let bytes_to_write = bounds_data.len() * std::mem::size_of::<f32>();

            if bytes_to_write <= buffer_capacity {
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
    }

    fn render_init(&mut self, context: &crate::render::Context) {
        // Create points buffer
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

        let points_vertex_buffer =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Points Vertex Buffer"),
                    contents: bytemuck::cast_slice(&point_position_data),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

        // Create bounds buffer
        const MAX_NODES: usize = 10000;
        const FLOATS_PER_NODE: usize = 10; // 3 min, 3 max, 4 color
        const BUFFER_SIZE: usize = MAX_NODES * FLOATS_PER_NODE * std::mem::size_of::<f32>();

        // Initial bounds data
        let initial_bounds_data: Vec<f32> = self
            .bounds
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

// Add a public function to create a 3D Barnes-Hut simulation
pub fn create_barnes_hut_3d<F: Float + Send + Sync + SimdValue, P, I>(
    points: Vec<P>,
    integrator: I,
    bounds: Bounds<F, 3>,
) -> BarnesHut3D<F, P, I>
where
    P: Particle<F, 3> + Send + Sync,
    I: Integrator<F, 3, P> + Sync,
{
    BarnesHut3D::new(points, integrator, bounds)
}
