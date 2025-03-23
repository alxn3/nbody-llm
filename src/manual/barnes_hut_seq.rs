use nalgebra::{SVector, SimdComplexField};

#[cfg(feature = "render")]
use crate::render::{BufferWrapper, PipelineType, Renderable};

use crate::shared::{
    AABB, Bounds, Float, Integrator, LeapFrogIntegrator, Particle, Simulation, SimulationSettings,
};

#[derive(Clone)]
struct OrthNode<F: Float, const D: usize> {
    center_of_mass: SVector<F, D>,
    bounds: Bounds<F, D>,
    mass: F,
    // TODO: use an array with size 2^D after const_generics stabilizes #![feature(const_generics)]
    children: Vec<Option<OrthNode<F, D>>>,
}

impl<F: Float, const D: usize> OrthNode<F, D> {
    fn new(bounds: Bounds<F, D>) -> Self {
        Self {
            bounds,
            // fill with none.
            children: vec![None; 1 << D],
            center_of_mass: SVector::<F, D>::zeros(),
            mass: F::from(0.0).unwrap(),
        }
    }
}

struct OrthNodeIterator<'a, F: Float, const D: usize> {
    current: Vec<&'a OrthNode<F, D>>,
    next: Vec<&'a OrthNode<F, D>>,
    current_index: usize,
    current_depth: usize,
}

impl<'a, F: Float, const D: usize> Iterator for OrthNodeIterator<'a, F, D> {
    type Item = (usize, &'a OrthNode<F, D>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.current.len() {
            let node = self.current[self.current_index];
            self.next
                .extend(node.children.iter().filter_map(|n| n.as_ref()));
            self.current_index += 1;
            Some((self.current_depth, node))
        } else if self.next.is_empty() {
            None
        } else {
            self.current = self.next.drain(..).collect();
            self.current_index = 0;
            self.current_depth += 1;
            self.next = Vec::new();
            self.next()
        }
    }
}

impl<'a, F: Float, const D: usize> IntoIterator for &'a OrthNode<F, D> {
    type Item = (usize, &'a OrthNode<F, D>);
    type IntoIter = OrthNodeIterator<'a, F, D>;

    fn into_iter(self) -> Self::IntoIter {
        OrthNodeIterator {
            current: vec![self],
            next: Vec::new(),
            current_index: 0,
            current_depth: 0,
        }
    }
}

#[derive(Clone)]
pub struct BarnesHutSeqSimulation<F: Float, const D: usize, P, I = LeapFrogIntegrator<F, D, P>>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    points: Vec<P>,
    root: Option<OrthNode<F, D>>,
    bounds: Bounds<F, D>,
    integrator: I,
    settings: SimulationSettings<F>,
    elapsed: F,
    #[cfg(feature = "render")]
    points_buffer: Option<BufferWrapper>,
    #[cfg(feature = "render")]
    bounds_buffer: Option<BufferWrapper>,
    #[cfg(feature = "render")]
    num_bounds: u32,
}

impl<F: Float, const D: usize, P, I> BarnesHutSeqSimulation<F, D, P, I>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    fn build_tree(points: &[&P], bounds: Bounds<F, D>) -> OrthNode<F, D> {
        match points.len() {
            0 => OrthNode::new(bounds),
            1 => {
                let point = &points[0];
                let mut node = OrthNode::new(bounds);
                node.center_of_mass = *point.position();
                node.mass = point.get_mass();
                node
            }
            _ => {
                let mut orthants = vec![Vec::new(); 1 << D];
                for point in points.iter() {
                    let orthant = bounds.get_orthant(point.position());
                    orthants[orthant].push(*point);
                }

                let children = orthants
                    .iter()
                    .enumerate()
                    .map(|(i, orthant)| match orthant.len() {
                        0 => None,
                        _ => {
                            let child_bounds = bounds.create_orthant(i);
                            Some(Self::build_tree(orthant, child_bounds))
                        }
                    })
                    .collect::<Vec<_>>();

                let mut node = OrthNode::new(bounds);
                node.children = children;
                node.center_of_mass = points
                    .iter()
                    .map(|p| *p.position() * p.get_mass())
                    .fold(SVector::<F, D>::zeros(), |acc, p| acc + p)
                    / F::from(points.len()).unwrap();
                node.mass = points.iter().map(|p| p.get_mass()).sum();
                node
            }
        }
    }

    fn calc_force(&self, node: &OrthNode<F, D>, point: &P) -> SVector<F, D> {
        let r = node.center_of_mass - *point.position();
        let r2 = r.norm_squared();
        if node.bounds.width * node.bounds.width < self.settings().theta2 * r2 {
            let r_dist =
                SimdComplexField::simd_sqrt(r2 + self.settings().g_soft * self.settings().g_soft);
            let r_cubed = r_dist * r_dist * r_dist;
            r * (self.settings().g * node.mass / r_cubed)
        } else {
            node.children
                .iter()
                .filter_map(|c| c.as_ref())
                .map(|c| self.calc_force(c, point))
                .fold(SVector::<F, D>::zeros(), |acc, f| acc + f)
        }
    }
}

impl<F: Float, const D: usize, P, I> Simulation<F, D, P, I> for BarnesHutSeqSimulation<F, D, P, I>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    fn new(points: Vec<P>, integrator: I, bounds: Bounds<F, D>) -> Self {
        Self {
            points,
            bounds,
            integrator,
            root: None,
            settings: SimulationSettings::default(),
            elapsed: F::from(0.0).unwrap(),
            #[cfg(feature = "render")]
            points_buffer: None,
            #[cfg(feature = "render")]
            bounds_buffer: None,
            #[cfg(feature = "render")]
            num_bounds: 0,
        }
    }

    fn init(&mut self) {
        self.integrator.init();
        self.elapsed = F::from(0.0).unwrap();
        self.root = Some(Self::build_tree(
            self.points.iter().collect::<Vec<_>>().as_slice(),
            self.bounds.clone(),
        ));
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
        self.root = Some(Self::build_tree(
            self.points.iter().collect::<Vec<_>>().as_slice(),
            self.bounds.clone(),
        ));
        for i in 0..self.points.len() {
            let force = self.calc_force(self.root.as_ref().unwrap(), &self.points[i]);
            *self.points[i].acceleration_mut() = force;
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

#[cfg(feature = "render")]
impl<F, P, I> Renderable for BarnesHutSeqSimulation<F, 3, P, I>
where
    F: Float,
    P: Particle<F, 3>,
    I: Integrator<F, 3, P>,
{
    fn render(&mut self, renderer: &mut crate::render::Renderer) {
        use crate::shared::AABB;

        if let Some(ref mut points_buffer) = self.points_buffer {
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

            points_buffer.update(&renderer.context, point_position_data.as_slice());

            renderer.set_pipeline(PipelineType::Points);
            let render_pass = renderer.get_render_pass();
            render_pass.set_vertex_buffer(0, points_buffer.buffer.slice(..));
            render_pass.draw(0..4, 0..self.points.len() as u32);
        }

        if let Some(ref mut bounds_buffer) = self.bounds_buffer {
            let bounds_data = match self.root {
                Some(ref root) => {
                    let mut bounds_data = Vec::new();
                    self.num_bounds = 0;
                    let bounds = root.into_iter().collect::<Vec<_>>();
                    let max_depth = bounds.last().unwrap().0;
                    for (depth, node) in bounds.iter() {
                        let s = (*depth as f32) / (max_depth as f32) * 0.7 + 0.3;
                        bounds_data.extend(
                            node.bounds
                                .min()
                                .iter()
                                .chain(node.bounds.max().iter())
                                .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                                .chain::<[f32; 4]>([(1. - s * s) * 0.5, s * s, (1. - s) * 0.5, s])
                                .collect::<Vec<f32>>(),
                        );
                        self.num_bounds += 1;
                    }
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

            bounds_buffer.update(&renderer.context, bounds_data.as_slice());

            renderer.set_pipeline(PipelineType::AABB);
            let render_pass: &mut wgpu::RenderPass<'_> = renderer.get_render_pass();
            render_pass.set_vertex_buffer(0, bounds_buffer.buffer.slice(..));
            render_pass.draw(0..16, 0..self.num_bounds);
        }
    }

    fn render_init(&mut self, context: &crate::render::Context) {
        self.points_buffer = Some(BufferWrapper::new(
            &context.device,
            Some("Point Buffer"),
            &[] as &[f32],
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        ));

        self.bounds_buffer = Some(BufferWrapper::new(
            &context.device,
            Some("Bounds Buffer"),
            &[] as &[f32],
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        ));
    }
}
