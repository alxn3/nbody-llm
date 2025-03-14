use nalgebra::SimdComplexField;

#[cfg(feature = "render")]
use crate::render::{BufferWrapper, PipelineType};

use crate::shared::{AABB, Bounds, Float, Integrator, LeapFrogIntegrator, Particle, Simulation};

#[derive(Clone)]
pub struct BruteForceSimulation<F: Float, const D: usize, P, I = LeapFrogIntegrator<F, D, P>>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    points: Vec<P>,
    bounds: Bounds<F, D>,
    integrator: I,
    g: F,
    dt: F,
    g_soft: F,
    elapsed: F,
    #[cfg(feature = "render")]
    points_buffer: Option<BufferWrapper>,
    #[cfg(feature = "render")]
    bounds_buffer: Option<BufferWrapper>,
}

impl<F: Float, const D: usize, P, I> Simulation<F, D, P, I> for BruteForceSimulation<F, D, P, I>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    fn new(points: Vec<P>, integrator: I, bounds: Bounds<F, D>) -> Self {
        Self {
            points,
            bounds,
            integrator,
            g: F::from(1.0).unwrap(),
            dt: F::from(0.001).unwrap(),
            g_soft: F::from(0.0).unwrap(),
            elapsed: F::from(0.0).unwrap(),
            #[cfg(feature = "render")]
            points_buffer: None,
            #[cfg(feature = "render")]
            bounds_buffer: None,
        }
    }

    fn init(&mut self) {
        self.integrator.init();
        self.elapsed = F::from(0.0).unwrap();
    }

    fn g(&self) -> F {
        self.g
    }

    fn g_soft(&self) -> F {
        self.g_soft
    }

    fn dt(&self) -> F {
        self.dt
    }

    fn g_mut(&mut self) -> &mut F {
        &mut self.g
    }

    fn g_soft_mut(&mut self) -> &mut F {
        &mut self.g_soft
    }

    fn dt_mut(&mut self) -> &mut F {
        &mut self.dt
    }

    fn elapsed(&self) -> F {
        self.elapsed
    }

    fn update_forces(&mut self) {
        for point in self.points.iter_mut() {
            point.acceleration_mut().fill(F::from(0.0).unwrap());
        }

        let g_soft2 = self.g_soft() * self.g_soft();
        for i in 0..self.points.len() {
            for j in 0..i {
                let r = self.points[i].position() - self.points[j].position();
                let r_dist = SimdComplexField::simd_sqrt(r.norm_squared() + g_soft2);
                let r_cubed = r_dist * r_dist * r_dist;
                let m_i = self.points[i].get_mass();
                let m_j = self.points[j].get_mass();
                let force = self.g() / r_cubed;
                *self.points[i].acceleration_mut() -= r * force * m_j;
                *self.points[j].acceleration_mut() += r * force * m_i;
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

    #[cfg(feature = "render")]
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

            bounds_buffer.update(&renderer.context, bounds_data.as_slice());

            renderer.set_pipeline(PipelineType::AABB);
            let render_pass: &mut wgpu::RenderPass<'_> = renderer.get_render_pass();
            render_pass.set_vertex_buffer(0, bounds_buffer.buffer.slice(..));
            render_pass.draw(0..16, 0..1);
        }
    }

    #[cfg(feature = "render")]
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
