use nalgebra::SimdComplexField;

#[cfg(feature = "render")]
use {
    crate::render::{PipelineType, Renderer},
    wgpu::util::DeviceExt,
};

use crate::shared::{Float, Integrator, LeapFrogIntegrator, Particle, Simulation};

#[derive(Debug, Clone)]
pub struct BruteForceSimulation<F: Float, const D: usize, P, I = LeapFrogIntegrator<F, D, P>>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    bodies: Vec<P>,
    integrator: I,
    g: F,
    dt: F,
    g_soft: F,
    elapsed: F,
    #[cfg(feature = "render")]
    bodies_vertex_buffer: Option<wgpu::Buffer>,
}

impl<F: Float, const D: usize, P, I> Simulation<F, D, P, I> for BruteForceSimulation<F, D, P, I>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    fn new(points: Vec<P>, integrator: I) -> Self {
        Self {
            bodies: points,
            integrator,
            g: F::from(1.0).unwrap(),
            dt: F::from(0.001).unwrap(),
            g_soft: F::from(0.0).unwrap(),
            elapsed: F::from(0.0).unwrap(),
            #[cfg(feature = "render")]
            bodies_vertex_buffer: None,
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
        for point in self.bodies.iter_mut() {
            point.acceleration_mut().fill(F::from(0.0).unwrap());
        }

        let g_soft2 = self.g_soft() * self.g_soft();
        for i in 0..self.bodies.len() {
            for j in 0..i {
                let r = self.bodies[i].position() - self.bodies[j].position();
                let r_dist = SimdComplexField::simd_sqrt(r.norm_squared() + g_soft2);
                let r_cubed = r_dist * r_dist * r_dist;
                let m_i = self.bodies[i].get_mass();
                let m_j = self.bodies[j].get_mass();
                let force = self.g() / r_cubed;
                *self.bodies[i].acceleration_mut() -= r * force * m_j;
                *self.bodies[j].acceleration_mut() += r * force * m_i;
            }
        }
    }

    fn step_by(&mut self, dt: F) {
        self.integrator.integrate_pre_force(&mut self.bodies, dt);
        self.update_forces();
        self.integrator.integrate_after_force(&mut self.bodies, dt);
        self.elapsed += dt;
    }

    fn add_point(&mut self, point: P) {
        self.bodies.push(point);
    }

    fn remove_point(&mut self, index: usize) {
        self.bodies.remove(index);
    }

    fn get_points(&self) -> &Vec<P> {
        &self.bodies
    }

    #[cfg(feature = "render")]
    fn render(&mut self, renderer: &mut crate::render::Renderer) {
        let queue = &renderer.context.queue;

        let position_data: Vec<f32> = self
            .bodies
            .iter()
            .flat_map(|p| {
                p.position()
                    .iter()
                    .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                    .collect::<Vec<f32>>()
            })
            .collect();
        let bodies_vertex_buffer = self.bodies_vertex_buffer.as_ref().unwrap();

        queue.write_buffer(
            bodies_vertex_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );

        renderer.set_pipeline(PipelineType::Points);

        let render_pass = renderer.get_render_pass();

        render_pass.set_vertex_buffer(0, bodies_vertex_buffer.slice(..));

        render_pass.draw(0..4, 0..self.bodies.len() as u32);
    }

    #[cfg(feature = "render")]
    fn render_init(&mut self, renderer: &Renderer) {
        let device = &renderer.context.device;
        let queue = &renderer.context.queue;

        let position_data: Vec<f32> = self
            .bodies
            .iter()
            .flat_map(|p| {
                p.position()
                    .iter()
                    .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                    .collect::<Vec<f32>>()
            })
            .collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&position_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&position_data));

        self.bodies_vertex_buffer = Some(vertex_buffer);
    }
}
