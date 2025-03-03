// This file defines shared behavior for the manual and LLM implementations of the N-body simulation.

use bytemuck::Pod;
use nalgebra::{RealField, SVector};

use std::{fmt::Debug, ops::AddAssign};

#[cfg(feature = "render")]
use crate::render::{Context, Drawable, PipelineType};
#[cfg(feature = "render")]
use wgpu::util::DeviceExt;

pub trait Float:
    num_traits::Float
    + std::fmt::Debug
    + std::fmt::Display
    + Clone
    + PartialEq
    + bytemuck::Zeroable
    + RealField
    + 'static
{
}

impl Float for f64 {}
impl Float for f32 {}

pub trait Particle<F: Float, const D: usize>: Debug {
    fn new(position: SVector<F, D>, velocity: SVector<F, D>, mass: F, radius: F) -> Self;
    fn position(&self) -> &SVector<F, D>;
    fn velocity(&self) -> &SVector<F, D>;
    fn acceleration(&self) -> &SVector<F, D>;
    fn position_mut(&mut self) -> &mut SVector<F, D>;
    fn velocity_mut(&mut self) -> &mut SVector<F, D>;
    fn acceleration_mut(&mut self) -> &mut SVector<F, D>;
    #[allow(unused)]
    fn get_radius(&self) -> F;
    fn get_mass(&self) -> F;
}

pub trait Simulation<F: Float, const D: usize, P, I: Integrator<F, D, P>>
where
    P: Particle<F, D>,
{
    fn new(points: Vec<P>, integrator: I) -> Self;
    fn init(&mut self);
    fn step(&mut self);
    fn add_point(&mut self, point: P);
    fn remove_point(&mut self, index: usize);
    fn get_points(&self) -> &Vec<P>;
    fn g(&self) -> F;
    fn dt(&self) -> F;
    fn set_g(&mut self, g: F);
    fn set_dt(&mut self, dt: F);
    #[cfg(feature = "render")]
    fn get_drawables(&mut self) -> Vec<&mut dyn Drawable>;
    #[cfg(feature = "render")]
    fn init_drawables(&mut self, context: &mut Context);
}

pub trait Integrator<F: Float, const D: usize, P: Particle<F, D>> {
    fn new() -> Self;
    fn init(&mut self) {}
    fn integrate_pre_force(&mut self, points: &mut Vec<P>, dt: F);
    fn integrate_after_force(&mut self, points: &mut Vec<P>, dt: F);
}

#[derive(Debug)]
pub struct LeapFrogIntegrator<F: Float, const D: usize, P>
where
    P: Particle<F, D>,
{
    _phantom: std::marker::PhantomData<(F, P)>,
}

impl<F: Float, const D: usize, P> Default for LeapFrogIntegrator<F, D, P>
where
    P: Particle<F, D>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float, const D: usize, P> Integrator<F, D, P> for LeapFrogIntegrator<F, D, P>
where
    P: Particle<F, D>,
{
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    fn integrate_pre_force(&mut self, points: &mut Vec<P>, dt: F) {
        for point in points.iter_mut() {
            let velocity = *point.velocity();
            point
                .position_mut()
                .add_assign(velocity * F::from(0.5).unwrap() * dt);
        }
    }
    fn integrate_after_force(&mut self, points: &mut Vec<P>, dt: F) {
        for point in points.iter_mut() {
            let acceleration = *point.acceleration();
            point.velocity_mut().add_assign(acceleration * dt);
            let velocity = *point.velocity();
            point
                .position_mut()
                .add_assign(velocity * F::from(0.5).unwrap() * dt);
        }
    }
}

#[derive(Debug, Clone, Copy, bytemuck::Zeroable)]
#[repr(C)]
pub struct PointParticle<F: Float, const D: usize> {
    pub position: SVector<F, D>,
    pub velocity: SVector<F, D>,
    pub acceleration: SVector<F, D>,
    pub mass: F,
}

unsafe impl<F: Float, const D: usize> Pod for PointParticle<F, D> {}

impl<F: Float, const D: usize> Particle<F, D> for PointParticle<F, D> {
    fn new(position: SVector<F, D>, velocity: SVector<F, D>, mass: F, _radius: F) -> Self {
        Self {
            position,
            velocity,
            acceleration: SVector::<F, D>::zeros(),
            mass,
        }
    }

    #[inline(always)]
    fn position(&self) -> &SVector<F, D> {
        &self.position
    }

    #[inline(always)]
    fn velocity(&self) -> &SVector<F, D> {
        &self.velocity
    }

    #[inline(always)]
    fn acceleration(&self) -> &SVector<F, D> {
        &self.acceleration
    }

    #[inline(always)]
    fn position_mut(&mut self) -> &mut SVector<F, D> {
        &mut self.position
    }

    #[inline(always)]
    fn velocity_mut(&mut self) -> &mut SVector<F, D> {
        &mut self.velocity
    }

    #[inline(always)]
    fn acceleration_mut(&mut self) -> &mut SVector<F, D> {
        &mut self.acceleration
    }

    #[inline(always)]
    fn get_mass(&self) -> F {
        self.mass
    }

    #[inline(always)]
    fn get_radius(&self) -> F {
        F::from(0.0).unwrap()
    }
}

#[cfg(feature = "render")]
#[derive(Debug)]
pub struct BufferData {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: Option<wgpu::Buffer>,
    num_indices: u32,
    needs_update: bool,
}

#[derive(Debug)]
pub struct Bodies<F: Float, const D: usize, P>
where
    P: Particle<F, D>,
{
    pub points: Vec<P>,
    #[cfg(feature = "render")]
    pub buffer_data: Option<BufferData>,
    pub _phantom: std::marker::PhantomData<F>,
}

impl<F: Float, const D: usize, P> Bodies<F, D, P>
where
    P: Particle<F, D>,
{
    pub fn new(points: Vec<P>) -> Self {
        Self {
            points,
            #[cfg(feature = "render")]
            buffer_data: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct BruteForceSimulation<F: Float, const D: usize, P, I = LeapFrogIntegrator<F, D, P>>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    bodies: Bodies<F, D, P>,
    integrator: I,
    g: F,
    dt: F,
}

#[cfg(feature = "render")]
impl<F: Float, const D: usize, P> Drawable for Bodies<F, D, P>
where
    P: Particle<F, D>,
{
    fn init(&mut self, context: &mut Context) {
        let position_buffer: Vec<f32> = self
            .points
            .iter()
            .map(|p| {
                p.position()
                    .iter()
                    .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                    .collect::<Vec<f32>>()
            })
            .flatten()
            .collect();

        let vertex_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&position_buffer),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        let index_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(
                    &(0..self.points.len() as u16).collect::<Vec<u16>>(),
                ),
                usage: wgpu::BufferUsages::INDEX,
            });

        self.buffer_data = Some(BufferData {
            vertex_buffer,
            index_buffer,
            instance_buffer: None,
            num_indices: self.points.len() as u32,
            needs_update: false,
        });
    }

    fn get_pipeline_type(&self) -> crate::render::PipelineType {
        PipelineType::Points
    }

    fn get_vertex_buffer(&self) -> &wgpu::Buffer {
        &self.buffer_data.as_ref().unwrap().vertex_buffer
    }

    fn get_index_buffer(&self) -> &wgpu::Buffer {
        &self.buffer_data.as_ref().unwrap().index_buffer
    }

    fn get_num_indices(&self) -> u32 {
        self.buffer_data.as_ref().unwrap().num_indices
    }

    fn update_buffers(&mut self, queue: &mut wgpu::Queue) {
        let buffer_data = self.buffer_data.as_ref().unwrap();

        let position_buffer: Vec<f32> = self
            .points
            .iter()
            .map(|p| {
                p.position()
                    .iter()
                    .map(|x| num_traits::cast::<F, f32>(*x).unwrap())
                    .collect::<Vec<f32>>()
            })
            .flatten()
            .collect();

        queue.write_buffer(
            &buffer_data.vertex_buffer,
            0,
            bytemuck::cast_slice(&position_buffer),
        );
    }

    fn buffer_needs_update(&self) -> bool {
        self.buffer_data.as_ref().unwrap().needs_update
    }
}

impl<F: Float, const D: usize, P, I> Simulation<F, D, P, I> for BruteForceSimulation<F, D, P, I>
where
    P: Particle<F, D>,
    I: Integrator<F, D, P>,
{
    fn new(points: Vec<P>, integrator: I) -> Self {
        Self {
            bodies: Bodies::new(points),
            integrator,
            g: F::from(1.0).unwrap(),
            dt: F::from(0.001).unwrap(),
        }
    }

    fn init(&mut self) {
        self.integrator.init();
    }

    fn g(&self) -> F {
        self.g
    }

    fn dt(&self) -> F {
        self.dt
    }

    fn set_g(&mut self, g: F) {
        self.g = g;
    }

    fn set_dt(&mut self, dt: F) {
        self.dt = dt;
    }

    fn step(&mut self) {
        let dt = self.dt();
        self.integrator
            .integrate_pre_force(&mut self.bodies.points, dt);

        for point in self.bodies.points.iter_mut() {
            point.acceleration_mut().fill(F::from(0.0).unwrap());
        }

        for i in 0..self.bodies.points.len() {
            for j in 0..self.bodies.points.len() {
                if i == j {
                    continue;
                }
                let r = self.bodies.points[i].position() - self.bodies.points[j].position();
                let r_dist = r.norm();
                let r_cubed = r_dist * r_dist * r_dist;
                let m_i = self.bodies.points[i].get_mass();
                let m_j = self.bodies.points[j].get_mass();
                let force = self.g() / r_cubed;
                *self.bodies.points[i].acceleration_mut() -= r * force * m_j;
                *self.bodies.points[j].acceleration_mut() += r * force * m_i;
            }
        }

        self.integrator
            .integrate_after_force(&mut self.bodies.points, dt);

        #[cfg(feature = "render")]
        {
            self.bodies.buffer_data.as_mut().unwrap().needs_update = true;
        }
    }

    fn add_point(&mut self, point: P) {
        self.bodies.points.push(point);
    }

    fn remove_point(&mut self, index: usize) {
        self.bodies.points.remove(index);
    }

    fn get_points(&self) -> &Vec<P> {
        &self.bodies.points
    }

    #[cfg(feature = "render")]
    fn get_drawables(&mut self) -> Vec<&mut dyn Drawable> {
        vec![&mut self.bodies]
    }

    #[cfg(feature = "render")]
    fn init_drawables(&mut self, context: &mut Context) {
        self.bodies.init(context);
    }
}
