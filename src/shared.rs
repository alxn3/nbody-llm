// This file defines shared behavior for the manual and LLM implementations of the N-body simulation.

use nalgebra::{SVector, SimdRealField};

#[cfg(feature = "render")]
pub trait RenderTraits: egui::emath::Numeric {}
#[cfg(not(feature = "render"))]
pub trait RenderTraits {}

pub trait Float:
    num_traits::Float
    + std::fmt::Debug
    + std::fmt::Display
    + Copy
    + Clone
    + PartialEq
    + bytemuck::Zeroable
    + SimdRealField
    + RenderTraits
    + 'static
{
}

macro_rules! impl_float {
    ($($t:ty),*) => {
        $(
            impl Float for $t {}
        )*
    };
}

macro_rules! impl_render {
    ($($t:ty),*) => {
        $(
            impl RenderTraits for $t {}
        )*
    };
}

impl_float!(f32, f64);
impl_render!(f32, f64);

pub trait Particle<F: Float, const D: usize>: Clone {
    fn new(position: SVector<F, D>, velocity: SVector<F, D>, mass: F, radius: F) -> Self;
    fn position(&self) -> &SVector<F, D>;
    fn velocity(&self) -> &SVector<F, D>;
    fn acceleration(&self) -> &SVector<F, D>;
    fn position_mut(&mut self) -> &mut SVector<F, D>;
    fn velocity_mut(&mut self) -> &mut SVector<F, D>;
    fn acceleration_mut(&mut self) -> &mut SVector<F, D>;
    fn get_mass(&self) -> F;
}

pub trait ParticleSized<F: Float, const D: usize>: Particle<F, D> {
    fn get_radius(&mut self, radius: F);
}

#[derive(Debug, Clone)]
pub struct SimulationSettings<F: Float> {
    pub g: F,
    pub g_soft: F,
    pub dt: F,
}

impl<F: Float> Default for SimulationSettings<F> {
    fn default() -> Self {
        SimulationSettings {
            g: F::from(1.0).unwrap(),
            g_soft: F::from(0.0).unwrap(),
            dt: F::from(1e-3).unwrap(),
        }
    }
}

pub trait Simulation<F: Float, const D: usize, P, I: Integrator<F, D, P>>: Clone
where
    P: Particle<F, D>,
{
    fn new(points: Vec<P>, integrator: I, bounds: Bounds<F, D>) -> Self;
    fn init(&mut self);
    fn step(&mut self) {
        self.step_by(self.settings().dt);
    }
    fn step_by(&mut self, dt: F);
    fn update_forces(&mut self);
    fn add_point(&mut self, point: P);
    fn remove_point(&mut self, index: usize);
    fn get_points(&self) -> &Vec<P>;
    fn elapsed(&self) -> F;
    fn settings(&self) -> &SimulationSettings<F>;
    fn settings_mut(&mut self) -> &mut SimulationSettings<F>;
}

pub trait Integrator<F: Float, const D: usize, P: Particle<F, D>>: Clone {
    fn new() -> Self;
    fn init(&mut self) {}
    fn integrate_pre_force(&mut self, points: &mut Vec<P>, dt: F);
    fn integrate_after_force(&mut self, points: &mut Vec<P>, dt: F);
}

#[derive(Debug, Clone)]
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
            *point.position_mut() += velocity * F::from(0.5).unwrap() * dt;
        }
    }
    fn integrate_after_force(&mut self, points: &mut Vec<P>, dt: F) {
        for point in points.iter_mut() {
            let acceleration = *point.acceleration();
            *point.velocity_mut() += acceleration * dt;
            let velocity = *point.velocity();
            *point.position_mut() += velocity * F::from(0.5).unwrap() * dt;
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PointParticle<F: Float, const D: usize> {
    pub position: SVector<F, D>,
    pub velocity: SVector<F, D>,
    pub acceleration: SVector<F, D>,
    pub mass: F,
}

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
}

pub trait AABB<F: Float, const D: usize, P: Particle<F, D>> {
    fn min(&self) -> SVector<F, D>;
    fn max(&self) -> SVector<F, D>;
    fn center(&self) -> SVector<F, D>;
    fn contains(&self, position: &SVector<F, D>) -> bool {
        *position >= self.min() && *position <= self.max()
    }
}

#[derive(Debug, Clone)]
pub struct Bounds<F: Float, const D: usize> {
    center: SVector<F, D>,
    pub half_width: F,
}

impl<F: Float, const D: usize> AABB<F, D, PointParticle<F, D>> for Bounds<F, D> {
    fn min(&self) -> SVector<F, D> {
        self.center.add_scalar(-self.half_width)
    }

    fn max(&self) -> SVector<F, D> {
        self.center.add_scalar(self.half_width)
    }

    fn center(&self) -> SVector<F, D> {
        self.center
    }
}

impl<F: Float, const D: usize> Bounds<F, D> {
    pub fn new(center: SVector<F, D>, width: F) -> Self {
        Self {
            center,
            half_width: width * F::from(0.5).unwrap(),
        }
    }

    pub fn get_orthant(&self, position: &SVector<F, D>) -> usize {
        let center = self.center();
        let mut orthant = 0;
        for i in 0..D {
            if position[i] > center[i] {
                orthant |= 1 << i;
            }
        }
        orthant
    }

    pub fn create_orthant(&self, orthant: usize) -> Self {
        let mut center = self.center();
        let half_width = self.half_width * F::from(0.5).unwrap();
        for i in 0..D {
            if orthant & (1 << i) != 0 {
                center[i] += half_width;
            } else {
                center[i] -= half_width;
            }
        }
        Self { center, half_width }
    }
}
