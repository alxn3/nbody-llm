mod barnes_hut;
mod brute_force;

pub use brute_force::*;

pub use barnes_hut::*;

use nalgebra::SVector;

use crate::shared::{Float, Particle, PointParticle};

#[derive(Clone)]
pub struct TreeParticle<F: Float, const D: usize> {
    pub position: SVector<F, D>,
    pub velocity: SVector<F, D>,
    pub acceleration: SVector<F, D>,
    pub mass: F,
    pub radius: F,
    node: Option<Box<OrthNode<F, D>>>,
}

impl<F: Float, const D: usize> Particle<F, D> for TreeParticle<F, D> {
    fn new(position: SVector<F, D>, velocity: SVector<F, D>, mass: F, radius: F) -> Self {
        Self {
            position,
            velocity,
            acceleration: SVector::<F, D>::zeros(),
            mass,
            radius,
            node: None,
        }
    }

    fn position(&self) -> &SVector<F, D> {
        &self.position
    }

    fn velocity(&self) -> &SVector<F, D> {
        &self.velocity
    }

    fn acceleration(&self) -> &SVector<F, D> {
        &self.acceleration
    }

    fn position_mut(&mut self) -> &mut SVector<F, D> {
        &mut self.position
    }

    fn velocity_mut(&mut self) -> &mut SVector<F, D> {
        &mut self.velocity
    }

    fn acceleration_mut(&mut self) -> &mut SVector<F, D> {
        &mut self.acceleration
    }

    fn get_mass(&self) -> F {
        self.mass
    }
}

impl<F: Float, const D: usize> From<PointParticle<F, D>> for TreeParticle<F, D> {
    fn from(particle: PointParticle<F, D>) -> Self {
        Self {
            position: *particle.position(),
            velocity: *particle.velocity(),
            acceleration: *particle.acceleration(),
            mass: particle.get_mass(),
            radius: F::from(0.0).unwrap(),
            node: None,
        }
    }
}
