// This file defines shared behavior for the manual and LLM implementations of the N-body simulation.

pub trait Particle<FLOAT, const D: usize> {
    fn get_position(&self) -> &[FLOAT; D];
    fn get_velocity(&self) -> &[FLOAT; D];
    fn get_acceleration(&self) -> [FLOAT; D];
    fn set_position(&mut self, position: [FLOAT; D]);
    fn set_velocity(&mut self, velocity: [FLOAT; D]);
    fn set_acceleration(&mut self, acceleration: [FLOAT; D]);
    #[allow(unused)]
    fn get_radius(&self) -> FLOAT;
    fn get_mass(&self) -> FLOAT;
}

pub trait Simulation<FLOAT, const D: usize, P, I: Integrator<FLOAT, D, P>>
where
    P: Particle<FLOAT, D>,
{
    fn init(&mut self);
    fn step(&mut self);
    fn add_point(&mut self, point: P);
    fn get_points(&self) -> &[P];
}

pub trait Integrator<FLOAT, const D: usize, P: Particle<FLOAT, D>> {
    fn init(&mut self);
    fn integrate_pre_force(&mut self, points: &mut Vec<P>);
    fn integrate_after_force(&mut self, points: &mut Vec<P>);
}
