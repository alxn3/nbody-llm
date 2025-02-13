// Barnes–Hut implementation module

use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn zero() -> Self {
        Vec2 { x: 0.0, y: 0.0 }
    }
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, other: Vec2) -> Vec2 {
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, other: Vec2) -> Vec2 {
        Vec2 { x: self.x - other.x, y: self.y - other.y }
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(self, scalar: f64) -> Vec2 {
        Vec2 { x: self.x * scalar, y: self.y * scalar }
    }
}

impl Div<f64> for Vec2 {
    type Output = Vec2;
    fn div(self, scalar: f64) -> Vec2 {
        Vec2 { x: self.x / scalar, y: self.y / scalar }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Quad {
    pub center: Vec2,
    pub half_dimension: f64,
}

impl Quad {
    /// Returns true if the point lies in this quadrant.
    pub fn contains(&self, point: &Vec2) -> bool {
        let left = self.center.x - self.half_dimension;
        let right = self.center.x + self.half_dimension;
        let bottom = self.center.y - self.half_dimension;
        let top = self.center.y + self.half_dimension;
        point.x >= left && point.x <= right && point.y >= bottom && point.y <= top
    }
}

#[derive(Clone, Debug)]
pub struct Body {
    pub position: Vec2,
    pub prev_position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub mass: f64,
}

#[derive(Clone, Debug)]
pub struct BHTree {
    pub quad: Quad,
    pub body: Option<Body>,
    pub mass: f64,
    pub com: Vec2, // center of mass of this node
    pub nw: Option<Box<BHTree>>,
    pub ne: Option<Box<BHTree>>,
    pub sw: Option<Box<BHTree>>,
    pub se: Option<Box<BHTree>>,
}

impl BHTree {
    pub fn new(quad: Quad) -> BHTree {
        BHTree {
            quad,
            body: None,
            mass: 0.0,
            com: Vec2::zero(),
            nw: None,
            ne: None,
            sw: None,
            se: None,
        }
    }

    /// Returns true if this node has no subdivided children.
    pub fn is_external(&self) -> bool {
        self.nw.is_none() && self.ne.is_none() && self.sw.is_none() && self.se.is_none()
    }

    /// Subdivide this quadrant into four children.
    pub fn subdivide(&mut self) {
        let new_half = self.quad.half_dimension / 2.0;
        let x = self.quad.center.x;
        let y = self.quad.center.y;
        self.nw = Some(Box::new(BHTree::new(Quad { center: Vec2 { x: x - new_half, y: y + new_half }, half_dimension: new_half })));
        self.ne = Some(Box::new(BHTree::new(Quad { center: Vec2 { x: x + new_half, y: y + new_half }, half_dimension: new_half })));
        self.sw = Some(Box::new(BHTree::new(Quad { center: Vec2 { x: x - new_half, y: y - new_half }, half_dimension: new_half })));
        self.se = Some(Box::new(BHTree::new(Quad { center: Vec2 { x: x + new_half, y: y - new_half }, half_dimension: new_half })));
    }

    /// Given a position, returns a mutable reference to the appropriate child node.
    pub fn get_child_mut(&mut self, pos: &Vec2) -> &mut Box<BHTree> {
        if pos.x <= self.quad.center.x {
            if pos.y >= self.quad.center.y {
                self.nw.as_mut().unwrap()
            } else {
                self.sw.as_mut().unwrap()
            }
        } else {
            if pos.y >= self.quad.center.y {
                self.ne.as_mut().unwrap()
            } else {
                self.se.as_mut().unwrap()
            }
        }
    }

    /// Insert a body into the Barnes–Hut tree.
    pub fn insert(&mut self, b: Body) {
        // If this body is not in our quadrant, ignore it.
        if !self.quad.contains(&b.position) {
            return;
        }
        // If the node is empty and is an external node, store this body.
        if self.body.is_none() && self.is_external() {
            let mass = b.mass;
            let pos = b.position;
            self.body = Some(b);
            self.mass = mass;
            self.com = pos;
            return;
        } else {
            // If this node is external but already contains a body,
            // subdivide and then reinsert the old one.
            if self.is_external() {
                self.subdivide();
                if let Some(existing) = self.body.take() {
                    let child = self.get_child_mut(&existing.position);
                    child.insert(existing);
                }
            }
            // Update the mass and center-of-mass.
            let total_mass = self.mass + b.mass;
            self.com = (self.com * self.mass + b.position * b.mass) / total_mass;
            self.mass = total_mass;
            // Finally, insert the new body in the appropriate quadrant.
            let child = self.get_child_mut(&b.position);
            child.insert(b);
        }
    }

    /// Recursively calculate the gravitational force exerted on body `b` from this tree node.
    /// Uses the Barnes–Hut approximation if the node is sufficiently far away.
    pub fn calc_force(&self, b: &Body, theta: f64, G: f64, softening: f64) -> Vec2 {
        if self.mass == 0.0 {
            return Vec2::zero();
        }
        let dx = self.com.x - b.position.x;
        let dy = self.com.y - b.position.y;
        let dist = (dx * dx + dy * dy).sqrt();
        // If this node is external or the node's size is small enough compared to the distance,
        // treat it as a single body.
        if self.is_external() || (self.quad.half_dimension * 2.0 / dist) < theta {
            // If the only body in the node is the same as b, then do not add any force.
            if let Some(body_in_node) = &self.body {
                if (body_in_node.position.x == b.position.x) && (body_in_node.position.y == b.position.y) {
                    return Vec2::zero();
                }
            }
            let factor = G * b.mass * self.mass / (dist * dist + softening * softening);
            // Normalize the direction and multiply.
            return Vec2 { x: factor * dx / dist, y: factor * dy / dist };
        } else {
            // Otherwise, aggregate the force contributions from each child.
            let mut force = Vec2::zero();
            if let Some(ref nw) = self.nw {
                force = force + nw.calc_force(b, theta, G, softening);
            }
            if let Some(ref ne) = self.ne {
                force = force + ne.calc_force(b, theta, G, softening);
            }
            if let Some(ref sw) = self.sw {
                force = force + sw.calc_force(b, theta, G, softening);
            }
            if let Some(ref se) = self.se {
                force = force + se.calc_force(b, theta, G, softening);
            }
            force
        }
    }
}