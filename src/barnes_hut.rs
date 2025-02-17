// 3D Barnes–Hut implementation

use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn zero() -> Self {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, scalar: f64) -> Vec3 {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, scalar: f64) -> Vec3 {
        Vec3 {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Oct {
    pub center: Vec3,
    pub half_dimension: f64,
}

impl Oct {
    pub fn contains(&self, point: &Vec3) -> bool {
        point.x >= self.center.x - self.half_dimension
            && point.x <= self.center.x + self.half_dimension
            && point.y >= self.center.y - self.half_dimension
            && point.y <= self.center.y + self.half_dimension
            && point.z >= self.center.z - self.half_dimension
            && point.z <= self.center.z + self.half_dimension
    }
}

#[derive(Clone, Debug)]
pub struct Body3D {
    pub position: Vec3,
    pub _prev_position: Vec3,
    pub velocity: Vec3,
    pub _acceleration: Vec3,
    pub mass: f64,
}

#[derive(Clone, Debug)]
pub struct BHOctree {
    pub oct: Oct,
    pub body: Option<Body3D>,
    pub mass: f64,
    pub com: Vec3,
    pub children: [Option<Box<BHOctree>>; 8],
}

impl BHOctree {
    const OCTANTS: [(f64, f64, f64); 8] = [
        (-1.0, -1.0, -1.0), // index 0: x<, y<, z<
        (1.0, -1.0, -1.0),  // index 1: x>=, y<, z<
        (-1.0, 1.0, -1.0),  // index 2: x<, y>=, z<
        (1.0, 1.0, -1.0),   // index 3: x>=, y>=, z<
        (-1.0, -1.0, 1.0),  // index 4: x<, y<, z>=
        (1.0, -1.0, 1.0),   // index 5: x>=, y<, z>=
        (-1.0, 1.0, 1.0),   // index 6: x<, y>=, z>=
        (1.0, 1.0, 1.0),    // index 7: x>=, y>=, z>=
    ];

    pub fn new(oct: Oct) -> BHOctree {
        BHOctree {
            oct,
            body: None,
            mass: 0.0,
            com: Vec3::zero(),
            children: [None, None, None, None, None, None, None, None],
        }
    }

    pub fn is_external(&self) -> bool {
        self.children.iter().all(|c| c.is_none())
    }

    pub fn subdivide(&mut self) {
        let new_half = self.oct.half_dimension / 2.0;
        for i in 0..8 {
            let (dx, dy, dz) = Self::OCTANTS[i];
            let center = Vec3 {
                x: self.oct.center.x + dx * new_half,
                y: self.oct.center.y + dy * new_half,
                z: self.oct.center.z + dz * new_half,
            };
            self.children[i] = Some(Box::new(BHOctree::new(Oct {
                center,
                half_dimension: new_half,
            })));
        }
    }

    fn get_child_index(&self, pos: &Vec3) -> usize {
        let mut index = 0;
        if pos.x >= self.oct.center.x {
            index |= 1
        }
        if pos.y >= self.oct.center.y {
            index |= 2
        }
        if pos.z >= self.oct.center.z {
            index |= 4
        }
        index
    }

    pub fn insert(&mut self, b: Body3D) {
        if !self.oct.contains(&b.position) {
            return;
        }

        if self.body.is_none() && self.is_external() {
            let mass = b.mass;
            let pos = b.position;
            self.body = Some(b);
            self.mass = mass;
            self.com = pos;
            return;
        } else {
            if self.is_external() {
                self.subdivide();
                if let Some(existing) = self.body.take() {
                    let index = self.get_child_index(&existing.position);
                    if let Some(child) = &mut self.children[index] {
                        child.insert(existing);
                    }
                }
            }

            let total_mass = self.mass + b.mass;
            self.com = (self.com * self.mass + b.position * b.mass) / total_mass;
            self.mass = total_mass;

            let index = self.get_child_index(&b.position);
            if let Some(child) = &mut self.children[index] {
                child.insert(b);
            }
        }
    }

    pub fn calc_force(&self, b: &Body3D, theta: f64, g: f64, softening: f64) -> Vec3 {
        if self.mass == 0.0 {
            return Vec3::zero();
        }

        let dx = self.com.x - b.position.x;
        let dy = self.com.y - b.position.y;
        let dz = self.com.z - b.position.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if self.is_external() || (self.oct.half_dimension * 2.0 / dist) < theta {
            if let Some(body_in_node) = &self.body {
                if (body_in_node.position.x == b.position.x)
                    && (body_in_node.position.y == b.position.y)
                    && (body_in_node.position.z == b.position.z)
                {
                    return Vec3::zero();
                }
            }
            let factor = g * b.mass * self.mass / (dist * dist + softening * softening).powf(1.5);
            Vec3 {
                x: factor * dx,
                y: factor * dy,
                z: factor * dz,
            }
        } else {
            let mut force = Vec3::zero();
            for child in self.children.iter().flatten() {
                force = force + child.calc_force(b, theta, g, softening);
            }
            force
        }
    }
}
