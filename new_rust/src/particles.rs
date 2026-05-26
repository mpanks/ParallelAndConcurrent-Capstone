use rand::{RngExt};

use crate::vertex::Vertex;
use rand;

const INITIAL_VEL_MAX: f32 = 3.0;
const INITIAL_VEL_MIN: f32 = 1.5;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub mass: f32,
    pub temp: f32,
    pub drag: f32,
    pub alive: bool,
    pub time: f32,
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            mass: 1.0,
            temp: 1.0,
            drag: 0.0,
            alive: false,
            time: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct Particles{
    pub particles: Vec<Particle>,
    pub free_indices: Vec<usize>,
}
impl Particles{
    pub fn new(spawn_number: i32, circle: &[f32; 2]) -> Particles{
        let mut particles = Vec::with_capacity(spawn_number as usize);
        let mut free_indices = Vec::new();
        
        for i in 0..spawn_number {
            let params = generate_particle(circle);
            particles.push(Particle {
                position: params[0],
                velocity: params[1],
                mass: 1.0,
                temp: 1.0,
                drag: 0.0,
                alive: false,
                time: 0.0,
            });
            free_indices.push(i as usize);
        }

        Particles { 
            particles,
            free_indices,
        }
    }

    pub fn spawn_some(&mut self, count: usize, circle: &[f32; 2]) {
        let mut spawned = 0;
        let mut rng = rand::rng();

        while spawned < count && !self.free_indices.is_empty() {
            let random_slot = rng.random_range(0..self.free_indices.len());
            let index = self.free_indices.swap_remove(random_slot);
            self.respawn(index, circle);
            spawned += 1;
        }
    }

    pub fn write_positions_sampled(
        &self,
        out: &mut Vec<Vertex>,
        step: usize,
    ) {
        out.clear();

        for (i, particle) in self.particles.iter().enumerate() {
            if !particle.alive {
                continue;
            }
            
            if i % step == 0 {
                out.push(
                    Vertex{
                        position: particle.position,
                        temp: particle.temp,
                        mass: particle.mass,
                        drag: particle.drag,
                    }
                );
            }
        }
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    pub fn live_particles(&self) -> i32{
        self.particles.iter().filter(|p| p.alive).count() as i32
    }

    pub fn dead_particles(&self) -> i32{
        self.particles.iter().filter(|p| !p.alive).count() as i32
    }

    pub fn respawn(&mut self, i: usize, circle: &[f32; 2]) {
        let params = generate_particle(circle);
        self.particles[i] = Particle {
            position: params[0],
            velocity: params[1],
            mass: 1.0,
            temp: 1.0,
            drag: 0.0,
            alive: true,
            time: 0.0,
        };
    }
}

pub fn generate_particle(circle: &[f32; 2]) -> [[f32; 3]; 2] {
    let mut rng = rand::rng();

    let centre_y = circle[0];
    let radius = circle[1];

    // Spawn particles randomly across the emitter disc in x/z plane.
    let theta = rng.random_range(0.0..2.0 * std::f32::consts::PI);
    let r = rng.random::<f32>().sqrt() * radius;

    let x = r * theta.cos();
    let z = 0.5 + r * theta.sin();
    let y = centre_y;

    // Randomize emission direction independently from spawn position.
    let azimuth = rng.random_range(0.0..2.0 * std::f32::consts::PI);
    let elevation = rng.random_range(0.0..30.0_f32).to_radians();
    let initial_vel = rng.random_range(INITIAL_VEL_MIN..INITIAL_VEL_MAX);
    let speed = initial_vel * rng.random_range(0.95..1.05);

    let x_vel = speed * elevation.sin() * azimuth.cos();
    let y_vel = -speed * elevation.cos();
    let z_vel = speed * elevation.sin() * azimuth.sin();

    // Return the position [x, y, z] and velocity [x_vel, y_vel, z_vel]
    return [[x, y, z], [x_vel, y_vel, z_vel]];
}