use crate::particles::*;
use crate::bounds;
use crate::spatial_grid;

use spatial_grid::SpatialGrid;
use bounds::Bounds;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use rayon::prelude::*;

const COOLING_RATE:f32 = 2.0;
const DRAG_K: f32 = 0.1; // linear drag coefficient (higher = more resistance)
pub struct Simulation{
    pub current: Particles,
    pub render: Particles
}
impl Simulation {
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.current, &mut self.render);
    }
}

pub fn physics_step(p: &mut Particles, dt: f32, circle: &[f32; 2], floor_counter: Arc<AtomicU64>, bounds: &Bounds, grid: &mut SpatialGrid) {
    let g = -9.81;

    // Process particles in parallel chunks
    p.particles
        .par_chunks_mut(256)
        .for_each(|chunk| {
            for particle in chunk.iter_mut() {
                if !particle.alive { continue; }

                //particle.time += dt;

                // update velocity (gravity)
                particle.velocity[1] += g * dt;

                // apply linear drag: a_drag = -DRAG_K * v / mass
                let vx = particle.velocity[0];
                let vy = particle.velocity[1];
                let vz = particle.velocity[2];
                let speed = (vx*vx + vy*vy + vz*vz).sqrt();

                if speed > 0.0 {
                    let drag_acc_factor = DRAG_K / particle.mass;
                    particle.velocity[0] += -drag_acc_factor * vx * dt;
                    particle.velocity[1] += -drag_acc_factor * vy * dt;
                    particle.velocity[2] += -drag_acc_factor * vz * dt;

                    // store a normalized drag value for rendering (clamped 0..1)
                    let drag_val = (drag_acc_factor * speed).abs();
                    particle.drag = drag_val.min(1.0);
                } else {
                    particle.drag = 0.0;
                }

                // update position
                particle.position[0] += particle.velocity[0] * dt;
                particle.position[1] += particle.velocity[1] * dt;
                particle.position[2] += particle.velocity[2] * dt;

                // floor collision
                if particle.position[1] <= bounds.min_y {
                    respawn_particle(particle, circle);
                    floor_counter.fetch_add(1, Ordering::Relaxed);
                    //particle.alive = false;
                    continue;
                }

                // boundary reflections - check position, not velocity
                if particle.position[0] <= bounds.min_x || particle.position[0] >= bounds.max_x {
                    particle.velocity[0] *= -1.0;
                }

                if particle.position[2] <= bounds.min_z || particle.position[2] >= bounds.max_z{
                    particle.velocity[2] *= -1.0;
                }
            }
        });
}

pub fn cooling_step(p: &mut Particles, dt: f32) {
    p.particles
        .par_iter_mut()
        .filter(|particle| particle.alive)
        .for_each(|particle| {
            // Cooling depends on temp diff. between particle and ambient - more realistic
            particle.temp -= (COOLING_RATE * dt) / particle.mass;

            if particle.temp < 0.0 {
                particle.temp = 0.0;
            }
        });
}

fn respawn_particle(
    particle: &mut Particle,
    circle: &[f32; 2],
) {
    let params = generate_particle(circle);

    particle.position = params[0];
    particle.velocity = params[1];
    particle.temp = 1.0;
        particle.drag = 0.0;
    particle.mass = 1.0;
    particle.time = 0.0;
}