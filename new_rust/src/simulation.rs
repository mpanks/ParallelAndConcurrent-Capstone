use crate::particles;
use crate::bounds;

use bounds::Bounds;
use particles::Particles;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
pub struct Simulation{
    pub current: Particles,
    pub render: Particles
}
impl Simulation {
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.current, &mut self.render);
    }
}

pub fn physics_step(p: &mut Particles, dt: f32, circle: &[f32; 2], floor_counter: Arc<AtomicU64>, bounds: &Bounds) {
    let g = -9.81;

    let len = p.pos_x.len();

    (0..len).into_iter().for_each(|i| {
        if !p.alive[i] {
            return;
        }

        // gravity
        p.vel_y[i] += g * dt;

        // integrate
        p.pos_x[i] += p.vel_x[i] * dt;
        p.pos_y[i] += p.vel_y[i] * dt;
        p.pos_z[i] += p.vel_z[i] * dt;

        // floor
        if p.pos_y[i] <= bounds.min_y {
            p.respawn(i, circle);
            p.alive[i] = false;
            floor_counter.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // reflect X
        if p.pos_x[i] < bounds.min_x || p.pos_x[i] > bounds.max_x {
            p.vel_x[i] = -p.vel_x[i];
        }

        // reflect Z
        if p.pos_z[i] < bounds.min_z || p.pos_z[i] > bounds.max_z{
            p.vel_z[i] = -p.vel_z[i];
        }
    });
}