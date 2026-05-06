use crate::spatial_grid::*;
use crate::particles::*;

#[derive(Clone, Copy)]
pub struct Collision {
    pub a: usize,
    pub b: usize,
}

#[derive(Clone)]
pub struct CollisionSnapshot {
    pub positions: Vec<[f32; 3]>,
    pub time: Vec<f32>,
    pub alive: Vec<bool>,
    pub grid: SpatialGrid,
}

impl CollisionSnapshot {
    pub fn new(p: &Particles, grid: &SpatialGrid) -> Self {
        let positions = p.particles.iter().map(|particle| particle.position).collect();
        let time = p.particles.iter().map(|particle| particle.time).collect();
        let alive = p.particles.iter().map(|particle| particle.alive).collect();

        Self {
            positions,
            time,
            alive,
            grid: SpatialGrid {
                cell_size: grid.cell_size,
                inv_cell_size: grid.inv_cell_size,
                dims: grid.dims,
                cells: grid.cells.clone(),
            },
        }
    }
}

pub fn detect_collisions_snapshot(snapshot: &CollisionSnapshot, start_idx: usize, end_idx: usize) -> Vec<Collision> {
    let offsets = [-1, 0, 1];
    let (nx, ny, nz) = snapshot.grid.dims;
    let mut collisions = Vec::new();

    for cell_idx in start_idx..end_idx {
        let indices = &snapshot.grid.cells[cell_idx];
        let z = cell_idx as i32 / (nx * ny);
        let rem = cell_idx as i32 % (nx * ny);
        let y = rem / nx;
        let x = rem % nx;

        for &i in indices {
            for dx in &offsets {
                for dy in &offsets {
                    for dz in &offsets {
                        let nx_ = x + dx;
                        let ny_ = y + dy;
                        let nz_ = z + dz;

                        if nx_ < 0 || ny_ < 0 || nz_ < 0 { continue; }
                        if nx_ >= nx || ny_ >= ny || nz_ >= nz { continue; }

                        let neighbor_idx = snapshot.grid.index(nx_, ny_, nz_);

                        for &j in &snapshot.grid.cells[neighbor_idx] {
                            if i >= j { continue; }
                            if snapshot.time[i] < 0.5 || snapshot.time[j] < 0.5 { continue; }
                            if !snapshot.alive[i] || !snapshot.alive[j] { continue; }

                            let dx = snapshot.positions[i][0] - snapshot.positions[j][0];
                            let dy = snapshot.positions[i][1] - snapshot.positions[j][1];
                            let dz = snapshot.positions[i][2] - snapshot.positions[j][2];

                            let dist2 = dx*dx + dy*dy + dz*dz;
                            let radius = 0.001;
                            if dist2 < (radius * 2.0) * (radius * 2.0) {
                                collisions.push(Collision { a: i, b: j });
                            }
                        }
                    }
                }
            }
        }
    }

    collisions
}

pub fn merge(a: usize, b: usize, p: &mut Particles) {
    let m0 = p.particles[a].mass;
    let m1 = p.particles[b].mass;
    let m_new = m0 + m1;

    // momentum conservation
    p.particles[a].velocity[0] = (m0 * p.particles[a].velocity[0] + m1 * p.particles[b].velocity[0]) / m_new;
    p.particles[a].velocity[1] = (m0 * p.particles[a].velocity[1] + m1 * p.particles[b].velocity[1]) / m_new;
    p.particles[a].velocity[2] = (m0 * p.particles[a].velocity[2] + m1 * p.particles[b].velocity[2]) / m_new;

    p.particles[a].mass = m_new;
    p.particles[a].temp = (m0 * p.particles[a].temp + m1 * p.particles[b].temp) / (m0 + m1);

    // kill particle b
    p.particles[b].alive = false;
    p.free_indices.push(b);
}