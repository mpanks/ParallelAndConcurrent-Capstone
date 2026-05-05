use crate::bounds;
use crate::particles;

use particles::Particles;
use bounds::Bounds;

#[derive(Clone)]
pub struct SpatialGrid {
    pub cell_size: f32,
    pub inv_cell_size: f32,
    pub dims: (i32, i32, i32),
    pub cells: Vec<Vec<usize>>,
}
impl SpatialGrid {
    pub fn new(bounds: &Bounds, cell_size: f32) -> Self {
        let nx = ((bounds.max_x - bounds.min_x) / cell_size).ceil() as i32;
        let ny = ((bounds.max_y - bounds.min_y) / cell_size).ceil() as i32;
        let nz = ((bounds.max_z - bounds.min_z) / cell_size).ceil() as i32;

        let total = (nx * ny * nz) as usize;

        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            dims: (nx, ny, nz),
            cells: vec![Vec::new(); total],
        }
    }

    pub fn index(&self, x: i32, y: i32, z: i32) -> usize {
        let (nx, ny, _) = self.dims;
        (x + y * nx + z * nx * ny) as usize
    }

    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    pub fn insert(&mut self, i: usize, x: f32, y: f32, z: f32, bounds: &Bounds) {
        let gx = ((x - bounds.min_x) * self.inv_cell_size) as i32;
        let gy = ((y - bounds.min_y) * self.inv_cell_size) as i32;
        let gz = ((z - bounds.min_z) * self.inv_cell_size) as i32;

        if gx < 0 || gy < 0 || gz < 0 { return; }
        if gx >= self.dims.0 || gy >= self.dims.1 || gz >= self.dims.2 { return; }

        let idx = self.index(gx, gy, gz);
        self.cells[idx].push(i);
    }
}

pub fn build_grid(grid: &mut SpatialGrid, p: &Particles, bounds: &Bounds) {
    grid.clear();

    for (i, particle) in p.particles.iter().enumerate() {
        if particle.alive {
            grid.insert(i, particle.position[0], particle.position[1], particle.position[2], bounds);
        }
    }
}