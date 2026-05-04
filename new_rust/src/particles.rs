use rand::{RngExt};

const INITIAL_VEL: f32 = 1.5;
pub struct Particles{
    pub pos_x: Vec<f32>,
    pub pos_y: Vec<f32>,
    pub pos_z: Vec<f32>,

    pub vel_x: Vec<f32>,
    pub vel_y: Vec<f32>,
    pub vel_z: Vec<f32>,

    pub mass: Vec<f32>,
    pub temp: Vec<f32>,

    pub alive: Vec<bool>,
}
impl Particles{
    pub fn new(spawn_number: i32, circle: &[f32; 2], skip: bool) -> Particles{
        if !skip{
            let mut pos_x = vec!();
            let mut pos_y = vec!();
            let mut pos_z = vec!();
            let mut vel_x = vec!();
            let mut vel_y = vec!();
            let mut vel_z = vec!();
            let mut mass = vec!();
            let mut temp = vec!();
            let mut alive = vec!();
            for _ in 0..spawn_number{
                let params = generate_particle(circle);
                pos_x.push(params[0][0]);
                pos_y.push(params[0][1]);
                pos_z.push(params[0][2]);

                vel_x.push(params[1][0]);
                vel_y.push(params[1][1]);
                vel_z.push(params[1][2]);

                mass.push(1.0);
                temp.push(1.0);
                alive.push(false);
            }

            return Particles { pos_x: pos_x, pos_y: pos_y, pos_z: pos_z, vel_x: vel_x, vel_y: vel_y, vel_z: vel_z, mass: mass, temp: temp, alive: alive }
        }
        return Particles { pos_x: vec!(), pos_y: vec!(), pos_z: vec!(), vel_x: vec!(), vel_y: vec!(), vel_z: vec!(), mass: vec!(), temp: vec!(), alive: vec!() }
    }

    pub fn spawn_some(&mut self, count: usize, circle: &[f32; 2]) {
        let mut spawned = 0;

        for i in 0..self.pos_x.len() {
            if !self.alive[i] {
                self.respawn(i, circle);
                spawned += 1;

                if spawned >= count {
                    break;
                }
            }
        }
    }

    pub fn write_positions_sampled(
        &self,
        out: &mut Vec<[f32; 3]>,
        step: usize,
    ) {
        out.clear();

        for i in (0..self.pos_x.len()).step_by(step) {
            if !self.alive[i] {
                continue;
            }

            out.push([
                self.pos_x[i],
                self.pos_y[i],
                self.pos_z[i],
            ]);
        }
    }

    pub fn len(&self) -> usize {
        self.pos_x.len()
    }

    pub fn respawn(&mut self, i: usize, circle: &[f32; 2]) {
        let params = generate_particle(circle);
        self.pos_x[i] = params[0][0];
        self.pos_y[i] = params[0][1];
        self.pos_z[i] = params[0][2];

        self.vel_x[i] = params[1][0];
        self.vel_y[i] = params[1][1];
        self.vel_z[i] = params[1][2];

        self.mass[i] = 1.0;
        self.temp[i] = 1.0;
        self.alive[i] = true;
    }
}

fn generate_particle(circle: &[f32; 2]) -> [[f32; 3]; 2] {
    let mut rng = rand::rng();

    let centre_y = circle[0];
    let radius = circle[1];

    // Generate random angle for particle position within a circle
    let theta = rng.random_range(0.0..2.0 * std::f32::consts::PI);

    // Correct radius distribution
    let r = rng.random::<f32>().sqrt();

    // Particle's initial position in unit circle, then scale it to ellipse
    let mut x = r * theta.cos();
    let mut y = r * theta.sin();

    // Scale to ellipse and translate to circle's center
    x = x * radius;
    y = y * radius;
    let z = 0.5 + y;

    y += centre_y;

    // Determine the range for the emission angle based on horizontal position (x)
    // For a full cone, azimuthal angle should be 0 to 360 degrees
    let angle: f32 = rng.random_range(0.0..360.0);

    // Convert angle to radians
    let angle_rad: f32 = angle.to_radians();

    // Randomly generate the vertical cone angle (polar angle from vertical)
    let vertical_angle: f32 = rng.random_range(-30.0..30.0); // 0° to 30° cone from vertical
    let vertical_angle_rad: f32 = vertical_angle.to_radians();

    // Calculate the velocity components in x, y, and z directions
    let speed = INITIAL_VEL; // The initial speed for the particles

    // Direction vector for the cone:
    // x = sin(θ) * cos(φ)
    // y = -cos(θ)  (negative because y positive is down, so upward is negative y)
    // z = sin(θ) * sin(φ)
    let x_vel = speed * vertical_angle_rad.sin() * angle_rad.cos();
    let y_vel = -speed * vertical_angle_rad.cos(); // Upward direction
    let z_vel = speed * vertical_angle_rad.sin() * angle_rad.sin();

    // We now have the components for x, y, and z velocities. To ensure a normalized direction:
    let total_velocity = (x_vel.powi(2) + y_vel.powi(2) + z_vel.powi(2)).sqrt();
    
    // Normalize velocities (optional, but it ensures consistent speed in all directions)
    let norm_x_vel = x_vel / total_velocity * speed;
    let norm_y_vel = y_vel / total_velocity * speed;
    let norm_z_vel = z_vel / total_velocity * speed;

    // Return the position [x, y, z] and velocity [x_vel, y_vel, z_vel]
    return [[x, y, z], [norm_x_vel, norm_y_vel, norm_z_vel]];
}