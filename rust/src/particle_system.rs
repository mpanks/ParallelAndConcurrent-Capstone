use crate::{particle, vertex::Vertex};
use particle::Particle;
use rand::{RngExt};

pub struct ParticleSystem{
    pub particles: Vec<Particle>
}
const INITIAL_VEL: f32 = 5.0;
impl ParticleSystem{
    pub fn new() -> ParticleSystem{
        return ParticleSystem{
            particles: vec!()
        }
    }
    pub fn spawn(&mut self, particle_count: i32, circle: [f32; 3]){
        // Circle: [centre-y, width, height] - oblong for perspective
        let mut temp: Vec<Particle> = vec!();
        let mut rng = rand::rng();
        let max_y = circle[0] + circle[2];
        let min_y = circle[0] - circle[2];

        for _ in 0..particle_count{
            //TODO: ensure particle within circle, this works for squares
            let x = rng.random_range(-circle[1]..circle[1]);
            let y = rng.random_range(min_y..max_y);
            //TODO: Figure that range out
            let z: f32 = 0.0;

            // Find emition angle
            let range;

            if x >= 0.0 {
                range = -30.0..=0.0;
            }else{
                range = 0.0..=30.0;
            }

            let angle = rng.random_range(range);
            // Calculate velocity for x and y
            let x_vel;
            let y_vel;
            if angle == 0.0{
                x_vel = 0.0;
                y_vel = INITIAL_VEL;
            }
            else{
                let vel_angle = 90.0 / angle;
                x_vel = (1.0 / vel_angle) * INITIAL_VEL;
                y_vel = INITIAL_VEL - x_vel.abs();
            }
            // 30 degrees is 1:2 ratio of x:y velocity

            //TODO: Random velocity and mass/size
            temp.push(Particle::new(Vertex::new([x, y, z]), [x_vel, y_vel, 0.0], 1.0));
            println!("{}, {}, vel: {}, {}, angle: {}", x, y, x_vel, y_vel, angle);
        }
        self.particles = temp;
    }
    pub fn test(&self, particle_count: i32){
        let mut count = 0;
        for particle in 0..particle_count as usize{
            let position = &self.particles[particle].position.position;
            println!("{}: {}, {}, {}", count, position[0], position[1], position[2]);
            count += 1;
        }
    }
}