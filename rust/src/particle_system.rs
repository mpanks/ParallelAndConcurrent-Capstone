use crate::{particle, vertex::Vertex};
use particle::Particle;
use rand::{RngExt, rng};

pub struct ParticleSystem{
    pub particles: Vec<Particle>
}

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
            let x = rng.random_range(-circle[1]..circle[1]);
            let y = rng.random_range(min_y..max_y);
            //TODO: Figure that range out
            let z: f32 = 0.0;

            //TODO: Random velocity and mass/size
            temp.push(Particle::new(Vertex::new([x, y, z]), [0.0, 0.0, 0.0], 1.0));
        }
        self.particles = temp;
        println!("Spawned");
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