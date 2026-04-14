use crate::vertex;
use vertex::Vertex;

pub struct Particle{
    pub position: Vertex,
    pub velocity: [f32; 3],
    pub mass: f32,
    pub temperature: f32
}

impl Particle{
    pub fn new(pos: Vertex, vel: [f32; 3], mass: f32) -> Particle{
        Particle{
            position: pos,
            velocity: vel,
            mass: mass,
            temperature: 1.0
        }
    }
}