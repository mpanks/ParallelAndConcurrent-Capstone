//Vertices
#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub temp: f32,
    pub mass: f32
}
impl Vertex{
    pub fn new(pos: [f32; 3], temp: f32, mass: f32) -> Vertex{
        Vertex{
            position: pos,
            temp: temp,
            mass: mass
        }
    }
}

implement_vertex!(Vertex, position, temp, mass);