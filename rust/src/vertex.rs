//Vertices
#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
}
impl Vertex{
    pub fn new(pos: [f32; 3]) -> Vertex{
        Vertex{
            position: pos
        }
    }
}

implement_vertex!(Vertex, position);
