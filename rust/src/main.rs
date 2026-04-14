#[macro_use]
extern crate glium;
mod vertex;
mod particle;
mod particle_system;
// Use the re-exported winit dependency to avoid version mismatches.
// Requires the `simple_window_builder` feature.
use vertex::Vertex;
use particle_system::ParticleSystem;
use glium::{Frame, Program, Surface, glutin::surface::WindowSurface, winit, Display};

fn main() {
    const PARTICLE_COUNT: i32 = 5;
    let back_wall_x = 0.3;
    let back_wall_y = 0.6;
    let z = -1.0;

    let outer_x = 0.8;
    let outer_y = 0.9;
    let circle_y = back_wall_y + (0.5 * (outer_y - back_wall_y));
    let circle_height = 0.05;
    let circle_width = 0.09;

    let dimensions = [back_wall_x, back_wall_y, z, outer_x, outer_y, circle_y, circle_height, circle_width];

    // Spawn particles
    let mut particle_system = ParticleSystem::new();
    particle_system.spawn(PARTICLE_COUNT, [circle_y, circle_width, circle_height]);
    //particle_system.test(PARTICLE_COUNT);
    
    // Define window and OpenGL stuff

    // The **winit::EventLoop** for handling events.
    let event_loop = winit::event_loop::EventLoop::builder().build().unwrap();
    // Create a glutin context and glium Display
    let (window, display) = glium::backend::glutin::SimpleWindowBuilder::new().with_title("Pain").build(&event_loop);

    //Shader stuff
    let vertex_shader_src = r#"
        #version 140

        in vec3 position;

        uniform mat4 matrix;

        void main() {
            gl_Position = matrix * vec4(position, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 140

        out vec4 color;

        void main() {
            color = vec4(1.0, 0.0, 0.0, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    #[allow(deprecated)]
    let _ = event_loop.run(move |event, window_target| {
        // Handle events here.
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                // Exit
                winit::event::WindowEvent::CloseRequested => window_target.exit(),
                // Resize
                winit::event::WindowEvent::Resized(window_size) => {
                    display.resize(window_size.into());
                },
                // Render loop
                winit::event::WindowEvent::RedrawRequested => {
                    //Wait for next frame
                    let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
                    winit::event_loop::ControlFlow::WaitUntil(next_frame_time);

                    //Draw window
                    let mut target = display.draw();

                    target.clear_color(0.0, 0.0, 0.0, 1.0);

                    draw_shower(&dimensions, &mut target, &program, &display);

                    draw_particles(&mut target, &program, &display, &particle_system);

                    // Finish drawing
                    target.finish().unwrap();
                },

                // Default, goes at end
                _ => (),
            },
            // Call render loop
            winit::event::Event::AboutToWait => {
                window.request_redraw();
            },
            // Default
            _ => (),
        }
    });
} 

fn draw_shower(dimensions: &[f32; 8], target: &mut Frame, program: &Program, display: &Display<WindowSurface>) {
    // Specify the position of the box
    // Create a 4x4 matrix to store the position and orientation of the objects
    let uniforms = uniform! {
        matrix: [
            [1.0 as f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    };

    //Draw main shower//
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::LineLoop);

    //Shape stuff
    let back_wall_x = dimensions[0];
    let back_wall_y = dimensions[1];
    let z = dimensions[2];

    let outer_x = dimensions[3];
    let outer_y = dimensions[4];

    // Back wall
    let back_vertex1 = Vertex { position: [ -back_wall_x, -back_wall_y, z] };
    let back_vertex2 = Vertex { position: [ -back_wall_x,  back_wall_y, z] };
    let back_vertex3 = Vertex { position: [ back_wall_x, back_wall_y, z] };
    let back_vertex4 = Vertex { position: [ back_wall_x, -back_wall_y, z] };

    let back_wall = vec![back_vertex1, back_vertex2, back_vertex3, back_vertex4];

    let back_vertex_buffer = glium::VertexBuffer::new(display, &back_wall).unwrap();

    target.draw(&back_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Left wall
    let left_vertex1 = Vertex { position: [-outer_x, -outer_y, z + 1.0] };
    let left_vertex2 = Vertex { position: [ -outer_x,  outer_y, z + 1.0] };
    let left_vertex3 = Vertex { position: [ -back_wall_x, back_wall_y, z] };
    let left_vertex4 = Vertex { position: [ -back_wall_x, -back_wall_y, z] };

    let left_wall = vec![left_vertex1, left_vertex2, left_vertex3, left_vertex4];
    let left_vertex_buffer = glium::VertexBuffer::new(display, &left_wall).unwrap();

    target.draw(&left_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Right wall
    let right_vertex1 = Vertex { position: [outer_x, outer_y, z] };
    let right_vertex2 = Vertex { position: [outer_x, -outer_y, z] };
    let right_vertex3 = Vertex { position: [back_wall_x, -back_wall_y, z] };
    let right_vertex4 = Vertex { position: [back_wall_x, back_wall_y, z] };

    let right_wall = vec![right_vertex1, right_vertex2, right_vertex3, right_vertex4];
    let right_vertex_buffer = glium::VertexBuffer::new(display, &right_wall).unwrap();

    target.draw(&right_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Floor
    let floor_vertex1 = Vertex { position: [-outer_x, -outer_y, z] };
    let floor_vertex2 = Vertex { position: [outer_x, -outer_y, z] };
    
    let floor = vec![floor_vertex1, floor_vertex2];
    let floor_vertex_buffer = glium::VertexBuffer::new(display, &floor).unwrap();

    target.draw(&floor_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Ceiling
    let ceiling_vertex1 = Vertex { position: [-outer_x, outer_y, z] };
    let ceiling_vertex2 = Vertex { position: [outer_x, outer_y, z] };

    let ceiling = vec![ceiling_vertex1, ceiling_vertex2];
    let ceiling_vertex_buffer = glium::VertexBuffer::new(display, &ceiling).unwrap();

    target.draw(&ceiling_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Draw circle -> shower head
    let num_segments = 15;
    let circle_y = dimensions[5];
    let circle_height = dimensions[6];
    let circle_width = dimensions[7];

    let mut circle= Vec::new();
    for i in 0..num_segments {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / (num_segments as f32);
        let x = circle_width * theta.cos();
        let y = circle_y - (circle_height * theta.sin());
        let vertex = Vertex { position: [x, y, 1.0] };
        circle.push(vertex);
    }
    let circle_vertex_buffer = glium::VertexBuffer::new(display, &circle).unwrap();

    target.draw(&circle_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();
}

fn draw_particles(target: &mut Frame, program: &Program, display: &Display<WindowSurface>, particle_system: &ParticleSystem){
    let radius = 0.0025;
    let y_width = 0.01;
    let uniforms = uniform! {
        matrix: [
            [1.0 as f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    };

    let indices = glium::index::NoIndices(glium::index::PrimitiveType::LineLoop);

    for particle in &particle_system.particles{
        let particle_position = particle.position.position;
        
        let mut circle= Vec::new();
        let num_segments = 10;
        for i in 0..num_segments {
            let theta = 2.0 * std::f32::consts::PI * (i as f32) / (num_segments as f32);
            let x = particle_position[0] + radius * theta.cos();
            let y = particle_position[1] + (radius * theta.sin());
            let vertex = Vertex { position: [x, y, 1.0] };
            circle.push(vertex);
        }
        let circle_vertex_buffer = glium::VertexBuffer::new(display, &circle).unwrap();

        target.draw(&circle_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();
    }
}