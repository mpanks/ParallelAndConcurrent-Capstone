#[macro_use]
extern crate glium;
mod vertex;
mod particles;
mod simulation;
mod emitter;
mod bounds;
/*extern crate queues;
mod particle;
mod particle_system;*/

/*use queues::*;
use particle::Particle;
use particle_system::{ParticleSystem, ChannelSystem};*/
use bounds::Bounds;
use emitter::Emitter;
use rayon::ThreadPoolBuilder;
use particles::Particles;
use simulation::{Simulation, physics_step};
use vertex::Vertex;
use glium::{Frame, Program, Surface, glutin::surface::WindowSurface, winit, Display};
use std::{sync::{Arc, atomic::AtomicU64}};

fn main() {
    let floor_collisions = Arc::new(AtomicU64::new(0));
    const MAX_PARTICLES: i32 = 50000;

    let width = 1.0;   // meters
    let height = 2.0;  // meters
    let depth = 1.0;   // meters

    let back_wall_x = width / 2.0;   // 0.5
    let back_wall_y = height / 2.0;  // 1.0
    let z = depth;                   // 1.0

    let circle_y = 2.0;
    let circle_radius = 0.05;

    let circle = [circle_y, circle_radius];

    let bounds = Bounds {
        min_x: -0.5,
        max_x:  0.5,
        min_y:  0.0,
        max_y:  2.0,
        min_z:  0.0,
        max_z:  1.0,
    };

    let mut particles = Particles::new(MAX_PARTICLES, &circle, false);
    let pool = ThreadPoolBuilder::new()
    .num_threads(2) 
    .build()
    .unwrap();

    let (tx, rx) = crossbeam::channel::bounded(2);

    std::thread::spawn(move || {
        let mut last = std::time::Instant::now();

        let mut emitter = Emitter {
            spawn_rate: 5000.0,
            accumulator: 0.0,
        };

        let mut buffer = Vec::new();

        loop {
            let now = std::time::Instant::now();
            let dt = (now - last).as_secs_f32();
            last = now;

            // Spawn logic
            emitter.accumulator += emitter.spawn_rate * dt;
            let to_spawn = emitter.accumulator.floor() as usize;
            emitter.accumulator -= to_spawn as f32;

            particles.spawn_some(to_spawn, &circle);

            // Simulation
            pool.install(|| {
                physics_step(&mut particles, dt, &circle, Arc::clone(&floor_collisions), &bounds);
            });

            // Send to renderer
            particles.write_positions_sampled(&mut buffer, 1);
            tx.send(buffer.clone()).ok();
        }
    });

    // Spawn particles
    //let channel_system = Arc::new(ChannelSystem::new());
    
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
        uniform mat4 perspective;
        uniform float particle_size; // in meters (0.1)
        uniform float temp;

        out vec3 vertex_color;

        void main() {
            vec4 world_pos = vec4(position, 1.0);
            vec4 view_pos = matrix * world_pos;
            vec4 clip_pos = perspective * view_pos;

            gl_Position = clip_pos;

            // Perspective correct point size
            float dist = -view_pos.z;
            gl_PointSize = particle_size / dist * 500.0; // scale factor tweak

            vertex_color = vec3(temp, 0.0, 1 - temp);
        }
    "#;

    let particle_fragment_shader_src = r#"
        #version 140

        in vec3 vertex_color;
        out vec4 color;

        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);

            if (dist > 0.5)
                discard;

            color = vec4(vertex_color, 1.0);
        }
    "#;

    let shower_fragment_shader_src = r#"
        #version 140

        in vec3 vertex_color;
        out vec4 color;

        void main() {
            color = vec4(vertex_color, 1.0);
        }
    "#;

    let particle_program = glium::Program::from_source(&display, vertex_shader_src, particle_fragment_shader_src, None).unwrap();
    let shower_program = glium::Program::from_source(&display, vertex_shader_src, shower_fragment_shader_src, None).unwrap();
    let matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 3.0, 1.0], // move back & center
    ];

    let mut latest_positions: Vec<[f32; 3]> = Vec::new();
    let vertex_buffer = glium::VertexBuffer::empty_dynamic(&display, MAX_PARTICLES as usize).unwrap();
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

                    let perspective = {
                        let (width, height) = target.get_dimensions();
                        let aspect_ratio = height as f32 / width as f32;

                        let fov: f32 = 3.141592 / 3.0;
                        let zfar = 1024.0;
                        let znear = 0.1;

                        let f = 1.0 / (fov / 2.0).tan();

                        [
                            [f *   aspect_ratio   ,    0.0,              0.0              ,   0.0],
                            [         0.0         ,     f ,              0.0              ,   0.0],
                            [         0.0         ,    0.0,  (zfar+znear)/(zfar-znear)    ,   1.0],
                            [         0.0         ,    0.0, -(2.0*zfar*znear)/(zfar-znear),   0.0],
                        ]
                    };

                    target.clear_color(0.0, 0.0, 0.0, 1.0);

                    //let draw_clone = Arc::clone(&particle_system);
                    //let draw_clone = Arc::clone(&channel_system);

                    draw_shower(&bounds, &circle, &mut target, &shower_program, &display, &perspective, &matrix);

                    //draw_particles(&mut target, &program, &display, &draw_clone, &mut to_draw, &perspective, &matrix);

                    if let Ok(new_data) = rx.try_recv() {
                        latest_positions = new_data;
                    }

                    // update GPU buffer
                    let vertices: Vec<Vertex> = latest_positions
                        .iter()
                        //.step_by(100)
                        .map(|&p| Vertex { position: p })
                        .collect();

                    let count = vertices.len();

                    if count > 0 {
                        vertex_buffer
                            .slice(0..count)
                            .unwrap()
                            .write(&vertices);


                        let params = glium::DrawParameters {
                            point_size: Some(10.0), // fallback
                            ..Default::default()
                        };

                        let uniforms = uniform! {
                            matrix: matrix,
                            perspective: perspective,
                            particle_size: 0.1f32,
                            temp: 1.0f32
                        };

                        target.draw(
                            vertex_buffer.slice(0..count).unwrap(),
                            glium::index::NoIndices(glium::index::PrimitiveType::Points),
                            &particle_program,
                            &uniforms,
                            &params,
                        ).unwrap();
                    }
                    //let sim = sim.lock().unwrap();

                    /*let positions = sim.render.positions_as_vec3();

                    drop(sim); // release lock ASAP

                    draw_particles(&positions);*/

                    // Finish drawing
                    target.finish().unwrap();

                    /*for t in threads{
                        t.join().unwrap();
                    }*/
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
fn build_vertex_buffer(display: &glium::Display<WindowSurface>, positions: &[[f32; 3]])
    -> glium::VertexBuffer<Vertex>
{
    let vertices: Vec<Vertex> = positions
        .iter()
        .map(|&p| Vertex { position: p })
        .collect();

    glium::VertexBuffer::dynamic(display, &vertices).unwrap()
}

fn draw_shower(bounds: &Bounds, circle_dimensions: &[f32;2], target: &mut Frame, program: &Program, display: &Display<WindowSurface>, perspective: &[[f32;4];4], matrix: &[[f32;4];4]) {
    //println!("Draw_shower");
    // Specify the position of the box
    let uniforms = uniform! {
        matrix: *matrix,
        perspective: *perspective,
        temp: 1.0 as f32
    };

    //Draw main shower//
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::LineLoop);

    // Back wall
    let back_wall = vec![
        Vertex { position: [bounds.min_x, bounds.min_y, bounds.max_z] },
        Vertex { position: [bounds.min_x, bounds.max_y, bounds.max_z]  },
        Vertex { position: [bounds.max_x, bounds.max_y, bounds.max_z] },
        Vertex { position: [bounds.max_x, bounds.min_y, bounds.max_z] },
    ];

    let back_vertex_buffer = glium::VertexBuffer::new(display, &back_wall).unwrap();

    target.draw(&back_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    let left_wall = vec![
        Vertex { position: [bounds.min_x, bounds.min_y, bounds.min_z] }, // bottom front
        Vertex { position: [bounds.min_x, bounds.max_y, bounds.min_z]  }, // top front
        Vertex { position: [bounds.min_x, bounds.max_y, bounds.max_z] }, // top back
        Vertex { position: [bounds.min_x, bounds.min_y, bounds.max_z] }, // bottom back
    ];
    let left_vertex_buffer = glium::VertexBuffer::new(display, &left_wall).unwrap();

    target.draw(&left_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Right wall
    let right_wall = vec![
        Vertex { position: [bounds.max_x, bounds.min_y, bounds.min_z] }, // bottom front
        Vertex { position: [bounds.max_x, bounds.max_y, bounds.min_z] }, // top front
        Vertex { position: [bounds.max_x, bounds.max_y, bounds.max_z] }, // top back
        Vertex { position: [bounds.max_x, bounds.min_y, bounds.max_z] }, // bottom back
    ];
    let right_vertex_buffer = glium::VertexBuffer::new(display, &right_wall).unwrap();

    target.draw(&right_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Floor
    let floor = vec![
        Vertex { position: [bounds.min_x, bounds.min_y, bounds.min_z] },
        Vertex { position: [bounds.max_x, bounds.min_y, bounds.min_z] },
    ];
    let floor_vertex_buffer = glium::VertexBuffer::new(display, &floor).unwrap();

    target.draw(&floor_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Ceiling
    let ceiling = vec![
        Vertex { position: [bounds.min_x, bounds.max_y, bounds.min_z] },
        Vertex { position: [bounds.max_x, bounds.max_y, bounds.min_z] },
    ];
    let ceiling_vertex_buffer = glium::VertexBuffer::new(display, &ceiling).unwrap();

    target.draw(&ceiling_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Draw circle -> shower head
    let num_segments = 15;
    let centre_y = circle_dimensions[0];
    let radius = circle_dimensions[1];

    let mut circle = Vec::new();

    for i in 0..num_segments {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / num_segments as f32;

        let x = radius * theta.cos();
        let z = 0.5 + radius * theta.sin();

        circle.push(Vertex {
            position: [x, centre_y, z],
        });
    }
    let circle_vertex_buffer = glium::VertexBuffer::new(display, &circle).unwrap();

    target.draw(&circle_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();
}
/*
fn draw_particles(target: &mut Frame, program: &Program, display: &Display<WindowSurface>, particle_system: &Arc<ChannelSystem>, to_draw: &mut Queue<particle::Particle>, perspective: &[[f32;4];4], matrix: &[[f32; 4]; 4]){
    
    let radius = 0.0025;

    let indices = glium::index::NoIndices(glium::index::PrimitiveType::LineLoop);
            
    let temp:Vec<particle::Particle> = particle_system.draw_channel.1.try_iter().collect();
    if to_draw.size() >= 900{
        //println!("{}: {}", to_draw.size(), temp.iter().count());
        for _ in 0..temp.iter().count(){
            to_draw.remove().unwrap();
        }
    }
    //println!("draw_particles");
    for particle in temp{
        to_draw.add(particle).unwrap();
    }
    //println!("{}", to_draw.size());
    //if receive.is_err() {return;}
    
    for _ in 0..to_draw.size(){
        //println!("particle");
        let particle = to_draw.remove().unwrap();
        let particle_position = particle.position;
        
        let mut circle= Vec::new();
        let num_segments = 10;
        for i in 0..num_segments {
            let theta = 2.0 * std::f32::consts::PI * (i as f32) / (num_segments as f32);
            let x = particle_position[0] + radius * theta.cos();
            let y = particle_position[1] + (radius * theta.sin());
            let vertex = Vertex { position: [x, y, particle_position[2]] };
            circle.push(vertex);
        }

        let uniforms = uniform! {
            matrix: *matrix,
            perspective: *perspective,
            temp: particle.temperature
        };

        let circle_vertex_buffer = glium::VertexBuffer::new(display, &circle).unwrap();

        target.draw(&circle_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

        to_draw.add(particle).unwrap();
    }
     //   }
   // }
}*/