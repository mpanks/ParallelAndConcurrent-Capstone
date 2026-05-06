#[macro_use]
extern crate glium;
mod vertex;
mod particles;
mod simulation;
mod emitter;
mod bounds;
mod spatial_grid;
mod collision;
mod render_mode;
/*extern crate queues;
mod particle;
mod particle_system;*/

/*use queues::*;
use particle::Particle;
use particle_system::{ParticleSystem, ChannelSystem};*/
use render_mode::RenderMode;
use collision::{ merge, CollisionSnapshot, detect_collisions_snapshot};
use spatial_grid::{SpatialGrid, build_grid};
use bounds::Bounds;
use emitter::Emitter;
use rayon::ThreadPoolBuilder;
use particles::Particles;
use simulation::{physics_step};
use vertex::Vertex;
use glium::{Frame, Program, Surface, glutin::surface::WindowSurface, winit, Display};
use std::sync::{Arc, Mutex, atomic::{AtomicU64, Ordering}};
use crossbeam::channel::{bounded, unbounded};

use crate::simulation::cooling_step;

fn main() {
    let floor_collisions = Arc::new(AtomicU64::new(0));
    const MAX_PARTICLES: i32 = 100000;

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

    let particles = Arc::new(Mutex::new(Particles::new(MAX_PARTICLES, &circle)));

    let (tx, rx) = bounded(2);
    let thread_collision = Arc::clone(&floor_collisions);

    let physics_pool = ThreadPoolBuilder::new()
    .num_threads(2)
    .build()
    .unwrap();
    let cooling_pool = ThreadPoolBuilder::new()
    .num_threads(2)
    .build()
    .unwrap();

    let (collision_job_tx, collision_job_rx) = unbounded::<(Arc<CollisionSnapshot>, usize, usize)>();
    let (collision_result_tx, collision_result_rx) = bounded::<Vec<collision::Collision>>(2);
    let (merge_job_tx, merge_job_rx) = unbounded::<Vec<collision::Collision>>();
    let (merge_done_tx, merge_done_rx) = bounded::<()>(1);

    let mut grid = SpatialGrid::new(&bounds, 0.1);

    for _ in 0..2 {
        let job_rx = collision_job_rx.clone();
        let result_tx = collision_result_tx.clone();

        std::thread::spawn(move || {
            while let Ok((snapshot, start_cell, end_cell)) = job_rx.recv() {
                let collisions = detect_collisions_snapshot(&snapshot, start_cell, end_cell);
                let _ = result_tx.send(collisions);
            }
        });
    //}

    //{
        let merge_particles = Arc::clone(&particles);
        let merge_done_tx = merge_done_tx.clone();
        let merge_job_rx = merge_job_rx.clone();

        std::thread::spawn(move || {
            while let Ok(collisions) = merge_job_rx.recv() {
                let mut particles = merge_particles.lock().unwrap();
                let mut taken = vec![false; particles.len()];
                let mut valid = Vec::new();

                for c in collisions {
                    if !taken[c.a] && !taken[c.b] {
                        taken[c.a] = true;
                        taken[c.b] = true;
                        valid.push(c);
                    }
                }

                for c in valid {
                    merge(c.a, c.b, &mut *particles);
                }

                let _ = merge_done_tx.send(());
            }
        });
    }

    std::thread::spawn(move || {
        let mut last = std::time::Instant::now();

        let mut emitter = Emitter {
            spawn_rate: 1000.0,
            accumulator: 0.0,
        };

        // Build initial grid for the first loop.
        build_grid(&mut grid, &particles.lock().unwrap(), &bounds);

        loop {
            let now = std::time::Instant::now();
            let dt = (now - last).as_secs_f32();
            last = now;

            let mut particles_guard = particles.lock().unwrap();
            //println!("Live: {}", particles_guard.live_particles());
            //println!("Dead: {}", particles_guard.dead_particles());

            emitter.accumulator += emitter.spawn_rate * dt;
            let mut to_spawn = emitter.accumulator.floor() as usize;
            emitter.accumulator -= to_spawn as f32;

            let available = particles_guard.dead_particles() as usize;
            if to_spawn > available {
                to_spawn = available;
            }

            if to_spawn > 0 {
                particles_guard.spawn_some(to_spawn, &circle);
            }
            drop(particles_guard);

            let particles_arc = Arc::clone(&particles);
            physics_pool.install(|| {
                let mut particles_guard = particles_arc.lock().unwrap();
                physics_step(&mut *particles_guard, dt, &circle, Arc::clone(&thread_collision), &bounds, &mut grid)
            });

            let particles_guard = particles.lock().unwrap();
            build_grid(&mut grid, &*particles_guard, &bounds);

            let snapshot = Arc::new(CollisionSnapshot::new(&*particles_guard, &grid));
            drop(particles_guard);

            let total_cells = grid.cells.len();
            let mid_cell = total_cells / 2;
            collision_job_tx.send((Arc::clone(&snapshot), 0, mid_cell)).unwrap();
            collision_job_tx.send((Arc::clone(&snapshot), mid_cell, total_cells)).unwrap();

            let particles_arc = Arc::clone(&particles);
            cooling_pool.install(move || {
                let mut particles_guard = particles_arc.lock().unwrap();
                cooling_step(&mut *particles_guard, dt)
            });

            let mut collisions = collision_result_rx.recv().unwrap();
            collisions.extend(collision_result_rx.recv().unwrap());

            merge_job_tx.send(collisions).unwrap();
            merge_done_rx.recv().unwrap();

            let particles_guard = particles.lock().unwrap();
            let mut buffer: Vec<vertex::Vertex> = Vec::new();
            particles_guard.write_positions_sampled(&mut buffer, 1000);
            //println!("{}", buffer.iter().count());
            tx.send(buffer).ok();
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
        in float temp;
        in float mass;

        uniform mat4 matrix;
        uniform mat4 perspective;
        uniform float particle_size; // in meters (0.1)

        out float vertex_temp;
        out float vertex_mass;

        void main() {
            vec4 world_pos = vec4(position, 1.0);
            vec4 view_pos = matrix * world_pos;
            vec4 clip_pos = perspective * view_pos;

            gl_Position = clip_pos;

            // Perspective correct point size
            float dist = -view_pos.z;
            gl_PointSize = particle_size / dist * 500.0; // scale factor tweak

            vertex_temp = temp;
            vertex_mass = mass;
        }
    "#;

    let particle_fragment_shader_src = r#"
        #version 140

        in float vertex_temp;
        in float vertex_mass;
        out vec4 color;

        uniform int render_mode;

        vec3 temperature_color(float t) {
        return mix(
            vec3(0.0, 0.0, 1.0), // cold
            vec3(1.0, 0.0, 0.0), // hot
            t
            );
        }

        vec3 mass_color(float m) {
            if (m < 1.0) return vec3(0.2, 0.8, 1.0);   // small
            if (m < 3.0) return vec3(0.2, 1.0, 0.2);   // medium
            if (m < 6.0) return vec3(1.0, 1.0, 0.2);   // large
            return vec3(1.0, 0.2, 0.2);                // very large
        }

        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);

            if (dist > 0.5)
                discard;

            vec3 col;

            if (render_mode == 0) {
                col = temperature_color(vertex_temp);
            } else {
                col = mass_color(vertex_mass);
            }

            color = vec4(col, 1.0);
        }
    "#;

    let shower_fragment_shader_src = r#"
        #version 140

        in float vertex_temp;
        in float vertex_mass;
        out vec4 color;

        void main() {
            color = vec4(1.0, 0.0, 0.0, 1.0);
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

    let mut latest_positions: Vec<Vertex> = Vec::new();
    let vertex_buffer = glium::VertexBuffer::empty_dynamic(&display, MAX_PARTICLES as usize).unwrap();
    let mut render_mode = RenderMode::Temperature;
    let collision_clone = Arc::clone(&floor_collisions);
    #[allow(deprecated)]
    let _ = event_loop.run(move |event, window_target| {
        // Handle events here.
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                // Exit
                winit::event::WindowEvent::CloseRequested => {
                    window_target.exit();
                    println!("Total floor collisions: {}", collision_clone.load(Ordering::Relaxed));
                },
                // Resize
                winit::event::WindowEvent::Resized(window_size) => {
                    display.resize(window_size.into());
                },
                // Keyboard input
                winit::event::WindowEvent::KeyboardInput { event, .. } => {
                    use winit::keyboard::{Key};

                    if event.state == winit::event::ElementState::Pressed {
                        match event.logical_key {
                            Key::Character(ref s) if s == "1" => {
                                render_mode = RenderMode::Temperature;
                            }
                            Key::Character(ref s) if s == "2" => {
                                render_mode = RenderMode::Mass;
                            }
                            _ => {}
                        }
                    }
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

                    draw_shower(&bounds, &circle, &mut target, &shower_program, &display, &perspective, &matrix, render_mode as i32);

                    if let Ok(new_data) = rx.try_recv() {
                        latest_positions = new_data;
                    }

                    // update GPU buffer
                    let vertices: Vec<Vertex> = latest_positions
                        .iter()
                        //.step_by(100)
                        .map(|&p| Vertex { position: p.position, temp: p.temp, mass: p.mass })
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
                            render_mode: render_mode as i32
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
        .map(|&p| Vertex { position: p, temp: 1.0, mass: 1.0 })
        .collect();

    glium::VertexBuffer::dynamic(display, &vertices).unwrap()
}

fn draw_shower(bounds: &Bounds, circle_dimensions: &[f32;2], target: &mut Frame, program: &Program, display: &Display<WindowSurface>, perspective: &[[f32;4];4], matrix: &[[f32;4];4], render_mode: i32) {
    //println!("Draw_shower");
    // Specify the position of the box
    let uniforms = uniform! {
        matrix: *matrix,
        perspective: *perspective,
        render_mode: render_mode
    };

    //Draw main shower//
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::LineLoop);

    // Back wall
    let back_wall = vec![
        Vertex { position: [bounds.min_x, bounds.min_y, bounds.max_z], temp: 1.0, mass: 1.0 },
        Vertex { position: [bounds.min_x, bounds.max_y, bounds.max_z], temp: 1.0, mass: 1.0 },
        Vertex { position: [bounds.max_x, bounds.max_y, bounds.max_z], temp: 1.0, mass: 1.0 },
        Vertex { position: [bounds.max_x, bounds.min_y, bounds.max_z], temp: 1.0, mass: 1.0 },
    ];

    let back_vertex_buffer = glium::VertexBuffer::new(display, &back_wall).unwrap();

    target.draw(&back_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    let left_wall = vec![
        Vertex { position: [bounds.min_x, bounds.min_y, bounds.min_z], temp: 1.0, mass: 1.0 }, 
        Vertex { position: [bounds.min_x, bounds.max_y, bounds.min_z], temp: 1.0, mass: 1.0 }, 
        Vertex { position: [bounds.min_x, bounds.max_y, bounds.max_z], temp: 1.0, mass: 1.0 }, 
        Vertex { position: [bounds.min_x, bounds.min_y, bounds.max_z], temp: 1.0, mass: 1.0 }, 
    ];
    let left_vertex_buffer = glium::VertexBuffer::new(display, &left_wall).unwrap();

    target.draw(&left_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Right wall
    let right_wall = vec![
        Vertex { position: [bounds.max_x, bounds.min_y, bounds.min_z], temp: 1.0, mass: 1.0 }, 
        Vertex { position: [bounds.max_x, bounds.max_y, bounds.min_z], temp: 1.0, mass: 1.0 }, 
        Vertex { position: [bounds.max_x, bounds.max_y, bounds.max_z], temp: 1.0, mass: 1.0 }, 
        Vertex { position: [bounds.max_x, bounds.min_y, bounds.max_z], temp: 1.0, mass: 1.0 }, 
    ];
    let right_vertex_buffer = glium::VertexBuffer::new(display, &right_wall).unwrap();

    target.draw(&right_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Floor
    let floor = vec![
        Vertex { position: [bounds.min_x, bounds.min_y, bounds.min_z], temp: 1.0, mass: 1.0 },
        Vertex { position: [bounds.max_x, bounds.min_y, bounds.min_z], temp: 1.0, mass: 1.0 },
    ];
    let floor_vertex_buffer = glium::VertexBuffer::new(display, &floor).unwrap();

    target.draw(&floor_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();

    // Ceiling
    let ceiling = vec![
        Vertex { position: [bounds.min_x, bounds.max_y, bounds.min_z], temp: 1.0, mass: 1.0 },
        Vertex { position: [bounds.max_x, bounds.max_y, bounds.min_z], temp: 1.0, mass: 1.0 },
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
            temp: 1.0,
            mass: 1.0
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