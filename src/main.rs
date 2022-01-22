#[macro_use]
extern crate glium;

#[path = "./geometry/consts/cube.rs"]
mod cube;

#[path = "./geometry/object.rs"]
mod object;

use object::Vertex;
use object::Object;
use std::collections::LinkedList;


fn main() {
    #[allow(unused_imports)]
    use glium::{glutin, Surface};

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24).with_multisampling(8);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();
    
    let mut cube = object::load_obj("./src/resources/cube.obj".to_string());
    let mut teapot = object::load_obj("./src/resources/teapot.obj".to_string());
    let mut sphere = object::load_obj("./src/resources/sphere.obj".to_string());

    cube.model = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [5.0, 1.0, 0.0, 1.0f32]
    ];

    sphere.model = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-5.0, 1.0, 0.0, 1.0f32]
    ];

    let mut objects: LinkedList<Object> = LinkedList::new();

    objects.push_back(teapot);
    objects.push_back(cube);
    objects.push_back(sphere);

    let vertex_shader_src = r#"
        #version 150
        in vec3 position;
        in vec3 normal;
        out vec3 v_normal;
        uniform mat4 perspective;
        uniform mat4 view;
        uniform mat4 model;
        void main() {
            mat4 modelview = view * model;
            v_normal = transpose(inverse(mat3(modelview))) * normal;
            gl_Position = perspective * modelview * vec4(position, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 150
        in vec3 v_normal;
        out vec4 color;
        uniform vec3 u_light;
        void main() {
            float brightness = dot(normalize(v_normal), normalize(u_light));
            vec3 dark_color = vec3(0.6, 0.0, 0.0);
            vec3 regular_color = vec3(1.0, 0.0, 0.0);
            color = vec4(mix(dark_color, regular_color, brightness), 1.0);
        }
    "#;

    let vertex_shader_axis = r#"
        #version 150
        in vec3 position;
        uniform mat4 perspective;
        uniform mat4 view;
        uniform mat4 model;
        void main() {
            mat4 modelview = view * model;
            gl_Position = perspective * modelview * vec4(position, 1.0);
        }
    "#;

    let fragment_shader_axis = r#"
        #version 140
        out vec4 color;
        void main() {
            color = vec4(0.5, 0.5, 0.5, 1.0);
        }
    "#;

    let fragment_shader_bounding_box = r#"
        #version 140
        out vec4 color;
        void main() {
            color = vec4(0.0, 1.0, 0.0, 1.0);
        }
    "#;
    let speed: f32 = 0.04;
    let rotate_speed: f32 = 0.05;
    const PI: f32 = std::f32::consts::PI;

    let axis = vec![
        Vertex { position: ( -10.0, 0.0, 0.0) },
        Vertex { position: ( 10.0, 0.0, 0.0) },
        Vertex { position: ( 0.0, -10.0, 0.0) },
        Vertex { position: ( 0.0, 10.0, 0.0) },
        Vertex { position: ( 0.0, 0.0, -10.0) },
        Vertex { position: ( 0.0, 0.0, 10.0) }
    ];

    let axis_model = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0f32]
    ];


    let mut x: f32 = 0.0;
    let mut y: f32 = 1.0;
    let mut z: f32 = -5.0;
    let mut c_angle: f32 = 0.0;
    let mut a_angle: f32 = 0.0;
    let mut w: bool = false;
    let mut a: bool = false;
    let mut s: bool = false;
    let mut d: bool = false;
    let mut right: bool = false;
    let mut left: bool = false;
    let mut up: bool = false;
    let mut down: bool = false;
    let mut lshift: bool = false;
    let mut space: bool = false;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src,
                                              None).unwrap();
    let axis_program = glium::Program::from_source(&display, vertex_shader_axis, fragment_shader_axis,
                                              None).unwrap();
    let bounding_box_program = glium::Program::from_source(&display, vertex_shader_axis, fragment_shader_bounding_box,
                                              None).unwrap();

    event_loop.run(move |event, _, control_flow| {
        let next_frame_time = std::time::Instant::now() +
            std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                glutin::event::WindowEvent::KeyboardInput { input, .. } => {
                    if let glutin::event::ElementState::Pressed = input.state {
                        match input.virtual_keycode {
                            Some(glutin::event::VirtualKeyCode::Q) => *control_flow = glutin::event_loop::ControlFlow::Exit,
                            Some(glutin::event::VirtualKeyCode::W) => w = true,
                            Some(glutin::event::VirtualKeyCode::A) => a = true,
                            Some(glutin::event::VirtualKeyCode::S) => s = true,
                            Some(glutin::event::VirtualKeyCode::D) => d = true,
                            Some(glutin::event::VirtualKeyCode::Right) => right = true,
                            Some(glutin::event::VirtualKeyCode::Left) => left = true,
                            Some(glutin::event::VirtualKeyCode::Up) => up = true,
                            Some(glutin::event::VirtualKeyCode::Down) => down = true,
                            Some(glutin::event::VirtualKeyCode::Space) => space = true,
                            Some(glutin::event::VirtualKeyCode::LShift) => lshift = true,
                            _ => ()
                        }
                    } else if let glutin::event::ElementState::Released = input.state {
                        match input.virtual_keycode {
                            Some(glutin::event::VirtualKeyCode::W) => w = false,
                            Some(glutin::event::VirtualKeyCode::A) => a = false,
                            Some(glutin::event::VirtualKeyCode::S) => s = false,
                            Some(glutin::event::VirtualKeyCode::D) => d = false,
                            Some(glutin::event::VirtualKeyCode::Right) => right = false,
                            Some(glutin::event::VirtualKeyCode::Left) => left = false,
                            Some(glutin::event::VirtualKeyCode::Up) => up = false,
                            Some(glutin::event::VirtualKeyCode::Down) => down = false,
                            Some(glutin::event::VirtualKeyCode::Space) => space = false,
                            Some(glutin::event::VirtualKeyCode::LShift) => lshift = false,
                            _ => ()
                        }
                    }
                },
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        if w {
            z += speed * c_angle.cos();
            x += speed * c_angle.sin();
        }
        if a {
            x -= speed * c_angle.cos();
            z += speed * c_angle.sin();
        }
        if s {
            z -= speed * c_angle.cos();
            x -= speed * c_angle.sin();
        }
        if d {
            x += speed * c_angle.cos();
            z -= speed * c_angle.sin();
        }

        if right {
            c_angle += rotate_speed;
        }
        if left {
            c_angle -= rotate_speed;
        }

        if up && a_angle < PI / 2.0 {
            a_angle += 0.02;
        }
        if down && -a_angle < PI / 2.0 {
            a_angle -= 0.02;
        }

        if lshift {
            y -= speed;
        }
        if space {
            y += speed;
        }

        let view = view_matrix(&[x, y, z], &[c_angle.sin(), a_angle.sin(), c_angle.cos()], &[0.0, 1.0, 0.0]);

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

        let light = [-2.0, 2.0, 2.0f32];

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            //backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
            .. Default::default()
        };

        let axis_vertex_buffer = glium::VertexBuffer::new(&display, &axis).unwrap();
        let axis_indices = glium::index::NoIndices(glium::index::PrimitiveType::LinesList);

        target.draw(&axis_vertex_buffer, &axis_indices, &axis_program,
                    &uniform! { model: axis_model, view: view, perspective: perspective, u_light: light },
                    &params).unwrap();

        for object in objects.iter() {
            let positions = glium::VertexBuffer::new(&display, &object.vertices).unwrap();
            let normals = glium::VertexBuffer::new(&display, &object.normals).unwrap();
            let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList,
                                          &object.indices).unwrap();
            let bounding_box_positions = glium::VertexBuffer::new(&display, &object.bounding_box_vertices).unwrap();
            let bounding_box_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::LinesList,
                &object.bounding_box_indices).unwrap();
            target.draw((&positions, &normals), &indices, &program,
                    &uniform! { model: object.model, view: view, perspective: perspective, u_light: light },
                    &params).unwrap();
            target.draw(&bounding_box_positions, &bounding_box_indices, &bounding_box_program,
                    &uniform! { model: object.model, view: view, perspective: perspective, u_light: light },
                    &params).unwrap();
        }
        target.finish().unwrap();
    });
}


fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let f = direction;
        let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let len = len.sqrt();
        [f[0] / len, f[1] / len, f[2] / len]
    };

    let s = [up[1] * f[2] - up[2] * f[1],
             up[2] * f[0] - up[0] * f[2],
             up[0] * f[1] - up[1] * f[0]];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [f[1] * s_norm[2] - f[2] * s_norm[1],
             f[2] * s_norm[0] - f[0] * s_norm[2],
             f[0] * s_norm[1] - f[1] * s_norm[0]];

    let p = [-position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
             -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
             -position[0] * f[0] - position[1] * f[1] - position[2] * f[2]];

    [
        [s_norm[0], u[0], f[0], 0.0],
        [s_norm[1], u[1], f[1], 0.0],
        [s_norm[2], u[2], f[2], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}