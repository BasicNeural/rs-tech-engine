#[macro_use]
extern crate glium;

#[path = "./geometry/consts/cube.rs"]
mod cube;

#[path = "./geometry/consts/plane.rs"]
pub mod plane;

#[path = "./geometry/object.rs"]
mod object;

use plane::get_plane;

use glium::*;
use glutin::event::VirtualKeyCode;
use rusttype::gpu_cache::Cache;
use rusttype::{point, vector, Font, PositionedGlyph, Rect, Scale};
use std::borrow::Cow;
use object::Vertex;
use object::Object;
use std::io::BufReader;
use std::collections::LinkedList;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use rodio;
use rodio::{Decoder, source::Source};


fn main() {
    #[allow(unused_imports)]
    use glium::{glutin, Surface};

    let font_data = std::include_bytes!("resources/Ubuntu-R.ttf");
    let font = Font::try_from_bytes(font_data as &[u8]).expect("err");

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new().with_inner_size(glium::glutin::dpi::LogicalSize::new(1280, 720));
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24).with_multisampling(4);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let (_stream, stream_handle) = rodio::OutputStream::try_default().unwrap();

    // let file = std::fs::File::open("./src/resources/examples_music.wav").unwrap();
    // let beep1 = stream_handle.play_once(BufReader::new(file)).unwrap();
    
    let mut objects: LinkedList<Object> = load_objects("./src/resources/world1.scene".to_string());
    let mut plane: Object = get_plane();
    objects.push_back(plane);

    let scale = display.gl_window().window().scale_factor();

    let (cache_width, cache_height) = ((512.0 * scale) as u32, (512.0 * scale) as u32);
    let mut cache: Cache<'static> = Cache::builder()
        .dimensions(cache_width, cache_height)
        .build();

    let cache_tex = glium::texture::Texture2d::with_format(
        &display,
        glium::texture::RawImage2d {
            data: Cow::Owned(vec![128u8; cache_width as usize * cache_height as usize]),
            width: cache_width,
            height: cache_height,
            format: glium::texture::ClientFormat::U8,
        },
        glium::texture::UncompressedFloatFormat::U8,
        glium::texture::MipmapsOption::NoMipmap,
    ).expect("err");

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

        out vec4 color;
        uniform vec3 v_color;
        void main() {
            color = vec4(v_color, 1.0);
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
    let speed: f32 = 0.2;
    let rotate_speed: f32 = 0.07;
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
    let mut z: f32 = -2.0;
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
    let mut bounding_box_mode: bool = false;

    let text_program = glium::Program::from_source(&display, "
        #version 140
        in vec2 position;
        in vec2 tex_coords;
        in vec4 colour;
        out vec2 v_tex_coords;
        out vec4 v_colour;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_tex_coords = tex_coords;
            v_colour = colour;
        }
        ", "
        #version 140
        uniform sampler2D tex;
        in vec2 v_tex_coords;
        in vec4 v_colour;
        out vec4 f_colour;
        void main() {
            f_colour = v_colour * vec4(1.0, 1.0, 1.0, texture(tex, v_tex_coords).r);
        }", None).unwrap();

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src,
                                              None).unwrap();
    let axis_program = glium::Program::from_source(&display, vertex_shader_axis, fragment_shader_axis,
                                              None).unwrap();
    let bounding_box_program = glium::Program::from_source(&display, vertex_shader_axis, fragment_shader_bounding_box,
                                              None).unwrap();

    let mut prev_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let start_time = std::time::Instant::now();
        let wait = (start_time - prev_time).as_millis();

        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                glutin::event::WindowEvent::KeyboardInput { input, .. } => {
                    if let glutin::event::ElementState::Pressed = input.state {
                        match input.virtual_keycode {
                            Some(VirtualKeyCode::Q) => *control_flow = glutin::event_loop::ControlFlow::Exit,
                            Some(VirtualKeyCode::W) => w = true,
                            Some(VirtualKeyCode::A) => a = true,
                            Some(VirtualKeyCode::S) => s = true,
                            Some(VirtualKeyCode::D) => d = true,
                            Some(VirtualKeyCode::Right) => right = true,
                            Some(VirtualKeyCode::Left) => left = true,
                            Some(VirtualKeyCode::Up) => up = true,
                            Some(VirtualKeyCode::Down) => down = true,
                            Some(VirtualKeyCode::Space) => space = true,
                            Some(VirtualKeyCode::LShift) => lshift = true,
                            Some(VirtualKeyCode::F) => bounding_box_mode = !bounding_box_mode,
                            Some(VirtualKeyCode::E) => {
                                let beep_file = BufReader::new(std::fs::File::open("./src/resources/test.wav").unwrap());
                                let source = Decoder::new(beep_file).unwrap();
                                let sample = source.convert_samples();
                                stream_handle.play_raw(sample).unwrap();
                            },
                            _ => ()
                        }
                    } else if let glutin::event::ElementState::Released = input.state {
                        match input.virtual_keycode {
                            Some(VirtualKeyCode::W) => w = false,
                            Some(VirtualKeyCode::A) => a = false,
                            Some(VirtualKeyCode::S) => s = false,
                            Some(VirtualKeyCode::D) => d = false,
                            Some(VirtualKeyCode::Right) => right = false,
                            Some(VirtualKeyCode::Left) => left = false,
                            Some(VirtualKeyCode::Up) => up = false,
                            Some(VirtualKeyCode::Down) => down = false,
                            Some(VirtualKeyCode::Space) => space = false,
                            Some(VirtualKeyCode::LShift) => lshift = false,
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
            z += speed * c_angle.cos() * (wait as f32 / 50.0);
            x += speed * c_angle.sin() * (wait as f32 / 50.0);
        }
        if a {
            x -= speed * c_angle.cos() * (wait as f32 / 50.0);
            z += speed * c_angle.sin() * (wait as f32 / 50.0);
        }
        if s {
            z -= speed * c_angle.cos() * (wait as f32 / 50.0);
            x -= speed * c_angle.sin() * (wait as f32 / 50.0);
        }
        if d {
            x += speed * c_angle.cos() * (wait as f32 / 50.0);
            z -= speed * c_angle.sin() * (wait as f32 / 50.0);
        }

        if right {
            c_angle += rotate_speed * (wait as f32 / 50.0);
        }
        if left {
            c_angle -= rotate_speed * (wait as f32 / 50.0);
        }

        if up && a_angle < PI {
            a_angle += 0.02;
        }
        if down && -a_angle < PI {
            a_angle -= 0.02;
        }

        if lshift {
            y -= speed;
        }
        if space {
            y += speed;
        }

        let view = view_matrix(&[x, y, z], &[c_angle.sin(), a_angle, c_angle.cos()], &[0.0, 1.0, 0.0]);

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

        let frustum = mat_mul(&view, &perspective);

        let frustums = [
            [   // left
                frustum[0][3] + frustum[0][0],
                frustum[1][3] + frustum[1][0],
                frustum[2][3] + frustum[2][0],
                frustum[3][3] + frustum[3][0],
            ], 
            [   // right
                frustum[0][3] - frustum[0][0],
                frustum[1][3] - frustum[1][0],
                frustum[2][3] - frustum[2][0],
                frustum[3][3] - frustum[3][0],
            ], 
            [   // near
                frustum[0][3] + frustum[0][2],
                frustum[1][3] + frustum[1][2],
                frustum[2][3] + frustum[2][2],
                frustum[3][3] + frustum[3][2],
            ]
        ].to_vec();

        let light = [-2.0, 2.0, 2.0f32];

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            // backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
            .. Default::default()
        };

        let scale = display.gl_window().window().scale_factor();
        let (width, _): (u32, _) = display
            .gl_window()
            .window()
            .inner_size()
            .into();
        let scale = scale as f32;

        let axis_vertex_buffer = glium::VertexBuffer::new(&display, &axis).unwrap();
        let axis_indices = glium::index::NoIndices(glium::index::PrimitiveType::LinesList);

        target.draw(&axis_vertex_buffer, &axis_indices, &axis_program,
                    &uniform! { model: axis_model, view: view, perspective: perspective },
                    &params).unwrap();

        let mut draw_count = 0;

        for object in objects.iter() {
            if is_draw(&frustums, object) {

                let positions = glium::VertexBuffer::new(&display, &object.vertices).unwrap();
                let normals = glium::VertexBuffer::new(&display, &object.normals).unwrap();
                let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList,
                                                      &object.indices).unwrap();
                let bounding_box_positions = glium::VertexBuffer::new(&display, &object.bounding_box_vertices).unwrap();
                let bounding_box_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::LinesList,
                                                                   &object.bounding_box_indices).unwrap();

                draw_count += 1;
                target.draw((&positions, &normals), &indices, &program,
                    &uniform! { model: object.model, view: view, perspective: perspective, u_light: light, v_color: object.color },
                    &params).unwrap();
                if bounding_box_mode {
                    target.draw(&bounding_box_positions, &bounding_box_indices, &bounding_box_program,
                        &uniform! { model: object.model, view: view, perspective: perspective, u_light: light },
                        &params).unwrap();
                }
            }
        }

        let fps = {
            if wait == 0 {
                0
            } else {
                1000 / wait
            }
        };

        let text = format!("FPS: {}\r\nvisible: {}", fps, draw_count);

        let glyphs = layout_paragraph(&font, Scale::uniform(24.0 * scale), width, &text);
        for glyph in &glyphs {
            cache.queue_glyph(0, glyph.clone());
        }
        cache.cache_queued(|rect, data| {
            cache_tex.main_level().write(
                glium::Rect {
                    left: rect.min.x,
                    bottom: rect.min.y,
                    width: rect.width(),
                    height: rect.height(),
                },
                glium::texture::RawImage2d {
                    data: Cow::Borrowed(data),
                    width: rect.width(),
                    height: rect.height(),
                    format: glium::texture::ClientFormat::U8,
                },
            );
        }).unwrap();

        let uniforms = uniform! {
            tex: cache_tex.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
        };

        let vertex_buffer = {
            #[derive(Copy, Clone)]
            struct Vertex {
                position: [f32; 2],
                tex_coords: [f32; 2],
                colour: [f32; 4],
            }

            implement_vertex!(Vertex, position, tex_coords, colour);
            let colour = [1.0, 1.0, 1.0, 1.0];
            let (screen_width, screen_height) = {
            let (w, h) = display.get_framebuffer_dimensions();
                (w as f32, h as f32)
            };
            let origin = point(0.0, 0.0);
            let vertices: Vec<Vertex> = glyphs
                .iter()
                .filter_map(|g| cache.rect_for(0, g).ok().flatten())
                .flat_map(|(uv_rect, screen_rect)| {
                    let gl_rect = Rect {
                        min: origin
                            + (vector(
                            screen_rect.min.x as f32 / screen_width - 0.5,
                                1.0 - screen_rect.min.y as f32 / screen_height - 0.5,
                            )) * 2.0,
                        max: origin
                            + (vector(
                                screen_rect.max.x as f32 / screen_width - 0.5,
                                1.0 - screen_rect.max.y as f32 / screen_height - 0.5,
                            )) * 2.0,
                    };
                    vec![
                        Vertex {
                            position: [gl_rect.min.x, gl_rect.max.y],
                            tex_coords: [uv_rect.min.x, uv_rect.max.y],
                            colour,
                        },
                        Vertex {
                            position: [gl_rect.min.x, gl_rect.min.y],
                            tex_coords: [uv_rect.min.x, uv_rect.min.y],
                            colour,
                        },
                        Vertex {
                            position: [gl_rect.max.x, gl_rect.min.y],
                            tex_coords: [uv_rect.max.x, uv_rect.min.y],
                            colour,
                        },
                        Vertex {
                            position: [gl_rect.max.x, gl_rect.min.y],
                            tex_coords: [uv_rect.max.x, uv_rect.min.y],
                            colour,
                        },
                        Vertex {
                            position: [gl_rect.max.x, gl_rect.max.y],
                            tex_coords: [uv_rect.max.x, uv_rect.max.y],
                            colour,
                        },
                        Vertex {
                            position: [gl_rect.min.x, gl_rect.max.y],
                            tex_coords: [uv_rect.min.x, uv_rect.max.y],
                            colour,
                        },
                    ]
                })
                .collect();

            glium::VertexBuffer::new(&display, &vertices).unwrap()
        };

        target.draw(
            &vertex_buffer,
            glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
            &text_program,
            &uniforms,
            &glium::DrawParameters {
                blend: glium::Blend::alpha_blending(),
                ..Default::default()
            },
        ).unwrap();
        target.finish().unwrap();

        match *control_flow {
            glutin::event_loop::ControlFlow::Exit => (),
            _ => {
                /*
                 * Grab window handle from the display (untested - based on API)
                 */
                // display.gl_window().window().request_redraw();
                /*
                 * Below logic to attempt hitting TARGET_FPS.
                 * Basically, sleep for the rest of our milliseconds
                 */
                // let elapsed_time = std::time::Instant::now().duration_since(start_time).as_nanos() as u64;
                // let wait_nanos = match 16_666_667 >= elapsed_time {
                //     true => 16_666_667 - elapsed_time,
                //     false => 0
                // };
                
                let new_inst = start_time + std::time::Duration::from_nanos(16_666_667);
                prev_time = start_time;
                *control_flow = glutin::event_loop::ControlFlow::WaitUntil(new_inst);
            }
        }
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

fn layout_paragraph<'a>(
    font: &Font<'a>,
    scale: Scale,
    width: u32,
    text: &str,
) -> Vec<PositionedGlyph<'a>> {
    let mut result = Vec::new();
    let v_metrics = font.v_metrics(scale);
    let advance_height = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap;
    let mut caret = point(0.0, v_metrics.ascent);
    let mut last_glyph_id = None;
    for c in text.chars() {
        if c.is_control() {
            match c {
                '\r' => {
                    caret = point(0.0, caret.y + advance_height);
                }
                '\n' => {}
                _ => {}
            }
            continue;
        }
        let base_glyph = font.glyph(c);
        if let Some(id) = last_glyph_id.take() {
            caret.x += font.pair_kerning(scale, id, base_glyph.id());
        }
        last_glyph_id = Some(base_glyph.id());
        let mut glyph = base_glyph.scaled(scale).positioned(caret);
        if let Some(bb) = glyph.pixel_bounding_box() {
            if bb.max.x > width as i32 {
                caret = point(0.0, caret.y + advance_height);
                glyph.set_position(caret);
                last_glyph_id = None;
            }
        }
        caret.x += glyph.unpositioned().h_metrics().advance_width;
        result.push(glyph);
    }
    result
}

fn is_draw(equations: &Vec<[f32; 4]>, object: &Object) -> bool {
    let mut drawable: bool;
    for vertex in object.bounding_box_vertices.iter() {
        drawable = true;
        for equation in equations.iter() {
            let v = transform_vec(&object.model, &vertex);
                if equation[0] * v.position.0 + equation[1] * v.position.1 + equation[2] * v.position.2 + equation[3] < 0.0 {
                    drawable = false;
                    break;
                }
        }
        if drawable {
            return true;
        }
    }
    return false;
}

fn transform_vec(a: &[[f32; 4]; 4], x: &Vertex) -> Vertex {
    Vertex { position: (
        a[0][0] * x.position.0 + a[1][0] * x.position.1 + a[2][0] * x.position.2 + a[3][0],
        a[0][1] * x.position.0 + a[1][1] * x.position.1 + a[2][1] * x.position.2 + a[3][1],
        a[0][2] * x.position.0 + a[1][2] * x.position.1 + a[2][2] * x.position.2 + a[3][2]
    ) }
}


fn mat_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut c: [[f32; 4]; 4] = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ];

    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    c
}

pub fn load_objects(path: String) -> LinkedList<Object> {
    let mut objects = LinkedList::new();
    if let Ok(lines) = read_lines(path) {
        for line in lines {
            if let Ok(data) = line {
                if data.contains("obj ") {
                    if let Some((src, r, g, b, a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33)) = sscanf::scanf!(data, "obj {} {} {} {}  {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}", String, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
                        let mut obj = object::Object::new(src);
                        obj.color = [r, g, b];
                        obj.model = [
                            [a00, a01, a02, a03],
                            [a10, a11, a12, a13],
                            [a20, a21, a22, a23],
                            [a30, a31, a32, a33]
                        ];
                        objects.push_back(obj);
                    }
                }
            }
        }
    }
    objects
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}