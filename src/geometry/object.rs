use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: (f32, f32, f32)
}

#[derive(Copy, Clone, Debug)]
pub struct Normal {
    pub normal: (f32, f32, f32)
}

implement_vertex!(Vertex, position);
implement_vertex!(Normal, normal);

pub struct Object {
    pub vertices: Vec<Vertex>,
    pub normals: Vec<Normal>,
    pub indices: Vec<u16>,
    pub bounding_box_vertices: Vec<Vertex>,
    pub bounding_box_indices: Vec<u16>,
    pub model: [[f32; 4]; 4]
}

pub fn load_obj(path: String) -> Object {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut min_x: f32 = 0.0;
    let mut min_y: f32 = 0.0;
    let mut min_z: f32 = 0.0;
    let mut max_x: f32 = 0.0;
    let mut max_y: f32 = 0.0;
    let mut max_z: f32 = 0.0;
    vertices.push(Vertex { position: (0.0, 0.0, 0.0) });
    normals.push(Normal { normal: (0.0, 0.0, 0.0) });
    if let Ok(lines) = read_lines(path) {
        for line in lines {
            if let Ok(data) = line {
                if data.contains("v ") {
                    if let Some((x, y, z)) = sscanf::scanf!(data, "v {} {} {}", f32, f32, f32) {
                        vertices.push(Vertex { position: (x, y, z) });
                        if min_x > x {
                            min_x = x;
                        } else if max_x < x {
                            max_x = x;
                        }
                        if min_y > y {
                            min_y = y;
                        } else if max_y < y {
                            max_y = y;
                        }
                        if min_z > z {
                            min_z = z;
                        } else if max_z < z {
                            max_z = z;
                        }
                        
                    }
                } else if data.contains("vn ") {
                    if let Some((x, y, z)) = sscanf::scanf!(data, "vn {} {} {}", f32, f32, f32) {
                        normals.push(Normal { normal: (x, y, z) });
                    }
                } else if data.contains("f ") {
                    if let Some((a, _, _, b, _, _, c, _, _)) = sscanf::scanf!(data, "f {}/{}/{} {}/{}/{} {}/{}/{}", u16, u16, u16, u16, u16, u16, u16, u16, u16) {
                        indices.push(a);
                        indices.push(b);
                        indices.push(c);
                    } else if let Some((a, _, b, _, c, _)) = sscanf::scanf!(data, "f {}//{} {}//{} {}//{}", u16, u16, u16, u16, u16, u16) {
                        indices.push(a);
                        indices.push(b);
                        indices.push(c);
                    } else if let Some((a, b, c)) = sscanf::scanf!(data, "f {} {} {}", u16, u16, u16) {
                        indices.push(a);
                        indices.push(b);
                        indices.push(c);
                    }
                }
            }
        }
    }
    Object {
        vertices: vertices,
        normals: normals,
        indices: indices,
        bounding_box_vertices: [
            Vertex { position: (min_x, min_y, min_z) },
            Vertex { position: (min_x, min_y, max_z) },
            Vertex { position: (min_x, max_y, min_z) },
            Vertex { position: (min_x, max_y, max_z) },
            Vertex { position: (max_x, min_y, min_z) },
            Vertex { position: (max_x, min_y, max_z) },
            Vertex { position: (max_x, max_y, min_z) },
            Vertex { position: (max_x, max_y, max_z) },
        ].to_vec(),
        bounding_box_indices: [
            0, 1,
            1, 3,
            3, 2,
            2, 0,
            4, 5,
            5, 7,
            7, 6,
            6, 4,
            0, 4,
            1, 5,
            3, 7,
            2, 6
        ].to_vec(),
        model:  [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0f32]
        ]
    }
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}