#[path = "../object.rs"]
mod object;

use object::Vertex;
use object::Normal;
use object::Object;

pub fn get_cube() -> Object {
    Object {
        vertices: [
            Vertex { position: (0.0, 0.0, 0.0) },
            Vertex { position: (1.0, 1.0, -1.0) },
            Vertex { position: (1.0, -1.0, -1.0) },
            Vertex { position: (1.0, 1.0, 1.0) },
            Vertex { position: (1.0, -1.0, 1.0) },
            Vertex { position: (-1.0, 1.0, -1.0) },
            Vertex { position: (-1.0, -1.0, -1.0) },
            Vertex { position: (-1.0, 1.0, 1.0) },
            Vertex { position: (-1.0, -1.0, 1.0) }
        ].to_vec(),
        normals: [
            Normal { normal: (0.0, 0.0, 0.0) },
            Normal { normal: (0.0, 1.0, 0.0) },
            Normal { normal: (0.0, 0.0, 1.0) },
            Normal { normal: (-1.0, 0.0, 0.0) },
            Normal { normal: (0.0, -1.0, 0.0) },
            Normal { normal: (1.0, 0.0, 0.0) },
            Normal { normal: (0.0, 0.0, -1.0) }
        ].to_vec(),
        indices: [
            5, 3, 1, 
            3, 8, 4, 
            7, 6, 8, 
            2, 8, 6, 
            1, 4, 2, 
            5, 2, 6, 
            5, 7, 3, 
            3, 7, 8, 
            7, 5, 6, 
            2, 4, 8, 
            1, 3, 4, 
            5, 1, 2u16
        ].to_vec(),
        bounding_box_vertices: [].to_vec(),
        bounding_box_indices: [].to_vec(),
        color: [1.0, 1.0, 1.0f32],
        model: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0f32]
        ],
    }
}
