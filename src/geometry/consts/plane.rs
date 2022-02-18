#[path = "../object.rs"]
pub mod object;

use object::Vertex;
use object::Object;

pub fn get_plane() -> Object {
    Object {
        vertices: [
            Vertex { position: (-1.0, -1.0, 0.0) },
            Vertex { position: (1.0, -1.0, 0.0) },
            Vertex { position: (1.0, 1.0, 0.0) },
            Vertex { position: (-1.0, 1.0, 0.0) },
        ].to_vec(),
        normals: [
        ].to_vec(),
        indices: [
            0, 1, 2,
            2, 3, 0u16
        ].to_vec(),
        bounding_box_vertices: [
            Vertex { position: (-1.0, -1.0, 0.0) },
            Vertex { position: (1.0, -1.0, 0.0) },
            Vertex { position: (1.0, 1.0, 0.0) },
            Vertex { position: (-1.0, 1.0, 0.0) },
        ].to_vec(),
        bounding_box_indices: [
            0, 1, 2,
            2, 3, 0u16
        ].to_vec(),
        color: [1.0, 1.0, 1.0f32],
        model: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0f32]
        ],
    }
}
