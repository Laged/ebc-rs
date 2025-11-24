@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var gradient_output: texture_storage_2d<r8unorm, write>;

struct EdgeParams {
    threshold: f32,
}

@group(0) @binding(2) var<uniform> params: EdgeParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(surface_texture);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Skip border pixels
    if (coords.x < 1 || coords.y < 1 || coords.x >= i32(dims.x) - 1 || coords.y >= i32(dims.y) - 1) {
        textureStore(gradient_output, coords, vec4<f32>(0.0));
        return;
    }

    // Load 3x3 neighborhood and extract timestamps
    var timestamps: array<f32, 9>;
    var idx = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let pos = coords + vec2<i32>(dx, dy);
            let packed = textureLoad(surface_texture, pos, 0).r;
            // Extract timestamp (ignore polarity bit)
            timestamps[idx] = f32(packed & 0x7FFFFFFFu);
            idx++;
        }
    }

    // Sobel kernels
    // Gx: [-1  0  1]    Gy: [-1 -2 -1]
    //     [-2  0  2]        [ 0  0  0]
    //     [-1  0  1]        [ 1  2  1]

    let gx = -timestamps[0] + timestamps[2]
             - 2.0 * timestamps[3] + 2.0 * timestamps[5]
             - timestamps[6] + timestamps[8];

    let gy = -timestamps[0] - 2.0 * timestamps[1] - timestamps[2]
             + timestamps[6] + 2.0 * timestamps[7] + timestamps[8];

    let magnitude = sqrt(gx * gx + gy * gy);

    // Threshold and write
    let edge_value = select(0.0, 1.0, magnitude > params.threshold);
    textureStore(gradient_output, coords, vec4<f32>(edge_value));
}
