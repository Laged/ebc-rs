// CMax-SLAM: Output the center IWE slice to Sobel texture
// Applies edge detection (Sobel) to motion-compensated image

struct CmaxSlamParams {
    centroid_x: f32,         // 0-3
    centroid_y: f32,         // 4-7
    t_ref: f32,              // 8-11
    omega: f32,              // 12-15
    delta_omega: f32,        // 16-19
    edge_weight: f32,        // 20-23
    window_start: u32,       // 24-27
    window_end: u32,         // 28-31
    event_count: u32,        // 32-35
    _pad1: vec3<u32>,        // 36-47 - alignment padding
    _padding: vec3<u32>,     // 48-59 - actual vec3 field
    _pad2: u32,              // 60-63 - trailing alignment
}

@group(0) @binding(0) var<storage, read> iwe_buffer: array<u32>;
@group(0) @binding(1) var<uniform> params: CmaxSlamParams;
@group(0) @binding(2) var output_texture: texture_storage_2d<r32float, write>;

const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;

fn get_iwe(x: i32, y: i32) -> f32 {
    if x < 0 || x >= i32(WIDTH) || y < 0 || y >= i32(HEIGHT) {
        return 0.0;
    }
    // Read from center slice (offset 0)
    let idx = u32(y) * WIDTH + u32(x);
    // Convert from bilinear-scaled (256x) back to float
    return f32(iwe_buffer[idx]) / 256.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    if gid.x >= WIDTH || gid.y >= HEIGHT {
        return;
    }

    // Skip border for Sobel
    if x < 1 || x >= i32(WIDTH) - 1 || y < 1 || y >= i32(HEIGHT) - 1 {
        textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(0.0));
        return;
    }

    // Check if center has events
    let center_val = get_iwe(x, y);
    if center_val < 0.5 {
        textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(0.0));
        return;
    }

    // Load 3x3 neighborhood
    let p00 = get_iwe(x - 1, y - 1);
    let p01 = get_iwe(x, y - 1);
    let p02 = get_iwe(x + 1, y - 1);
    let p10 = get_iwe(x - 1, y);
    let p12 = get_iwe(x + 1, y);
    let p20 = get_iwe(x - 1, y + 1);
    let p21 = get_iwe(x, y + 1);
    let p22 = get_iwe(x + 1, y + 1);

    // Sobel kernels
    let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
    let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

    let magnitude = sqrt(gx * gx + gy * gy);

    // Normalize: IWE values can be high, so use log scale
    // Then threshold to binary edge
    let normalized = log2(magnitude + 1.0) / 10.0;
    let edge_val = select(0.0, 1.0, normalized > 0.1);

    textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(edge_val));
}
