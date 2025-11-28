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
    _pad0: u32,              // 36-39
    _pad1: u32,              // 40-43
    _pad2: u32,              // 44-47
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

    // DEBUG: Just output raw IWE value to see if anything is accumulated
    let iwe_val = get_iwe(x, y);

    // Normalize for visualization (IWE values are event counts, can be 0-100+)
    // Use log scale to see low values
    let viz = log2(iwe_val + 1.0) / 8.0;

    textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(viz));
}
