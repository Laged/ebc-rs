// CMax-SLAM: Apply Sobel edge detection to motion-compensated IWE
// Outputs edge magnitude to Sobel texture slot

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

    // Skip border pixels (Sobel needs 3x3 neighborhood)
    if x < 1 || x >= i32(WIDTH) - 1 || y < 1 || y >= i32(HEIGHT) - 1 {
        textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(0.0));
        return;
    }

    // Sample 3x3 neighborhood from IWE
    let p00 = get_iwe(x - 1, y - 1);
    let p10 = get_iwe(x,     y - 1);
    let p20 = get_iwe(x + 1, y - 1);
    let p01 = get_iwe(x - 1, y);
    // p11 = center, not used in Sobel
    let p21 = get_iwe(x + 1, y);
    let p02 = get_iwe(x - 1, y + 1);
    let p12 = get_iwe(x,     y + 1);
    let p22 = get_iwe(x + 1, y + 1);

    // Sobel kernels
    // Gx = [-1 0 1]    Gy = [-1 -2 -1]
    //      [-2 0 2]         [ 0  0  0]
    //      [-1 0 1]         [ 1  2  1]
    let gx = -p00 + p20 - 2.0 * p01 + 2.0 * p21 - p02 + p22;
    let gy = -p00 - 2.0 * p10 - p20 + p02 + 2.0 * p12 + p22;

    // Edge magnitude
    let magnitude = sqrt(gx * gx + gy * gy);

    // Output Sobel edge magnitude
    textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(magnitude));
}
