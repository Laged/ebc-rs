// Contrast Maximization: Compute gradient-based contrast for each IWE slice
// Uses sum of squared Sobel gradient magnitudes

struct ContrastParams {
    n_omega: u32,
    width: u32,
    height: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> iwe_buffer: array<u32>;
@group(0) @binding(1) var<uniform> params: ContrastParams;
@group(0) @binding(2) var<storage, read_write> contrast: array<atomic<u32>>;

@group(0) @binding(3) var edge_texture: texture_2d<f32>;

const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;

fn get_iwe(omega_idx: u32, x: i32, y: i32) -> f32 {
    if x < 0 || x >= i32(WIDTH) || y < 0 || y >= i32(HEIGHT) {
        return 0.0;
    }
    let idx = omega_idx * (WIDTH * HEIGHT) + u32(y) * WIDTH + u32(x);
    return f32(iwe_buffer[idx]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let omega_idx = gid.z;

    if gid.x >= WIDTH || gid.y >= HEIGHT || omega_idx >= params.n_omega {
        return;
    }

    // 1. Compute Contrast (Gradient Magnitude Squared)
    // Skip border pixels for Sobel
    var contrast_val = 0.0;
    if x >= 1 && x < i32(WIDTH) - 1 && y >= 1 && y < i32(HEIGHT) - 1 {
        // Load 3x3 neighborhood
        let p00 = get_iwe(omega_idx, x - 1, y - 1);
        let p01 = get_iwe(omega_idx, x, y - 1);
        let p02 = get_iwe(omega_idx, x + 1, y - 1);
        let p10 = get_iwe(omega_idx, x - 1, y);
        let p12 = get_iwe(omega_idx, x + 1, y);
        let p20 = get_iwe(omega_idx, x - 1, y + 1);
        let p21 = get_iwe(omega_idx, x, y + 1);
        let p22 = get_iwe(omega_idx, x + 1, y + 1);

        // Sobel kernels
        let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
        let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

        // Squared magnitude
        contrast_val = gx * gx + gy * gy;
    }
    // 2. Compute Correlation with Edge Map (Edge-Informed)
    // Sample edge texture (Sobel magnitude) with 5x5 Box Blur to handle sparse events
    var edge_sum = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            edge_sum += textureLoad(edge_texture, vec2<i32>(x + dx, y + dy), 0).r;
        }
    }
    let edge_val = edge_sum / 25.0; // Average magnitude

    let iwe_val = get_iwe(omega_idx, x, y);
    
    // Correlation: IWE * Edge
    let correlation_val = iwe_val * edge_val;

    // Combine objectives
    // Total = Contrast + Correlation
    // Weight correlation heavily (50.0) because edge_val is smoothed (smaller)
    let total_val = contrast_val + correlation_val * 50.0;

    // Atomic add to contrast sum
    // Reduced scale to 10.0 to avoid overflow, but kept precision
    let scaled = u32(total_val * 10.0);
    if scaled > 0u {
        atomicAdd(&contrast[omega_idx], scaled);
    }
}
