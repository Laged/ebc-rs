@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var canny_output: texture_storage_2d<r32float, write>;

struct CannyParams {
    sigma: f32,           // Not used - kernel is fixed for sigma=1.4
    low_threshold: f32,   // Lower threshold for weak edges
    high_threshold: f32,  // Upper threshold for strong edges
    _padding: f32,        // Alignment padding
}

@group(0) @binding(2) var<uniform> params: CannyParams;

// 5x5 Gaussian kernel (sigma ≈ 1.4)
// Normalized to sum to 1.0 for proper averaging
const GAUSSIAN_KERNEL: array<f32, 25> = array<f32, 25>(
    0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
    0.015625,   0.0625,   0.09375,   0.0625,   0.015625,
    0.0234375,  0.09375,  0.140625,  0.09375,  0.0234375,
    0.015625,   0.0625,   0.09375,   0.0625,   0.015625,
    0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625
);

// Sobel kernels for gradient computation
// Gx: [-1  0  1]    Gy: [-1 -2 -1]
//     [-2  0  2]        [ 0  0  0]
//     [-1  0  1]        [ 1  2  1]

const PI: f32 = 3.14159265359;

// Quantize gradient direction to one of 4 directions: 0°, 45°, 90°, 135°
// Returns 0, 1, 2, or 3
fn quantize_direction(angle: f32) -> u32 {
    // Normalize angle to [0, PI)
    var normalized = angle;
    if (normalized < 0.0) {
        normalized += PI;
    }

    // Map to 4 directions
    let degrees = normalized * 180.0 / PI;

    if (degrees < 22.5 || degrees >= 157.5) {
        return 0u; // 0° (horizontal)
    } else if (degrees >= 22.5 && degrees < 67.5) {
        return 1u; // 45° (diagonal /)
    } else if (degrees >= 67.5 && degrees < 112.5) {
        return 2u; // 90° (vertical)
    } else {
        return 3u; // 135° (diagonal \)
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(surface_texture);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Skip border pixels (need 2-pixel border for 5x5 Gaussian + 1 for Sobel = 3 total)
    if (coords.x < 3 || coords.y < 3 || coords.x >= i32(dims.x) - 3 || coords.y >= i32(dims.y) - 3) {
        textureStore(canny_output, coords, vec4<f32>(0.0));
        return;
    }

    // Step 1: Apply Gaussian blur to 5x5 neighborhood
    // Surface texture packs: (timestamp << 1) | polarity
    var blurred = 0.0;
    var kernel_idx = 0u;

    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let pos = coords + vec2<i32>(dx, dy);
            let packed = textureLoad(surface_texture, pos, 0).r;
            let timestamp = f32(packed >> 1u);

            blurred += timestamp * GAUSSIAN_KERNEL[kernel_idx];
            kernel_idx++;
        }
    }

    // Step 2: Load 3x3 neighborhood around blurred center for Sobel
    var neighborhood: array<f32, 9>;
    var idx = 0u;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let pos = coords + vec2<i32>(dx, dy);
            let packed = textureLoad(surface_texture, pos, 0).r;
            let timestamp = f32(packed >> 1u);
            neighborhood[idx] = timestamp;
            idx++;
        }
    }

    // Step 3: Compute Sobel gradients (Gx, Gy)
    let gx = -neighborhood[0] + neighborhood[2]
             - 2.0 * neighborhood[3] + 2.0 * neighborhood[5]
             - neighborhood[6] + neighborhood[8];

    let gy = -neighborhood[0] - 2.0 * neighborhood[1] - neighborhood[2]
             + neighborhood[6] + 2.0 * neighborhood[7] + neighborhood[8];

    // Step 4: Compute magnitude and direction
    let magnitude = sqrt(gx * gx + gy * gy);
    let direction = atan2(gy, gx);
    let dir_quantized = quantize_direction(direction);

    // Step 5: Non-maximum suppression
    // Compare magnitude with neighbors in gradient direction
    var neighbor1_mag = 0.0;
    var neighbor2_mag = 0.0;

    // Load neighbors based on quantized direction
    if (dir_quantized == 0u) {
        // 0° (horizontal): compare with left and right
        let left_pos = coords + vec2<i32>(-1, 0);
        let right_pos = coords + vec2<i32>(1, 0);

        let left_packed = textureLoad(surface_texture, left_pos, 0).r;
        let right_packed = textureLoad(surface_texture, right_pos, 0).r;

        let left_ts = f32(left_packed >> 1u);
        let right_ts = f32(right_packed >> 1u);

        // Approximate gradient magnitude for neighbors
        neighbor1_mag = abs(left_ts - neighborhood[4]);
        neighbor2_mag = abs(right_ts - neighborhood[4]);
    } else if (dir_quantized == 1u) {
        // 45° (diagonal /): compare with top-right and bottom-left
        let tr_pos = coords + vec2<i32>(1, -1);
        let bl_pos = coords + vec2<i32>(-1, 1);

        let tr_packed = textureLoad(surface_texture, tr_pos, 0).r;
        let bl_packed = textureLoad(surface_texture, bl_pos, 0).r;

        let tr_ts = f32(tr_packed >> 1u);
        let bl_ts = f32(bl_packed >> 1u);

        neighbor1_mag = abs(tr_ts - neighborhood[4]);
        neighbor2_mag = abs(bl_ts - neighborhood[4]);
    } else if (dir_quantized == 2u) {
        // 90° (vertical): compare with top and bottom
        let top_pos = coords + vec2<i32>(0, -1);
        let bottom_pos = coords + vec2<i32>(0, 1);

        let top_packed = textureLoad(surface_texture, top_pos, 0).r;
        let bottom_packed = textureLoad(surface_texture, bottom_pos, 0).r;

        let top_ts = f32(top_packed >> 1u);
        let bottom_ts = f32(bottom_packed >> 1u);

        neighbor1_mag = abs(top_ts - neighborhood[4]);
        neighbor2_mag = abs(bottom_ts - neighborhood[4]);
    } else {
        // 135° (diagonal \): compare with top-left and bottom-right
        let tl_pos = coords + vec2<i32>(-1, -1);
        let br_pos = coords + vec2<i32>(1, 1);

        let tl_packed = textureLoad(surface_texture, tl_pos, 0).r;
        let br_packed = textureLoad(surface_texture, br_pos, 0).r;

        let tl_ts = f32(tl_packed >> 1u);
        let br_ts = f32(br_packed >> 1u);

        neighbor1_mag = abs(tl_ts - neighborhood[4]);
        neighbor2_mag = abs(br_ts - neighborhood[4]);
    }

    // Suppress if not a local maximum
    if (magnitude < neighbor1_mag || magnitude < neighbor2_mag) {
        textureStore(canny_output, coords, vec4<f32>(0.0));
        return;
    }

    // Step 6: Double threshold (simplified hysteresis)
    // Strong edges = 1.0, weak edges = 0.5, non-edges = 0.0
    var edge_value = 0.0;

    if (magnitude >= params.high_threshold) {
        edge_value = 1.0; // Strong edge
    } else if (magnitude >= params.low_threshold) {
        edge_value = 0.5; // Weak edge
    }
    // else: edge_value = 0.0 (non-edge)

    // Step 7: Write output
    textureStore(canny_output, coords, vec4<f32>(edge_value));
}
