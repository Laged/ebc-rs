// Canny edge detection shader
// Input: Pre-filtered surface texture from preprocess stage
// Output: Edge map with strong edges (1.0), weak edges (0.5), non-edges (0.0)

@group(0) @binding(0) var filtered_texture: texture_2d<u32>;
@group(0) @binding(1) var canny_output: texture_storage_2d<r32float, write>;

struct CannyParams {
    sigma: f32,           // Not used - kernel is fixed for sigma=1.4
    low_threshold: f32,   // Lower threshold for weak edges
    high_threshold: f32,  // Upper threshold for strong edges
    _padding: f32,        // Alignment padding
}

@group(0) @binding(2) var<uniform> params: CannyParams;

// 5x5 Gaussian kernel (sigma approx 1.4)
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

// Quantize gradient direction to one of 4 directions: 0 deg, 45 deg, 90 deg, 135 deg
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
        return 0u; // 0 deg (horizontal)
    } else if (degrees >= 22.5 && degrees < 67.5) {
        return 1u; // 45 deg (diagonal /)
    } else if (degrees >= 67.5 && degrees < 112.5) {
        return 2u; // 90 deg (vertical)
    } else {
        return 3u; // 135 deg (diagonal \)
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(filtered_texture);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Skip border pixels (need 2-pixel border for 5x5 Gaussian + 1 for Sobel = 3 total)
    if (coords.x < 3 || coords.y < 3 || coords.x >= i32(dims.x) - 3 || coords.y >= i32(dims.y) - 3) {
        textureStore(canny_output, coords, vec4<f32>(0.0));
        return;
    }

    // Check if center pixel was filtered out by preprocess stage
    let center_packed = textureLoad(filtered_texture, coords, 0).r;
    if (center_packed == 0u) {
        textureStore(canny_output, coords, vec4<f32>(0.0));
        return;
    }

    // Step 1: Load 3x3 neighborhood using binary event presence
    // Filtered texture packs: (timestamp << 1) | polarity (or 0 if filtered out)
    var has_event: array<f32, 9>;
    var idx = 0u;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let pos = coords + vec2<i32>(dx, dy);
            let packed = textureLoad(filtered_texture, pos, 0).r;
            // Binary: 1.0 if event present, 0.0 if not
            has_event[idx] = select(0.0, 1.0, packed > 0u);
            idx++;
        }
    }

    // Check if center pixel has an event
    if (has_event[4] < 0.5) {
        textureStore(canny_output, coords, vec4<f32>(0.0));
        return;
    }

    // Step 2: Compute Sobel gradients (Gx, Gy) on binary event presence
    let gx = -has_event[0] + has_event[2]
             - 2.0 * has_event[3] + 2.0 * has_event[5]
             - has_event[6] + has_event[8];

    let gy = -has_event[0] - 2.0 * has_event[1] - has_event[2]
             + has_event[6] + 2.0 * has_event[7] + has_event[8];

    // Step 4: Compute magnitude and direction
    let magnitude = sqrt(gx * gx + gy * gy);
    let direction = atan2(gy, gx);
    let dir_quantized = quantize_direction(direction);

    // Step 5: Non-maximum suppression using binary event presence
    // Compare current magnitude with neighbors in gradient direction
    var neighbor1_mag = 0.0;
    var neighbor2_mag = 0.0;

    // Use binary event presence for NMS
    if (dir_quantized == 0u) {
        // 0 deg (horizontal): compare with left and right
        neighbor1_mag = abs(has_event[3] - has_event[4]); // left - center
        neighbor2_mag = abs(has_event[5] - has_event[4]); // right - center
    } else if (dir_quantized == 1u) {
        // 45 deg (diagonal /): compare with top-right and bottom-left
        neighbor1_mag = abs(has_event[2] - has_event[4]); // top-right - center
        neighbor2_mag = abs(has_event[6] - has_event[4]); // bottom-left - center
    } else if (dir_quantized == 2u) {
        // 90 deg (vertical): compare with top and bottom
        neighbor1_mag = abs(has_event[1] - has_event[4]); // top - center
        neighbor2_mag = abs(has_event[7] - has_event[4]); // bottom - center
    } else {
        // 135 deg (diagonal \): compare with top-left and bottom-right
        neighbor1_mag = abs(has_event[0] - has_event[4]); // top-left - center
        neighbor2_mag = abs(has_event[8] - has_event[4]); // bottom-right - center
    }

    // Suppress if not a local maximum
    if (magnitude < neighbor1_mag || magnitude < neighbor2_mag) {
        textureStore(canny_output, coords, vec4<f32>(0.0));
        return;
    }

    // Step 6: Double threshold (simplified hysteresis)
    // With binary inputs, magnitude range is 0-5.66
    // Thresholds are used directly - typical values: low=0.5, high=2.0
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
