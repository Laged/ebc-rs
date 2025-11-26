// Sobel edge detection shader
// Input: Pre-filtered surface texture from preprocess stage
// Output: Binary edge map (1.0 = edge, 0.0 = not edge)

@group(0) @binding(0) var filtered_texture: texture_2d<u32>;
@group(0) @binding(1) var gradient_output: texture_storage_2d<r32float, write>;

struct EdgeParams {
    filter_dead_pixels: u32,
    filter_density: u32,
    filter_temporal: u32,
    min_density_count: u32,
    min_temporal_spread: f32,
    sobel_threshold: f32,
    canny_low_threshold: f32,
    canny_high_threshold: f32,
    log_threshold: f32,
    filter_bidirectional: u32,
    bidirectional_ratio: f32,
    _padding: f32,
}

@group(0) @binding(2) var<uniform> params: EdgeParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(filtered_texture);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Skip border pixels
    if (coords.x < 1 || coords.y < 1 || coords.x >= i32(dims.x) - 1 || coords.y >= i32(dims.y) - 1) {
        textureStore(gradient_output, coords, vec4<f32>(0.0));
        return;
    }

    // Load 3x3 neighborhood - use binary event presence for edge detection
    // Filtered texture packs: (timestamp << 1) | polarity (or 0 if filtered out)
    // For edge detection, we care about WHERE events are, not WHEN exactly
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
        textureStore(gradient_output, coords, vec4<f32>(0.0));
        return;
    }

    // Sobel kernels on binary event presence
    // Detects boundaries between event-active and event-inactive regions
    // Gx: [-1  0  1]    Gy: [-1 -2 -1]
    //     [-2  0  2]        [ 0  0  0]
    //     [-1  0  1]        [ 1  2  1]

    let gx = -has_event[0] + has_event[2]
             - 2.0 * has_event[3] + 2.0 * has_event[5]
             - has_event[6] + has_event[8];

    let gy = -has_event[0] - 2.0 * has_event[1] - has_event[2]
             + has_event[6] + 2.0 * has_event[7] + has_event[8];

    let magnitude = sqrt(gx * gx + gy * gy);

    // With binary event inputs, magnitude range is 0-5.66 (max Sobel response)
    // Threshold is used directly - typical values: 0.5 (weak edges), 2.0 (strong edges)
    // Note: UI may use 0-10000 scale, in which case divide by 1000 to get 0-10 range
    // But hypersearch uses direct values like 0.5, 1.0, 2.0

    // Post-processing: Bidirectional gradient check
    // Requires significant gradient in both X and Y directions
    if (params.filter_bidirectional == 1u) {
        let gx_abs = abs(gx);
        let gy_abs = abs(gy);
        let min_directional = params.sobel_threshold * params.bidirectional_ratio;
        if (gx_abs < min_directional || gy_abs < min_directional) {
            textureStore(gradient_output, coords, vec4<f32>(0.0));
            return;
        }
    }

    // Threshold and write - use threshold directly
    let edge_value = select(0.0, 1.0, magnitude > params.sobel_threshold);
    textureStore(gradient_output, coords, vec4<f32>(edge_value));
}
