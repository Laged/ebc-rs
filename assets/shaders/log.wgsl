// Laplacian of Gaussian (LoG) edge detection shader
// Input: Pre-filtered surface texture from preprocess stage
// Output: Binary edge map (1.0 = edge, 0.0 = not edge)

@group(0) @binding(0) var filtered_texture: texture_2d<u32>;
@group(0) @binding(1) var log_output: texture_storage_2d<r32float, write>;

struct LogParams {
    sigma: f32,          // Not used - kernel is fixed for sigma=1.4
    threshold: f32,      // Threshold for edge magnitude
}

@group(0) @binding(2) var<uniform> params: LogParams;

// 5x5 Laplacian of Gaussian kernel (approximation for sigma approx 1.4)
// This combines Gaussian smoothing and Laplacian second derivative
// Sum = 0 (as required for LoG), normalized for edge detection
const LOG_KERNEL: array<f32, 25> = array<f32, 25>(
    0.0,   0.0,  -1.0,   0.0,   0.0,
    0.0,  -1.0,  -2.0,  -1.0,   0.0,
   -1.0,  -2.0,  16.0,  -2.0,  -1.0,
    0.0,  -1.0,  -2.0,  -1.0,   0.0,
    0.0,   0.0,  -1.0,   0.0,   0.0
);

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(filtered_texture);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Skip border pixels (need 2-pixel border for 5x5 kernel)
    if (coords.x < 2 || coords.y < 2 || coords.x >= i32(dims.x) - 2 || coords.y >= i32(dims.y) - 2) {
        textureStore(log_output, coords, vec4<f32>(0.0));
        return;
    }

    // Check if center pixel was filtered out by preprocess stage
    let center_packed = textureLoad(filtered_texture, coords, 0).r;
    if (center_packed == 0u) {
        textureStore(log_output, coords, vec4<f32>(0.0));
        return;
    }

    // Load 5x5 neighborhood using binary event presence
    // Filtered texture packs: (timestamp << 1) | polarity (or 0 if filtered out)
    var has_event: array<f32, 25>;
    var kernel_idx = 0u;

    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let pos = coords + vec2<i32>(dx, dy);
            let packed = textureLoad(filtered_texture, pos, 0).r;
            // Binary: 1.0 if event present, 0.0 if not
            has_event[kernel_idx] = select(0.0, 1.0, packed > 0u);
            kernel_idx++;
        }
    }

    // Check if center pixel has an event (index 12 in 5x5 grid)
    if (has_event[12] < 0.5) {
        textureStore(log_output, coords, vec4<f32>(0.0));
        return;
    }

    // Apply LoG kernel to binary event presence
    var log_response = 0.0;
    for (var i = 0u; i < 25u; i++) {
        log_response += has_event[i] * LOG_KERNEL[i];
    }

    // Output magnitude of LoG response
    // Positive or negative responses indicate edges
    // Higher magnitude = stronger edge
    let edge_strength = abs(log_response);

    // With binary inputs, LoG kernel produces values in roughly 0-16 range (center weight is 16)
    // Threshold is used directly - typical values: 1.0 (weak edges), 8.0 (strong edges)
    // Threshold and output binary edge map
    let edge_value = select(0.0, 1.0, edge_strength > params.threshold);
    textureStore(log_output, coords, vec4<f32>(edge_value));
}
