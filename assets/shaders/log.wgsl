@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var log_output: texture_storage_2d<r32float, write>;

struct LogParams {
    sigma: f32,          // Not used - kernel is fixed for sigma=1.4
    threshold: f32,      // Threshold for edge magnitude
}

@group(0) @binding(2) var<uniform> params: LogParams;

// 5x5 Laplacian of Gaussian kernel (approximation for sigma ≈ 1.4)
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
    let dims = textureDimensions(surface_texture);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Skip border pixels (need 2-pixel border for 5x5 kernel)
    if (coords.x < 2 || coords.y < 2 || coords.x >= i32(dims.x) - 2 || coords.y >= i32(dims.y) - 2) {
        textureStore(log_output, coords, vec4<f32>(0.0));
        return;
    }

    // Apply LoG kernel to detect edges
    // LoG detects regions of rapid intensity change (second derivative)
    // Surface texture packs: (timestamp << 1) | polarity
    var log_response = 0.0;
    var kernel_idx = 0u;

    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let pos = coords + vec2<i32>(dx, dy);
            let packed = textureLoad(surface_texture, pos, 0).r;
            let timestamp = f32(packed >> 1u);

            log_response += timestamp * LOG_KERNEL[kernel_idx];
            kernel_idx++;
        }
    }

    // Output magnitude of LoG response
    // Positive or negative responses indicate edges
    // Higher magnitude = stronger edge
    let edge_strength = abs(log_response);

    // Threshold and output binary edge map
    let edge_value = select(0.0, 1.0, edge_strength > params.threshold);
    textureStore(log_output, coords, vec4<f32>(edge_value));
}
