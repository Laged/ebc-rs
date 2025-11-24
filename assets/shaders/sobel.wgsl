@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var gradient_output: texture_storage_2d<r32float, write>;

struct EdgeParams {
    threshold: f32,
    filter_dead_pixels: u32,
    filter_density: u32,
    filter_bidirectional: u32,
    filter_temporal: u32,
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

    // Filter 1: Dead pixel check
    if (params.filter_dead_pixels == 1u) {
        let center_timestamp = timestamps[4]; // Center of 3x3 grid
        if (center_timestamp < 1.0) {  // No events at center
            textureStore(gradient_output, coords, vec4<f32>(0.0));
            return;
        }
    }

    // Filter 2: Event density check
    if (params.filter_density == 1u) {
        var active_count = 0u;
        for (var i = 0u; i < 9u; i++) {
            if (timestamps[i] > 1.0) {
                active_count++;
            }
        }
        if (active_count < 5u) {  // Need at least 5/9 pixels with events
            textureStore(gradient_output, coords, vec4<f32>(0.0));
            return;
        }
    }

    // Filter 4: Temporal variance
    if (params.filter_temporal == 1u) {
        var min_ts = timestamps[0];
        var max_ts = timestamps[0];
        for (var i = 1u; i < 9u; i++) {
            if (timestamps[i] > 0.0) {
                min_ts = min(min_ts, timestamps[i]);
                max_ts = max(max_ts, timestamps[i]);
            }
        }
        let ts_range = max_ts - min_ts;
        if (ts_range < 500.0) {  // Minimum 500μs timestamp spread
            textureStore(gradient_output, coords, vec4<f32>(0.0));
            return;
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

    // Filter 3: Bidirectional gradient check
    if (params.filter_bidirectional == 1u) {
        let gx_abs = abs(gx);
        let gy_abs = abs(gy);
        let min_directional = params.threshold * 0.3;
        if (gx_abs < min_directional || gy_abs < min_directional) {
            textureStore(gradient_output, coords, vec4<f32>(0.0));
            return;
        }
    }

    let magnitude = sqrt(gx * gx + gy * gy);

    // Threshold and write
    let edge_value = select(0.0, 1.0, magnitude > params.threshold);
    textureStore(gradient_output, coords, vec4<f32>(edge_value));
}
