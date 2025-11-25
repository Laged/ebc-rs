@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var filtered_output: texture_storage_2d<r32uint, write>;

struct GpuParams {
    // Pre-processing
    filter_dead_pixels: u32,
    filter_density: u32,
    filter_temporal: u32,
    min_density_count: u32,
    min_temporal_spread: f32,

    // Sobel
    sobel_threshold: f32,

    // Canny
    canny_low_threshold: f32,
    canny_high_threshold: f32,

    // LoG
    log_threshold: f32,

    // Post-processing
    filter_bidirectional: u32,
    bidirectional_ratio: f32,

    // Padding for 16-byte alignment
    _padding: f32,
}

@group(0) @binding(2) var<uniform> params: GpuParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(surface_texture);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Load center pixel
    let center_packed = textureLoad(surface_texture, coords, 0).r;
    let center_timestamp = center_packed >> 1u;

    // Filter 1: Dead pixel check
    if (params.filter_dead_pixels == 1u && center_timestamp < 1u) {
        textureStore(filtered_output, coords, vec4<u32>(0u));
        return;
    }

    // Skip border for neighborhood filters
    if (coords.x < 1 || coords.y < 1 || coords.x >= i32(dims.x) - 1 || coords.y >= i32(dims.y) - 1) {
        textureStore(filtered_output, coords, vec4<u32>(center_packed));
        return;
    }

    // Load 3x3 neighborhood timestamps
    var timestamps: array<f32, 9>;
    var idx = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let pos = coords + vec2<i32>(dx, dy);
            let packed = textureLoad(surface_texture, pos, 0).r;
            timestamps[idx] = f32(packed >> 1u);
            idx++;
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
        if (active_count < params.min_density_count) {
            textureStore(filtered_output, coords, vec4<u32>(0u));
            return;
        }
    }

    // Filter 3: Temporal variance check
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
        if (ts_range < params.min_temporal_spread) {
            textureStore(filtered_output, coords, vec4<u32>(0u));
            return;
        }
    }

    // Passed all filters - copy original value
    textureStore(filtered_output, coords, vec4<u32>(center_packed));
}
