#import bevy_pbr::forward_io::VertexOutput

struct Params {
    width: f32,
    height: f32,
    time: f32,
    decay_tau: f32,
    show_sobel: u32,
    show_raw: u32,
    show_canny: u32,
    show_log: u32,
    show_ground_truth: u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: Params;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var surface_texture: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var sobel_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var sobel_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var canny_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var canny_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var log_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var log_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var ground_truth_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var ground_truth_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map UV to pixel coordinates
    let x = i32(in.uv.x * params.width);
    let y = i32(in.uv.y * params.height);

    // Initialize output color
    var output_color = vec3<f32>(0.0, 0.0, 0.0);

    // Layer 0: Raw events (red/blue) - only if enabled
    if (params.show_raw == 1u) {
        let packed_val = textureLoad(surface_texture, vec2<i32>(x, y), 0).r;

        if (packed_val != 0u) {
            let timestamp = packed_val >> 1u;
            let polarity = packed_val & 1u;

            let dt = params.time - f32(timestamp);
            let intensity = exp(-dt / params.decay_tau);

            // Polarity 1 = Positive (Red), 0 = Negative (Blue)
            if (polarity == 1u) {
                output_color = vec3<f32>(1.0, 0.2, 0.2) * intensity; // Reddish
            } else {
                output_color = vec3<f32>(0.2, 0.2, 1.0) * intensity; // Bluish
            }
        }
    }

    // Layer 1: Sobel edges (yellow) with 50% alpha blend compositing
    if (params.show_sobel == 1u) {
        let edge_val = textureSample(sobel_texture, sobel_sampler, in.uv).r;
        if (edge_val > 0.0) {
            let yellow = vec3<f32>(1.0, 1.0, 0.0);
            output_color = mix(output_color, yellow, 0.5);  // 50% alpha blend
        }
    }

    // Layer 2: Canny edges (cyan) with 50% alpha blend compositing
    if (params.show_canny == 1u) {
        let edge_val = textureSample(canny_texture, canny_sampler, in.uv).r;
        if (edge_val > 0.0) {
            let cyan = vec3<f32>(0.0, 1.0, 1.0);
            output_color = mix(output_color, cyan, 0.5);  // 50% alpha blend
        }
    }

    // Layer 3: LoG edges (magenta) with 50% alpha blend compositing
    if (params.show_log == 1u) {
        let edge_val = textureSample(log_texture, log_sampler, in.uv).r;
        if (edge_val > 0.0) {
            let magenta = vec3<f32>(1.0, 0.0, 1.0);
            output_color = mix(output_color, magenta, 0.5);  // 50% alpha blend
        }
    }

    // Layer 4: Ground truth edges (green) with 50% alpha blend
    if (params.show_ground_truth == 1u) {
        let gt_val = textureSample(ground_truth_texture, ground_truth_sampler, in.uv);
        // R channel = edge pixels
        if (gt_val.r > 0.0) {
            let green = vec3<f32>(0.0, 1.0, 0.0);
            output_color = mix(output_color, green, 0.5);
        }
    }

    return vec4<f32>(output_color, 1.0);
};
