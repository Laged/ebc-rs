#import bevy_pbr::forward_io::VertexOutput

struct Params {
    width: f32,
    height: f32,
    time: f32,
    decay_tau: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: Params;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var surface_texture: texture_2d<u32>;
// Sampler removed as we use textureLoad with integer coordinates

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map UV to pixel coordinates
    let x = i32(in.uv.x * params.width);
    let y = i32(in.uv.y * params.height); // Flip Y if needed? Bevy UVs are bottom-left 0,0.

    // Load packed value from texture
    let packed_val = textureLoad(surface_texture, vec2<i32>(x, y), 0).r;

    if (packed_val == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let timestamp = packed_val >> 1u;
    let polarity = packed_val & 1u;

    let dt = params.time - f32(timestamp);
    let intensity = exp(-dt / params.decay_tau);

    // Polarity 1 = Positive (Red), 0 = Negative (Blue)
    // Using a nice hot/cold palette
    var color: vec3<f32>;
    if (polarity == 1u) {
        color = vec3<f32>(1.0, 0.2, 0.2); // Reddish
    } else {
        color = vec3<f32>(0.2, 0.2, 1.0); // Bluish
    }

    return vec4<f32>(color * intensity, 1.0);
};
