// Ground truth blade geometry shader
// Analytically computes exact blade positions for synthetic data validation

struct Params {
    center_x: f32,
    center_y: f32,
    r_min: f32,
    r_max: f32,
    blade_count: u32,
    angular_velocity: f32,
    current_time: f32,
    sweep_k: f32,
    width_root: f32,
    width_tip: f32,
    edge_thickness: f32,
    _padding: f32,
}

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// Wrap angle to [-PI, PI]
fn wrap_angle(angle: f32) -> f32 {
    var a = angle;
    while (a > PI) { a -= TWO_PI; }
    while (a < -PI) { a += TWO_PI; }
    return a;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    // Bounds check (1280x720)
    if (x >= 1280 || y >= 720) {
        return;
    }

    let coords = vec2<i32>(x, y);

    // Convert to polar coordinates relative to fan center
    let dx = f32(x) - params.center_x;
    let dy = f32(y) - params.center_y;
    let r = sqrt(dx * dx + dy * dy);
    let theta = atan2(dy, dx);

    // Default: background (black)
    var output = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    // Skip pixels outside fan radius
    if (r < params.r_min || r > params.r_max) {
        textureStore(output_texture, coords, output);
        return;
    }

    // Calculate current base rotation angle
    let base_angle = params.angular_velocity * params.current_time;

    // Blade width varies with radius (wider at root, narrower at tip)
    let r_norm = (r - params.r_min) / (params.r_max - params.r_min);
    let half_width = (params.width_root + (params.width_tip - params.width_root) * r_norm) * 0.5;

    // Edge thickness in angular units at this radius
    let edge_angular_thickness = params.edge_thickness / r;

    // Check each blade
    for (var blade = 0u; blade < params.blade_count; blade++) {
        // Calculate this blade's center angle
        let blade_spacing = TWO_PI / f32(params.blade_count);
        let blade_base_angle = base_angle + f32(blade) * blade_spacing;

        // Logarithmic spiral: blade curves as radius increases
        let sweep_angle = params.sweep_k * log(r / params.r_min);
        let blade_center = blade_base_angle + sweep_angle;

        // Check if pixel angle is within blade
        let angle_diff = wrap_angle(theta - blade_center);

        // Check if interior pixel (inside blade)
        if (abs(angle_diff) < half_width) {
            // Interior pixel - G channel
            output.g = 1.0;
        }

        // Check if edge pixel - within thickness of either blade boundary
        // This matches the synthesis which generates events AT the blade edges
        // Leading edge at +half_width, trailing edge at -half_width
        let dist_to_leading = abs(angle_diff - half_width);
        let dist_to_trailing = abs(angle_diff + half_width);

        if (dist_to_leading < edge_angular_thickness || dist_to_trailing < edge_angular_thickness) {
            // Edge pixel - R channel
            output.r = 1.0;
        }
    }

    textureStore(output_texture, coords, output);
}
