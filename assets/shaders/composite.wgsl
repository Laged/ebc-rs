// Composite shader: combines 4 detector outputs into 2x2 grid
// Output: 2560x1440 (2x base resolution of 1280x720)

// Input textures have different formats:
// - raw_texture: R32Uint (event counts)
// - sobel/canny/log: R32Float (edge magnitudes)
@group(0) @binding(0) var raw_texture: texture_2d<u32>;
@group(0) @binding(1) var sobel_texture: texture_2d<f32>;
@group(0) @binding(2) var canny_texture: texture_2d<f32>;
@group(0) @binding(3) var log_texture: texture_2d<f32>;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba8unorm, write>;

const BASE_WIDTH: u32 = 1280u;
const BASE_HEIGHT: u32 = 720u;

// Color scheme for each detector
const RAW_COLOR: vec3<f32> = vec3<f32>(0.8, 0.8, 0.8);    // Light gray
const SOBEL_COLOR: vec3<f32> = vec3<f32>(1.0, 0.4, 0.4);  // Red
const CANNY_COLOR: vec3<f32> = vec3<f32>(0.4, 1.0, 0.4);  // Green
const LOG_COLOR: vec3<f32> = vec3<f32>(0.4, 0.4, 1.0);    // Blue

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_x = global_id.x;
    let output_y = global_id.y;

    // Bounds check for 2560x1440
    if (output_x >= BASE_WIDTH * 2u || output_y >= BASE_HEIGHT * 2u) {
        return;
    }

    let coords = vec2<i32>(i32(output_x), i32(output_y));

    // Determine which quadrant we're in
    let is_right = output_x >= BASE_WIDTH;
    let is_bottom = output_y >= BASE_HEIGHT;

    // Map to source texture coordinates
    let source_x = output_x % BASE_WIDTH;
    let source_y = output_y % BASE_HEIGHT;
    let source_coords = vec2<i32>(i32(source_x), i32(source_y));

    var intensity: f32 = 0.0;
    var color: vec3<f32>;

    if (!is_right && !is_bottom) {
        // Top-left: Raw events (u32 -> normalize)
        let raw_value = textureLoad(raw_texture, source_coords, 0).r;
        intensity = clamp(f32(raw_value) / 10.0, 0.0, 1.0);
        color = RAW_COLOR;
    } else if (is_right && !is_bottom) {
        // Top-right: Sobel
        let value = textureLoad(sobel_texture, source_coords, 0).r;
        intensity = clamp(value / 500.0, 0.0, 1.0);
        color = SOBEL_COLOR;
    } else if (!is_right && is_bottom) {
        // Bottom-left: Canny
        let value = textureLoad(canny_texture, source_coords, 0).r;
        intensity = clamp(value / 500.0, 0.0, 1.0);
        color = CANNY_COLOR;
    } else {
        // Bottom-right: LoG
        let value = textureLoad(log_texture, source_coords, 0).r;
        intensity = clamp(value / 500.0, 0.0, 1.0);
        color = LOG_COLOR;
    }

    // Draw border between quadrants (2px wide)
    let border_x = output_x == BASE_WIDTH - 1u || output_x == BASE_WIDTH;
    let border_y = output_y == BASE_HEIGHT - 1u || output_y == BASE_HEIGHT;

    var output_color: vec4<f32>;
    if (border_x || border_y) {
        output_color = vec4<f32>(0.3, 0.3, 0.3, 1.0); // Dark gray border
    } else {
        output_color = vec4<f32>(color * intensity, 1.0);
    }

    textureStore(output_texture, coords, output_color);
}
