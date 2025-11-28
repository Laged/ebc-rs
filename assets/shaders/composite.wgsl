// Composite shader: combines 4 detector outputs into 2x2 grid
// Output: 2560x1440 (2x base resolution of 1280x720)
//
// Layout:
//   Top-left:     RAW events (polarity colors: blue=-1, red=+1)
//   Top-right:    Sobel edges (green)
//   Bottom-left:  LoG edges (white)
//   Bottom-right: CMax-SLAM motion-compensated edges (purple)

// Input textures have different formats:
// - raw_texture: R32Uint (packed: timestamp << 1 | polarity)
// - sobel/cm/log: R32Float (edge magnitudes)
@group(0) @binding(0) var raw_texture: texture_2d<u32>;
@group(0) @binding(1) var sobel_texture: texture_2d<f32>;
@group(0) @binding(2) var cm_texture: texture_2d<f32>;
@group(0) @binding(3) var log_texture: texture_2d<f32>;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba8unorm, write>;

struct CompositeParams {
    show_raw: u32,
    show_sobel: u32,
    show_canny: u32,  // Now used for CMax-SLAM (bottom-right)
    show_log: u32,
}

@group(0) @binding(5) var<uniform> params: CompositeParams;

const BASE_WIDTH: u32 = 1280u;
const BASE_HEIGHT: u32 = 720u;

// Color scheme for each detector
const RAW_COLOR_POS: vec3<f32> = vec3<f32>(1.0, 0.2, 0.2);   // Red for +1 polarity
const RAW_COLOR_NEG: vec3<f32> = vec3<f32>(0.2, 0.2, 1.0);   // Blue for -1 polarity
const SOBEL_COLOR: vec3<f32> = vec3<f32>(0.0, 1.0, 1.0);     // Bright cyan
const LOG_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);       // White
const CMAX_COLOR: vec3<f32> = vec3<f32>(0.2, 1.0, 0.2);      // Green

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

    var output_color: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05); // Dark background

    if (!is_right && !is_bottom) {
        // Top-left: Raw events with polarity colors
        if (params.show_raw == 1u) {
            // FilteredSurface packs: (timestamp << 1) | polarity
            // polarity bit: 0 = negative (-1), 1 = positive (+1)
            let raw_value = textureLoad(raw_texture, source_coords, 0).r;
            if (raw_value > 0u) {
                let polarity = raw_value & 1u;
                if (polarity == 1u) {
                    output_color = RAW_COLOR_POS;  // Red for +1
                } else {
                    output_color = RAW_COLOR_NEG;  // Blue for -1
                }
            }
        }
    } else if (is_right && !is_bottom) {
        // Top-right: Sobel (green)
        if (params.show_sobel == 1u) {
            let value = textureLoad(sobel_texture, source_coords, 0).r;
            if (value > 0.0) {
                output_color = SOBEL_COLOR * value;
            }
        }
    } else if (!is_right && is_bottom) {
        // Bottom-left: LoG (white)
        if (params.show_log == 1u) {
            let value = textureLoad(log_texture, source_coords, 0).r;
            if (value > 0.0) {
                output_color = LOG_COLOR * value;
            }
        }
    } else {
        // Bottom-right: CMax-SLAM (purple)
        if (params.show_canny == 1u) {
            let value = textureLoad(cm_texture, source_coords, 0).r;
            if (value > 0.0) {
                output_color = CMAX_COLOR * value;
            }
        }
    }

    // Draw border between quadrants (2px wide)
    let border_x = output_x == BASE_WIDTH - 1u || output_x == BASE_WIDTH;
    let border_y = output_y == BASE_HEIGHT - 1u || output_y == BASE_HEIGHT;

    if (border_x || border_y) {
        output_color = vec3<f32>(0.3, 0.3, 0.3); // Dark gray border
    }

    textureStore(output_texture, coords, vec4<f32>(output_color, 1.0));
}
