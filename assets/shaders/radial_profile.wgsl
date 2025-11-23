// Radial Profile Analysis Shader
// Analyzes accumulated surface buffer to detect fan radius

struct RadialResult {
    radial_bins: array<atomic<u32>, 400>,  // Bins for 0-400px radius
    total_intensity: atomic<u32>,
}

@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> result: RadialResult;
@group(0) @binding(2) var<uniform> centroid: vec2<f32>;

const MAX_RADIUS: u32 = 400u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(surface_texture);
    let pos = global_id.xy;

    // Bounds check
    if (pos.x >= dims.x || pos.y >= dims.y) {
        return;
    }

    // Read pixel intensity from surface buffer
    let pixel_value = textureLoad(surface_texture, pos, 0).r;

    if (pixel_value == 0u) {
        return;  // Skip empty pixels
    }

    // Calculate distance from centroid
    let dx = f32(pos.x) - centroid.x;
    let dy = f32(pos.y) - centroid.y;
    let distance = sqrt(dx * dx + dy * dy);

    // Bin the distance
    let bin_index = min(u32(distance), MAX_RADIUS - 1u);

    // Accumulate intensity in the appropriate bin
    atomicAdd(&result.radial_bins[bin_index], pixel_value);
    atomicAdd(&result.total_intensity, pixel_value);
}
