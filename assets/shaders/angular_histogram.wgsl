// Angular Histogram Shader
// Analyzes event distribution around centroid to detect blade positions
//
// Event timestamps are in MICROSECONDS (1us units)

struct AngularResult {
    bins: array<atomic<u32>, 360>,  // 1 degree per bin
}

struct GpuEvent {
    timestamp: u32,  // In microseconds
    x: u32,
    y: u32,
    polarity: u32,
}

struct AnalysisParams {
    centroid_x: f32,
    centroid_y: f32,
    radius: f32,           // Minimum radius (exclude events closer than this)
    radius_tolerance: f32, // Unused (kept for struct alignment)
    window_start: u32,     // In microseconds
    window_end: u32,       // In microseconds
}

@group(0) @binding(0) var<storage, read> events: array<GpuEvent>;
@group(0) @binding(1) var<storage, read_write> result: AngularResult;
@group(0) @binding(2) var<uniform> params: AnalysisParams;

const PI: f32 = 3.14159265359;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Stride must match dispatch logic: x_workgroups = min(total, 65535)
    // So stride for next row is 65535 * workgroup_size(64)
    let stride = 65535u * 64u;
    let event_index = global_id.x + global_id.y * stride;

    if (event_index >= arrayLength(&events)) {
        return;
    }

    let event = events[event_index];

    // Filter by time window
    if (event.timestamp < params.window_start || event.timestamp > params.window_end) {
        return;
    }

    // Calculate distance from centroid
    let dx = f32(event.x) - params.centroid_x;
    let dy = f32(event.y) - params.centroid_y;
    let distance = sqrt(dx * dx + dy * dy);

    // Only count events beyond minimum radius (params.radius = min_radius)
    // This excludes events too close to the center
    if (distance < params.radius) {
        return;
    }

    // Calculate angle (atan2 returns [-π, π], normalize to [0, 2π])
    var angle = atan2(dy, dx);
    if (angle < 0.0) {
        angle = angle + 2.0 * PI;
    }

    // Convert to bin index (0-359 degrees)
    let bin_index = u32((angle / (2.0 * PI)) * 360.0) % 360u;

    // Increment bin
    atomicAdd(&result.bins[bin_index], 1u);
}
