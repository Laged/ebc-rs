// Angular Histogram Shader
// Analyzes event distribution around centroid to detect blade positions

struct AngularResult {
    bins: array<atomic<u32>, 360>,  // 1 degree per bin
}

struct GpuEvent {
    timestamp: u32,
    x: u32,
    y: u32,
    polarity: u32,
}

struct AnalysisParams {
    centroid_x: f32,
    centroid_y: f32,
    radius: f32,
    radius_tolerance: f32,
    window_start: u32,
    window_end: u32,
}

@group(0) @binding(0) var<storage, read> events: array<GpuEvent>;
@group(0) @binding(1) var<storage, read_write> result: AngularResult;
@group(0) @binding(2) var<uniform> params: AnalysisParams;

const PI: f32 = 3.14159265359;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let event_index = global_id.x + global_id.y * 65535u;

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

    // Only count events near the detected radius
    if (abs(distance - params.radius) > params.radius_tolerance) {
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
