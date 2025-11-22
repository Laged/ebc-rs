// Contrast Maximization (CMax) Optimization Shader
// Warps events based on test angular velocity and calculates variance
//
// CRITICAL: Event timestamps are in 100 nanosecond units, NOT microseconds!
// This affects the omega scaling when converting to RPM.

struct Event {
    timestamp: u32,         // Timestamp in 100ns units (0.1 microseconds)
    x: u32,
    y: u32,
    polarity: u32,
}

struct OptimizationParams {
    width: u32,
    height: u32,
    window_start: u32,      // In 100ns units
    window_end: u32,        // In 100ns units
    centroid_x: f32,
    centroid_y: f32,
    test_omega: f32,        // Angular velocity to test (radians per 100ns unit)
    reference_time: f32,    // Reference timestamp (in 100ns units)
}

struct VarianceResult {
    sum_intensity: atomic<u32>,
    sum_squared: atomic<u32>,
    pixel_count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> events: array<Event>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> result: VarianceResult;
@group(0) @binding(3) var<uniform> params: OptimizationParams;

// Rotate point around centroid by angle theta
fn rotate_point(x: f32, y: f32, cx: f32, cy: f32, theta: f32) -> vec2<f32> {
    let dx = x - cx;
    let dy = y - cy;

    let cos_theta = cos(theta);
    let sin_theta = sin(theta);

    let rotated_x = dx * cos_theta - dy * sin_theta + cx;
    let rotated_y = dx * sin_theta + dy * cos_theta + cy;

    return vec2<f32>(rotated_x, rotated_y);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + global_id.y * 65535u;

    if index >= arrayLength(&events) {
        return;
    }

    let event = events[index];

    // Only process events within the current time window
    if event.timestamp < params.window_start || event.timestamp > params.window_end {
        return;
    }

    // Calculate time difference from reference (in microseconds)
    let dt = f32(event.timestamp) - params.reference_time;

    // Calculate rotation angle: theta = omega * dt
    let theta = params.test_omega * dt;

    // Warp the event position
    let warped = rotate_point(
        f32(event.x),
        f32(event.y),
        params.centroid_x,
        params.centroid_y,
        theta
    );

    // Check if warped position is within bounds
    let wx = u32(warped.x);
    let wy = u32(warped.y);

    if wx < params.width && wy < params.height {
        let pixel_idx = wy * params.width + wx;

        // Accumulate the event (using polarity: +1 or -1)
        if event.polarity == 1u {
            atomicAdd(&accumulation[pixel_idx], 1u);
        } else {
            atomicSub(&accumulation[pixel_idx], 1u);
        }
    }
}

// Second pass: Calculate variance
@compute @workgroup_size(64)
fn calculate_variance(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + global_id.y * 65535u;
    let total_pixels = params.width * params.height;

    if index >= total_pixels {
        return;
    }

    let intensity = atomicLoad(&accumulation[index]);

    if intensity > 0u {
        atomicAdd(&result.sum_intensity, intensity);
        atomicAdd(&result.sum_squared, intensity * intensity);
        atomicAdd(&result.pixel_count, 1u);
    }
}
