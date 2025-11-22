// Centroid Tracking Compute Shader
// Calculates the spatial mean (centroid) of events in the current time window
//
// CRITICAL: Event timestamps are in 100 nanosecond units, NOT microseconds!

struct Event {
    timestamp: u32,         // Timestamp in 100ns units (0.1 microseconds)
    x: u32,
    y: u32,
    polarity: u32,
}

struct Dimensions {
    width: u32,
    height: u32,
    window_start: u32,      // In 100ns units
    window_end: u32,        // In 100ns units
}

struct CentroidResult {
    sum_x: atomic<u32>,  // Accumulated X * 1000 (to preserve precision)
    sum_y: atomic<u32>,  // Accumulated Y * 1000 (to preserve precision)
    count: atomic<u32>,  // Number of events
}

@group(0) @binding(0) var<storage, read> events: array<Event>;
@group(0) @binding(1) var<storage, read_write> result: CentroidResult;
@group(0) @binding(2) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + global_id.y * 65535u;

    if index >= arrayLength(&events) {
        return;
    }

    let event = events[index];

    // Only process events within the current time window
    if event.timestamp >= dimensions.window_start && event.timestamp <= dimensions.window_end {
        // Scale by 1000 to preserve sub-pixel precision when using atomic integers
        let x_scaled = event.x * 1000u;
        let y_scaled = event.y * 1000u;

        atomicAdd(&result.sum_x, x_scaled);
        atomicAdd(&result.sum_y, y_scaled);
        atomicAdd(&result.count, 1u);
    }
}
