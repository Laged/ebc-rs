struct GpuEvent {
    timestamp: u32,
    x: u32,
    y: u32,
    polarity: u32,
}

@group(0) @binding(0) var<storage, read> events: array<GpuEvent>;
@group(0) @binding(1) var<storage, read_write> surface_buffer: array<atomic<u32>>;
// dimensions.x = width
// dimensions.y = height
// dimensions.z = window_start_time
// dimensions.w = window_end_time
@group(0) @binding(2) var<uniform> dimensions: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Handle 2D dispatch for large event counts (exceeding 65535 workgroups in X)
    // Stride must match the max X workgroups (65535) * workgroup_size (64)
    let threads_per_row = 65535u * 64u;
    let idx = global_id.x + global_id.y * threads_per_row;
    
    if (idx >= arrayLength(&events)) {
        return;
    }
    
    let e = events[idx];

    // Filter events outside the time window
    if (e.timestamp < dimensions.z || e.timestamp > dimensions.w) {
        return;
    }
    // Boundary check
    if (e.x >= dimensions.x || e.y >= dimensions.y) {
        return;
    }
    
    let linear_idx = e.y * dimensions.x + e.x;
    
    // Pack timestamp and polarity: 31 bits for timestamp, 1 bit for polarity
    // We assume timestamp fits in 31 bits (up to ~35 minutes)
    let packed_val = (e.timestamp << 1u) | (e.polarity & 1u);
    
    atomicMax(&surface_buffer[linear_idx], packed_val);
}
