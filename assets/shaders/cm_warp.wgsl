// Contrast Maximization: Warp events by candidate angular velocities
// Builds Image of Warped Events (IWE) for each omega candidate

struct GpuEvent {
    timestamp: u32,
    data: u32,  // packed x[13:0], y[27:14], polarity[31:28]
}

struct CmParams {
    centroid_x: f32,
    centroid_y: f32,
    t_ref: f32,
    omega_min: f32,
    omega_step: f32,
    n_omega: u32,
    window_start: u32,
    window_end: u32,
    event_count: u32,
    _padding: vec3<u32>,
}

@group(0) @binding(0) var<storage, read> events: array<GpuEvent>;
@group(0) @binding(1) var<uniform> params: CmParams;
@group(0) @binding(2) var<storage, read_write> iwe_buffer: array<atomic<u32>>;

// IWE dimensions
const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;

fn unpack_x(data: u32) -> u32 {
    return data & 0x3FFFu;  // bits 0-13
}

fn unpack_y(data: u32) -> u32 {
    return (data >> 14u) & 0x3FFFu;  // bits 14-27
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let event_idx = gid.x;
    if (event_idx >= params.event_count) {
        return;
    }

    let event = events[event_idx];

    // Filter by time window
    if (event.timestamp < params.window_start || event.timestamp > params.window_end) {
        return;
    }

    // Unpack event coordinates
    let ex = f32(unpack_x(event.data));
    let ey = f32(unpack_y(event.data));

    // Convert to polar around centroid
    let dx = ex - params.centroid_x;
    let dy = ey - params.centroid_y;
    let r = sqrt(dx * dx + dy * dy);

    // Skip events at center (undefined angle)
    if (r < 1.0) {
        return;
    }

    let theta = atan2(dy, dx);
    let dt = f32(event.timestamp) - params.t_ref;

    // For each omega candidate, warp and accumulate
    for (var i = 0u; i < params.n_omega; i++) {
        let omega = params.omega_min + f32(i) * params.omega_step;
        let theta_warped = theta - omega * dt;

        // Convert back to Cartesian
        let x_warped = params.centroid_x + r * cos(theta_warped);
        let y_warped = params.centroid_y + r * sin(theta_warped);

        let ix = u32(x_warped);
        let iy = u32(y_warped);

        if (ix < WIDTH && iy < HEIGHT) {
            // Calculate buffer index: slice * (WIDTH * HEIGHT) + y * WIDTH + x
            let buffer_idx = i * (WIDTH * HEIGHT) + iy * WIDTH + ix;
            atomicAdd(&iwe_buffer[buffer_idx], 1u);
        }
    }
}
