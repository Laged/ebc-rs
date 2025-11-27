// CMax-SLAM: Warp events and compute contrast for 3 omega values
// Computes IWE for omega, omega+delta, omega-delta simultaneously

struct GpuEvent {
    timestamp: u32,
    data: u32,  // packed x[13:0], y[27:14], polarity[31:28]
}

struct CmaxSlamParams {
    centroid_x: f32,         // 0-3
    centroid_y: f32,         // 4-7
    t_ref: f32,              // 8-11
    omega: f32,              // 12-15
    delta_omega: f32,        // 16-19
    edge_weight: f32,        // 20-23
    window_start: u32,       // 24-27
    window_end: u32,         // 28-31
    event_count: u32,        // 32-35
    _pad0: u32,              // 36-39
    _pad1: u32,              // 40-43
    _pad2: u32,              // 44-47
}

struct ContrastResult {
    contrast_center: f32,
    contrast_plus: f32,
    contrast_minus: f32,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> events: array<GpuEvent>;
@group(0) @binding(1) var<uniform> params: CmaxSlamParams;
@group(0) @binding(2) var<storage, read_write> iwe_buffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> contrast_result: ContrastResult;

const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;
const SLICE_SIZE: u32 = 1280u * 720u;

fn unpack_x(data: u32) -> u32 {
    return data & 0x3FFFu;
}

fn unpack_y(data: u32) -> u32 {
    return (data >> 14u) & 0x3FFFu;
}

// Warp event to IWE coordinate for given omega
fn warp_event(ex: f32, ey: f32, dt: f32, omega: f32) -> vec2<f32> {
    let dx = ex - params.centroid_x;
    let dy = ey - params.centroid_y;
    let r = sqrt(dx * dx + dy * dy);

    if r < 1.0 {
        return vec2<f32>(-1.0, -1.0);  // Skip center events
    }

    let theta = atan2(dy, dx);
    let theta_warped = theta - omega * dt;

    let x_warped = params.centroid_x + r * cos(theta_warped);
    let y_warped = params.centroid_y + r * sin(theta_warped);

    return vec2<f32>(x_warped, y_warped);
}

// Add event to IWE with bilinear voting
fn accumulate_bilinear(pos: vec2<f32>, slice_offset: u32) {
    let ix = i32(floor(pos.x));
    let iy = i32(floor(pos.y));

    // Bilinear weights
    let fx = pos.x - f32(ix);
    let fy = pos.y - f32(iy);

    // Four corners with weights
    let corners = array<vec2<i32>, 4>(
        vec2<i32>(ix, iy),
        vec2<i32>(ix + 1, iy),
        vec2<i32>(ix, iy + 1),
        vec2<i32>(ix + 1, iy + 1)
    );

    let weights = array<f32, 4>(
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy
    );

    for (var i = 0u; i < 4u; i++) {
        let cx = corners[i].x;
        let cy = corners[i].y;

        if cx >= 0 && cx < i32(WIDTH) && cy >= 0 && cy < i32(HEIGHT) {
            let idx = slice_offset + u32(cy) * WIDTH + u32(cx);
            // Scale weight to integer (precision: 1/256)
            let w = u32(weights[i] * 256.0);
            if w > 0u {
                atomicAdd(&iwe_buffer[idx], w);
            }
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let event_idx = gid.x;
    if event_idx >= params.event_count {
        return;
    }

    let event = events[event_idx];

    // Filter by time window
    if event.timestamp < params.window_start || event.timestamp > params.window_end {
        return;
    }

    // Unpack coordinates
    let ex = f32(unpack_x(event.data));
    let ey = f32(unpack_y(event.data));
    let dt = f32(event.timestamp) - params.t_ref;

    // Warp for three omega values
    let omega_center = params.omega;
    let omega_plus = params.omega + params.delta_omega;
    let omega_minus = params.omega - params.delta_omega;

    let pos_center = warp_event(ex, ey, dt, omega_center);
    let pos_plus = warp_event(ex, ey, dt, omega_plus);
    let pos_minus = warp_event(ex, ey, dt, omega_minus);

    // Accumulate to respective slices
    if pos_center.x >= 0.0 {
        accumulate_bilinear(pos_center, 0u);
    }
    if pos_plus.x >= 0.0 {
        accumulate_bilinear(pos_plus, SLICE_SIZE);
    }
    if pos_minus.x >= 0.0 {
        accumulate_bilinear(pos_minus, 2u * SLICE_SIZE);
    }
}
