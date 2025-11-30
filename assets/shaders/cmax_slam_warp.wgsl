// CMax-SLAM: Warp events and compute contrast for 7 parameter variations
// Computes IWE for omega, omega+/-, cx+/-, cy+/-

const IDX_CENTER: u32 = 0u;
const IDX_OMEGA_PLUS: u32 = 1u;
const IDX_OMEGA_MINUS: u32 = 2u;
const IDX_CX_PLUS: u32 = 3u;
const IDX_CX_MINUS: u32 = 4u;
const IDX_CY_PLUS: u32 = 5u;
const IDX_CY_MINUS: u32 = 6u;

struct GpuEvent {
    timestamp: u32,
    x: u32,
    y: u32,
    polarity: u32,
}

struct CmaxSlamParams {
    centroid_x: f32,         // 0-3
    centroid_y: f32,         // 4-7
    t_ref: f32,              // 8-11
    omega: f32,              // 12-15
    delta_omega: f32,        // 16-19
    delta_pos: f32,          // 20-23
    edge_weight: f32,        // 24-27
    window_start: u32,       // 28-31
    window_end: u32,         // 32-35
    event_count: u32,        // 36-39
    _pad0: u32,              // 40-43
    _pad1: u32,              // 44-47
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

// Warp event to IWE coordinate for given omega and centroid
fn warp_event(ex: f32, ey: f32, dt: f32, omega: f32, cx: f32, cy: f32) -> vec2<f32> {
    let dx = ex - cx;
    let dy = ey - cy;
    let r = sqrt(dx * dx + dy * dy);

    if r < 1.0 {
        return vec2<f32>(-1.0, -1.0);  // Skip center events
    }

    let theta = atan2(dy, dx);
    let theta_warped = theta - omega * dt;

    return vec2<f32>(cx + r * cos(theta_warped), cy + r * sin(theta_warped));
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

    // Use coordinates directly from struct
    let ex = f32(event.x);
    let ey = f32(event.y);
    let dt = f32(event.timestamp) - params.t_ref;

    // Extract parameters
    let cx = params.centroid_x;
    let cy = params.centroid_y;
    let omega = params.omega;
    let d_omega = params.delta_omega;
    let d_pos = params.delta_pos;

    // Compute 7 warps
    let pos0 = warp_event(ex, ey, dt, omega, cx, cy);
    let pos1 = warp_event(ex, ey, dt, omega + d_omega, cx, cy);
    let pos2 = warp_event(ex, ey, dt, omega - d_omega, cx, cy);
    let pos3 = warp_event(ex, ey, dt, omega, cx + d_pos, cy);
    let pos4 = warp_event(ex, ey, dt, omega, cx - d_pos, cy);
    let pos5 = warp_event(ex, ey, dt, omega, cx, cy + d_pos);
    let pos6 = warp_event(ex, ey, dt, omega, cx, cy - d_pos);

    // Accumulate to 7 slices
    if pos0.x >= 0.0 { accumulate_bilinear(pos0, IDX_CENTER * SLICE_SIZE); }
    if pos1.x >= 0.0 { accumulate_bilinear(pos1, IDX_OMEGA_PLUS * SLICE_SIZE); }
    if pos2.x >= 0.0 { accumulate_bilinear(pos2, IDX_OMEGA_MINUS * SLICE_SIZE); }
    if pos3.x >= 0.0 { accumulate_bilinear(pos3, IDX_CX_PLUS * SLICE_SIZE); }
    if pos4.x >= 0.0 { accumulate_bilinear(pos4, IDX_CX_MINUS * SLICE_SIZE); }
    if pos5.x >= 0.0 { accumulate_bilinear(pos5, IDX_CY_PLUS * SLICE_SIZE); }
    if pos6.x >= 0.0 { accumulate_bilinear(pos6, IDX_CY_MINUS * SLICE_SIZE); }
}
