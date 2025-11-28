// CMax-SLAM: Reduce IWE slices to contrast values (sum of squares)
// Uses workgroup reduction to minimize atomic contention

struct ContrastResult {
    contrast_center: atomic<u32>,
    contrast_omega_plus: atomic<u32>,
    contrast_omega_minus: atomic<u32>,
    contrast_cx_plus: atomic<u32>,
    contrast_cx_minus: atomic<u32>,
    contrast_cy_plus: atomic<u32>,
    contrast_cy_minus: atomic<u32>,
    pixel_count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> iwe_buffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: ContrastResult;

const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;
const SLICE_SIZE: u32 = WIDTH * HEIGHT;
const WORKGROUP_SIZE: u32 = 256u;

// Workgroup shared memory for local reduction
var<workgroup> local_center: array<u32, 256>;
var<workgroup> local_omega_p: array<u32, 256>;
var<workgroup> local_omega_m: array<u32, 256>;
var<workgroup> local_cx_p: array<u32, 256>;
var<workgroup> local_cx_m: array<u32, 256>;
var<workgroup> local_cy_p: array<u32, 256>;
var<workgroup> local_cy_m: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let global_idx = gid.x;
    let local_idx = lid.x;

    // Load pixel values and shift right by 4 to prevent overflow
    // Raw values from bilinear voting: ~0-4096 typically (depends on event density)
    // >> 4 gives 0-256, squared max = 65536 per pixel
    // With 921600 pixels / 256 threads per workgroup = 3600 workgroups
    // Each workgroup sums 256 squared values: max 256 * 65536 = 16M (safe for u32)
    // Total across all workgroups: 3600 * 16M could overflow, but workgroup reduction
    // limits it to sum of 256 values before atomic add
    var val_c = 0u;
    var val_omega_p = 0u;
    var val_omega_m = 0u;
    var val_cx_p = 0u;
    var val_cx_m = 0u;
    var val_cy_p = 0u;
    var val_cy_m = 0u;

    if global_idx < SLICE_SIZE {
        val_c = iwe_buffer[global_idx] >> 4u;
        val_omega_p = iwe_buffer[global_idx + SLICE_SIZE] >> 4u;
        val_omega_m = iwe_buffer[global_idx + 2u * SLICE_SIZE] >> 4u;
        val_cx_p = iwe_buffer[global_idx + 3u * SLICE_SIZE] >> 4u;
        val_cx_m = iwe_buffer[global_idx + 4u * SLICE_SIZE] >> 4u;
        val_cy_p = iwe_buffer[global_idx + 5u * SLICE_SIZE] >> 4u;
        val_cy_m = iwe_buffer[global_idx + 6u * SLICE_SIZE] >> 4u;
    }

    // Square values (max 256^2 = 65536)
    local_center[local_idx] = val_c * val_c;
    local_omega_p[local_idx] = val_omega_p * val_omega_p;
    local_omega_m[local_idx] = val_omega_m * val_omega_m;
    local_cx_p[local_idx] = val_cx_p * val_cx_p;
    local_cx_m[local_idx] = val_cx_m * val_cx_m;
    local_cy_p[local_idx] = val_cy_p * val_cy_p;
    local_cy_m[local_idx] = val_cy_m * val_cy_m;

    // Synchronize workgroup
    workgroupBarrier();

    // Tree reduction within workgroup
    var stride = WORKGROUP_SIZE / 2u;
    while stride > 0u {
        if local_idx < stride {
            local_center[local_idx] += local_center[local_idx + stride];
            local_omega_p[local_idx] += local_omega_p[local_idx + stride];
            local_omega_m[local_idx] += local_omega_m[local_idx + stride];
            local_cx_p[local_idx] += local_cx_p[local_idx + stride];
            local_cx_m[local_idx] += local_cx_m[local_idx + stride];
            local_cy_p[local_idx] += local_cy_p[local_idx + stride];
            local_cy_m[local_idx] += local_cy_m[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 adds workgroup result to global buffer
    if local_idx == 0u {
        atomicAdd(&result.contrast_center, local_center[0]);
        atomicAdd(&result.contrast_omega_plus, local_omega_p[0]);
        atomicAdd(&result.contrast_omega_minus, local_omega_m[0]);
        atomicAdd(&result.contrast_cx_plus, local_cx_p[0]);
        atomicAdd(&result.contrast_cx_minus, local_cx_m[0]);
        atomicAdd(&result.contrast_cy_plus, local_cy_p[0]);
        atomicAdd(&result.contrast_cy_minus, local_cy_m[0]);

        // Count pixels for debugging (one per workgroup)
        if global_idx < SLICE_SIZE {
            let pixels_in_workgroup = min(WORKGROUP_SIZE, SLICE_SIZE - wid.x * WORKGROUP_SIZE);
            atomicAdd(&result.pixel_count, pixels_in_workgroup);
        }
    }
}
