// CMax-SLAM: Reduce IWE slices to contrast values (sum of squares)
// Uses workgroup reduction to minimize atomic contention

struct ContrastResult {
    sum_sq_center: atomic<u32>,
    sum_sq_plus: atomic<u32>,
    sum_sq_minus: atomic<u32>,
    pixel_count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> iwe_buffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: ContrastResult;

const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;
const SLICE_SIZE: u32 = WIDTH * HEIGHT;
const WORKGROUP_SIZE: u32 = 256u;

// Workgroup shared memory for local reduction
var<workgroup> local_center: array<u32, WORKGROUP_SIZE>;
var<workgroup> local_plus: array<u32, WORKGROUP_SIZE>;
var<workgroup> local_minus: array<u32, WORKGROUP_SIZE>;

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
    var val_p = 0u;
    var val_m = 0u;

    if global_idx < SLICE_SIZE {
        val_c = iwe_buffer[global_idx] >> 4u;
        val_p = iwe_buffer[global_idx + SLICE_SIZE] >> 4u;
        val_m = iwe_buffer[global_idx + 2u * SLICE_SIZE] >> 4u;
    }

    // Square values (max 256^2 = 65536)
    local_center[local_idx] = val_c * val_c;
    local_plus[local_idx] = val_p * val_p;
    local_minus[local_idx] = val_m * val_m;

    // Synchronize workgroup
    workgroupBarrier();

    // Tree reduction within workgroup
    var stride = WORKGROUP_SIZE / 2u;
    while stride > 0u {
        if local_idx < stride {
            local_center[local_idx] += local_center[local_idx + stride];
            local_plus[local_idx] += local_plus[local_idx + stride];
            local_minus[local_idx] += local_minus[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 adds workgroup result to global buffer
    if local_idx == 0u {
        atomicAdd(&result.sum_sq_center, local_center[0]);
        atomicAdd(&result.sum_sq_plus, local_plus[0]);
        atomicAdd(&result.sum_sq_minus, local_minus[0]);

        // Count pixels for debugging (one per workgroup)
        if global_idx < SLICE_SIZE {
            let pixels_in_workgroup = min(WORKGROUP_SIZE, SLICE_SIZE - wid.x * WORKGROUP_SIZE);
            atomicAdd(&result.pixel_count, pixels_in_workgroup);
        }
    }
}
