// Contrast Maximization: Find best omega and copy its IWE to output
// Single workgroup performs parallel reduction

struct SelectParams {
    n_omega: u32,
    width: u32,
    height: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> contrast: array<u32>;
@group(0) @binding(1) var<storage, read> iwe_buffer: array<u32>;
@group(0) @binding(2) var<uniform> params: SelectParams;
@group(0) @binding(3) var output_texture: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;  // [best_idx, best_contrast]

const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;

var<workgroup> shared_idx: array<u32, 128>;
var<workgroup> shared_val: array<u32, 128>;

@compute @workgroup_size(128)
fn find_best(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Each thread handles multiple omega values
    var local_best_idx = 0u;
    var local_best_val = 0u;

    var i = tid;
    while (i < params.n_omega) {
        let val = contrast[i];
        if (val > local_best_val) {
            local_best_val = val;
            local_best_idx = i;
        }
        i += 128u;
    }

    shared_idx[tid] = local_best_idx;
    shared_val[tid] = local_best_val;
    workgroupBarrier();

    // Parallel reduction
    var stride = 64u;
    while (stride > 0u) {
        if (tid < stride && tid + stride < 128u) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 writes result
    if (tid == 0u) {
        result[0] = shared_idx[0];
        result[1] = shared_val[0];
    }
}

@compute @workgroup_size(8, 8)
fn copy_best(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let x = gid.x;
    let y = gid.y;

    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }

    let best_idx = result[0];
    let iwe_idx = best_idx * (WIDTH * HEIGHT) + y * WIDTH + x;
    let value = f32(iwe_buffer[iwe_idx]);

    // Normalize for visualization (log scale for better contrast)
    let normalized = log2(value + 1.0) / 10.0;

    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(normalized, 0.0, 0.0, 1.0));
}
