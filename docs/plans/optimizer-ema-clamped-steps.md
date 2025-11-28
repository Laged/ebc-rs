# CMax-SLAM Optimizer Fix: EMA + Clamped Steps

## Problem Statement

The CMax-SLAM optimizer oscillates wildly between 1.26e-4 and 1.79e-4 rad/μs instead of converging to the ground truth value of 1.2566e-4 rad/μs. This is caused by:

1. **Weak gradient signal**: Contrast values for center/plus/minus are within 2-5% of each other
2. **Noisy parabolic estimation**: Small differences amplify into large step sizes
3. **No dampening**: Each frame's update is independent, causing oscillation

## Selected Approach: EMA + Clamped Steps

Two complementary mechanisms to stabilize convergence:

### 1. Exponential Moving Average (EMA) on Omega Updates

Instead of directly applying `omega += step`, smooth updates:
```
omega_raw = omega + step
omega = alpha * omega_raw + (1 - alpha) * omega_prev
```

Where `alpha` controls responsiveness:
- Lower alpha (0.1) = more stable, slower tracking
- Higher alpha (0.3) = less stable, faster tracking
- Default: 0.2 (balance for 10-20 frame tracking)

### 2. Hard Step Clamping

Before applying any update, clamp the step magnitude:
```
max_step = omega * max_step_fraction  // 2% of current omega
clamped_step = step.clamp(-max_step, max_step)
```

This prevents:
- Runaway divergence from noisy gradients
- Large jumps from poorly conditioned parabolic fits
- Oscillation amplitude > 2% per frame

## Implementation Tasks

### Task 1: Add EMA fields to CmaxSlamState

**File**: `src/cmax_slam/resources.rs`

Add to `CmaxSlamState`:
```rust
// EMA smoothing
pub ema_alpha: f32,           // Smoothing factor (default 0.2)

// Step clamping
pub max_step_fraction: f32,   // Max step as fraction of omega (default 0.02)
pub last_raw_step: f32,       // For debugging/UI display
pub step_was_clamped: bool,   // Flag for UI
```

Note: `omega` itself is smoothed via EMA - no separate `omega_ema` needed.

Update `Default` impl with initial values.

### Task 2: Implement clamped step in receive_contrast_results

**File**: `src/cmax_slam/systems.rs`

In `receive_contrast_results`, after parabolic step calculation:

```rust
// 1. Calculate raw step (existing code)
let raw_step = if denominator.abs() > 1e-6 {
    -(v_p - v_m) / (2.0 * denominator) * current_delta
} else {
    params.learning_rate * gradient.signum() * current_delta
};

// 2. CLAMP step to ±max_step_fraction of omega
let max_step = state.omega.abs() * state.max_step_fraction;
let clamped_step = raw_step.clamp(-max_step, max_step);
state.last_raw_step = raw_step;
state.step_was_clamped = raw_step.abs() > max_step;

// 3. Apply clamped step
let omega_raw = state.omega + clamped_step;

// 4. EMA smooth the update
state.omega = state.ema_alpha * omega_raw
            + (1.0 - state.ema_alpha) * state.omega;
```

### Task 3: Update UI to display new debug fields

**File**: `src/compare/ui.rs`

Add display for:
- Raw step value
- Whether step was clamped
- EMA alpha value
- Current vs EMA omega (if different tracking needed)

### Task 4: Add EMA parameters to CmaxSlamParams for UI tuning

**File**: `src/cmax_slam/resources.rs`

Add to `CmaxSlamParams`:
```rust
pub ema_alpha: f32,           // Default 0.2
pub max_step_fraction: f32,   // Default 0.02
```

These can be exposed in UI for runtime tuning.

### Task 5: Integration test - verify convergence

Run compare_live with logging, verify:
1. Omega converges within 20 frames
2. Final omega within 5% of ground truth
3. No oscillation after convergence

### Task 6: Regression test - existing tests pass

Run full test suite to ensure no regressions.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ema_alpha` | 0.2 | EMA smoothing factor (0=frozen, 1=no smoothing) |
| `max_step_fraction` | 0.02 | Max step as fraction of omega (2%) |
| `convergence_threshold` | 0.001 | Variance threshold for convergence (0.1%) |

## Expected Behavior

1. **Cold start**: Initialize omega from GT with +10% offset
2. **Frames 1-10**: Rapid convergence toward GT (clamped steps)
3. **Frames 10-20**: Fine-tuning with EMA smoothing
4. **Steady state**: Omega within 1% of GT, low variance, `converged=true`

## Success Criteria

- [ ] Omega converges within 20 frames
- [ ] Final omega within 5% of ground truth
- [ ] No oscillation after convergence (variance < 0.1%)
- [ ] All regression tests pass
- [ ] Frame rate not significantly impacted
