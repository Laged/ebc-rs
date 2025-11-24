# Angular Histogram Pipeline Diagnostics

**Date:** 2025-11-24
**Task:** Phase 1: Task 1.1 - Add Diagnostic Instrumentation
**Goal:** Investigate why blade angles aren't being detected

---

## Diagnostic Output

### System Information
- **Events loaded:** 26,439,977 events
- **Timestamp range:** 4096 to 9,508,095 (span: 9,503,999 units = 9.504 seconds)
- **GPU:** NVIDIA GeForce GTX 1080 Ti
- **OS:** Linux (NixOS 25.11)

### Key Diagnostic Logs

#### 1. Shader Execution (AngularHistogramNode::run)
```
[DIAG] AngularHistogramNode::run() called
[DIAG] Total events in buffer: 26439977
[DIAG] Dispatching: x_groups=65535, y_groups=7, total_threads=29359680
```

**Status:** ✅ Shader IS running
- Called consistently every frame
- Total events correctly identified
- Dispatch parameters are using x_groups=65535 (max dimension)

#### 2. Angular Parameters (prepare_angular_bind_group)
```
[DIAG] Angular params: centroid=(640.0, 360.0), radius=30.0, window=[9507995, 9508095] (span=100μs)
[DIAG] Angular params: centroid=(638.5, 358.0), radius=30.0, window=[9507995, 9508095] (span=100μs)
[DIAG] Angular params: centroid=(637.0, 356.1), radius=30.0, window=[9507995, 9508095] (span=100μs)
```

**Status:** ⚠️ TIME WINDOW TOO SMALL
- Centroid position is reasonable (near center of 1280x720 image)
- **CRITICAL ISSUE: Time window is only 100μs (0.0001 seconds)**
- Total data span is 9,503,999μs (9.504 seconds)
- Window captures only 0.001% of total time range!

#### 3. Result Readback (read_angular_result_render)
```
[DIAG] Angular result buffer mapped successfully
[DIAG] Angular histogram received: 0 total events across 360 bins
[DIAG] Top 5 bins: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]

[DIAG] Angular result buffer mapped successfully
[DIAG] Angular histogram received: 251 total events across 360 bins
[DIAG] Top 5 bins: [(206, 8), (239, 8), (212, 7), (214, 7), (220, 7)]

[DIAG] Angular result buffer mapped successfully
[DIAG] Angular histogram received: 247 total events across 360 bins
[DIAG] Top 5 bins: [(214, 9), (205, 8), (211, 7), (220, 7), (222, 7)]

[DIAG] Angular result buffer mapped successfully
[DIAG] Angular histogram received: 248 total events across 360 bins
[DIAG] Top 5 bins: [(212, 9), (206, 7), (240, 7), (203, 6), (214, 6)]
```

**Status:** ⚠️ SHADER RETURNS VERY FEW EVENTS
- First readback: 0 events (window likely empty)
- Subsequent readbacks: 247-251 events
- Events are concentrated around bins 204-240 (approximately 204-240 degrees)
- Peak bin counts are 6-9 events (very low!)

---

## Pattern Matched

**Pattern: TIME WINDOW WRONG**

From the plan's expected patterns, this matches:
```
If time window wrong:
[DIAG] Angular params: window=[0, 0]  // Empty window
// OR
[DIAG] Angular histogram received: 0 total events  // No events in range
```

Our case shows:
1. ✅ Shader is running (not Pattern: "shader not running")
2. ✅ Dispatch is correct (x_groups=65535, not Pattern: "dispatch wrong")
3. ❌ **Time window is 100μs when it should be much larger** (Pattern: "time window wrong")
4. ⚠️ Shader returns very few events (251 avg) from 26M total events

---

## Root Cause Analysis

### Primary Issue: Time Window Too Small

**Current behavior:**
```rust
window=[9507995, 9508095] (span=100μs)
```

**Problem:**
- The time window captures only the last 100 microseconds of data
- This is 0.001% of the total 9.504-second recording
- With 26,439,977 events over 9.5 seconds, average event rate is ~2.78M events/second
- Expected events in 100μs: 2,780,000 * 0.0001 = **278 events** ✓ (matches observed 247-251)

**Why this causes missing blades:**
- The histogram only sees ~250 events in a tiny time slice
- Not enough data to detect meaningful angular patterns
- Blade detection requires analyzing sustained patterns over time
- Current window is too narrow to capture even one complete blade rotation

### Secondary Observations

**Peak Detection:**
- With only 250 events spread across 360 bins, peak bins have 6-9 events
- MIN_PEAK_HEIGHT threshold likely too high for such sparse data
- Even if peaks are detected, they're not statistically significant

**Dispatch Calculation:**
- ✅ Dispatch is working correctly now: x_groups=65535, y_groups=7
- This should process all 26M events if they're in the time window

---

## Hypothesis: Window Size Configuration

**Looking at the code pattern:**
```rust
let window_end = playback_state.current_time as u32;
let window_start = if window_end > playback_state.window_size as u32 {
    window_end - playback_state.window_size as u32
} else {
    0
};
```

**Two possible causes:**

1. **`playback_state.window_size` is set to 100 instead of a larger value**
   - Should be 1,000,000 for 1-second window
   - Or larger (e.g., 100,000 for 100ms minimum)

2. **Initial window is centered at end of data**
   - Window is at [9507995, 9508095] - the very last 100μs
   - If playback starts at end, this explains why first readback has 0 events
   - Subsequent frames have 247-251 as centroid stabilizes

---

## Next Investigation Steps

### Task 1.2: Find Window Size Configuration

**Files to check:**
- Search for `window_size` initialization in codebase
- Find where `PlaybackState` is created
- Check for constants like `TIME_WINDOW_MS` or similar

**Expected findings:**
- Likely a constant set to 100 (microseconds) when it should be 1,000,000 (1 second)
- Or a slider/UI value defaulting too small

### Task 1.3: Verify Shader Logic

**Once window is fixed:**
- Re-run diagnostics
- Should see 100,000+ events in histogram (not 250)
- Bin peaks should be 1000+ events
- Peak detection should find 3-4 significant peaks

---

## Verification Checklist

✅ **Diagnostics added correctly:**
- [x] AngularHistogramNode::run() logging
- [x] prepare_angular_bind_group() logging
- [x] read_angular_result_render() logging
- [x] All use [DIAG] prefix as specified

✅ **Pattern identification:**
- [x] Pattern matched: "time window wrong"
- [x] Evidence documented
- [x] Root cause hypothesis formed

❌ **Issues NOT found:**
- [ ] Shader not running (it IS running)
- [ ] Dispatch incorrect (it's correct: 65535 x 7)
- [ ] Shader returns zeros (it returns ~250 events, not zero)

---

## Summary

**Root Cause:** Time window is only 100μs, capturing ~250 events instead of the thousands needed for reliable blade detection.

**Impact:**
- Histogram is statistically insignificant (too sparse)
- Peak detection cannot find meaningful patterns
- Blade angles remain empty, visualization shows no blades

**Next Action:**
- Find and fix `window_size` configuration
- Should be 1,000,000μs (1 second) minimum
- Re-run diagnostics to verify events increase to expected levels

**Pattern from Plan:** Matches "time window wrong" pattern exactly.
