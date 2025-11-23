
Detailed Implementation Plan: Real-Time Fan RPM & Border Tracking
Target Branch: fan-rpm Stack: Rust 1.91, Bevy 0.17, WGSL Goal: Transform ebc-rs from a raw event visualizer into an analytical tool capable of tracking high-speed rotating objects (fans) using Event-Based Vision techniques.
1. Project Overview & Objectives
The ebc-rs project visualizes asynchronous event streams from neuromorphic cameras. The current implementation renders events using temporal accumulation. This update will introduce a Physics-Aware Analysis Pipeline that solves for the angular velocity of a rotating fan in real-time.
Core Objectives
UI Enhancements: Add a "Motion Analysis" panel with an RPM counter and "Show Borders" toggle.
Real-Time RPM Calculation: Implement a robust algorithm to determine the fan's speed, resilient to noise and camera movement.
Border Visualization: Render the physical outline of the fan blades overlaying the event stream, synchronized with the calculated RPM.
Tip Velocity: Calculate and display the physical speed of the blade tips.
2. Research Approach: Motion-Compensated Contrast Maximization (CMax)
Standard frame-based optical flow fails at high RPMs due to motion blur. We will use Motion Compensation, utilizing the high temporal resolution of event cameras (microsecond precision).
The Algorithm
The core hypothesis is that correct motion parameters maximize the sharpness (contrast) of the accumulated image.
Centroid Tracking (Translation):
Before calculating rotation, we must know the center of the fan (C_x, C_y).
We use a compute shader to calculate the spatial mean of incoming events.
Why: This allows the fan to move across the screen without breaking the RPM calculation.
Warping (Rotation):
For a candidate angular velocity \omega (omega), we rotate every event (x, y, t) backward in time to a reference time t_{ref}.
Formula:
Optimization (RPM Solver):
We warp a batch of events using a test \omega.
We calculate the Variance of the resulting image.
High Variance = Sharp Image = Correct RPM.
Low Variance = Blurred Image = Wrong RPM.
We use a gradient ascent or Golden Section Search to find the optimal \omega.
3. Architecture & File Responsibilities
Modified Files
Cargo.toml: Ensure bevy_gizmos is included.
src/main.rs: Register the new AnalysisPlugin.
src/ui.rs: Add the new "Motion Analysis" UI section (RPM label, Toggles, Blade Count slider).
New Files
1. src/analysis.rs (The Brain)
Responsibilities:
Manages the FanAnalysis resource (state).
Controls the "Search Loop" (deciding which \omega to test next).
Dispatches the compute shaders.
Reads back the variance score from the GPU.
Key Structs:
#[derive(Resource)]
pub struct FanAnalysis {
    pub is_tracking: bool,
    pub show_borders: bool,
    pub current_rpm: f32,
    pub blade_count: u32, // Default: 3
    pub centroid: Vec2,
    pub tip_velocity: f32,
}


2. assets/shaders/centroid.wgsl (Compute Shader A)
Responsibilities:
Input: Raw Event Buffer.
Operation: Performs an atomic addition of all X and Y coordinates.
Output: sum_x, sum_y, count. Used to calculate the average position (Centroid).
3. assets/shaders/cmax_optimization.wgsl (Compute Shader B)
Responsibilities:
Input: Raw Event Buffer, Uniforms (centroid, test_omega).
Operation: Warps events based on test_omega and accumulates them onto a temporary grid. Calculates the grid's variance.
Output: A single f32 (Variance Score).
4. src/gizmos.rs (The Visualizer)
Responsibilities:
Reads FanAnalysis state.
Calculates the current rotation angle \theta_{now} = \theta_0 + \omega \times t.
Uses bevy_gizmos to draw:
A cross at the centroid.
N lines representing the blade borders radiating from the center.
4. Phased Implementation Steps
Phase 1: Foundation (UI & State)
Goal: Create the controls and data structures. No math yet.
Create src/analysis.rs and define the FanAnalysis resource.
Initialize AnalysisPlugin in main.rs.
Update src/ui.rs to display:
"Detected RPM": Bind to FanAnalysis.current_rpm.
"Show Borders": Bind to FanAnalysis.show_borders.
"Blade Count": Slider (0-8).
Deliverable: A UI that compiles and modifies the resource values.
Phase 2: Centroid Tracking
Goal: Automatically track the fan center.
Implement assets/shaders/centroid.wgsl. Note: Use atomic operations carefully. If atomicAdd for float is unavailable, scale floats to u32.
Create a system in analysis.rs to dispatch this shader every frame using bevy_render.
Read the buffer back to CPU to update FanAnalysis.centroid.
Deliverable: A red debug cross (gizmo) that follows the fan center as it moves.
Phase 3: The Physics Engine (RPM Solver)
Goal: Real-time RPM calculation.
Implement assets/shaders/cmax_optimization.wgsl (The Warper).
Implement the Optimizer Loop in Rust:
Step A: Guess \omega. Dispatch shader.
Step B: Guess \omega + \delta. Dispatch shader.
Step C: Compare variances. Update \omega in the direction of higher variance.
Convert \omega (rad/s) to RPM: RPM = (omega * 60) / (2 * PI).
Deliverable: The "Detected RPM" value in the UI stabilizes near the real fan speed.
Phase 4: Visualization
Goal: Render the blade borders.
Create src/gizmos.rs.
Implement draw_fan_borders system:
Get centroid, blade_count, and current_rpm.
Calculate current angle based on time.elapsed_seconds().
Draw lines using Gizmos::lines.
Calculate tip_velocity using the max radius of events.
Deliverable: Visual overlay of blades spinning in sync with the event data.
5. Technical Constraints
Rust Version: 1.91.0
Bevy Version: 0.17 (Use updated App syntax and SystemParam).
WGPU: Ensure compute shader compatibility (WGSL) for the target hardware.
Performance: All heavy loops (warping, accumulation) must remain on the GPU. The CPU should only handle the high-level optimization logic.
