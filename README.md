# Event-Based Camera Visualizer (ebc-rs)

<img src="fan-demo.gif" width="100%" alt="Fan Demo">

## Project Goal
This project is a high-performance visualizer for event-based camera data, built with Rust and Bevy. It is designed to efficiently load, process, and render millions of events using a GPU-accelerated pipeline.

## Data Format (.dat)
The application expects data in a custom binary format:

1.  **Header**: Text lines starting with `%`.
2.  **Event Type & Size**: 2 bytes (Type + Size=8).
3.  **Binary Data**: Sequence of 8-byte events.

### Event Structure (8 bytes)
*   **Timestamp (4 bytes)**: `u32` (microseconds).
*   **Data (4 bytes)**: `u32` containing packed fields:
    *   **X**: Bits 0-13 (14 bits)
    *   **Y**: Bits 14-27 (14 bits)
    *   **Polarity**: Bits 28-31 (4 bits)

## Rendering Pipeline
The visualization pipeline leverages WGPU compute shaders for high performance:

1.  **Binary Load**: `DatLoader` reads the `.dat` file and parses events into `GpuEvent` structs.
2.  **GPU Upload**: Events are uploaded to a `StorageBuffer` on the GPU (`GpuEventBuffer`).
3.  **Compute Shader**:
    *   A compute shader (`accumulation.wgsl`) processes the events.
    *   It iterates through the event buffer.
    *   Events falling within the current time window are accumulated onto a `SurfaceBuffer`.
4.  **Texture Copy**: The `SurfaceBuffer` is copied to a `GpuImage` texture.
5.  **Visualization**: A custom material (`EventMaterial`) renders the texture onto a quad, applying color mapping and decay effects based on the accumulated values.

## Controls
*   **Play/Pause**: Toggle playback.
*   **Loop**: Toggle looping.
*   **Time Slider**: Scrub through the dataset.
*   **Window Slider**: Adjust the integration time window (accumulation duration).
*   **Speed Slider**: Adjust playback speed.
