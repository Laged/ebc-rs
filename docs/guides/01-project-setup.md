# Project Setup: Zero to Hero

This guide explains how to set up the `ebc-rs` development environment from scratch. We use **Nix** to provide a hermetic, reproducible environment that includes the Rust toolchain and all necessary system libraries (Vulkan, Wayland, X11, etc.).

## 1. The Nix Advantage

The `flake.nix` file is the heart of our environment. It ensures that every developer (and CI machine) uses the exact same versions of tools and libraries.

### Key Components of `flake.nix`

- **Inputs**: We pin `nixpkgs` (system packages) and `fenix` (Rust toolchain) to specific commits/versions.
- **`devShells`**: Defines the interactive shell environment.
  - **`rustToolchain`**: Includes `rustc`, `cargo`, `clippy`, `rustfmt`, and `rust-src`.
  - **System Libraries**: Automatically sets `LD_LIBRARY_PATH` and `PKG_CONFIG_PATH` for libraries like `vulkan-loader`, `wayland`, `alsa-lib`, and `udev`. This eliminates "missing shared library" errors common in graphics programming.
- **`apps`**: Defines runnable commands like `cargo run --bin generate_synthetic` as `nix run .#generate_data`.

### Getting Started

1.  **Install Nix**: Follow the instructions at [nixos.org](https://nixos.org/download.html).
2.  **Enable Flakes**: Ensure `experimental-features = nix-command flakes` is in your `/etc/nix/nix.conf`.
3.  **Enter the Shell**:
    ```bash
    nix develop
    ```
    This drops you into a bash shell with `cargo`, `rustc`, and all system deps available.

## 2. Dependency Management (`Cargo.toml`)

Our `Cargo.toml` defines the Rust-level dependencies.

### Core Dependencies

-   **`bevy = "0.17"`**: The game engine. We use the latest version to leverage modern rendering features.
    -   `features = ["file_watcher"]`: Enables hot-reloading of assets.
-   **`bytemuck`**: Critical for GPU interop. It allows us to safely cast Rust structs (like `Vec3`, `Color`) into byte slices that can be uploaded to GPU buffers.
-   **`wgpu`**: (Implicitly via Bevy) The underlying graphics API. We interact with it directly for custom compute shaders.

### Project-Specific

-   **`rayon`**: For parallel CPU processing (e.g., generating synthetic data).
-   **`memmap2`**: For efficient memory-mapped file I/O when reading large `.dat` event files.

## 3. Project Structure

The codebase is organized into a library and several binaries.

### `src/lib.rs`
The core logic. It exposes modules like:
-   `loader`: For reading event data.
-   `gpu`: Contains the custom render graph nodes and compute pipelines.
-   `edge_detection`: The main plugin that ties everything together.

### `src/bin/`
Executable entry points for different workflows:
-   `compare_live.rs`: Runs the main visualization and comparison tool.
-   `generate_synthetic.rs`: Generates `.dat` files for testing.
-   `hypersearch.rs`: Runs optimization algorithms.

### `src/main.rs`
A minimal entry point that runs the default visualization.

```rust
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EdgeDetectionPlugin) // Our custom plugin
        .run();
}
```

## 4. The "Plugin" Philosophy

Bevy is built on the Entity Component System (ECS) and Plugins. We follow this pattern:

1.  **Define a Plugin**: Create a struct implementing `Plugin`.
2.  **Register Resources**: Add global data (e.g., `EventFilePath`, `GpuParams`).
3.  **Add Systems**: Functions that run every frame (e.g., `update_camera`, `process_events`).
4.  **Configure Rendering**: We extend the `RenderApp` (a sub-app in Bevy) to add custom render graph nodes.

This modularity allows us to easily swap out components (e.g., switching from `SpinningTrianglePlugin` to `BoidsPlugin`) without rewriting the core engine setup.
