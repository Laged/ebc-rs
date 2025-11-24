pub mod accumulation;
pub mod sobel;
pub mod resources;
pub mod types;

// Re-export commonly used items
pub use accumulation::{
    EventAccumulationNode, EventBindGroup, EventComputePipeline, EventLabel, prepare_events,
    queue_bind_group,
};
pub use sobel::{
    EdgeParamsBuffer, SobelBindGroup, SobelLabel, SobelNode, SobelPipeline,
    prepare_sobel,
};
pub use resources::{CannyImage, EdgeParams, EventData, GpuEventBuffer, LogImage, SobelImage, SurfaceImage};
pub use types::{GpuEdgeParams, GpuEvent};
