pub mod accumulation;
pub mod gradient;
pub mod resources;
pub mod types;

// Re-export commonly used items
pub use accumulation::{
    EventAccumulationNode, EventBindGroup, EventComputePipeline, EventLabel, prepare_events,
    queue_bind_group,
};
pub use gradient::{
    EdgeParamsBuffer, GradientBindGroup, GradientLabel, GradientNode, GradientPipeline,
    prepare_gradient,
};
pub use resources::{EdgeParams, EventData, GpuEventBuffer, GradientImage, SurfaceImage};
pub use types::{GpuEdgeParams, GpuEvent};
