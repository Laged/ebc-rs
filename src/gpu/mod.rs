pub mod accumulation;
pub mod canny;
pub mod log;
pub mod sobel;
pub mod preprocess;
pub mod resources;
pub mod types;
pub mod readback;

// Re-export commonly used items
pub use accumulation::{
    EventAccumulationNode, EventBindGroup, EventComputePipeline, EventLabel, prepare_events,
    queue_bind_group,
};
pub use canny::{
    CannyBindGroup, CannyLabel, CannyNode, CannyParamsBuffer, CannyPipeline,
    prepare_canny,
};
pub use log::{
    LogBindGroup, LogLabel, LogNode, LogParamsBuffer, LogPipeline,
    prepare_log,
};
pub use sobel::{
    EdgeParamsBuffer, SobelBindGroup, SobelLabel, SobelNode, SobelPipeline,
    prepare_sobel,
};
pub use preprocess::{
    PreprocessLabel, PreprocessNode, PreprocessPipeline, PreprocessBindGroup,
    prepare_preprocess,
};
pub use resources::{ActiveDetector, CannyImage, EdgeParams, EdgeReadbackBuffer, EventData, FilteredSurfaceImage, GpuEventBuffer, GroundTruthImage, LogImage, SobelImage, SurfaceImage};
pub use types::{GpuParams, GpuEdgeParams, GpuEvent};
pub use readback::{ReadbackLabel, ReadbackNode, prepare_readback, read_readback_result};
