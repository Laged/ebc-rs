pub mod composite;
pub mod config;
pub mod multi_readback;
pub mod plugin;
pub mod ui;

pub use composite::{CompositeImage, CompositeLabel, CompositeNode, CompositePipeline, CompositeBindGroup, prepare_composite};
pub use config::CompareConfig;
pub use multi_readback::{
    AllDetectorMetrics, DetectorMetrics, MetricsSender, MetricsReceiver, AllEdgeData, DetectorEdgeData,
    receive_metrics, setup_metrics_channel, PendingMetricsSender,
    MultiReadbackLabel, MultiReadbackBuffers, MultiReadbackNode,
    prepare_multi_readback, read_multi_readback,
};
pub use plugin::CompositeRenderPlugin;
pub use ui::{CompareUiPlugin, DataFileState, draw_metrics_overlay, handle_file_input};
