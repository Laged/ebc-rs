pub mod composite;
pub mod config;
pub mod multi_readback;
pub mod ui;

pub use composite::{CompositeImage, CompositeLabel, CompositeNode, CompositePipeline, CompositeBindGroup, prepare_composite};
pub use config::CompareConfig;
pub use multi_readback::{AllDetectorMetrics, DetectorMetrics, MetricsSender, MetricsReceiver, AllEdgeData, DetectorEdgeData, receive_metrics};
pub use ui::{CompareUiPlugin, DataFileState, draw_metrics_overlay, handle_file_input};
