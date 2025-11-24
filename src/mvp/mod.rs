pub mod gpu;
pub mod playback;
pub mod render;

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;

pub use gpu::{EventData, SurfaceImage, GradientImage, EdgeParams};
pub use playback::PlaybackState;

pub struct MvpPlugin;

impl Plugin for MvpPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<EventData>::default())
            .add_plugins(ExtractResourcePlugin::<SurfaceImage>::default())
            .add_plugins(ExtractResourcePlugin::<GradientImage>::default())
            .add_plugins(ExtractResourcePlugin::<PlaybackState>::default())
            .add_plugins(ExtractResourcePlugin::<EdgeParams>::default())
            .add_plugins(render::EventRenderPlugin)
            .init_resource::<SurfaceImage>()
            .init_resource::<GradientImage>()
            .init_resource::<PlaybackState>()
            .init_resource::<EdgeParams>();
    }
}
