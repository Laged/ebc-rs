use crate::render::EventRenderPlugin;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::{prelude::*, window::WindowResolution};
use bevy_framepace::{FramepacePlugin, FramepaceSettings, Limiter};

pub struct CorePlugins;

impl Plugin for CorePlugins {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Event-Based Camera Visualizer".into(),
                        resolution: WindowResolution::new(1280, 720),
                        present_mode: bevy::window::PresentMode::Immediate,
                        ..default()
                    }),
                    ..default()
                })
                .set(bevy::render::RenderPlugin {
                    render_creation: bevy::render::settings::RenderCreation::Automatic(
                        bevy::render::settings::WgpuSettings {
                            power_preference:
                                bevy::render::settings::PowerPreference::HighPerformance,
                            ..default()
                        },
                    ),
                    ..default()
                }),
        )
        .add_plugins(FramepacePlugin)
        .insert_resource(FramepaceSettings {
            limiter: Limiter::Off,
            ..default()
        })
        .add_plugins(EventRenderPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin::default());
    }
}
