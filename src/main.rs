use bevy::{prelude::*, window::WindowResolution};

mod gpu;
mod loader;
mod render;

fn main() {
    App::new()
        .add_plugins(
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
        .add_plugins(bevy_framepace::FramepacePlugin)
        .insert_resource(bevy_framepace::FramepaceSettings {
            limiter: bevy_framepace::Limiter::Off,
            ..default()
        })
        .add_plugins(render::EventRenderPlugin)
        .add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
        .run();
}
