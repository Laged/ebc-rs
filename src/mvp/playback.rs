use bevy::{
    prelude::*,
    render::extract_resource::ExtractResource,
};

#[derive(Resource, ExtractResource, Clone)]
pub struct PlaybackState {
    pub is_playing: bool,
    pub current_time: f32,   // Microseconds
    pub window_size: f32,    // Microseconds
    pub playback_speed: f32, // Real-time multiplier
    pub looping: bool,
    pub max_timestamp: u32,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            is_playing: true,
            current_time: 20000.0,
            window_size: 1_000_000.0,  // 1 second
            playback_speed: 0.1,        // 0.1x speed
            looping: true,
            max_timestamp: 1_000_000,
        }
    }
}

pub fn playback_system(time: Res<Time>, mut playback_state: ResMut<PlaybackState>) {
    if playback_state.is_playing {
        let delta_us = time.delta_secs() * 1_000_000.0 * playback_state.playback_speed;
        playback_state.current_time += delta_us;

        if playback_state.current_time > playback_state.max_timestamp as f32 {
            if playback_state.looping {
                playback_state.current_time = 0.0;
            } else {
                playback_state.current_time = playback_state.max_timestamp as f32;
                playback_state.is_playing = false;
            }
        }
    }
}
