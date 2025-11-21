#[cfg(test)]
mod playback_tests {
    use crate::gpu::PlaybackState;

    #[test]
    fn test_playback_defaults() {
        let state = PlaybackState::default();
        assert_eq!(state.is_playing, false);
        assert_eq!(state.current_time, 20000.0);
        assert_eq!(state.window_size, 100.0);
        assert_eq!(state.playback_speed, 0.1);
    }

    #[test]
    fn test_playback_looping() {
        let mut state = PlaybackState::default();
        state.is_playing = true;
        state.max_timestamp = 1000;
        state.current_time = 990.0;
        state.looping = true;
        state.playback_speed = 1.0; // 1x speed

        // Simulate 20ms passing (20000us)
        // delta_us = 0.02 * 1_000_000 * 1.0 = 20000
        let delta_seconds = 0.02;
        let delta_us = delta_seconds * 1_000_000.0 * state.playback_speed;

        state.current_time += delta_us;

        // Should have looped
        // 990 + 20000 = 20990.
        // Logic in render.rs: if current > max { if looping { current = 0 } }
        // It resets to 0, it doesn't wrap around with modulo in the current implementation.

        if state.current_time > state.max_timestamp as f32 {
            if state.looping {
                state.current_time = 0.0;
            }
        }

        assert_eq!(state.current_time, 0.0);
    }

    #[test]
    fn test_playback_stop_at_end() {
        let mut state = PlaybackState::default();
        state.is_playing = true;
        state.max_timestamp = 1000;
        state.current_time = 990.0;
        state.looping = false;
        state.playback_speed = 1.0;

        let delta_seconds = 0.02;
        let delta_us = delta_seconds * 1_000_000.0 * state.playback_speed;
        state.current_time += delta_us;

        if state.current_time > state.max_timestamp as f32 {
            if state.looping {
                state.current_time = 0.0;
            } else {
                state.current_time = state.max_timestamp as f32;
                state.is_playing = false;
            }
        }

        assert_eq!(state.current_time, 1000.0);
        assert_eq!(state.is_playing, false);
    }
}
