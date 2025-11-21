//! # Data Loader
//!
//! This module handles loading event data from `.dat` files.
//!
//! ## File Format
//! The `.dat` file format consists of:
//! 1.  **Header**: Text lines starting with `%`.
//! 2.  **Event Type & Size**: 2 bytes immediately following the header.
//!     *   Byte 0: Event Type (ignored)
//!     *   Byte 1: Event Size (must be 8 bytes)
//! 3.  **Binary Data**: Sequence of 8-byte events.
//!
//! ### Event Structure (8 bytes)
//! *   **Timestamp (t32)**: 4 bytes (u32, little-endian) - Microseconds.
//! *   **Data (w32)**: 4 bytes (u32, little-endian).
//!     *   Bits 0-13: X coordinate (14 bits)
//!     *   Bits 14-27: Y coordinate (14 bits)
//!     *   Bits 28-31: Polarity (4 bits)

use crate::gpu::GpuEvent;
use anyhow::{Context, Result};
use memmap2::MmapOptions;
use std::fs::File;
use std::path::Path;

pub struct DatLoader;

impl DatLoader {
    pub fn load(path: impl AsRef<Path>) -> Result<Vec<GpuEvent>> {
        let path = path.as_ref();
        let file = File::open(path).context("Failed to open .dat file")?;

        // Read the whole file into mmap
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let mut events = Vec::new();
        let mut offset = 0;

        // Skip header lines starting with %
        while offset < mmap.len() {
            if mmap[offset] == b'%' {
                // find newline
                while offset < mmap.len() && mmap[offset] != b'\n' {
                    offset += 1;
                }
                offset += 1; // skip \n
            } else {
                break;
            }
        }

        // Check event type and size
        if offset + 2 > mmap.len() {
            return Err(anyhow::anyhow!("File too short"));
        }

        let _event_type = mmap[offset];
        let event_size = mmap[offset + 1];
        offset += 2;

        if event_size != 8 {
            return Err(anyhow::anyhow!("Unsupported event size: {}", event_size));
        }

        // Read events
        while offset + 8 <= mmap.len() {
            let chunk = &mmap[offset..offset + 8];
            // t32 is first 4 bytes (little endian)
            let t32 = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
            // w32 is next 4 bytes
            let w32 = u32::from_le_bytes(chunk[4..8].try_into().unwrap());

            // Decode w32
            // x: bits 0-13 (14 bits)
            // y: bits 14-27 (14 bits)
            // p: bits 28-31 (4 bits)
            let x = w32 & 0x3FFF;
            let y = (w32 >> 14) & 0x3FFF;
            let polarity = (w32 >> 28) & 0xF;

            events.push(GpuEvent {
                timestamp: t32,
                x,
                y,
                polarity,
            });

            offset += 8;
        }

        // Sort by timestamp
        events.sort_by_key(|e| e.timestamp);

        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_dat_loader_parsing() -> anyhow::Result<()> {
        // Create a dummy .dat file
        let mut file = tempfile::NamedTempFile::new()?;

        // Header
        writeln!(file, "% This is a header")?;
        writeln!(file, "% Another header line")?;

        // Event type (0) and size (8)
        file.write_all(&[0x00, 0x08])?;

        // Event 1: t=100, x=10, y=20, p=1
        // w32 = 10 | (20 << 14) | (1 << 28)
        // w32 = 10 | 327680 | 268435456 = 268763146
        let t1: u32 = 100;
        let w1: u32 = 10 | (20 << 14) | (1 << 28);
        file.write_all(&t1.to_le_bytes())?;
        file.write_all(&w1.to_le_bytes())?;

        // Event 2: t=50, x=5, y=5, p=0 (should be sorted first)
        let t2: u32 = 50;
        let w2: u32 = 5 | (5 << 14) | (0 << 28);
        file.write_all(&t2.to_le_bytes())?;
        file.write_all(&w2.to_le_bytes())?;

        let path = file.path().to_str().unwrap();
        let events = DatLoader::load(path)?;

        assert_eq!(events.len(), 2);

        // Check sorting
        assert_eq!(events[0].timestamp, 50);
        assert_eq!(events[0].x, 5);
        assert_eq!(events[0].y, 5);
        assert_eq!(events[0].polarity, 0);

        assert_eq!(events[1].timestamp, 100);
        assert_eq!(events[1].x, 10);
        assert_eq!(events[1].y, 20);
        assert_eq!(events[1].polarity, 1);

        Ok(())
    }
}
