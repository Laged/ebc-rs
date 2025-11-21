use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuEvent {
    pub timestamp: u32,
    pub x: u32,
    pub y: u32,
    pub polarity: u32,
}
