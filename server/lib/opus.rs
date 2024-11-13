use anyhow::Result;
use ogg::PacketWriter;
use opus::{Application, Channels, Encoder};
// use rubato::{FftFixedInOut, Resampler};

pub struct OpusEncoder {
    encoder: Encoder,
    packet_writer: PacketWriter<'static, Vec<u8>>,
    frame_size: usize,
}

impl OpusEncoder {
    pub fn new() -> Result<Self> {
        // Create encoder - 24kHz mono optimized for voice
        let mut encoder = Encoder::new(24000, Channels::Mono, Application::Voip)?;

        // Set bitrate to 24kbps (good for voice)
        encoder.set_bitrate(opus::Bitrate::Bits(24000))?;

        // Frame size is 20ms of audio at 24kHz
        let frame_size = 480; // 24000 * 0.02

        // Initialize Ogg stream
        let mut packet_writer = PacketWriter::new(Vec::new());

        // Write proper Opus header (19 bytes total)
        let header = {
            let mut h = Vec::with_capacity(19);
            h.extend_from_slice(b"OpusHead"); // 8 bytes
            h.push(1); // Version (1 byte)
            h.push(1); // Channel count (1 byte)
            h.extend_from_slice(&3840u16.to_le_bytes()); // Pre-skip (2 bytes)
            h.extend_from_slice(&24000u32.to_le_bytes()); // Input sample rate (4 bytes)
            h.extend_from_slice(&0i16.to_le_bytes()); // Output gain (2 bytes)
            h.push(0); // Channel mapping family (1 byte)
            h
        };

        // Write the ID header
        packet_writer.write_packet(header, 0, ogg::PacketWriteEndInfo::EndPage, 0)?;

        // Write comment header
        let comments = {
            let mut c = Vec::new();
            c.extend_from_slice(b"OpusTags");
            c.extend_from_slice(&8u32.to_le_bytes()); // Vendor string length
            c.extend_from_slice(b"fish-tts"); // Vendor string
            c.extend_from_slice(&0u32.to_le_bytes()); // User comment list length
            c
        };

        packet_writer.write_packet(comments, 0, ogg::PacketWriteEndInfo::EndPage, 0)?;

        Ok(Self {
            encoder,
            packet_writer,
            frame_size,
        })
    }

    pub fn encode_pcm(&mut self, pcm_data: &[f32]) -> Result<Vec<u8>> {
        // Process in 20ms frames
        for chunk in pcm_data.chunks(self.frame_size) {
            // Pad last frame if needed
            let mut frame = chunk.to_vec();
            if frame.len() < self.frame_size {
                frame.resize(self.frame_size, 0.0);
            }

            // Encode frame
            let mut opus_frame = vec![0u8; 1275]; // Max size of an Opus frame
            let encoded_len = self.encoder.encode_float(&frame, &mut opus_frame)?;

            // Write to Ogg stream if we got data
            if encoded_len > 0 {
                // Clone the encoded data to avoid lifetime issues
                let encoded_data = opus_frame[..encoded_len].to_vec();
                self.packet_writer.write_packet(
                    encoded_data,
                    0,
                    ogg::PacketWriteEndInfo::EndPage,
                    frame.len() as u64,
                )?;
            }
        }

        // Get the encoded data
        let output = self.packet_writer.inner_mut().clone();
        self.packet_writer.inner_mut().clear();

        Ok(output)
    }
}
