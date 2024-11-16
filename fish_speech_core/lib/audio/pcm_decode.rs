use std::io::Cursor;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::CODEC_TYPE_NULL;
use symphonia::core::conv::FromSample;
use symphonia::core::io::MediaSourceStream;

fn average_channels<T>(buffer: &symphonia::core::audio::AudioBuffer<T>) -> Vec<f32>
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    let num_channels = buffer.spec().channels.count();
    let num_frames = buffer.frames();
    let mut averaged_data = Vec::with_capacity(num_frames);

    for frame in 0..num_frames {
        let sum: f32 = (0..num_channels)
            .map(|ch| f32::from_sample(*buffer.chan(ch).get(frame).unwrap()))
            .sum();
        averaged_data.push(sum / num_channels as f32);
    }

    averaged_data
}
// Original file-based decoder
pub fn decode_audio_file<P: AsRef<std::path::Path>>(
    path: P,
) -> anyhow::Result<(Vec<f32>, u32, usize, usize)> {
    let src = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    decode_audio_inner(mss)
}

// New memory-based decoder for multipart
pub fn decode_audio_bytes(bytes: Vec<u8>) -> anyhow::Result<(Vec<f32>, u32, usize, usize)> {
    let cursor = Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    decode_audio_inner(mss)
}

// Shared decoding logic
fn decode_audio_inner(mss: MediaSourceStream) -> anyhow::Result<(Vec<f32>, u32, usize, usize)> {
    let probed = symphonia::default::get_probe().format(
        &Default::default(),
        mss,
        &Default::default(),
        &Default::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("no supported audio tracks"))?;

    let mut decoder =
        match symphonia::default::get_codecs().make(&track.codec_params, &Default::default()) {
            Ok(d) => d,
            Err(_) => return Err(anyhow::anyhow!("unsupported codec")),
        };

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let averaged = match decoded {
            AudioBufferRef::F32(buf) => average_channels(&buf),
            AudioBufferRef::U8(buf) => average_channels(&buf),
            AudioBufferRef::U16(buf) => average_channels(&buf),
            AudioBufferRef::U24(buf) => average_channels(&buf),
            AudioBufferRef::U32(buf) => average_channels(&buf),
            AudioBufferRef::S8(buf) => average_channels(&buf),
            AudioBufferRef::S16(buf) => average_channels(&buf),
            AudioBufferRef::S24(buf) => average_channels(&buf),
            AudioBufferRef::S32(buf) => average_channels(&buf),
            AudioBufferRef::F64(buf) => average_channels(&buf),
        };
        pcm_data.extend(averaged);
    }

    let num_frames = pcm_data.len();
    Ok((pcm_data, sample_rate, num_frames, 1))
}
