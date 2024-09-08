use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::conv::FromSample;

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

pub fn pcm_decode<P: AsRef<std::path::Path>>(
    path: P,
) -> anyhow::Result<(Vec<f32>, u32, usize, usize)> {
    let src = std::fs::File::open(path)?;
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());
    let hint = symphonia::core::probe::Hint::new();
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("unsupported codec");

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();

    while let Ok(packet) = format.next_packet() {
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

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

    // Normalize to (-1,1) using a numerically stable method
    // let epsilon = 1e-10;
    // let max_abs = pcm_data.iter().map(|&x| x.abs()).fold(0f32, f32::max);
    // if max_abs > epsilon {
    //     let scale = 1.0 / max_abs;
    //     pcm_data.iter_mut().for_each(|x| *x *= scale);
    // }

    let num_frames = pcm_data.len();

    Ok((pcm_data, sample_rate, num_frames, 1))
}
