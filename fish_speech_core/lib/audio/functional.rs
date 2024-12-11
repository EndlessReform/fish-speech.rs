use candle_core::{Result, Tensor};

pub fn resample(pcm_data: &Tensor, from_rate: u32, to_rate: u32) -> Result<Tensor> {
    let device = pcm_data.device();
    let (num_channels, num_frames) = pcm_data.dims2()?;

    // Use f64 for ratio and length calculations for better precision with large numbers
    let resample_ratio = to_rate as f64 / from_rate as f64;
    let output_len = (num_frames as f64 * resample_ratio).ceil() as usize;

    // Create index tensors
    let input_indices = (Tensor::arange(0f64, output_len as f64, device)?
        // .to_dtype(candle_core::DType::F64)?
        / resample_ratio)?;
    let input_indices_floor = input_indices.floor()?.to_dtype(candle_core::DType::U32)?;
    let input_indices_ceil = input_indices
        .ceil()?
        .to_dtype(candle_core::DType::U32)?
        .minimum((num_frames - 1) as u32)?;

    // Calculate interpolation weights
    let t = input_indices
        .sub(&input_indices_floor.to_dtype(candle_core::DType::F64)?)?
        .to_dtype(candle_core::DType::F32)?;
    let one_minus_t = Tensor::ones(t.shape(), pcm_data.dtype(), device)?.sub(&t)?;

    // Gather values for interpolation
    let pcm_data_flat = pcm_data.flatten_all()?;
    let values_floor = pcm_data_flat.gather(&input_indices_floor, 0)?;
    let values_ceil = pcm_data_flat.gather(&input_indices_ceil, 0)?;

    // Perform linear interpolation
    let interpolated = values_floor.mul(&one_minus_t)?.add(&values_ceil.mul(&t)?)?;

    // Reshape to (num_channels, output_len)
    interpolated.reshape((num_channels, output_len))
}
