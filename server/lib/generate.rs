use anyhow::Result;
use async_stream::try_stream;
use candle_core::{DType, Device, Tensor, D};
use fish_speech_core::models::vqgan::decoder::FireflyDecoder;
use futures_util::Stream;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;

const WINDOW_SIZE: usize = 20; // ~300ms of audio
const OVERLAP_SIZE: usize = 0; // Previous context to maintain
const HOP_SIZE: usize = WINDOW_SIZE - OVERLAP_SIZE;

struct WindowedTokenProcessor {
    previous_window: VecDeque<Tensor>,
    current_buffer: Vec<Tensor>,
}

impl WindowedTokenProcessor {
    pub fn new() -> Self {
        Self {
            previous_window: VecDeque::with_capacity(OVERLAP_SIZE),
            current_buffer: Vec::with_capacity(WINDOW_SIZE),
        }
    }

    /// Process tokens as they arrive, yielding complete windows when ready
    pub fn process_token(&mut self, token: Tensor) -> Option<Result<Tensor>> {
        self.current_buffer.push(
            token
                .broadcast_sub(&Tensor::ones_like(&token).unwrap())
                .unwrap(),
        );

        if self.current_buffer.len() >= HOP_SIZE {
            println!("Create chunk");
            // Create full window by combining previous context and new tokens
            let mut window_tokens = Vec::with_capacity(WINDOW_SIZE);
            window_tokens.extend(self.previous_window.iter().cloned());
            window_tokens.extend(self.current_buffer.iter().cloned());

            // Create concatenated tensor for vocoder
            let window_tensor = match Tensor::cat(&window_tokens, D::Minus1) {
                Ok(t) => t,
                Err(e) => {
                    return Some(Err(anyhow::anyhow!(
                        "Failed to concatenate window tokens: {}",
                        e
                    )))
                }
            };

            // Update previous window with the overlap portion
            self.previous_window.clear();
            self.previous_window.extend(
                self.current_buffer
                    .iter()
                    .skip(self.current_buffer.len().saturating_sub(OVERLAP_SIZE))
                    .cloned(),
            );

            // Clear current buffer
            self.current_buffer.clear();
            println!("Sending window of size {:?}", window_tensor.shape());

            Some(Ok(window_tensor))
        } else {
            None
        }
    }

    /// Flush any remaining tokens when the iterator is done
    pub fn flush(&mut self) -> Option<Result<Tensor>> {
        if !self.current_buffer.is_empty() {
            let mut final_tokens = Vec::new();
            final_tokens.extend(self.previous_window.iter().cloned());
            final_tokens.extend(self.current_buffer.iter().cloned());
            println!("flushing window of size {:?}", final_tokens.len());

            match Tensor::cat(&final_tokens, D::Minus1) {
                Ok(t) => Some(Ok(t)),
                Err(e) => Some(Err(anyhow::anyhow!(
                    "Failed to concatenate final tokens: {}",
                    e
                ))),
            }
        } else {
            None
        }
    }
}

/// Process an iterator of tokens using windowing
pub fn process_token_stream<I>(
    token_iter: I,
    device: Device,
    vocoder: Arc<Mutex<FireflyDecoder>>,
) -> impl Stream<Item = Result<Vec<f32>>>
where
    I: Iterator<Item = candle_core::Result<Tensor>>,
{
    let mut processor = WindowedTokenProcessor::new();

    try_stream! {
        for token_result in token_iter {
            let token = token_result?;
            if let Some(window_result) = processor.process_token(token) {
                let window = window_result?;

                // Process window through vocoder
                let feature_lengths = Tensor::from_slice(
                    &[window.dim(D::Minus1)? as u32],
                    1,
                    &device,
                )?;

                let vocoder = vocoder.lock().await;
                let audio = vocoder.decode(
                    &window.unsqueeze(0)?,
                    &feature_lengths,
                )?
                .to_dtype(DType::F32)?
                .squeeze(0)?
                .squeeze(0)?
                .to_vec1()?;

                yield audio;
            }
        }

        // Handle any remaining tokens
        if let Some(final_window_result) = processor.flush() {
            let final_window = final_window_result?;
            let feature_lengths = Tensor::from_slice(
                &[final_window.dim(D::Minus1)? as u32],
                1,
                &device,
            )?;

            println!("Final window: {:?}, feature lengths: {:?}", final_window.to_vec2::<u32>().unwrap(), feature_lengths);
            let vocoder = vocoder.lock().await;
            let audio = vocoder.decode(
                &final_window.unsqueeze(0)?,
                &feature_lengths,
            )?
            .to_dtype(DType::F32)?
            .squeeze(0)?
            .squeeze(0)?
            .to_vec1()?;

            yield audio;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_tokens() -> Result<()> {
        // First, create a small sequence of semantic tokens similar to what the model would output
        let mut processor = WindowedTokenProcessor::new();

        // Create dummy semantic tokens - these should match what your text2semantic model outputs
        // Let's create values that would make it obvious if they're being processed wrong
        let input_tokens: Vec<Tensor> = (0..34)
            .map(|i| {
                // Each semantic token should be shape [codebook_size]
                // Using distinctive values to track ordering
                Tensor::new(&[i as f32; 8], &Device::Cpu).unwrap()
            })
            .collect();

        println!("Input token shapes:");
        for (i, t) in input_tokens.iter().enumerate() {
            println!(
                "Token {}: shape {:?}, values: {:?}",
                i,
                t.shape(),
                t.to_vec1::<f32>()?
            );
        }

        // Now process through windowing
        let mut windows = Vec::new();
        for tensor in input_tokens.iter().cloned() {
            if let Some(window_result) = processor.process_token(tensor) {
                let window = window_result?;
                println!("\nWindow shape: {:?}", window.shape());
                println!("Window values: {:?}", window.to_vec1::<f32>()?);
                windows.push(window);
            }
        }

        if let Some(final_window) = processor.flush() {
            let window = final_window?;
            println!("\nFinal window shape: {:?}", window.shape());
            println!("Final window values: {:?}", window.to_vec1::<f32>()?);
            windows.push(window);
        }

        // Now let's simulate what the blocking version does
        let all_tokens_tensor = Tensor::cat(&input_tokens, D::Minus1)?;
        println!(
            "\nBlocking version full tensor shape: {:?}",
            all_tokens_tensor.shape()
        );
        println!(
            "Blocking version values: {:?}",
            all_tokens_tensor.to_vec1::<f32>()?
        );

        Ok(())
    }
}
