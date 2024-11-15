use anyhow::Result;
use async_stream::try_stream;
use candle_core::{DType, Device, Tensor};
use fish_speech_core::models::vqgan::decoder::FireflyDecoder;
use futures_util::Stream;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;

const WINDOW_SIZE: usize = 10; // ~300ms of audio
const OVERLAP_SIZE: usize = 4; // Previous context to maintain
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
        self.current_buffer.push(token);

        if self.current_buffer.len() >= HOP_SIZE {
            // Create full window by combining previous context and new tokens
            let mut window_tokens = Vec::with_capacity(WINDOW_SIZE);
            window_tokens.extend(self.previous_window.iter().cloned());
            window_tokens.extend(self.current_buffer.iter().cloned());

            // Create concatenated tensor for vocoder
            let window_tensor = match Tensor::cat(&window_tokens, 0) {
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

            match Tensor::cat(&final_tokens, 0) {
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
    I: Iterator<Item = Result<Tensor>>,
{
    let mut processor = WindowedTokenProcessor::new();

    try_stream! {
        for token_result in token_iter {
            let token = token_result?;
            if let Some(window_result) = processor.process_token(token) {
                let window = window_result?;

                // Process window through vocoder
                let feature_lengths = Tensor::from_slice(
                    &[window.dim(0)? as u32],
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
                &[final_window.dim(0)? as u32],
                1,
                &device,
            )?;

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
