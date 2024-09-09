## Fish Speech Candle implementation

A port of [Fish Speech 1.2](https://github.com/fishaudio/fish-speech) to [Candle.rs](https://github.com/huggingface/candle).

## Roadmap

- [ ] Implement core Fish Speech 1.2 architecture in Candle.rs
    - [x] FireflyArchitecture model for generating speaker conditioning tokens
        - [x] ConvNext
        - [X] GroupedResidualFSQ
    - [ ] DualAR Transformer backbone
    - [ ] HifiGAN decoding
- [ ] Nvidia and Metal GPU support using Candle
- [ ] Python bindings
- [ ] Pure Rust utilities
    - [ ] Audio preprocessing to log mel spectrogram
    - [ ] CLI
    - [ ] OpenAI-compatible TTS server with Axum