# fish-speech.rs

## Initial setup

This repo requires a working Rust installation ([see official docs](https://www.rust-lang.org/tools/install)). Packaging for homebrew and Linux to come.

Unfortunately, we also still need a basic Python environment until Rust-native audio preprocessing is implemented. Let's set up our Python FFI for the speaker conditioning encoder:

```bash
python -m venv .venv
pip install -r ./fish_speech_python/requirements.txt
cd ./fish_speech_python && maturin develop
```

Save the Fish Speech checkpoints to `./checkpoints`. I recommend using [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli):

```bash
# If it's not already on system
brew install huggingface-cli

mkdir -p checkpoints/fish-speech-1.4
huggingface-cli download jkeisling/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

Note that we don't support the official `.pth` weights.

### System requirements

Nvidia GPU or Apple Silicon are highly recommended. CPU inference is supported as a fallback, but it's pretty slow. Please raise an issue if you want CPU accelerated.

## Usage

For now, we're keeping compatibility with the official Fish Speech inference CLI scripts. (Inference server and Python bindings coming soon!)

Generate speaker conditioning tokens (NOTE: this will be replaced with fully native Rust soon):

```bash
python fish_speech_python/encode_audio.py --output_path ./fake.npy ./tests/resources/sky.wav
```

NOTE: Fish 1.4 support will be added ASAP in the next PR. Until then,

- You can use `tests/resources/sky.npy` as an example conditioning prompt. - You can also create a prompt using the official repo, but be sure to convert the weights to `np.float32` beforehand as Candle can't handle integer npy files.

### Generate semantic codebook tokens

For Fish 1.4 (default):

```bash
# Switch to --features cuda for Nvidia GPUs
cargo run --release --features metal --bin llama_generate -- \
  --text "That is not dead which can eternal lie, and with strange aeons even death may die." \
  --prompt-text "When I heard the release demo, I was shocked, angered, and in disbelief that Mr. Altman would pursue a voice that sounded so eerily similar to mine that my closest friends and news outlets could not tell the difference." \
  --prompt-tokens fake.npy
```

For Fish 1.2, you'll have to specify version and checkpoint explicitly:

```bash
cargo run --release --features metal --bin llama_generate -- --text "That is not dead which can eternal lie, and with strange aeons even death may die." --fish-version 1.2 --checkpoint ./checkpoints/fish-speech-1.2-sft
```

### Decode tokens to WAV

For Fish 1.4 (default):

```bash
# Switch to --features cuda for Nvidia GPUs
cargo run --release --features metal --bin vocoder -- -i out.npy -o fake.wav
```

For Fish 1.2:

```bash
cargo run --release --bin vocoder -- --fish-version 1.2 --checkpoint ./checkpoints/fish-speech-1.2-sft
```

## License

> [!WARNING]
> This codebase is licensed under the original CC-BY-NC-SA-4.0 license. For non-commercial use only!
>
> Please support the original authors by using the [official API](https://fish.audio/go-api/) for production.

This model is permissively licensed under the BY-CC-NC-SA-4.0 license.
The source code is released under BSD-3-Clause license.

## Original README below

**Fish Speech V1.4** is a leading text-to-speech (TTS) model trained on 700k hours of audio data in multiple languages.

Supported languages:

- English (en) ~300k hours
- Chinese (zh) ~300k hours
- German (de) ~20k hours
- Japanese (ja) ~20k hours
- French (fr) ~20k hours
- Spanish (es) ~20k hours
- Korean (ko) ~20k hours
- Arabic (ar) ~20k hours

Please refer to [Fish Speech Github](https://github.com/fishaudio/fish-speech) for more info.  
Demo available at [Fish Audio](https://fish.audio/).

## Citation

If you found this repository useful, please consider citing this work:

```
@misc{fish-speech-v1.4,
  author = {Shijia Liao, Tianyu Li, etc},
  title = {Fish Speech V1.4},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fishaudio/fish-speech}}
}
```
