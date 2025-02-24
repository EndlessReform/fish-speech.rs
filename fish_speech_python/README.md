# fish_speech_python

Python bindings for the Fish Speech Candle implementation, using PyO3.

## Supported Platforms

> [!WARNING]
> Read this list very carefully. Hardware support is very limited.
> If you try to use this library on unsupported hardware, it will probably not work.
>
> You have been warned.

Python: 3.9+.

OS + hardware:
- Linux:
  - CPU: x86_64, glibc 2.34+
    - Example: Ubuntu 22.04 IS supported, Ubuntu 20.04 IS NOT supported
  - GPU: Nvidia CUDA 11.8+ with compute capability >= 8.0 (RTX 30 series+, A100 series+)
    - Example: 2080 Ti is NOT supported (Turing)
    - Example: RX5700 XT is NOT supported (AMD)
    - NOTE: We don't currently have a CUDA build matrix, so it's compiled with CUDA 11.8; sorry. It should be compatible with newer CUDA versions, but please use the Rust runtime if full optimization is required.
- macOS (M1+, 14.0+ (Monterey))

Windows and AMD hardware will never be supported, so don't ask.
Feel free to raise an issue if you need ARM or Alpine Linux.

## Installation

```bash
# From PyPI
pip install fish_speech_rs
```

and done.

## Usage

### Codec

This is the low-level API. You feed it PCM audio, it compresses it into codes, and then decompresses it back into PCM.

```python
from fish_speech import FireflyCodec
import numpy as np
# optional but highly recommended
from huggingface_hub import snapshot_download

# This just returns a directory path.
# Substitute with your own directory path if you don't want to download from Hugging Face.
dir = snapshot_download("jkeisling/fish-speech-1.5")

# Load the codec model (set device to "cuda" for speed)
codec = FireflyCodec(
    dir,
    version="1.5",  # Supports 1.2 to 1.5; 1.5 is default
    device="cuda"    # Or "cpu" (much slower), "metal" on Apple Silicon
)

# 1s of random audio. Substitute with your own audio.
# You will need to resample to codec.sample_rate yourself. Soundfile is recommended.
pcm = np.random.randn(1, 1, codec.sample_rate).astype(np.float32)  # (batch, channels, samples)
# Encode raw PCM into compressed codes
codes = codec.encode(pcm)

# Decode the compressed codes back into PCM
decoded_pcm = codec.decode(codes)
```

- Input: Raw PCM audio (please handle resampling to 44.1 kHz yourself)
- Output: Encoded Numpy uint32 “codes” (compressed speech)

## LM

The language model (LM) takes text and turns it into speech codes, which you then decode back to audio.

```python
from fish_speech import LM
from typing import List

# Load the TTS model
lm = LM(
    dir,
    version="1.5",
    device="cuda",
    # bf16 only recommended for CUDA, otherwise leave it default (f32)
    dtype="bf16"
)

# Extract the speaker prompt from reference audio
speaker_prompt = lm.get_speaker_prompt([{
    'text': 'foobar',
    'codes': codes  # From previous encoding step
}], sysprompt="Speak out the provided text.")

# Generate speech codes
# Text chunking and normalization are your responsibility (sorry!);
# official text preprocessing helper function coming soon
generated_codes = lm.generate(["This is a test", "This is another test"], speaker_prompt=speaker_prompt)

# Decode to PCM audio using codec from earlier
pcm = codec.decode(generated_codes)
```

If you're in a Jupyter notebook, you can use the following code to play the audio in a widget:

```python
# assumes you ran the above code
from IPython.display import Audio

Audio(pcm.flatten(), rate=codec.sample_rate)
```

### Developing

Requires Python and Rust toolchains. Clone this repo, set up a Rust and Python toolchain.

1. `python -m venv .venv`
2. `pip install -r requirements.txt`
3. `maturin develop`
