# fish_speech_python

Python bindings for the Fish Speech Candle implementation, using PyO3.

## Installation
Supports Python 3.9+.

**Supported Platforms**
- Linux:
  - x86_64, glibc 2.34+
  - CUDA 12+ with compute capability >= 8.0 (RTX 30 series+, A100 series+)
- macOS (M1+, 14.0+ (Monterey))

Windows and AMD hardware will never be supported, so don't ask.
Feel free to raise an issue if you need ARM or Alpine Linux.

```bash
# From PyPI
pip install fish_speech_python
```

## Usage

### Codec

This is the low-level API. You feed it PCM audio, it compresses it into codes, and then decompresses it back into PCM.

```python
from fish_speech_rs import FireflyCodec
from huggingface_hub import snapshot_dir
import numpy as np

# Download weights from Hugging Face
dir = snapshot_dir("jkeisling/fish-speech-1.5")

# Load the codec model (set device to "cuda" for speed)
codec = FireflyCodec(
    dir,
    version="1.5",  # Supports 1.4 and 1.5
    device="cuda"    # Or "cpu" if you hate yourself, "metal" on Apple Silicon
)

# Encode raw PCM into compressed codes
pcm = np.random.randn(1, 1, 44_100).astype(np.float32)  # (batch, channels, samples)
codes = codec.encode(pcm)

# Decode the compressed codes back into PCM
decoded_pcm = codec.decode(codes)
```

- Input: Raw PCM audio (please handle resampling to 44.1 kHz yourself)
- Output: Encoded Numpy uint32 “codes” (compressed speech)

## LM

The language model (LM) takes text and turns it into speech codes, which you then decode back to audio.

```python
from fish_speech_rs import LM, preprocess_text
from typing import List

# Load the TTS model
lm = LM(
    dir,
    version="1.5",
    device="cuda"
)

# Extract the speaker prompt from reference audio
speaker_prompt = lm.get_speaker_prompt([{
    'text': 'foobar',
    'audio': codes  # From previous encoding step
}], sysprompt="Speak out the provided text.")

# Preprocess text (splits into chunks)
chunks: List[str] = preprocess_text("Hello world. This is fast as hell.")

# Generate speech codes (you can stream this too)
generated_codes = lm.generate(chunks, speaker_prompt=speaker_prompt)

# Decode to PCM audio using codec from earlier
pcm = codec.decode(generated_codes)
```


### Local installation

Requires Python and Rust toolchains.

1. `python -m venv .venv`
2. `pip install -r requirements.txt`
3. `maturin develop`
