## fish_speech_python

Python bindings for the Fish Speech Candle implementation, using PyO3.

### Local installation

Requires Python and Rust toolchains.

1. `python -m venv .venv`
2. `pip install -r requirements.txt`
3. `maturin develop`

TODO: Library packaging

### Scripts

Generate speaker conditioning tokens as a `.npy` file:

```python
# Saves to input.npy by default
python encode_audio.py --output_path ../fake.npy ../tests/resources/sky.wav
```

You can use these drop-in with the [official Fish Audio inference script](https://github.com/fishaudio/fish-speech):

```bash
# Follow their steps for inference.
# If anything goes wrong take it up with them, the whole point of this repo is to not use that inference stack
python -m tools.vqgan.inference -i ./output.npy --checkpoint-path "checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"
```

### Entry point

As of 2024-09-08, speaker encoding is implemented (the rest to come):

```python
import numpy as np
from fish_speech import FishSpeechModel

# Downloads VQGan the first time using HF Hub
model = FishSpeechModel()

# Fake input ndarray of "audio".
# In reality, you'd run preprocessing with librosa
mels_shape = (1, 160, 400)
random_array = np.random.uniform(-1, 1, size=mels_shape)

indices = model.encode(mels)
np.save("output.py", indices)
```
