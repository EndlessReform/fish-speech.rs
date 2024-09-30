import torch
import torchaudio.functional as F
import numpy as np

# Parameters to match your filterbank
n_fft = 2048
f_min = 0
f_max = 22050
n_mels = 160
sample_rate = 44100

# Step 1: Generate the original filterbank using torchaudio
original_fb = F.melscale_fbanks(
    n_freqs=n_fft // 2 + 1,
    f_min=f_min,
    f_max=f_max,
    n_mels=n_mels,
    sample_rate=sample_rate,
    norm="slaney",
    mel_scale="slaney",
)

# Step 2: Load the bytes from the file
with open("fish_speech_core/lib/audio/melfilters160_old.bytes", "rb") as f:
    fb_bytes = f.read()

# Convert bytes back to a NumPy array (assuming float32)
fb_np = np.frombuffer(fb_bytes, dtype=np.float32)

# Step 3: Reshape to match the original filterbank shape
fb_np = fb_np.reshape(original_fb.shape)

# Step 4: Convert NumPy array to PyTorch tensor
loaded_fb = torch.from_numpy(fb_np)

# Step 5: Compare the two tensors
if torch.allclose(original_fb, loaded_fb, atol=1e-6):
    print("The loaded filterbank matches the original one!")
else:
    print("The loaded filterbank does NOT match the original one.")

# Optionally, print out the difference for debugging
print("Difference between tensors:", (original_fb - loaded_fb).abs().max())
