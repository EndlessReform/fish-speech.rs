import librosa
import numpy as np
import pickle

# Parameters (replace these with your actual values)
n_fft = 2048  # Example value, this is your self.n_fft
f_min = 0  # Minimum frequency
f_max = 22050  # Maximum frequency, or use None to calculate from sample_rate
n_mels = 1600  # Number of mel bands
sample_rate = 44100  # Sampling rate

# Generate the mel filterbank using librosa
fb = librosa.filters.mel(
    sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max, norm="slaney"
)

# Serialize the filterbank as bytes using pickle
fb_bytes = fb.tobytes()

# Optionally, you can save the bytes to a file
with open("mels160.bytes", "wb") as f:
    f.write(fb_bytes)

# Or, if you just want to return or manipulate the bytes directly
print(fb_bytes[:100])  # Just print first 100 bytes as a preview
