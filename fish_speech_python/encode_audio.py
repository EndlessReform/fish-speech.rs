import argparse
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from fish_speech import FishSpeechModel


def process_audio(
    input_path, target_sr=44100, n_mels=160, n_fft=2048, hop_length=512, win_length=2048
):
    # Load audio file
    audio, sr = sf.read(input_path)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if necessary
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Compute log mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=target_sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=False,
    )

    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # No need to transpose now, as librosa returns in (n_mels, time) format

    print(f"Audio shape: {audio.shape}")
    print(
        f"Audio min: {audio.min():.4f}, max: {audio.max():.4f}, mean: {audio.mean():.4f}"
    )
    print(f"Log Mel Spectrogram shape: {log_mel_spec.shape}")

    return log_mel_spec, audio


def main():
    parser = argparse.ArgumentParser(
        description="Process audio file and run FishSpeech model"
    )
    parser.add_argument("input_path", type=str, help="Path to input audio file")
    parser.add_argument(
        "--output_path", type=str, help="Path to save output NPY file (optional)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = input_path.with_stem(input_path.stem + "_output").with_suffix(
            ".npy"
        )

    # Process audio
    log_mel_spec, _ = process_audio(input_path)

    # Initialize and run FishSpeechModel
    model = FishSpeechModel()
    indices = model.forward(log_mel_spec[np.newaxis, :, :].astype(np.int32))

    # NOTE: Original saves to int32, but Candle.rs can't read int32 ndarrays (for some reason)
    np.save(output_path, np.array(indices).astype(np.float32))
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
