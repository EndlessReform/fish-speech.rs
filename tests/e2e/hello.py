import numpy as np
from scipy import signal


def compare_audio(wav1, wav2):
    # Ensure same length
    min_len = min(len(wav1), len(wav2))
    wav1 = wav1[:min_len]
    wav2 = wav2[:min_len]

    # Mean Squared Error
    mse = np.mean((wav1 - wav2) ** 2)
    print(f"Mean Squared Error: {mse:.2e}")

    # Root Mean Square Error
    rmse = np.sqrt(mse)
    print(f"Root Mean Square Error: {rmse:.2e}")

    # Maximum Absolute Error
    max_error = np.max(np.abs(wav1 - wav2))
    print(f"Maximum Absolute Error: {max_error}")

    # Cross-correlation
    correlation = signal.correlate(wav1, wav2, mode="full")
    lags = signal.correlation_lags(len(wav1), len(wav2), mode="full")
    lag = lags[np.argmax(correlation)]
    max_corr = np.max(correlation)
    print(f"Maximum Cross-correlation: {max_corr} at lag {lag}")

    # Signal-to-Noise Ratio (treating the difference as noise)
    signal_power = np.mean(wav1**2)
    noise_power = np.mean((wav1 - wav2) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    print(f"Signal-to-Noise Ratio: {snr} dB")


def main():
    print("Hello from e2e!")
    wav1 = np.load("../resources/fish_speech_preprocessed_audio.npy")
    wav2 = np.load("../resources/candle_fish_preprocessed_audio.npy")
    compare_audio(wav1, wav2)


if __name__ == "__main__":
    main()
