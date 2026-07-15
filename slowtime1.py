from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x).squeeze()

    if x.ndim != 1:
        raise ValueError(f"x must be 1-D after squeeze, but got shape {x.shape}")

    nyq = fs / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(
            f"Need 0 < low < high < fs/2; got low={low}, high={high}, fs/2={nyq}"
        )

    b, a = butter(order, [low, high], btype='bandpass', fs=fs, output='ba')
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        raise ValueError(
            f"Signal too short for filtfilt: len(x)={len(x)}, padlen={padlen}"
        )

    return filtfilt(b, a, x)


def estimate_bpm(sig: np.ndarray, fs: float, fmin: float, fmax: float) -> Tuple[float, np.ndarray, np.ndarray]:
    f, pxx = welch(sig, fs=fs, nperseg=min(len(sig), 256))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return float('nan'), f, pxx

    band_freqs = f[mask]
    band_psd = pxx[mask]
    peak_index = np.argmax(band_psd)
    f_peak = band_freqs[peak_index]
    return float(f_peak * 60.0), f, pxx


def main() -> None:
    c = 3e8
    fc = 77e9
    lam = c / fc

    # slow-time sampling, 例如每個 frame 取一次 phase
    fs_slow = 20.0          # 20 Hz
    T = 60                  # 60 秒
    t = np.arange(0, T, 1 / fs_slow)

    # 人體參數
    Ar = 5e-3               # 呼吸位移振幅 5 mm
    Ah = 0.5e-3             # 心跳位移振幅 0.5 mm
    fr = 0.25               # 呼吸 0.25 Hz = 15 bpm
    fh = 1.2                # 心跳 1.2 Hz = 72 bpm

    # 胸部位移模型
    x = Ar * np.sin(2 * np.pi * fr * t) + Ah * np.sin(2 * np.pi * fh * t)

    # target range bin 的 complex signal
    A = 1.0
    noise_std = 0.05
    noise = noise_std * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    s = A * np.exp(1j * 4 * np.pi * x / lam) + noise

    # phase extraction
    phase = np.unwrap(np.angle(s))
    phase = phase - np.mean(phase)

    # RR
    rr_sig = bandpass(phase, fs_slow, 0.1, 0.5)
    rr_bpm, f_rr, p_rr = estimate_bpm(rr_sig, fs_slow, 0.1, 0.5)

    # HR
    hr_sig = bandpass(phase, fs_slow, 0.8, 2.0)
    hr_bpm, f_hr, p_hr = estimate_bpm(hr_sig, fs_slow, 0.8, 2.0)

    print("Estimated RR:", f"{rr_bpm:.2f}", "bpm", "| Actual RR:", f"{fr * 60:.2f}", "bpm")
    print("Estimated HR:", f"{hr_bpm:.2f}", "bpm", "| Actual HR:", f"{fh * 60:.2f}", "bpm")

    fig, axs = plt.subplots(4, 1, figsize=(10, 6))

    axs[0].plot(t, x * 1000)
    axs[0].set_title("Chest displacement (mm)")

    axs[1].plot(t, phase)
    axs[1].set_title("Unwrapped phase")

    axs[2].plot(f_rr * 60, p_rr, label='RR PSD')
    axs[2].set_xlabel('Frequency (bpm)')
    axs[2].set_ylabel('Power')
    axs[2].set_xlim(0, 50)
    axs[2].axvline(rr_bpm, color='r', linestyle='--', label=f'Peak: {rr_bpm:.2f} bpm')
    axs[2].legend()

    axs[3].plot(f_hr * 60, p_hr, label='HR PSD')
    axs[3].set_xlabel('Frequency (bpm)')
    axs[3].set_ylabel('Power')
    axs[3].set_xlim(40, 100)
    axs[3].axvline(hr_bpm, color='r', linestyle='--', label=f'Peak: {hr_bpm:.2f} bpm')
    axs[3].legend()

    fig.tight_layout()

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output_filename = f'slowtime1_plot_{timestamp}.png'
    fig.savefig(output_filename, dpi=200)
    print(f'Saved plot to {output_filename}')


if __name__ == '__main__':
    main()
