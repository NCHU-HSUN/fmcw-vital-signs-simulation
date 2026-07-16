from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, detrend, filtfilt, welch
from scipy.stats import pearsonr, spearmanr


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


def bandstop(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x).squeeze()

    if x.ndim != 1:
        raise ValueError(f"x must be 1-D after squeeze, but got shape {x.shape}")

    nyq = fs / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(
            f"Need 0 < low < high < fs/2; got low={low}, high={high}, fs/2={nyq}"
        )

    b, a = butter(order, [low, high], btype='bandstop', fs=fs, output='ba')
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        raise ValueError(
            f"Signal too short for filtfilt: len(x)={len(x)}, padlen={padlen}"
        )

    return filtfilt(b, a, x)


def highpass(x: np.ndarray, fs: float, cutoff: float, order: int = 2) -> np.ndarray:
    x = np.asarray(x).squeeze()

    if x.ndim != 1:
        raise ValueError(f"x must be 1-D after squeeze, but got shape {x.shape}")

    nyq = fs / 2.0
    if not (0 < cutoff < nyq):
        raise ValueError(f"Need 0 < cutoff < fs/2; got cutoff={cutoff}, fs/2={nyq}")

    b, a = butter(order, cutoff, btype='highpass', fs=fs, output='ba')
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        raise ValueError(
            f"Signal too short for filtfilt: len(x)={len(x)}, padlen={padlen}"
        )

    return filtfilt(b, a, x)


def estimate_bpm(sig: np.ndarray, fs: float, fmin: float, fmax: float) -> Tuple[float, np.ndarray, np.ndarray]:
    #f, pxx = welch(sig, fs=fs, nperseg=min(len(sig), 1200))
    f, pxx = welch(sig, fs=fs, nperseg=len(sig))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return float('nan'), f, pxx

    band_freqs = f[mask]
    band_psd = pxx[mask]
    peak_index = np.argmax(band_psd)
    f_peak = band_freqs[peak_index]
    return float(f_peak * 60.0), f, pxx


def summarize_metrics(estimates: np.ndarray, references: np.ndarray) -> dict:
    estimates = np.asarray(estimates, dtype=float)
    references = np.asarray(references, dtype=float)
    valid_mask = ~np.isnan(estimates) & ~np.isnan(references)

    valid_estimates = estimates[valid_mask]
    valid_references = references[valid_mask]
    coverage = float(np.mean(~np.isnan(estimates))) if estimates.size else float('nan')

    if valid_estimates.size == 0:
        return {
            'N': int(estimates.size),
            'ValidN': 0,
            'Coverage': coverage,
            'MAE': float('nan'),
            'RMSE': float('nan'),
            'Bias': float('nan'),
            'Pearson': float('nan'),
            'Spearman': float('nan'),
        }

    errors = valid_estimates - valid_references
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    if valid_estimates.size > 1 and np.std(valid_estimates) > 0 and np.std(valid_references) > 0:
        pearson = float(pearsonr(valid_estimates, valid_references).statistic)
        spearman = float(spearmanr(valid_estimates, valid_references).statistic)
    else:
        pearson = float('nan')
        spearman = float('nan')

    return {
        'N': int(estimates.size),
        'ValidN': int(valid_estimates.size),
        'Coverage': coverage,
        'MAE': mae,
        'RMSE': rmse,
        'Bias': bias,
        'Pearson': pearson,
        'Spearman': spearman,
    }

def suppress_resp_harmonics(
    freqs: np.ndarray,
    psd: np.ndarray,
    rr_bpm: float,
    hr_fmin: float,
    hr_fmax: float,
    notch_width_hz: float = 0.08,
    max_harmonics: int = 6,
) -> np.ndarray:
    """
    根據已估測的呼吸頻率，將 HR 頻譜中呼吸基頻與其諧波附近的 PSD 壓低。
    """
    cleaned_psd = psd.copy()

    if np.isnan(rr_bpm):
        return cleaned_psd

    fr = rr_bpm / 60.0  # bpm -> Hz
    if fr <= 0:
        return cleaned_psd

    for k in range(1, max_harmonics + 1):
        harmonic_freq = k * fr
        if harmonic_freq > hr_fmax:
            break

        # 只在 HR 頻段內做 suppression
        if harmonic_freq < hr_fmin:
            continue

        mask = np.abs(freqs - harmonic_freq) <= notch_width_hz
        cleaned_psd[mask] = 0.0

    return cleaned_psd


def estimate_bpm_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float
) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return float("nan")

    band_freqs = freqs[mask]
    band_psd = psd[mask]

    if band_psd.size == 0 or np.all(band_psd == 0):
        return float("nan")

    peak_index = np.argmax(band_psd)
    return float(band_freqs[peak_index] * 60.0)


def main() -> None:
    rng = np.random.default_rng(42)

    c = 3e8
    fc = 77e9
    lam = c / fc

    # slow-time sampling, 例如每個 frame 取一次 phase
    fs_slow = 20.0         # 提高到 20 Hz 以更好地捕捉心跳
    T = 60                 # 增加到 60 秒以提高頻率分辨率
    t = np.arange(0, T, 1 / fs_slow)

    # 人體參數
    Ar = 5e-3               # 呼吸位移振幅 5 mm
    Ah = 0.5e-3             # 心跳位移振幅 0.5 mm
    hr = 230                # 心跳 bpm
    rr = 15                 # 呼吸 bpm
    fh = hr / 60            # 心跳 Hz
    fr = rr / 60            # 呼吸 Hz
    hmax = 240              # 最大心跳 bpm
    hmin = 30                # 最小心跳 bpm
    rmax = 120              # 最大呼吸 bpm
    rmin = 4                # 最小呼吸 bpm      
    fhrmax = hmax / 60       # 最大心跳 Hz
    fhrmin = hmin / 60       # 最小心跳 Hz
    frrmax = rmax / 60       # 最大呼吸 Hz
    frrmin = rmin / 60       # 最小呼吸 Hz

    n_trials = 100
    rr_true_values = []
    rr_est_values = []
    hr_true_values = []
    hr_est_values = []
    first_plot_data = None

    for trial_index in range(n_trials):
        #rr = rng.uniform(4.0, 120.0)
        #hr = rng.uniform(30.0, 240.0)
        #fr = rr / 60.0
        #fh = hr / 60.0

        # 胸部位移模型
        x = Ar * np.sin(2 * np.pi * fr * t) + Ah * np.sin(2 * np.pi * fh * t)

        # target range bin 的 complex signal
        A = 1.0
        noise_std = 0.05
        noise = noise_std * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
        s = A * np.exp(1j * 4 * np.pi * x / lam) + noise

        # phase extraction
        phase = np.unwrap(np.angle(s))
        #phase = detrend(phase, type='linear')
        phase = phase - np.mean(phase)
        #phase = highpass(phase, fs_slow, cutoff=0.05)

        # RR
        rr_sig = bandpass(phase, fs_slow, frrmin, frrmax)
        rr_bpm, f_rr, p_rr = estimate_bpm(rr_sig, fs_slow, frrmin, frrmax)

        # HR
        hr_sig = bandpass(phase, fs_slow, fhrmin, fhrmax)
        #_, f_hr, p_hr = estimate_bpm(hr_sig, fs_slow, fhrmin, fhrmax)
        hr_bpm, f_hr, p_hr = estimate_bpm(hr_sig, fs_slow, fhrmin, fhrmax)

        # 利用 RR 結果抑制呼吸基頻與諧波
        p_hr_clean = suppress_resp_harmonics(
            freqs=f_hr,
            psd=p_hr,
            rr_bpm=rr_bpm,
            hr_fmin=fhrmin,
            hr_fmax=fhrmax,
            notch_width_hz=0.08,
            max_harmonics=6,
        )

        # 再從處理後 PSD 估 HR
        #hr_bpm = estimate_bpm_from_psd(f_hr, p_hr_clean, fhrmin, fhrmax)

        rr_true_values.append(rr)
        rr_est_values.append(rr_bpm)
        hr_true_values.append(hr)
        hr_est_values.append(hr_bpm)

        if first_plot_data is None:
            first_plot_data = {
                'x': x,
                'phase': phase,
                'f_rr': f_rr,
                'p_rr': p_rr,
                'rr_bpm': rr_bpm,
                'f_hr': f_hr,
                'p_hr': p_hr,
                'p_hr_clean': p_hr_clean,
                'hr_bpm': hr_bpm,
                'rr_true': rr,
                'hr_true': hr,
            }

    rr_metrics = summarize_metrics(np.asarray(rr_est_values), np.asarray(rr_true_values))
    hr_metrics = summarize_metrics(np.asarray(hr_est_values), np.asarray(hr_true_values))

    print("RR metrics:")
    print(
        f"  N={rr_metrics['N']} | ValidN={rr_metrics['ValidN']} | Coverage={rr_metrics['Coverage']:.2%} | "
        f"MAE={rr_metrics['MAE']:.2f} bpm | RMSE={rr_metrics['RMSE']:.2f} bpm | "
        f"Bias={rr_metrics['Bias']:.2f} bpm | Pearson={rr_metrics['Pearson']:.3f} | "
        f"Spearman={rr_metrics['Spearman']:.3f}"
    )
    print("HR metrics:")
    print(
        f"  N={hr_metrics['N']} | ValidN={hr_metrics['ValidN']} | Coverage={hr_metrics['Coverage']:.2%} | "
        f"MAE={hr_metrics['MAE']:.2f} bpm | RMSE={hr_metrics['RMSE']:.2f} bpm | "
        f"Bias={hr_metrics['Bias']:.2f} bpm | Pearson={hr_metrics['Pearson']:.3f} | "
        f"Spearman={hr_metrics['Spearman']:.3f}"
    )
    print(f"RR valid rate: {rr_metrics['Coverage']:.2%}")
    print(f"HR valid rate: {hr_metrics['Coverage']:.2%}")

    if first_plot_data is None:
        raise RuntimeError('No successful trial was generated for plotting.')

    plot_data = first_plot_data

    fig, axs = plt.subplots(4, 1, figsize=(10, 6))

    x = plot_data['x']
    phase = plot_data['phase']
    f_rr = plot_data['f_rr']
    p_rr = plot_data['p_rr']
    rr_bpm = plot_data['rr_bpm']
    f_hr = plot_data['f_hr']
    p_hr = plot_data['p_hr']
    p_hr_clean = plot_data['p_hr_clean']
    hr_bpm = plot_data['hr_bpm']

    axs[0].plot(t, x * 1000)
    axs[0].set_title(f"Chest displacement (mm) | RR ref={plot_data['rr_true']:.2f} bpm, HR ref={plot_data['hr_true']:.2f} bpm")

    axs[1].plot(t, phase)
    axs[1].set_title("Unwrapped phase")

    axs[2].plot(f_rr * 60, p_rr, label='PSD')
    axs[2].set_xlabel('RR Frequency (bpm)')
    axs[2].set_ylabel('Power')
    axs[2].set_xlim(frrmin * 60, frrmax * 60)
    axs[2].axvline(rr_bpm, color='r', linestyle='--', label=f'Peak: {rr_bpm:.2f} bpm')
    axs[2].legend()

    axs[3].plot(f_hr * 60, p_hr, label='Original HR PSD', alpha=0.5)
    axs[3].plot(f_hr * 60, p_hr_clean, label='Cleaned HR PSD', linewidth=1.5)
    axs[3].set_xlabel('HR Frequency (bpm)')
    axs[3].set_ylabel('Power')
    axs[3].set_xlim(fhrmin * 60, fhrmax * 60)
    axs[3].axvline(hr_bpm, color='r', linestyle='--', label=f'Peak: {hr_bpm:.2f} bpm')
    axs[3].legend()

    fig.tight_layout()

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output_filename = f'slowtime1_plot_{timestamp}.png'
    fig.savefig(output_filename, dpi=200)
    print(f'Saved plot to {output_filename}')


if __name__ == '__main__':
    main()
