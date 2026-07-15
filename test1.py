import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# =========================
# 1. 基本工具
# =========================
def bandpass_filter(x, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)

def estimate_rate_welch(x, fs, fmin, fmax, nperseg=None):
    if nperseg is None:
        nperseg = min(len(x), int(fs * 16))

    freqs, psd = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend='constant'
    )

    mask = (freqs >= fmin) & (freqs <= fmax)
    if np.sum(mask) == 0:
        return np.nan, freqs, psd

    band_freqs = freqs[mask]
    band_psd = psd[mask]
    peak_freq = band_freqs[np.argmax(band_psd)]
    bpm = peak_freq * 60.0
    return bpm, freqs, psd

def bland_altman_stats(est, ref):
    diff = est - ref
    mean_val = (est + ref) / 2
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    return mean_val, diff, bias, loa_upper, loa_lower

def compute_metrics(est, ref):
    est = np.asarray(est)
    ref = np.asarray(ref)
    mask = ~np.isnan(est) & ~np.isnan(ref)
    est = est[mask]
    ref = ref[mask]

    mae = np.mean(np.abs(est - ref))
    rmse = np.sqrt(np.mean((est - ref) ** 2))
    bias = np.mean(est - ref)
    corr = np.corrcoef(est, ref)[0, 1] if len(est) > 1 else np.nan

    return {
        "N": len(est),
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "Corr": corr
    }

# =========================
# 2. 讀資料
# 假設你的 CSV 至少有:
# time, radar_signal, ref_rr, ref_hr
# ref_rr/ref_hr 可先是每段對應值；若沒有可先只驗 RR
# =========================
df = pd.read_csv("your_data.csv")

time = df["time"].values
radar_signal = df["radar_signal"].values

# 若 time 等間距，可由 time 推 fs
fs = 1.0 / np.mean(np.diff(time))

# =========================
# 3. 切 segment
# =========================
segment_sec = 30
segment_len = int(segment_sec * fs)

results = []

for start in range(0, len(radar_signal) - segment_len + 1, segment_len):
    end = start + segment_len

    seg_t = time[start:end]
    seg_x = radar_signal[start:end]

    # 去平均 / detrend 簡化版
    seg_x = seg_x - np.mean(seg_x)

    # RR 頻帶約 0.1 - 0.5 Hz (6 - 30 bpm)
    rr_sig = bandpass_filter(seg_x, fs, 0.1, 0.5, order=4)
    est_rr, rr_freqs, rr_psd = estimate_rate_welch(rr_sig, fs, 0.1, 0.5)

    # HR 頻帶約 0.8 - 2.0 Hz (48 - 120 bpm)
    hr_sig = bandpass_filter(seg_x, fs, 0.8, 2.0, order=4)
    est_hr, hr_freqs, hr_psd = estimate_rate_welch(hr_sig, fs, 0.8, 2.0)

    # 假設 ref_rr / ref_hr 是逐點資料，這裡取 segment 平均
    ref_rr = df["ref_rr"].iloc[start:end].mean() if "ref_rr" in df.columns else np.nan
    ref_hr = df["ref_hr"].iloc[start:end].mean() if "ref_hr" in df.columns else np.nan

    # 簡單品質分數: 頻帶 peak / 頻帶總能量
    rr_mask = (rr_freqs >= 0.1) & (rr_freqs <= 0.5)
    hr_mask = (hr_freqs >= 0.8) & (hr_freqs <= 2.0)

    rr_quality = (
        np.max(rr_psd[rr_mask]) / np.sum(rr_psd[rr_mask])
        if np.sum(rr_mask) > 0 and np.sum(rr_psd[rr_mask]) > 0 else np.nan
    )
    hr_quality = (
        np.max(hr_psd[hr_mask]) / np.sum(hr_psd[hr_mask])
        if np.sum(hr_mask) > 0 and np.sum(hr_psd[hr_mask]) > 0 else np.nan
    )

    results.append({
        "segment_id": len(results) + 1,
        "start_time": seg_t[0],
        "end_time": seg_t[-1],
        "est_rr": est_rr,
        "ref_rr": ref_rr,
        "est_hr": est_hr,
        "ref_hr": ref_hr,
        "rr_quality": rr_quality,
        "hr_quality": hr_quality
    })

results_df = pd.DataFrame(results)
print(results_df.head())

# =========================
# 4. 指標評估
# =========================
if "ref_rr" in results_df.columns:
    rr_metrics = compute_metrics(results_df["est_rr"], results_df["ref_rr"])
    print("RR metrics:", rr_metrics)

if "ref_hr" in results_df.columns:
    hr_metrics = compute_metrics(results_df["est_hr"], results_df["ref_hr"])
    print("HR metrics:", hr_metrics)

# =========================
# 5. 畫 Bland-Altman
# =========================
def plot_bland_altman(est, ref, title):
    est = np.asarray(est)
    ref = np.asarray(ref)
    mask = ~np.isnan(est) & ~np.isnan(ref)
    est = est[mask]
    ref = ref[mask]

    mean_val, diff, bias, loa_upper, loa_lower = bland_altman_stats(est, ref)

    plt.figure(figsize=(7, 5))
    plt.scatter(mean_val, diff, alpha=0.7)
    plt.axhline(bias, color='red', linestyle='--', label=f'Bias = {bias:.2f}')
    plt.axhline(loa_upper, color='gray', linestyle='--', label=f'+1.96 SD = {loa_upper:.2f}')
    plt.axhline(loa_lower, color='gray', linestyle='--', label=f'-1.96 SD = {loa_lower:.2f}')
    plt.xlabel("Mean of Estimate and Reference")
    plt.ylabel("Estimate - Reference")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if results_df["ref_rr"].notna().any():
    plot_bland_altman(results_df["est_rr"], results_df["ref_rr"], "Bland-Altman Plot for RR")

if results_df["ref_hr"].notna().any():
    plot_bland_altman(results_df["est_hr"], results_df["ref_hr"], "Bland-Altman Plot for HR")

# =========================
# 6. 存結果
# =========================
results_df.to_csv("vital_signs_validation_results.csv", index=False)