from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.signal.windows import hamming

# ============================ 初始化 ============================ #
# Python 不需要 clc / clear / close all
plt.close("all")

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
output_dir = Path("outputs") / f"slowtime2_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

# ========================== 發射參數設置 ========================= #
c = 3e8                       # 光速
fc = 77e9                     # IWRL6432: 60 GHz
k = 75e12                     # 75 MHz/us = 75e12 Hz/s

# ============== Fast Time（單個 Chirp 內部的採樣） ============== #
Tc = 10.24e-6                 # Chirp 時間週期
R_fs = 25e6                   # 取樣率 25 MHz
R_Fs = 1 / R_fs               # 取樣週期 40 ns

# 避免 np.arange 因浮點誤差產生多一點或少一點
S = int(round(Tc / R_Fs))     # 256
fast_time = np.arange(S) * R_Fs

bw = k * Tc                   # 768 MHz

# ================= Slow Time（Chirp 與 Chirp 之間） ============== #
Ts = 13.84e-6                 # 單個 Chirp 總耗時
num_chirps_per_loop = 7
PRI_Fs = Ts * num_chirps_per_loop
PRI_fs = 1 / PRI_Fs

num_loops = 1                 # 原 MATLAB 程式目前設定為 1
slow_time = np.arange(num_loops) * PRI_Fs
P = len(slow_time)

# ====================== Frame（幀與幀之間） ====================== #
frame_periodicity = 80e-3
frame_Fs = 1 / frame_periodicity
frame_len = 128
frame_time = np.arange(frame_len) * frame_periodicity

# ========================== 反射訊號參數 ========================= #
distance = np.array([1.0])    # 目標距離 (m)
velocity = np.array([0.0])    # 相對速度 (m/s)

# ====================== 生命徵象參數 ============================ #
# 呼吸
breat_amp = 2e-3
breat_bpm = 93 
breat_freq = breat_bpm / 60

# 心跳
heart_amp = 0.5e-3
heart_bpm = 122
heart_freq = heart_bpm / 60

# ===============頻帶濾波器設置======================== #
breat_max_bpm = 120
breat_min_bpm = 10
heart_max_bpm = 240
heart_min_bpm = 30

breat_max_freq = breat_max_bpm / 60
breat_min_freq = breat_min_bpm / 60
heart_max_freq = heart_max_bpm / 60
heart_min_freq = heart_min_bpm / 60

# ======================== 陣列天線參數 ========================== #
M = 3                         # 接收天線數
N = 2                         # 發射天線數

Theta = np.array([0.0])       # DOA，單位：degree
Phi = np.array([0.0])         # DOD，單位：degree

lambda_ = c / fc
d = lambda_ / 2
Q = 1                         # 目標數目

SNR = 30
Np = 1
An = np.sqrt(Np)
Sp = Np * 10 ** (SNR / 10)
As = np.sqrt(Sp)

# ========================== 產生 Radar 資料 ====================== #
# shape: [Fast time, Slow time, virtual antenna, frame]
Y1 = np.zeros((S, P, M * N, frame_len), dtype=np.complex128)

antenna_idx = 0

for f in range(frame_len):
    antenna_idx = 0

    # 本 frame 的胸腔位移
    vibration = (
        breat_amp * np.sin(2 * np.pi * breat_freq * frame_time[f])
        + heart_amp * np.sin(2 * np.pi * heart_freq * frame_time[f])
    )

    for n in range(N):
        for m in range(M):

            X_target = np.zeros((S, P), dtype=np.complex128)

            for q in range(Q):
                IF_data = np.zeros((S, P), dtype=np.complex128)

                new_dis = distance[q] + vibration

                # Beat frequency 與 Doppler frequency
                fb = 2 * k * new_dis / c
                fd = 2 * velocity[q] * fc / c

                # 天線發射/接收角度相位
                D1 = np.exp(
                    1j * 2 * np.pi
                    * n * d * np.sin(np.deg2rad(Phi[q])) / lambda_
                )

                D2 = np.exp(
                    1j * 2 * np.pi
                    * m * d * np.sin(np.deg2rad(Theta[q])) / lambda_
                )

                # 快時間與慢時間訊號
                for p in range(P):
                    A1 = np.exp(1j * 2 * np.pi * (fb + fd) * fast_time)
                    B1 = np.exp(1j * 2 * np.pi * fd * p * Ts)
                    C1 = np.exp(1j * 2 * np.pi * fc * 2 * new_dis / c)

                    IF_signal = As * A1 * B1 * C1 * D1 * D2

                    # 原 MATLAB 程式最後沒有真正加入雜訊：
                    # IF_data[:, p] = IF;
                    #
                    # 若要加入實數高斯雜訊，可使用：
                    # noise = An * np.random.randn(S)
                    # IF_data[:, p] = IF_signal + noise

                    IF_data[:, p] = IF_signal

                X_target += IF_data

            Y1[:, :, antenna_idx, f] = X_target
            antenna_idx += 1

# ====================== 1. Range FFT =========================== #
range_win = hamming(S, sym=True)
range_profile_matrix = np.zeros((S, frame_len), dtype=np.complex128)

for i in range(frame_len):
    # Python index 從 0 開始
    # MATLAB: Y1(:, 1, 1, i)
    temp = Y1[:, 0, 0, i] * range_win
    range_profile_matrix[:, i] = np.fft.fft(temp)

# 找出平均能量最強的 Range Bin
max_idx = np.argmax(np.mean(np.abs(range_profile_matrix), axis=1))

# MATLAB 顯示的 index 從 1 開始，因此印出時 +1
# print(f"目標鎖定在 Range Bin: {int(max_idx) + 1}")

# ====================== 2. 相位提取 ============================= #
target_complex_data = range_profile_matrix[max_idx, :]
extracted_phase2 = np.angle(target_complex_data)
extracted_phase = np.unwrap(extracted_phase2)

# 去除 DC 成分
phase_vibration = extracted_phase - np.mean(extracted_phase)

# ====================== 帶通濾波 ================================ #
# 呼吸頻帶：0.1 ~ 0.5 Hz
b_resp, a_resp = butter(
    2,
    [breat_min_freq, breat_max_freq],
    btype="bandpass",
    fs=frame_Fs
)
resp_filtered = filtfilt(b_resp, a_resp, phase_vibration)

# 心跳頻帶：1.0 ~ 3.0 Hz
b_heart, a_heart = butter(
    2,
    [heart_min_freq, heart_max_freq],
    btype="bandpass",
    fs=frame_Fs
)
heart_filtered = filtfilt(b_heart, a_heart, phase_vibration)

# ====================== FFT 頻譜 ================================ #
f1 = np.arange(frame_len) * (frame_Fs / frame_len)

fft_resp = np.abs(np.fft.fft(resp_filtered))
fft_heart = np.abs(np.fft.fft(heart_filtered))

half_len = frame_len // 2
freq_axis_hz = f1[:half_len]

def estimate_bpm_from_psd(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return float("nan")

    band_freqs = freqs[mask]
    band_psd = psd[mask]
    if band_psd.size == 0 or np.all(band_psd == 0):
        return float("nan")

    peak_index = int(np.argmax(band_psd))
    return float(band_freqs[peak_index] * 60.0)


def suppress_resp_harmonics(
    freqs: np.ndarray,
    psd: np.ndarray,
    rr_bpm: float,
    hr_fmin: float,
    hr_fmax: float,
    notch_width_hz: float = 0.08,
    max_harmonics: int = 6,
) -> np.ndarray:
    cleaned_psd = psd.copy()

    if np.isnan(rr_bpm):
        return cleaned_psd

    rr_hz = rr_bpm / 60.0
    if rr_hz <= 0:
        return cleaned_psd

    for harmonic_index in range(1, max_harmonics + 1):
        harmonic_freq = harmonic_index * rr_hz
        if harmonic_freq > hr_fmax:
            break
        if harmonic_freq < hr_fmin:
            continue

        mask = np.abs(freqs - harmonic_freq) <= notch_width_hz
        cleaned_psd[mask] = 0.0

    return cleaned_psd


resp_freqs, resp_psd = welch(resp_filtered, fs=frame_Fs, nperseg=len(resp_filtered))
heart_freqs, heart_psd = welch(heart_filtered, fs=frame_Fs, nperseg=len(heart_filtered))

resp_est_min_freq = 70 / 60
resp_est_max_freq = 110 / 60
heart_est_min_freq = 110 / 60
heart_est_max_freq = 160 / 60

estimated_breat_bpm = estimate_bpm_from_psd(resp_freqs, resp_psd, resp_est_min_freq, resp_est_max_freq)
heart_psd_clean = suppress_resp_harmonics(
    heart_freqs,
    heart_psd,
    estimated_breat_bpm,
    heart_est_min_freq,
    heart_est_max_freq,
)
estimated_heart_bpm = estimate_bpm_from_psd(heart_freqs, heart_psd_clean, heart_est_min_freq, heart_est_max_freq)

print(
    f"原始呼吸 BPM: {breat_bpm:.2f}, 估算呼吸 BPM: {estimated_breat_bpm:.2f}\n"
    f"原始心跳 BPM: {heart_bpm:.2f}, 估算心跳 BPM: {estimated_heart_bpm:.2f}"
)

# ====================== 合併總覽圖 ============================== #
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].plot(phase_vibration)
axes[0, 0].set_title("Phase Vibration")
axes[0, 0].set_xlabel("Frame Index")
axes[0, 0].set_ylabel("Phase (rad)")
axes[0, 0].grid(True)

axes[0, 1].plot(f1[:half_len], fft_resp[:half_len], "b-")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("Magnitude")
axes[0, 1].set_xlim(0, frame_Fs / 2)
axes[0, 1].set_title(
    f"Respiration | True: {breat_bpm:.2f} bpm, Est: {estimated_breat_bpm:.2f} bpm"
)
axes[0, 1].grid(True)

axes[1, 0].plot(f1[:half_len], fft_heart[:half_len], "r-")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("Magnitude")
axes[1, 0].set_xlim(0, frame_Fs / 2)
axes[1, 0].set_title(
    f"Heartbeat | True: {heart_bpm:.2f} bpm, Est: {estimated_heart_bpm:.2f} bpm"
)
axes[1, 0].grid(True)

phase_fft = np.abs(np.fft.fft(phase_vibration))
phase_fft_freq_hz = np.arange(half_len) * (frame_Fs / frame_len)

axes[1, 1].plot(phase_fft_freq_hz, phase_fft[:half_len])
axes[1, 1].set_title("Phase Vibration FFT")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Magnitude")
axes[1, 1].set_xlim(0, frame_Fs / 2)
axes[1, 1].grid(True)

fig.tight_layout()
combined_path = output_dir / "slowtime2_overview.png"
fig.savefig(combined_path, dpi=200)
print(f"已將合併圖片儲存至: {combined_path}")

plt.show()