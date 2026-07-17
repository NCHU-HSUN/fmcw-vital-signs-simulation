import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal.windows import hamming

# ============================ 初始化 ============================ #
# Python 不需要 clc / clear / close all
plt.close("all")

# ========================== 發射參數設置 ========================= #
c = 3e8                       # 光速
fc = 60e9                     # IWRL6432: 60 GHz
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
breat_freq = 0.25

# 心跳
heart_amp = 0.5e-3
heart_freq = 3.0

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
print(f"目標鎖定在 Range Bin: {max_idx + 1}")

# ====================== 2. 相位提取 ============================= #
target_complex_data = range_profile_matrix[max_idx, :]
extracted_phase2 = np.angle(target_complex_data)
extracted_phase = np.unwrap(extracted_phase2)

# 去除 DC 成分
phase_vibration = extracted_phase - np.mean(extracted_phase)

plt.figure(1)
plt.plot(phase_vibration)
plt.title("Phase Vibration")
plt.xlabel("Frame Index")
plt.ylabel("Phase (rad)")
plt.grid(True)

# ====================== 帶通濾波 ================================ #
# 呼吸頻帶：0.1 ~ 0.5 Hz
b_resp, a_resp = butter(
    2,
    [0.1, 0.5],
    btype="bandpass",
    fs=frame_Fs
)
resp_filtered = filtfilt(b_resp, a_resp, phase_vibration)

# 心跳頻帶：1.0 ~ 3.0 Hz
b_heart, a_heart = butter(
    2,
    [1.0, 3.0],
    btype="bandpass",
    fs=frame_Fs
)
heart_filtered = filtfilt(b_heart, a_heart, phase_vibration)

# ====================== FFT 頻譜 ================================ #
f1 = np.arange(frame_len) * (frame_Fs / frame_len)

fft_resp = np.abs(np.fft.fft(resp_filtered))
fft_heart = np.abs(np.fft.fft(heart_filtered))

half_len = frame_len // 2

# ====================== 呼吸頻譜圖 ============================== #
plt.figure(2)
plt.plot(f1[:half_len], fft_resp[:half_len], "b-")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim([0, frame_Fs / 2])
plt.title("呼吸")
plt.grid(True)

# ====================== 心跳頻譜圖 ============================== #
plt.figure(3)
plt.plot(f1[:half_len], fft_heart[:half_len], "r-")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim([0, frame_Fs / 2])
plt.title("心跳")
plt.grid(True)

# ====================== 原始相位 FFT ============================ #
plt.figure(4)
plt.plot(np.abs(np.fft.fft(phase_vibration[:64])))
plt.title("Phase Vibration FFT")
plt.xlabel("FFT Bin")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()