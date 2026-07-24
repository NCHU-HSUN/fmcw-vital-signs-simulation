import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# 為了確保圖表能正常顯示中文字 (如果沒有設定，中文會變成方塊)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 微軟正黑體 (Windows)
plt.rcParams['axes.unicode_minus'] = False # 讓負號正常顯示


"參數設定"
c: float = 3e+8;
fc: float = 60e+9;                         # IWRL6432 是 60GHz 頻段 (原本寫 77G)
k: float = 75e+12;                         # 斜率 75 MHz/us = 75e6 / 1e-6 = 75e12 Hz/s
#------------------- Fast Time (單個 Chirp 內部的採樣) --------------------
Tc: float = 10.24e-6;                      # 時間週期,  S*R_Fs = 10.24 us (256 * 40 ns)
R_fs: float = 25e+6;                       # 取樣率(sampling rate), 25.6 MHz
R_Fs: float = 1/R_fs;                      # 取樣週期(sampling period), 40 ns
# fast_time = 0:R_Fs:Tc-R_Fs;         # 時間向量 (共256個點)
fast_time = np.arange(0, Tc, R_Fs, dtype=float);        # 時間向量 (共256個點)

# for i in range(10):
#     print(fast_time[i]);

S: int = len(fast_time);                    # 取樣頻率點數 (共256個點)
bw: float = k*Tc;                           # 掃頻頻寬, 75e12*10.24e-6 = 768 MHz
# print(bw);
#-------------------- Slow Time (Chirp 與 Chirp 之間) ---------------------
# 這裡要計算「同一個 Chirp 序列」重複的時間間隔
# 單個 Chirp 週期約 13.84 us，但你的 frameCfg 包含 7 個 Chirp (從 index 2 到 8)
Ts: float = 13.84e-6;                       #單個 Chirp 總耗時 13.84 us
num_chirps_per_loop: int = 7;               # frameCfg 2 8 代表 2,3,4,5,6,7,8 共 7 個 Chirp
PRI_Fs: float = Ts*num_chirps_per_loop;     # 相同 Chirp 重複的時間間隔 (Doppler 取樣週期/968.8 ms)
PRI_fs: float = 1/PRI_Fs;                   # 取樣率, 10.322 KHz
num_loops: float = 1;                       # frameCfg 中的 580 loops
# slow_time = (0:num_loops-1)*PRI_Fs;         # 慢時間向量
slow_time = np.arange(0, num_loops * PRI_Fs, PRI_Fs, dtype=float);
P: int = len(slow_time);                    # 多普勒維度點數 (580)
#--------------------------- Frame (幀與幀之間) ---------------------------
frame_periodicity: float = 80e-3;
frame_Fs: float = 1/frame_periodicity;      # 7.5188 Hz
frame_len: int = 128;
# frame_time = (0:frame_len-1)*frame_periodicity;
frame_time = np.arange(0, frame_len * frame_periodicity, frame_periodicity, dtype=float);
#----------------------------- 反射訊號參數 -------------------------------
distance: float = 1.42;                     # 目標距離雷達 0m
velocity: float = 0;                        # 目標距離雷達的相對速度 0m/s
#----------------------- 生命徵象參數 (加入心跳呼吸) -----------------------
# 呼吸 (2mm, 0.25Hz)
# breat_amp = 2e-3;
# breat_freq = 0.25;
breat_amp: float = 0.78e-3;
breat_freq: float = 1.8372;

# 心跳 (0.5mm, 1.2Hz)
# heart_amp = 0.5e-3;
# heart_freq = 3.0;
heart_amp: float = 1.24e-3;
heart_freq: float = 0.4532;
#----------------------------- 陣列天線參數 -------------------------------
M: int = 1;                                     # 接收天線數目
N: int = 1;                                     # 發射天線數目
# dm = (0:M-1).';
# dn = (0:N-1).';
dm = np.arange(0, M, dtype=int);
dn = np.arange(0, N, dtype=int);
Theta: float = 0;                               # 目標入射角度, DOA
Phi: float = 0;                                 # 目標發射角度, DOD
lamda: float = c/fc;                            # 訊號波長
d: float = lamda/2.0;                           # 天線間距
Q: int = 1;                                     # 目標數目
SNR: float = 30;                                # SNR
Np: float = 1;                                  # Noise power
An: float = np.sqrt(Np);                        # Noise amplitude
Sp: float = Np*np.power(10, (SNR/10));          # Signal power
As: float = np.sqrt(Sp);                        # Signal amplitude
#------------------------------- 產生資料 ---------------------------------
Y1 = np.zeros((S, P, M * N, frame_len), dtype=complex)
for f in range(frame_len):
    time_idx = 0  # 替代原先的 time = 1，Python 的 index 從 0 開始
    for n in range(N):
        for m in range(M):
            X_target = np.zeros((S, P), dtype=complex)      # 存儲所有目標的中頻訊號
            for q in range(Q):
                IF_data = np.zeros((S, P), dtype=complex)   # 存儲帶有雜訊的中頻訊號
                for p in range(P):
                    # 取出對應的值 (注意 MATLAB 的 () 要改成 Python 的 [])
                    vibration: float = breat_amp * np.sin(2 * np.pi * breat_freq * frame_time[f]) + \
                                       heart_amp * np.sin(2 * np.pi * heart_freq * frame_time[f])
                    
                    new_dis: float = distance + vibration
                    # new_dis = distance[q]
                    
                    fb: float = 2 * k * new_dis / c
                    fd: float = 2 * velocity * fc / c
                    
                    # 虛數單位 1i 轉換為 1j
                    A1 = np.exp(1j * 2 * np.pi * ((fb + fd) * fast_time))
                    
                    # 核心改動：MATLAB 的 p 範圍是 1~P，所以用 (p-1)
                    # Python 的 p 範圍預設就是 0~(P-1)，所以直接寫 p 即可！以下 n, m 同理。
                    B1 = np.exp(1j * 2 * np.pi * fd * p * Ts)
                    C1 = np.exp(1j * 2 * np.pi * fc * 2 * new_dis / c)
                    
                    # MATLAB 的 sind (度數正弦) 需替換為 np.sin(np.deg2rad(...))
                    # lambda 是 Python 保留字，建議變數名稱改為 lambda_ 或 wave_len
                    D1 = np.exp(1j * 2 * np.pi * n * d * np.sin(np.deg2rad(Phi)) / lamda)
                    D2 = np.exp(1j * 2 * np.pi * m * d * np.sin(np.deg2rad(Theta)) / lamda)
                    
                    IF = As * (A1 * B1 * C1 * D1 * D2)
                    noise = An * np.random.randn(S)  # np.random.randn(S) 產生長度 S 的 1D 陣列
                    
                    # IF_data[:, p] = IF + noise
                    IF_data[:, p] = IF
                    
                X_target = X_target + IF_data
                
            Y1[:, :, time_idx, f] = X_target
            time_idx += 1
#--------------------------- 訊號處理與繪圖 --------------------------------
# 1. Range Profile (距離維度 FFT)
range_win = np.hamming(S);

# 宣告存儲結果的矩陣 (必須明確指定 dtype=complex)
range_profile_matrix = np.zeros((S, frame_len), dtype=complex);

#---------------------------------------------------------------------------------------------------
# for i in range(frame_len):
#     # 注意：MATLAB 的 index 1，在 Python 對應為 0
#     temp = Y1[:, 0, 0, i] * range_win;
#     range_profile_matrix[:, i] = np.fft.fft(temp);

# [免迴圈的高效寫法]
# np.newaxis 讓一維的 range_win (S,) 變成二維 (S, 1)，這樣就能直接和 (S, frame_len) 矩陣相乘
temp_matrix = Y1[:, 0, 0, :] * range_win[:, np.newaxis]; 
# 直接對第 0 個維度 (Fast Time) 進行整體的 FFT
range_profile_matrix = np.fft.fft(temp_matrix, axis=0);
#---------------------------------------------------------------------------------------------------

# --- 繪圖部分 (對應 MATLAB 的 figure(5); mesh(...)) ---
fig = plt.figure(num=1, figsize=(10, 6));
ax = fig.add_subplot(111, projection='3d');

# 建立 3D 圖形需要的 X, Y 網格 (對應 Frame 與 距離點數)
X, Y = np.meshgrid(np.arange(frame_len), np.arange(S));

# 取絕對值並繪製網格圖 (plot_wireframe 最接近 MATLAB 的 mesh)
# 如果想要實體的曲面，可以改成 ax.plot_surface
ax.plot_wireframe(X, Y, np.abs(range_profile_matrix), color='b', linewidth=0.5);
ax.set_title('Range Profile (Fast Time FFT)');
ax.set_xlabel('Frame Index (Slow Time)');
ax.set_ylabel('Range Bin (Fast Time)');
ax.set_zlabel('Amplitude');

#-------------------------------------------------------------------------------------
# 1. 找出能量最強的距離點 (Range Bin)
# np.mean 中 axis=1 對應 MATLAB 的 2 (對矩陣的橫向/Frame維度取平均)
mean_profile = np.mean(np.abs(range_profile_matrix), axis=1)

# np.argmax 取代了 MATLAB 的 [~, max_idx] = max(...)，直接回傳最大值的索引
max_idx = np.argmax(mean_profile)
print(f'目標鎖定在 Range Bin: {max_idx}')

# 2. 提取該點的相位 (Vital Sign Extraction)
target_complex_data = range_profile_matrix[max_idx, :]
extracted_phase2 = np.angle(target_complex_data)
extracted_phase = np.unwrap(extracted_phase2)  # 解捲繞

# 3. 去除直流 & 轉為位移 (目前程式碼僅完成去除直流)
phase_vibration = extracted_phase - np.mean(extracted_phase)

# 繪圖 (對應 figure(1) 與 plot)
fig = plt.figure(num=2, figsize=(10, 4))
plt.plot(phase_vibration)
plt.title('Phase Vibration (DC Removed)')
plt.xlabel('Frame Index')
plt.ylabel('Phase (radians)')
plt.grid(True)  # 加上網格會讓生命體徵的波形更容易觀察


#------------------ 訊號處理：帶通濾波 (Bandpass Filtering) ----------------
# 直接傳入截止頻率 [0.1, 0.5] 與取樣頻率 fs=frame_Fs
# 注意：MATLAB 的 'bandpass' 在 Python 中要明確指定給 btype 參數
b_resp, a_resp = butter(2, [0.1, 0.5], btype='bandpass', fs=frame_Fs);
# filtfilt 的用法與 MATLAB 完全一致 (零相位濾波)
resp_filtered = filtfilt(b_resp, a_resp, phase_vibration);


b_heart, a_heart = butter(2, [0.8, 3.0], btype='bandpass', fs=frame_Fs);
# filtfilt 的用法與 MATLAB 完全一致 (零相位濾波)
heart_filtered = filtfilt(b_heart, a_heart, phase_vibration);
#-------------------------------------------------------------------------

# 計算頻率軸 f1
f1 = np.arange(frame_len) * (frame_Fs / frame_len)

# 呼吸 FFT (取絕對值)
fft_resp = np.abs(np.fft.fft(resp_filtered))

# 心跳 FFT (取絕對值)
fft_heart = np.abs(np.fft.fft(heart_filtered))

# 設定取前半段 (正頻率部分) 的索引長度，必須使用整除 (//)
half_len = frame_len // 2

# --- 繪製圖 2：呼吸頻譜 ---
plt.figure(num=3, figsize=(8, 4))
plt.plot(f1[:half_len], fft_resp[:half_len], 'b-')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, frame_Fs / 2)
# plt.ylim(0, 4)
plt.title('呼吸')
plt.grid(True) # 加上網格方便觀察峰值

# --- 繪製圖 3：心跳頻譜 ---
plt.figure(num=4, figsize=(8, 4))
plt.plot(f1[:half_len], fft_heart[:half_len], 'r-')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, frame_Fs / 2)
# plt.ylim(0, 4)
plt.title('心跳')
plt.grid(True)




plt.show()
