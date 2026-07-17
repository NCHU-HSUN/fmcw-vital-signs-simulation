from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt


FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


@dataclass(frozen=True)
class RadarConfig:
    """FMCW Radar 與生命徵象模擬參數。"""

    # -------------------------- 雷達發射參數 -------------------------- #
    c: float = 3.0e8
    fc: float = 60.0e9
    chirp_slope: float = 75.0e12

    # --------------------------- Fast Time --------------------------- #
    chirp_duration: float = 10.24e-6
    adc_sampling_rate: float = 25.0e6

    # --------------------------- Slow Time --------------------------- #
    chirp_period: float = 13.84e-6
    chirps_per_loop: int = 7
    num_loops: int = 1

    # -------------------------- Frame Time --------------------------- #
    frame_periodicity: float = 80.0e-3
    frame_length: int = 128

    # --------------------------- 目標參數 ----------------------------- #
    distance_m: float = 1.0
    velocity_mps: float = 0.0

    # -------------------------- 呼吸／心跳參數 ------------------------- #
    breath_amplitude_m: float = 2.0e-3
    breath_frequency_hz: float = 0.25

    heart_amplitude_m: float = 0.5e-3
    heart_frequency_hz: float = 3.0

    # --------------------------- 雜訊參數 ----------------------------- #
    snr_db: float = 30.0
    add_noise: bool = False
    random_seed: int = 42

    @property
    def wavelength_m(self) -> float:
        return self.c / self.fc

    @property
    def frame_sampling_rate(self) -> float:
        return 1.0 / self.frame_periodicity

    @property
    def adc_sample_period(self) -> float:
        return 1.0 / self.adc_sampling_rate

    @property
    def num_fast_time_samples(self) -> int:
        return int(round(self.chirp_duration * self.adc_sampling_rate))

    @property
    def bandwidth_hz(self) -> float:
        return self.chirp_slope * self.chirp_duration


@dataclass(frozen=True)
class PlotConfig:
    """
    控制圖片是否儲存與是否顯示。

    save_range_profile:
        Range FFT 結果，通常用於確認目標所在 Range Bin。

    save_displacement:
        原始理論位移與雷達估計位移，建議保留。

    save_filtered_signals:
        呼吸與心跳帶通濾波結果，建議保留。

    save_spectrum:
        呼吸與心跳頻譜、峰值與估測頻率，建議保留。
    """

    output_dir: Path = Path("output")
    show_figures: bool = True
    dpi: int = 300

    # FMCW 發射 / 接收 / IF 波形
    save_fmcw_waveform: bool = True

    # Range Profile、位移、濾波結果、頻譜合併圖
    save_vital_sign_summary: bool = True


def plot_fmcw_waveform(
    config: RadarConfig,
    plot_config: PlotConfig,
) -> None:
    """
    繪製 FMCW Chirp 時頻圖。

    上半部：
        顯示 2 個 Frame 的壓縮時間軸。

    下半部：
        放大顯示 Frame 1 的 Chirp。
        每個 Chirp 從 0 MHz 線性掃頻至 Bandwidth。
        Chirp 結束後為 Guard Time。
    """

    # 上半部：只顯示兩個 Frame
    frames_to_show: int = 2

    # 下半部：顯示 Frame 1 的所有 Chirp
    chirps_to_show: int = (
        config.chirps_per_loop * config.num_loops
    )

    bandwidth_hz: float = config.bandwidth_hz
    bandwidth_mhz: float = bandwidth_hz / 1.0e6

    chirp_duration_us: float = config.chirp_duration * 1.0e6
    chirp_period_us: float = config.chirp_period * 1.0e6
    guard_time_us: float = chirp_period_us - chirp_duration_us

    frame_period_us: float = config.frame_periodicity * 1.0e6

    frame_colors: tuple[str, str] = (
        "#007C7C",  # Frame 1: Teal
        "#D99000",  # Frame 2: Orange
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 11),
        gridspec_kw={
            "height_ratios": [1.0, 1.35],
        },
    )

    # ====================================================================== #
    # 上半部：2 個 Frame 壓縮時間軸
    # ====================================================================== #
    top_ax = axes[0]

    for frame_index in range(frames_to_show):
        frame_number: int = frame_index + 1
        frame_start_us: float = frame_index * frame_period_us
        color: str = frame_colors[frame_index]

        # 壓縮時間軸中，每個 Frame 使用一條垂直線表示 FMCW 掃頻
        top_ax.plot(
            [frame_start_us, frame_start_us],
            [0.0, bandwidth_mhz],
            color=color,
            linewidth=2.2,
            label=f"Frame {frame_number}",
        )

        # Frame 名稱標記
        label_x_us: float = frame_start_us + frame_period_us * 0.50

        top_ax.text(
            label_x_us,
            bandwidth_mhz * 1.05,
            f"F{frame_number}",
            color=color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    top_ax.set_title(
        "FMCW Frame Timeline (Compressed View)",
        fontsize=13,
        fontweight="bold",
    )

    top_ax.set_xlabel(
        "Time (µs, compressed — "
        f"1 frame = {config.frame_periodicity * 1.0e3:.0f} ms)"
    )

    top_ax.set_ylabel("Relative Frequency (MHz from $f_c$)")

    # 顯示 F1 與 F2，x 軸延伸到第二個 Frame 結束
    top_ax.set_xlim(
        0.0,
        frames_to_show * frame_period_us,
    )

    top_ax.set_ylim(
        -bandwidth_mhz * 0.05,
        bandwidth_mhz * 1.15,
    )

    top_ax.grid(True, linestyle="--", alpha=0.5)

    top_ax.legend(
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.0, 1.18),
        frameon=False,
    )

    # ====================================================================== #
    # 下半部：Frame 1 的全部 Chirp
    # ====================================================================== #
    bottom_ax = axes[1]

    for chirp_index in range(chirps_to_show):
        chirp_number: int = chirp_index + 1

        chirp_start_us: float = chirp_index * chirp_period_us
        chirp_end_us: float = chirp_start_us + chirp_duration_us
        chirp_period_end_us: float = chirp_start_us + chirp_period_us

        # FMCW Up-Chirp: 0 MHz -> Bandwidth MHz
        chirp_time_us: FloatArray = np.array(
            [chirp_start_us, chirp_end_us],
            dtype=np.float64,
        )

        chirp_frequency_mhz: FloatArray = np.array(
            [0.0, bandwidth_mhz],
            dtype=np.float64,
        )

        bottom_ax.plot(
            chirp_time_us,
            chirp_frequency_mhz,
            color="#007C7C",
            linewidth=2.0,
        )

        # Chirp 結束後頻率歸零
        bottom_ax.plot(
            [chirp_end_us, chirp_end_us],
            [bandwidth_mhz, 0.0],
            color="#007C7C",
            linewidth=2.0,
        )

        # Guard Time 平坦區域
        bottom_ax.plot(
            [chirp_end_us, chirp_period_end_us],
            [0.0, 0.0],
            color="#007C7C",
            linewidth=2.0,
        )

        # Guard Time 灰色背景
        bottom_ax.axvspan(
            chirp_end_us,
            chirp_period_end_us,
            color="lightgray",
            alpha=0.35,
        )

        # Guard 標示
        guard_center_us: float = (
            chirp_end_us + chirp_period_end_us
        ) / 2.0

        bottom_ax.text(
            guard_center_us,
            bandwidth_mhz * 0.56,
            "Guard",
            color="gray",
            fontsize=8,
            ha="center",
            va="center",
        )

        # Chirp 編號
        chirp_center_us: float = (
            chirp_start_us + chirp_end_us
        ) / 2.0

        bottom_ax.text(
            chirp_center_us,
            bandwidth_mhz * 1.04,
            f"Chirp {chirp_number}",
            color="#007C7C",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    # 僅在第一個 Chirp 標記 Tc / Ts / Guard Time
    first_chirp_center_us: float = chirp_duration_us / 2.0

    bottom_ax.text(
        first_chirp_center_us,
        -bandwidth_mhz * 0.13,
        f"Tc = {chirp_duration_us:.2f} µs",
        color="#007C7C",
        fontsize=9,
        ha="center",
        va="top",
    )

    bottom_ax.text(
        chirp_period_us / 2.0,
        -bandwidth_mhz * 0.25,
        f"Ts = {chirp_period_us:.2f} µs",
        color="#D99000",
        fontsize=9,
        ha="center",
        va="top",
    )

    bottom_ax.text(
        chirp_duration_us + guard_time_us / 2.0,
        -bandwidth_mhz * 0.13,
        f"Guard = {guard_time_us:.2f} µs",
        color="gray",
        fontsize=9,
        ha="center",
        va="top",
    )

    total_detail_time_us: float = chirps_to_show * chirp_period_us

    bottom_ax.set_title(
        f"Frame 1 Detail ({chirps_to_show} Chirps)",
        fontsize=13,
        fontweight="bold",
    )

    bottom_ax.set_xlabel("Time (µs)")
    bottom_ax.set_ylabel("Relative Frequency (MHz from $f_c$)")

    bottom_ax.set_xlim(
        0.0,
        total_detail_time_us,
    )

    bottom_ax.set_ylim(
        -bandwidth_mhz * 0.32,
        bandwidth_mhz * 1.12,
    )

    bottom_ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "FMCW Chirp Structure",
        fontsize=16,
        fontweight="bold",
    )

    fig.subplots_adjust(
        top=0.90,
        hspace=0.65,
    )

    save_or_show_figure(
        figure=fig,
        file_path=plot_config.output_dir / "01_fmcw_waveform.png",
        should_save=plot_config.save_fmcw_waveform,
        should_show=plot_config.show_figures,
        dpi=plot_config.dpi,
    )


@dataclass(frozen=True)
class VitalSignResult:
    """生命徵象處理結果。"""

    time_s: FloatArray
    range_axis_m: FloatArray
    range_profile: ComplexArray
    target_range_bin: int

    ground_truth_breath_mm: FloatArray
    ground_truth_heart_mm: FloatArray
    ground_truth_total_mm: FloatArray

    estimated_displacement_mm: FloatArray
    estimated_respiration_mm: FloatArray
    estimated_heartbeat_mm: FloatArray

    frequency_axis_hz: FloatArray
    respiration_spectrum: FloatArray
    heartbeat_spectrum: FloatArray

    estimated_breath_frequency_hz: float
    estimated_heart_frequency_hz: float


def save_or_show_figure(
    figure: Figure,
    file_path: Path,
    should_save: bool,
    should_show: bool,
    dpi: int,
) -> None:
    """依設定決定圖片是否儲存及顯示。"""

    figure.tight_layout()

    if should_save:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(file_path), dpi=dpi, bbox_inches="tight")
        print(f"[已儲存圖片] {file_path}")

    if should_show:
        plt.show()

    plt.close(figure)


def bandpass_filter(
    signal: FloatArray,
    sampling_rate_hz: float,
    low_cut_hz: float,
    high_cut_hz: float,
    order: int = 2,
) -> FloatArray:
    """Butterworth Bandpass + zero-phase filtfilt。"""

    nyquist_hz: float = sampling_rate_hz / 2.0

    if low_cut_hz <= 0.0:
        raise ValueError("low_cut_hz 必須大於 0。")

    if high_cut_hz >= nyquist_hz:
        raise ValueError(
            f"high_cut_hz={high_cut_hz} 必須小於 Nyquist frequency={nyquist_hz}。"
        )

    normalized_band: list[float] = [
        low_cut_hz / nyquist_hz,
        high_cut_hz / nyquist_hz,
    ]

    b: FloatArray
    a: FloatArray
    b, a = butter(order, normalized_band, btype="bandpass")

    filtered: FloatArray = filtfilt(b, a, signal)
    return filtered


def estimate_peak_frequency(
    signal: FloatArray,
    sampling_rate_hz: float,
    search_low_hz: float,
    search_high_hz: float,
) -> tuple[float, FloatArray, FloatArray]:
    """
    使用 rFFT 搜尋指定頻率範圍的最大頻譜峰值。

    回傳:
        estimated_frequency_hz,
        frequency_axis_hz,
        magnitude_spectrum
    """

    signal_length: int = signal.size
    window: FloatArray = np.hamming(signal_length)
    windowed_signal: FloatArray = (signal - np.mean(signal)) * window

    spectrum_complex: ComplexArray = np.fft.rfft(windowed_signal)
    spectrum: FloatArray = np.abs(spectrum_complex)
    frequency_axis: FloatArray = np.fft.rfftfreq(
        signal_length,
        d=1.0 / sampling_rate_hz,
    )

    valid_mask: npt.NDArray[np.bool_] = (
        (frequency_axis >= search_low_hz)
        & (frequency_axis <= search_high_hz)
    )

    valid_indices: npt.NDArray[np.int64] = np.where(valid_mask)[0]

    if valid_indices.size == 0:
        raise ValueError("指定頻率搜尋範圍內沒有 FFT bin。")

    local_peak_index: int = int(np.argmax(spectrum[valid_indices]))
    peak_index: int = int(valid_indices[local_peak_index])

    estimated_frequency_hz: float = float(frequency_axis[peak_index])

    return estimated_frequency_hz, frequency_axis, spectrum


def simulate_and_process(config: RadarConfig) -> VitalSignResult:
    """模擬 FMCW 雷達生命徵象訊號，並進行 Range FFT、相位解調與頻率估測。"""

    rng: np.random.Generator = np.random.default_rng(config.random_seed)

    # ---------------------- 建立 Fast Time / Frame Time ---------------------- #
    fast_time: FloatArray = (
        np.arange(config.num_fast_time_samples, dtype=np.float64)
        * config.adc_sample_period
    )

    frame_time: FloatArray = (
        np.arange(config.frame_length, dtype=np.float64)
        * config.frame_periodicity
    )

    # ------------------------- 原始呼吸 / 心跳位移 -------------------------- #
    ground_truth_breath_m: FloatArray = (
        config.breath_amplitude_m
        * np.sin(2.0 * np.pi * config.breath_frequency_hz * frame_time)
    )

    ground_truth_heart_m: FloatArray = (
        config.heart_amplitude_m
        * np.sin(2.0 * np.pi * config.heart_frequency_hz * frame_time)
    )

    vibration_m: FloatArray = ground_truth_breath_m + ground_truth_heart_m
    target_distance_m: FloatArray = config.distance_m + vibration_m

    # -------------------------- FMCW IF Signal 模擬 ------------------------- #
    # fb = 2 * slope * range / c
    beat_frequency_hz: FloatArray = (
        2.0 * config.chirp_slope * target_distance_m / config.c
    )

    # fd = 2 * velocity * fc / c
    doppler_frequency_hz: float = (
        2.0 * config.velocity_mps * config.fc / config.c
    )

    # C1 = exp(j * 2*pi*fc*2R/c)
    carrier_phase: FloatArray = (
        2.0
        * np.pi
        * config.fc
        * 2.0
        * target_distance_m
        / config.c
    )

    # 建立 [fast_time, frame] 的矩陣
    phase_fast_time: FloatArray = (
        2.0
        * np.pi
        * (beat_frequency_hz[np.newaxis, :] + doppler_frequency_hz)
        * fast_time[:, np.newaxis]
    )

    phase_total: FloatArray = phase_fast_time + carrier_phase[np.newaxis, :]

    signal_power: float = 10.0 ** (config.snr_db / 10.0)
    signal_amplitude: float = np.sqrt(signal_power)

    if_signal: ComplexArray = signal_amplitude * np.exp(1j * phase_total)

    if config.add_noise:
        noise_power: float = 1.0
        noise_amplitude: float = np.sqrt(noise_power)

        noise_real: FloatArray = rng.normal(
            loc=0.0,
            scale=noise_amplitude,
            size=if_signal.shape,
        )
        noise_imag: FloatArray = rng.normal(
            loc=0.0,
            scale=noise_amplitude,
            size=if_signal.shape,
        )

        complex_noise: ComplexArray = noise_real + 1j * noise_imag
        if_signal = if_signal + complex_noise

    # ------------------------------- Range FFT ------------------------------ #
    range_window: FloatArray = np.hamming(config.num_fast_time_samples)

    range_profile: ComplexArray = np.fft.fft(
        if_signal * range_window[:, np.newaxis],
        axis=0,
    )

    average_range_energy: FloatArray = np.mean(np.abs(range_profile), axis=1)
    target_range_bin: int = int(np.argmax(average_range_energy))

    # FMCW range resolution = c / (2 * BW)
    range_resolution_m: float = config.c / (2.0 * config.bandwidth_hz)
    range_axis_m: FloatArray = (
        np.arange(config.num_fast_time_samples, dtype=np.float64)
        * range_resolution_m
    )

    # ------------------------- Range Bin 相位擷取 --------------------------- #
    target_complex_data: ComplexArray = range_profile[target_range_bin, :]
    extracted_phase_rad: FloatArray = np.unwrap(np.angle(target_complex_data))

    phase_vibration_rad: FloatArray = (
        extracted_phase_rad - np.mean(extracted_phase_rad)
    )

    # Phase -> displacement:
    # phase = 4*pi*displacement/lambda
    estimated_displacement_m: FloatArray = (
        phase_vibration_rad * config.wavelength_m / (4.0 * np.pi)
    )
    estimated_displacement_mm: FloatArray = estimated_displacement_m * 1000.0

    # ---------------------------- 呼吸與心跳濾波 ---------------------------- #
    respiration_phase_rad: FloatArray = bandpass_filter(
        signal=phase_vibration_rad,
        sampling_rate_hz=config.frame_sampling_rate,
        low_cut_hz=0.1,
        high_cut_hz=0.5,
        order=2,
    )

    heartbeat_phase_rad: FloatArray = bandpass_filter(
        signal=phase_vibration_rad,
        sampling_rate_hz=config.frame_sampling_rate,
        low_cut_hz=1.0,
        high_cut_hz=3.0,
        order=2,
    )

    respiration_mm: FloatArray = (
        respiration_phase_rad * config.wavelength_m / (4.0 * np.pi) * 1000.0
    )

    heartbeat_mm: FloatArray = (
        heartbeat_phase_rad * config.wavelength_m / (4.0 * np.pi) * 1000.0
    )

    # -------------------------- 頻率 / BPM 估測 ----------------------------- #
    estimated_breath_frequency_hz: float
    frequency_axis_hz: FloatArray
    respiration_spectrum: FloatArray

    (
        estimated_breath_frequency_hz,
        frequency_axis_hz,
        respiration_spectrum,
    ) = estimate_peak_frequency(
        signal=respiration_mm,
        sampling_rate_hz=config.frame_sampling_rate,
        search_low_hz=0.1,
        search_high_hz=0.5,
    )

    estimated_heart_frequency_hz: float
    heartbeat_frequency_axis_hz: FloatArray
    heartbeat_spectrum: FloatArray

    (
        estimated_heart_frequency_hz,
        heartbeat_frequency_axis_hz,
        heartbeat_spectrum,
    ) = estimate_peak_frequency(
        signal=heartbeat_mm,
        sampling_rate_hz=config.frame_sampling_rate,
        search_low_hz=1.0,
        search_high_hz=3.0,
    )

    # 確認兩者 FFT frequency axis 相同
    if not np.allclose(frequency_axis_hz, heartbeat_frequency_axis_hz):
        raise RuntimeError("呼吸與心跳的頻率軸不一致。")

    return VitalSignResult(
        time_s=frame_time,
        range_axis_m=range_axis_m,
        range_profile=range_profile,
        target_range_bin=target_range_bin,
        ground_truth_breath_mm=ground_truth_breath_m * 1000.0,
        ground_truth_heart_mm=ground_truth_heart_m * 1000.0,
        ground_truth_total_mm=vibration_m * 1000.0,
        estimated_displacement_mm=estimated_displacement_mm,
        estimated_respiration_mm=respiration_mm,
        estimated_heartbeat_mm=heartbeat_mm,
        frequency_axis_hz=frequency_axis_hz,
        respiration_spectrum=respiration_spectrum,
        heartbeat_spectrum=heartbeat_spectrum,
        estimated_breath_frequency_hz=estimated_breath_frequency_hz,
        estimated_heart_frequency_hz=estimated_heart_frequency_hz,
    )


def plot_range_profile(
    result: VitalSignResult,
    plot_config: PlotConfig,
) -> None:
    """繪製平均 Range Profile。"""

    mean_magnitude: FloatArray = np.mean(np.abs(result.range_profile), axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.range_axis_m, mean_magnitude, color="tab:blue")
    ax.axvline(
        result.range_axis_m[result.target_range_bin],
        color="red",
        linestyle="--",
        label=(
            f"Target bin = {result.target_range_bin}, "
            f"range = {result.range_axis_m[result.target_range_bin]:.3f} m"
        ),
    )

    ax.set_title("Range Profile")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    ax.legend()

    save_or_show_figure(
        figure=fig,
        file_path=plot_config.output_dir / "01_range_profile.png",
        should_save=plot_config.save_range_profile,
        should_show=plot_config.show_figures,
        dpi=plot_config.dpi,
    )


def plot_displacement(
    result: VitalSignResult,
    plot_config: PlotConfig,
) -> None:
    """繪製理論真值與雷達估計位移。"""

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        result.time_s,
        result.ground_truth_total_mm,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Ground Truth Total (Breath + Heart)",
    )

    ax.plot(
        result.time_s,
        result.estimated_displacement_mm,
        color="tab:blue",
        linewidth=1.5,
        label="Radar Estimated Displacement",
    )

    ax.plot(
        result.time_s,
        result.ground_truth_breath_mm,
        color="tab:green",
        alpha=0.7,
        label="Ground Truth Breath",
    )

    ax.plot(
        result.time_s,
        result.ground_truth_heart_mm,
        color="tab:red",
        alpha=0.7,
        label="Ground Truth Heart",
    )

    ax.set_title("Original and Estimated Vital Sign Displacement")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm)")
    ax.grid(True)
    ax.legend(loc="upper right")

    save_or_show_figure(
        figure=fig,
        file_path=plot_config.output_dir / "02_displacement_comparison.png",
        should_save=plot_config.save_displacement,
        should_show=plot_config.show_figures,
        dpi=plot_config.dpi,
    )


def plot_filtered_signals(
    result: VitalSignResult,
    plot_config: PlotConfig,
) -> None:
    """繪製濾波分離後的呼吸與心跳訊號。"""

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
    )

    axes[0].plot(
        result.time_s,
        result.ground_truth_breath_mm,
        color="black",
        linestyle="--",
        label="Ground Truth Breath",
    )
    axes[0].plot(
        result.time_s,
        result.estimated_respiration_mm,
        color="tab:blue",
        label="Estimated Breath",
    )
    axes[0].set_title("Respiration Signal")
    axes[0].set_ylabel("Displacement (mm)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        result.time_s,
        result.ground_truth_heart_mm,
        color="black",
        linestyle="--",
        label="Ground Truth Heart",
    )
    axes[1].plot(
        result.time_s,
        result.estimated_heartbeat_mm,
        color="tab:red",
        label="Estimated Heart",
    )
    axes[1].set_title("Heartbeat Signal")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Displacement (mm)")
    axes[1].grid(True)
    axes[1].legend()

    save_or_show_figure(
        figure=fig,
        file_path=plot_config.output_dir / "03_filtered_vital_signals.png",
        should_save=plot_config.save_filtered_signals,
        should_show=plot_config.show_figures,
        dpi=plot_config.dpi,
    )


def plot_spectrum(
    config: RadarConfig,
    result: VitalSignResult,
    plot_config: PlotConfig,
) -> None:
    """繪製呼吸與心跳頻譜，以及估測峰值。"""

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
    )

    breath_peak_index: int = int(
        np.argmin(
            np.abs(
                result.frequency_axis_hz
                - result.estimated_breath_frequency_hz
            )
        )
    )

    axes[0].plot(
        result.frequency_axis_hz,
        result.respiration_spectrum,
        color="tab:blue",
        label="Respiration Spectrum",
    )
    axes[0].plot(
        result.estimated_breath_frequency_hz,
        result.respiration_spectrum[breath_peak_index],
        marker="o",
        color="red",
        label=(
            f"Estimated: {result.estimated_breath_frequency_hz:.3f} Hz "
            f"({result.estimated_breath_frequency_hz * 60.0:.1f} BPM)"
        ),
    )
    axes[0].axvline(
        config.breath_frequency_hz,
        color="green",
        linestyle="--",
        label=f"Ground truth: {config.breath_frequency_hz:.3f} Hz",
    )
    axes[0].set_xlim(0.0, 0.8)
    axes[0].set_title("Respiration Spectrum")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True)
    axes[0].legend()

    heart_peak_index: int = int(
        np.argmin(
            np.abs(
                result.frequency_axis_hz
                - result.estimated_heart_frequency_hz
            )
        )
    )

    axes[1].plot(
        result.frequency_axis_hz,
        result.heartbeat_spectrum,
        color="tab:red",
        label="Heartbeat Spectrum",
    )
    axes[1].plot(
        result.estimated_heart_frequency_hz,
        result.heartbeat_spectrum[heart_peak_index],
        marker="o",
        color="blue",
        label=(
            f"Estimated: {result.estimated_heart_frequency_hz:.3f} Hz "
            f"({result.estimated_heart_frequency_hz * 60.0:.1f} BPM)"
        ),
    )
    axes[1].axvline(
        config.heart_frequency_hz,
        color="green",
        linestyle="--",
        label=f"Ground truth: {config.heart_frequency_hz:.3f} Hz",
    )
    axes[1].set_xlim(0.0, 4.0)
    axes[1].set_title("Heartbeat Spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True)
    axes[1].legend()

    save_or_show_figure(
        figure=fig,
        file_path=plot_config.output_dir / "04_vital_sign_spectrum.png",
        should_save=plot_config.save_spectrum,
        should_show=plot_config.show_figures,
        dpi=plot_config.dpi,
    )

def plot_vital_sign_summary(
    config: RadarConfig,
    result: VitalSignResult,
    plot_config: PlotConfig,
) -> None:
    """
    將生命徵象結果合成一張 4x1 垂直圖片：

    第 1 張：Range Profile
    第 2 張：原始位移與雷達估計位移
    第 3 張：呼吸與心跳帶通濾波結果
    第 4 張：呼吸與心跳頻譜與峰值估計
    """

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(14, 20),
    )

    # ====================================================================== #
    # 圖 1：Range Profile
    # ====================================================================== #
    range_ax = axes[0]

    mean_magnitude: FloatArray = np.mean(
        np.abs(result.range_profile),
        axis=1,
    )

    target_range_m: float = result.range_axis_m[result.target_range_bin]

    range_ax.plot(
        result.range_axis_m,
        mean_magnitude,
        color="tab:blue",
        linewidth=1.5,
        label="Mean Range FFT Magnitude",
    )

    range_ax.axvline(
        target_range_m,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Target Bin = {result.target_range_bin}, "
            f"Estimated Range = {target_range_m:.3f} m"
        ),
    )

    range_ax.set_title("1. Range Profile")
    range_ax.set_xlabel("Range (m)")
    range_ax.set_ylabel("Magnitude")
    range_ax.grid(True)
    range_ax.legend()

    # ====================================================================== #
    # 圖 2：原始位移與雷達估計位移
    # ====================================================================== #
    displacement_ax = axes[1]

    displacement_ax.plot(
        result.time_s,
        result.ground_truth_total_mm,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Ground Truth Total (Breath + Heart)",
    )

    displacement_ax.plot(
        result.time_s,
        result.estimated_displacement_mm,
        color="tab:blue",
        linewidth=1.5,
        label="Radar Estimated Displacement",
    )

    displacement_ax.plot(
        result.time_s,
        result.ground_truth_breath_mm,
        color="tab:green",
        alpha=0.7,
        linewidth=1.0,
        label="Ground Truth Breath",
    )

    displacement_ax.plot(
        result.time_s,
        result.ground_truth_heart_mm,
        color="tab:red",
        alpha=0.7,
        linewidth=1.0,
        label="Ground Truth Heart",
    )

    displacement_ax.set_title("2. Ground Truth and Estimated Displacement")
    displacement_ax.set_xlabel("Time (s)")
    displacement_ax.set_ylabel("Displacement (mm)")
    displacement_ax.grid(True)
    displacement_ax.legend(loc="upper right", fontsize=8)

    # ====================================================================== #
    # 圖 3：呼吸與心跳帶通濾波結果
    # ====================================================================== #
    filtered_ax = axes[2]

    filtered_ax.plot(
        result.time_s,
        result.ground_truth_breath_mm,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Ground Truth Breath",
    )

    filtered_ax.plot(
        result.time_s,
        result.estimated_respiration_mm,
        color="tab:blue",
        linewidth=1.5,
        label="Estimated Breath",
    )

    filtered_ax.plot(
        result.time_s,
        result.ground_truth_heart_mm,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Ground Truth Heart",
    )

    filtered_ax.plot(
        result.time_s,
        result.estimated_heartbeat_mm,
        color="tab:red",
        linewidth=1.2,
        label="Estimated Heart",
    )

    filtered_ax.set_title("3. Bandpass Filtered Respiration and Heartbeat")
    filtered_ax.set_xlabel("Time (s)")
    filtered_ax.set_ylabel("Displacement (mm)")
    filtered_ax.grid(True)
    filtered_ax.legend(loc="upper right", fontsize=8)

    # ====================================================================== #
    # 圖 4：呼吸與心跳頻譜
    # ====================================================================== #
    spectrum_ax = axes[3]

    breath_peak_index: int = int(
        np.argmin(
            np.abs(
                result.frequency_axis_hz
                - result.estimated_breath_frequency_hz
            )
        )
    )

    heart_peak_index: int = int(
        np.argmin(
            np.abs(
                result.frequency_axis_hz
                - result.estimated_heart_frequency_hz
            )
        )
    )

    spectrum_ax.plot(
        result.frequency_axis_hz,
        result.respiration_spectrum,
        color="tab:blue",
        linewidth=1.5,
        label="Respiration Spectrum",
    )

    spectrum_ax.plot(
        result.frequency_axis_hz,
        result.heartbeat_spectrum,
        color="tab:red",
        linewidth=1.5,
        label="Heartbeat Spectrum",
    )

    spectrum_ax.plot(
        result.estimated_breath_frequency_hz,
        result.respiration_spectrum[breath_peak_index],
        marker="o",
        markersize=8,
        color="blue",
        label=(
            f"Estimated Breath = "
            f"{result.estimated_breath_frequency_hz:.3f} Hz "
            f"({result.estimated_breath_frequency_hz * 60.0:.1f} BPM)"
        ),
    )

    spectrum_ax.plot(
        result.estimated_heart_frequency_hz,
        result.heartbeat_spectrum[heart_peak_index],
        marker="o",
        markersize=8,
        color="red",
        label=(
            f"Estimated Heart = "
            f"{result.estimated_heart_frequency_hz:.3f} Hz "
            f"({result.estimated_heart_frequency_hz * 60.0:.1f} BPM)"
        ),
    )

    spectrum_ax.axvline(
        config.breath_frequency_hz,
        color="tab:green",
        linestyle="--",
        linewidth=1.2,
        label=(
            f"True Breath = {config.breath_frequency_hz:.3f} Hz "
            f"({config.breath_frequency_hz * 60.0:.1f} BPM)"
        ),
    )

    spectrum_ax.axvline(
        config.heart_frequency_hz,
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label=(
            f"True Heart = {config.heart_frequency_hz:.3f} Hz "
            f"({config.heart_frequency_hz * 60.0:.1f} BPM)"
        ),
    )

    spectrum_ax.set_title("4. Respiration and Heartbeat Spectrum")
    spectrum_ax.set_xlabel("Frequency (Hz)")
    spectrum_ax.set_ylabel("Magnitude")
    spectrum_ax.set_xlim(0.0, 4.0)
    spectrum_ax.grid(True)
    spectrum_ax.legend(loc="upper right", fontsize=8)

    # 整張圖標題
    fig.suptitle(
        "FMCW Radar Vital Sign Detection Summary",
        fontsize=16,
        fontweight="bold",
    )

    # 調整各子圖距離，避免標題、legend 重疊
    fig.subplots_adjust(
        top=0.95,
        bottom=0.05,
        hspace=0.45,
    )

    save_or_show_figure(
        figure=fig,
        file_path=plot_config.output_dir / "02_vital_sign_summary_4x1.png",
        should_save=plot_config.save_vital_sign_summary,
        should_show=plot_config.show_figures,
        dpi=plot_config.dpi,
    )

def print_result_summary(
    config: RadarConfig,
    result: VitalSignResult,
) -> None:
    """輸出原始與估測頻率結果。"""

    original_breath_bpm: float = config.breath_frequency_hz * 60.0
    original_heart_bpm: float = config.heart_frequency_hz * 60.0

    estimated_breath_bpm: float = result.estimated_breath_frequency_hz * 60.0
    estimated_heart_bpm: float = result.estimated_heart_frequency_hz * 60.0

    target_range_m: float = result.range_axis_m[result.target_range_bin]

    print("\n" + "=" * 70)
    print("FMCW Radar Vital Sign Estimation Result")
    print("=" * 70)
    print(f"Frame sampling rate : {config.frame_sampling_rate:.4f} Hz")
    print(f"Frame length        : {config.frame_length}")
    print(f"Target Range Bin    : {result.target_range_bin}")
    print(f"Estimated Range     : {target_range_m:.4f} m")
    print("-" * 70)
    print("Respiration")
    print(
        f"  Ground Truth : {config.breath_frequency_hz:.3f} Hz "
        f"({original_breath_bpm:.2f} BPM)"
    )
    print(
        f"  Estimated    : {result.estimated_breath_frequency_hz:.3f} Hz "
        f"({estimated_breath_bpm:.2f} BPM)"
    )
    print("-" * 70)
    print("Heartbeat")
    print(
        f"  Ground Truth : {config.heart_frequency_hz:.3f} Hz "
        f"({original_heart_bpm:.2f} BPM)"
    )
    print(
        f"  Estimated    : {result.estimated_heart_frequency_hz:.3f} Hz "
        f"({estimated_heart_bpm:.2f} BPM)"
    )
    print("=" * 70 + "\n")


def main() -> None:
    radar_config = RadarConfig(
        distance_m=1.0,
        breath_frequency_hz=0.25,
        heart_frequency_hz=3.0,
        add_noise=False,
    )

    plot_config = PlotConfig(
        output_dir=Path("output"),
        show_figures=True,
        save_fmcw_waveform=True,
        save_vital_sign_summary=True,
    )

    result: VitalSignResult = simulate_and_process(radar_config)

    print_result_summary(radar_config, result)

    # 圖 1：FMCW Chirp / Rx / IF Beat Signal
    plot_fmcw_waveform(
        config=radar_config,
        plot_config=plot_config,
    )

    # 圖 2：生命徵象總覽圖
    plot_vital_sign_summary(
        config=radar_config,
        result=result,
        plot_config=plot_config,
    )


if __name__ == "__main__":
    main()