from __future__ import annotations

# pyright: reportUnknownMemberType=false

from dataclasses import asdict, dataclass, replace
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


# ------------------------------- Data models ------------------------------ #
@dataclass(frozen=True)
class RadarConfig:
    """FMCW Radar 與生命徵象模擬參數。"""

    # -------------------------- 雷達發射參數 -------------------------- #
    c: float = 3.0e8
    fc: float = 77.0e9
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
    breath_frequency_bpm: float = 15.0

    heart_amplitude_m: float = 0.5e-3
    heart_frequency_bpm: float = 180.0

    # --------------------------- 雜訊參數 ----------------------------- #
    snr_db: float = 30.0
    add_noise: bool = False
    random_seed: int = 42

    # --------------------------- 頻率搜尋範圍 ------------------------- #
    breath_cut_search_low_bpm: float = 6.0
    breath_cut_search_high_bpm: float = 30.0
    heart_cut_search_low_bpm: float = 48.0
    heart_cut_search_high_bpm: float = 120.0

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

    @property
    def heart_frequency_hz(self) -> float:
        return self.heart_frequency_bpm / 60.0

    @property
    def breath_frequency_hz(self) -> float:
        return self.breath_frequency_bpm / 60.0

    @property
    def breath_cut_search_low_hz(self) -> float:
        return self.breath_cut_search_low_bpm / 60.0

    @property
    def breath_cut_search_high_hz(self) -> float:
        return self.breath_cut_search_high_bpm / 60.0

    @property
    def heart_cut_search_low_hz(self) -> float:
        return self.heart_cut_search_low_bpm / 60.0

    @property
    def heart_cut_search_high_hz(self) -> float:
        return self.heart_cut_search_high_bpm / 60.0

    def randomized_for_run(self, run_index: int) -> RadarConfig:
        """從頻率搜尋範圍內產生可重現的單次測試設定。"""

        run_seed: int = self.random_seed + run_index
        rng: np.random.Generator = np.random.default_rng(run_seed)

        return replace(
            self,
            distance_m=float(rng.uniform(0.5, 2.0)),
            velocity_mps=float(rng.uniform(-0.1, 0.1)),
            breath_amplitude_m=float(rng.uniform(1.0e-3, 4.0e-3)),
            breath_frequency_bpm=float(
                rng.uniform(
                    self.breath_cut_search_low_bpm,
                    self.breath_cut_search_high_bpm,
                )
            ),
            heart_amplitude_m=float(rng.uniform(0.2e-3, 0.8e-3)),
            heart_frequency_bpm=float(
                rng.uniform(
                    self.heart_cut_search_low_bpm,
                    self.heart_cut_search_high_bpm,
                )
            ),
            snr_db=float(rng.uniform(10.0, 30.0)),
            add_noise=True,
            random_seed=run_seed,
        )


@dataclass(frozen=True)
class PlotConfig:
    """控制圖片是否儲存與是否顯示。"""

    output_dir: Path = Path("output")
    show_figures: bool = True
    dpi: int = 300

    # FMCW 發射 / 接收 / IF 波形
    save_fmcw_waveform: bool = True

    # Range Profile、位移、濾波結果、頻譜合併圖
    save_vital_sign_summary: bool = True


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


@dataclass(frozen=True)
class BatchRunRecord:
    """單次批次模擬的設定與估測結果。"""

    run_number: int
    random_seed: int
    distance_m: float
    velocity_mps: float
    snr_db: float
    breath_amplitude_mm: float
    breath_bpm: float
    heart_amplitude_mm: float
    heart_bpm: float
    estimated_breath_bpm: float
    estimated_heart_bpm: float
    breath_absolute_error_bpm: float
    heart_absolute_error_bpm: float
    bpm_resolution: float
    target_range_bin: int
    estimated_range_m: float
    range_absolute_error_m: float


# ----------------------------- Output helpers ----------------------------- #
def save_waveform_viewer_config(
    config: RadarConfig,
    plot_config: PlotConfig,
) -> None:
    """輸出網頁 FMCW 波形檢視器所需的雷達設定。"""

    viewer_config: dict[str, float | int] = {
        "carrierHz": config.fc,
        "slopeHzPerSecond": config.chirp_slope,
        "chirpDuration": config.chirp_duration,
        "chirpPeriod": config.chirp_period,
        "chirpsPerFrame": config.chirps_per_loop * config.num_loops,
        "framePeriod": config.frame_periodicity,
        "frameCount": config.frame_length,
    }
    file_path: Path = plot_config.output_dir / "fmcw_waveform_config.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(viewer_config, indent=2),
        encoding="utf-8",
    )
    print(f"[已儲存網頁設定] {file_path}")


def save_first_run_data(
    result: VitalSignResult,
    output_dir: Path,
) -> None:
    """儲存第一輪生命徵象估算的逐點資料。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    time_domain_path: Path = output_dir / "first_run_time_domain.csv"

    with time_domain_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "time_s",
                "ground_truth_breath_mm",
                "ground_truth_heart_mm",
                "estimated_respiration_mm",
                "estimated_heartbeat_mm",
            ]
        )
        writer.writerows(
            (
                result.time_s[index],
                result.ground_truth_breath_mm[index],
                result.ground_truth_heart_mm[index],
                result.estimated_respiration_mm[index],
                result.estimated_heartbeat_mm[index],
            )
            for index in range(result.time_s.size)
        )

    frequency_domain_path: Path = output_dir / "first_run_frequency_domain.csv"

    with frequency_domain_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frequency_hz",
                "respiration_spectrum",
                "heartbeat_spectrum",
            ]
        )
        writer.writerows(
            (
                result.frequency_axis_hz[index],
                result.respiration_spectrum[index],
                result.heartbeat_spectrum[index],
            )
            for index in range(result.frequency_axis_hz.size)
        )

    print(f"[已儲存時域資料] {time_domain_path}")
    print(f"[已儲存頻域資料] {frequency_domain_path}")


def save_batch_results(
    records: list[BatchRunRecord],
    output_dir: Path,
    success_tolerance_bpm: float,
) -> None:
    """將所有批次測試結果與頻率誤差統計輸出為檔案。"""

    if not records:
        raise ValueError("至少需要一筆批次測試結果。")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path: Path = output_dir / "batch_results.csv"
    field_names: list[str] = list(asdict(records[0]))

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    breath_errors_bpm: FloatArray = np.array(
        [record.breath_absolute_error_bpm for record in records],
        dtype=np.float64,
    )
    heart_errors_bpm: FloatArray = np.array(
        [record.heart_absolute_error_bpm for record in records],
        dtype=np.float64,
    )
    range_errors_m: FloatArray = np.array(
        [record.range_absolute_error_m for record in records],
        dtype=np.float64,
    )
    breath_success_rate: float = float(
        np.mean(breath_errors_bpm <= success_tolerance_bpm) * 100.0
    )
    heart_success_rate: float = float(
        np.mean(heart_errors_bpm <= success_tolerance_bpm) * 100.0
    )
    overall_success_rate: float = float(
        np.mean(
            (breath_errors_bpm <= success_tolerance_bpm)
            & (heart_errors_bpm <= success_tolerance_bpm)
        )
        * 100.0
    )

    report: str = "\n".join(
        [
            "FMCW Vital-Sign Batch Statistics",
            "=" * 40,
            f"Case count: {len(records)}",
            f"BPM resolution: {records[0].bpm_resolution:.6f} BPM",
            "",
            "Respiration",
            f"  Mean absolute error: {np.mean(breath_errors_bpm):.3f} BPM",
            f"  Max absolute error:  {np.max(breath_errors_bpm):.3f} BPM",
            f"  Success rate:        {breath_success_rate:.2f}%",
            "",
            "Heartbeat (Original Flow)",
            f"  Mean absolute error: {np.mean(heart_errors_bpm):.3f} BPM",
            f"  Max absolute error:  {np.max(heart_errors_bpm):.3f} BPM",
            f"  Success rate:        {heart_success_rate:.2f}%",
            "",
            "Range",
            f"  Mean absolute error: {np.mean(range_errors_m):.4f} m",
            f"  Max absolute error:  {np.max(range_errors_m):.4f} m",
            "",
            "Overall",
            (
                "  Success rate (both respiration and heartbeat within "
                f"tolerance): {overall_success_rate:.2f}%"
            ),
        ]
    )
    report_path: Path = output_dir / "batch_statistics_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"[已儲存批次結果] {csv_path}")
    print(f"[已儲存統計報告] {report_path}")
    print("\n" + report)


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
        figure.savefig(  # pyright: ignore[reportUnknownMemberType]
            file_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        print(f"[已儲存圖片] {file_path}")

    if should_show:
        plt.show()

    plt.close(figure)


# --------------------------- Signal processing ---------------------------- #
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

    * 需要大於頻率避免被篩掉

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

    estimated_frequency_hz: float = estimate_peak_from_spectrum(
        frequency_axis_hz=frequency_axis,
        magnitude_spectrum=spectrum,
        search_low_hz=search_low_hz,
        search_high_hz=search_high_hz,
    )

    return estimated_frequency_hz, frequency_axis, spectrum


def estimate_peak_from_spectrum(
    frequency_axis_hz: FloatArray,
    magnitude_spectrum: FloatArray,
    search_low_hz: float,
    search_high_hz: float,
) -> float:
    """在指定頻率搜尋範圍內回傳頻譜最大峰值。"""

    valid_mask: npt.NDArray[np.bool_] = (frequency_axis_hz >= search_low_hz) & (
        frequency_axis_hz <= search_high_hz
    )

    valid_indices: npt.NDArray[np.int64] = np.where(valid_mask)[0]

    if valid_indices.size == 0:
        raise ValueError("指定頻率搜尋範圍內沒有 FFT bin。")

    local_peak_index: int = int(np.argmax(magnitude_spectrum[valid_indices]))
    peak_index: int = int(valid_indices[local_peak_index])

    return float(frequency_axis_hz[peak_index])


# -------------------------- Simulation and analysis ----------------------- #
def simulate_and_process(config: RadarConfig) -> VitalSignResult | None:
    """模擬 FMCW 雷達生命徵象訊號，並進行 Range FFT、相位解調與頻率估測。"""

    rng: np.random.Generator = np.random.default_rng(config.random_seed)

    # ---------------------- 建立 Fast Time / Frame Time ---------------------- #
    fast_time: FloatArray = (
        np.arange(config.num_fast_time_samples, dtype=np.float64)
        * config.adc_sample_period
    )

    frame_time: FloatArray = (
        np.arange(config.frame_length, dtype=np.float64) * config.frame_periodicity
    )

    # ------------------------- 原始呼吸 / 心跳位移 -------------------------- #
    ground_truth_breath_m: FloatArray = config.breath_amplitude_m * np.sin(
        2.0 * np.pi * config.breath_frequency_hz * frame_time
    )

    ground_truth_heart_m: FloatArray = config.heart_amplitude_m * np.sin(
        2.0 * np.pi * config.heart_frequency_hz * frame_time
    )

    vibration_m: FloatArray = ground_truth_breath_m + ground_truth_heart_m
    target_distance_m: FloatArray = config.distance_m + vibration_m

    # -------------------------- FMCW IF Signal 模擬 ------------------------- #
    # fb = 2 * slope * range / c
    beat_frequency_hz: FloatArray = (
        2.0 * config.chirp_slope * target_distance_m / config.c
    )

    # fd = 2 * velocity * fc / c
    doppler_frequency_hz: float = 2.0 * config.velocity_mps * config.fc / config.c

    # C1 = exp(j * 2*pi*fc*2R/c)
    carrier_phase: FloatArray = (
        2.0 * np.pi * config.fc * 2.0 * target_distance_m / config.c
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

    if_signal: ComplexArray = (signal_amplitude * np.exp(1j * phase_total)).astype(
        np.complex128
    )

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
        np.arange(config.num_fast_time_samples, dtype=np.float64) * range_resolution_m
    )

    # ------------------------- Range Bin 相位擷取 --------------------------- #
    target_complex_data: ComplexArray = range_profile[target_range_bin, :]
    wrapped_phase_rad: FloatArray = np.angle(target_complex_data)
    wrapped_phase_difference_rad: FloatArray = np.angle(
        target_complex_data[1:] * np.conj(target_complex_data[:-1])
    )

    # 相位差索引 i 代表 frame i 到 i + 1，因此將 i + 1 標記為警告 frame。
    phase_jump_threshold_rad: float = 0.9 * np.pi
    warning_frame_indices: npt.NDArray[np.intp] = (
        np.flatnonzero(np.abs(wrapped_phase_difference_rad) >= phase_jump_threshold_rad)
        + 1
    )

    max_phase_step_rad = float(np.max(np.abs(wrapped_phase_difference_rad)))

    print(f"Maximum adjacent wrapped phase step: " f"{max_phase_step_rad:.4f} rad")

    wrapped_phase_warning: bool = warning_frame_indices.size > 0
    if wrapped_phase_warning:
        print(
            "[警告] 相鄰 Frame 相位差已接近 π，"
            "相位解包可能發生 cycle slip；"
            f"已標記 Frame {warning_frame_indices.tolist()}。"
        )

    true_vibration_phase_rad: FloatArray = (
        4.0 * np.pi * vibration_m / config.wavelength_m
    )

    true_phase_step_rad: FloatArray = np.diff(true_vibration_phase_rad)

    print(
        "Maximum true phase step:",
        np.max(np.abs(true_phase_step_rad)),
        "rad",
    )

    true_phase_warning: bool = bool(np.max(np.abs(true_phase_step_rad)) >= np.pi)
    if true_phase_warning:
        print("[警告] 真實相鄰相位變化達到或超過 π，" "np.unwrap 無法保證正確。")

    if wrapped_phase_warning or true_phase_warning:
        print("[略過] 本次模擬不進行後續估算，也不納入 Pass/Fail 統計。")
        return None

    extracted_phase_rad: FloatArray = np.unwrap(wrapped_phase_rad)

    phase_vibration_rad: FloatArray = extracted_phase_rad - np.mean(extracted_phase_rad)

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
        low_cut_hz=config.breath_cut_search_low_hz,
        high_cut_hz=config.breath_cut_search_high_hz,
        order=2,
    )

    heartbeat_phase_rad: FloatArray = bandpass_filter(
        signal=phase_vibration_rad,
        sampling_rate_hz=config.frame_sampling_rate,
        low_cut_hz=config.heart_cut_search_low_hz,
        high_cut_hz=config.heart_cut_search_high_hz,
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
        search_low_hz=config.breath_cut_search_low_hz,
        search_high_hz=config.breath_cut_search_high_hz,
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
        search_low_hz=config.heart_cut_search_low_hz,
        search_high_hz=config.heart_cut_search_high_hz,
    )
    # 確認呼吸與心跳 FFT frequency axis 相同
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


# --------------------------------- Plotting -------------------------------- #
def plot_transmitted_fmcw_waveform(
    config: RadarConfig,
    plot_config: PlotConfig,
) -> None:
    """繪製一個 frame 中尚未混頻的 FMCW 發射 RF 波形。"""

    chirps_per_frame: int = config.chirps_per_loop * config.num_loops
    active_frame_duration_s: float = chirps_per_frame * config.chirp_period
    capture_duration_s: float = config.frame_length * config.frame_periodicity
    frame_start_times_s: FloatArray = (
        np.arange(config.frame_length, dtype=np.float64) * config.frame_periodicity
    )

    transmit_gate_time_s: list[float] = []
    transmit_gate_amplitude: list[float] = []

    for frame_start_s in frame_start_times_s:
        for chirp_index in range(chirps_per_frame):
            chirp_start_s: float = frame_start_s + chirp_index * config.chirp_period
            chirp_end_s: float = chirp_start_s + config.chirp_duration
            transmit_gate_time_s.extend(
                [chirp_start_s, chirp_start_s, chirp_end_s, chirp_end_s]
            )
            transmit_gate_amplitude.extend([0.0, 1.0, 1.0, 0.0])

    transmit_gate_time: FloatArray = np.array(
        transmit_gate_time_s,
        dtype=np.float64,
    )
    transmit_gate: FloatArray = np.array(
        transmit_gate_amplitude,
        dtype=np.float64,
    )

    # 60 GHz RF 訊號週期極短，因此只放大顯示 chirp 起始的 0.2 ns。
    waveform_duration_s: float = 0.2e-9
    waveform_time: FloatArray = np.linspace(
        0.0,
        waveform_duration_s,
        num=2_000,
        dtype=np.float64,
    )
    transmitted_signal: FloatArray = np.cos(
        2.0
        * np.pi
        * (config.fc * waveform_time + 0.5 * config.chirp_slope * waveform_time**2)
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 11),
    )

    axes[0].plot(
        transmit_gate_time,
        transmit_gate,
        color="#007C7C",
        linewidth=1.5,
    )
    axes[0].set_title("Transmit Gate Across the Complete Slow-Time Capture")
    axes[0].set_xlabel("Slow time (s)")
    axes[0].set_ylabel("Transmit enabled")
    axes[0].set_ylim(-0.15, 1.15)
    axes[0].set_yticks([0.0, 1.0])
    axes[0].grid(True, linestyle="--", alpha=0.5)

    for chirp_index in range(chirps_per_frame):
        chirp_start_s: float = chirp_index * config.chirp_period
        chirp_end_s: float = chirp_start_s + config.chirp_duration
        chirp_frequency_ghz: FloatArray = (
            np.array(
                [config.fc, config.fc + config.bandwidth_hz],
                dtype=np.float64,
            )
            / 1.0e9
        )

        axes[1].plot(
            np.array([chirp_start_s, chirp_end_s]) * 1.0e6,
            chirp_frequency_ghz,
            color="#007C7C",
            linewidth=1.8,
        )
        axes[1].plot(
            np.full(2, chirp_end_s * 1.0e6),
            chirp_frequency_ghz[::-1],
            color="#007C7C",
            linewidth=1.2,
        )
        axes[1].axvspan(
            chirp_end_s * 1.0e6,
            (chirp_start_s + config.chirp_period) * 1.0e6,
            color="lightgray",
            alpha=0.35,
        )

    axes[1].set_title("FMCW Transmit Frequency Across All Chirps in Frame 1")
    axes[1].set_xlabel("Time within active frame burst (µs)")
    axes[1].set_ylabel("Transmit frequency (GHz)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(
        waveform_time * 1.0e9,
        transmitted_signal,
        color="#D99000",
        linewidth=1.2,
    )
    axes[2].set_title("FMCW Transmit RF Waveform (First 0.2 ns of Chirp 1)")
    axes[2].set_xlabel("Time within chirp (ns)")
    axes[2].set_ylabel("Normalized amplitude")
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "FMCW Transmit Signal Before IF Mixing and Range FFT "
        f"({config.frame_length} Frames, {capture_duration_s:.2f} s Capture, "
        f"{active_frame_duration_s * 1.0e6:.2f} µs Active per Frame)",
        fontsize=15,
        fontweight="bold",
    )

    save_or_show_figure(
        figure=fig,
        file_path=plot_config.output_dir / "01_fmcw_transmit_waveform.png",
        should_save=plot_config.save_fmcw_waveform,
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
    第 3 張：呼吸與心跳帶通結果
    第 4 張：呼吸與心跳頻譜
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
        label="Estimated Heart (Original Flow)",
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
            np.abs(result.frequency_axis_hz - result.estimated_breath_frequency_hz)
        )
    )

    heart_peak_index: int = int(
        np.argmin(
            np.abs(result.frequency_axis_hz - result.estimated_heart_frequency_hz)
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
        label="Heartbeat Spectrum (Original Flow)",
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
    fig.suptitle(  # pyright: ignore[reportUnknownMemberType]
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
    num_runs: int = 10
    output_dir: Path = Path("output")
    batch_records: list[BatchRunRecord] = []
    skipped_run_count: int = 0
    first_successful_run_saved: bool = False

    radar_config = RadarConfig(
        distance_m=1.0,
        breath_frequency_bpm=24.0,
        heart_frequency_bpm=96.0,
        add_noise=False,
    )
    specific_configs: tuple[RadarConfig, ...] = ()
    run_configs: tuple[RadarConfig, ...] = specific_configs or tuple(
        radar_config.randomized_for_run(run_index) for run_index in range(num_runs)
    )

    for run_index, run_config in enumerate(run_configs):
        run_number: int = run_index + 1

        print(f"\n開始第 {run_number}/{len(run_configs)} 次模擬")

        result: VitalSignResult | None = simulate_and_process(run_config)
        if result is None:
            skipped_run_count += 1
            continue

        print_result_summary(run_config, result)

        estimated_breath_bpm: float = result.estimated_breath_frequency_hz * 60.0
        estimated_heart_bpm: float = result.estimated_heart_frequency_hz * 60.0
        heart_absolute_error_bpm: float = abs(
            estimated_heart_bpm - run_config.heart_frequency_bpm
        )
        bpm_resolution: float = (
            run_config.frame_sampling_rate / run_config.frame_length * 60.0
        )
        estimated_range_m: float = float(result.range_axis_m[result.target_range_bin])
        batch_records.append(
            BatchRunRecord(
                run_number=run_number,
                random_seed=run_config.random_seed,
                distance_m=run_config.distance_m,
                velocity_mps=run_config.velocity_mps,
                breath_amplitude_mm=run_config.breath_amplitude_m * 1000.0,
                breath_bpm=run_config.breath_frequency_bpm,
                heart_amplitude_mm=run_config.heart_amplitude_m * 1000.0,
                heart_bpm=run_config.heart_frequency_bpm,
                snr_db=run_config.snr_db,
                estimated_breath_bpm=estimated_breath_bpm,
                breath_absolute_error_bpm=abs(
                    estimated_breath_bpm - run_config.breath_frequency_bpm
                ),
                estimated_heart_bpm=estimated_heart_bpm,
                heart_absolute_error_bpm=heart_absolute_error_bpm,
                bpm_resolution=bpm_resolution,
                target_range_bin=result.target_range_bin,
                estimated_range_m=estimated_range_m,
                range_absolute_error_m=abs(estimated_range_m - run_config.distance_m),
            )
        )

        if not first_successful_run_saved:
            plot_config = PlotConfig(
                output_dir=output_dir,
                show_figures=False,
                save_fmcw_waveform=True,
                save_vital_sign_summary=True,
            )
            save_waveform_viewer_config(
                config=run_config,
                plot_config=plot_config,
            )

            #save_first_run_data(
            #    result=result,
            #    output_dir=output_dir,
            #)

            # 僅儲存第一次測試的 FMCW 與生命徵象圖。
            plot_transmitted_fmcw_waveform(
                config=run_config,
                plot_config=plot_config,
            )
            plot_vital_sign_summary(
                config=run_config,
                result=result,
                plot_config=plot_config,
            )
            first_successful_run_saved = True

    print(
        f"\n模擬完成：有效 {len(batch_records)} 次，"
        f"因相位警告略過 {skipped_run_count} 次。"
    )
    if batch_records:
        save_batch_results(
            records=batch_records,
            output_dir=output_dir,
            success_tolerance_bpm=batch_records[0].bpm_resolution,
        )
    else:
        print("[警告] 沒有可納入 Pass/Fail 統計的有效模擬結果。")


if __name__ == "__main__":
    main()
