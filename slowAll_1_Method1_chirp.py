from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
from functools import lru_cache
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt, sosfreqz

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
    chirps_per_loop: int = 1
    num_loops: int = 7

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
    heart_frequency_bpm: float = 120.0

    # --------------------------- 雜訊參數 ----------------------------- #
    snr_db: float = 30.0
    add_noise: bool = False
    random_seed: int = 42

    # --------------------------- 頻率搜尋範圍 ------------------------- #
    breath_cut_search_low_bpm: float = 6.0
    breath_cut_search_high_bpm: float = 30.0
    heart_cut_search_low_bpm: float = 48.0
    heart_cut_search_high_bpm: float = 120.0

    # 搜尋範圍邊界經過雙向 Butterworth 濾波後，允許的最大衰減。
    filter_order: int = 2
    filter_max_passband_attenuation_db: float = 0.2

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

    # True / Recovered phase 與分支變更位置
    save_phase_branch_diagnostics: bool = True

    # Range FFT 強度隨距離與 frame time 的 3D 圖
    save_range_time_3d: bool = True


@dataclass(frozen=True)
class VitalSignResult:
    """生命徵象處理結果。"""

    # ``time_s`` 是依 Frame、再依 Chirp 排序的非均勻量測時間軸。
    time_s: FloatArray
    # 濾波與 FFT 必須使用均勻的 Frame 時間軸，不能把所有 Chirp 當成
    # 固定 87.5 Hz 的均勻取樣。
    frame_time_s: FloatArray
    frame_index_per_measurement: npt.NDArray[np.int64]
    chirp_index_per_measurement: npt.NDArray[np.int64]
    range_axis_m: FloatArray
    range_profile: ComplexArray
    target_range_bin: int

    ground_truth_breath_mm: FloatArray
    ground_truth_heart_mm: FloatArray
    ground_truth_total_mm: FloatArray
    true_vibration_phase_rad: FloatArray
    recovered_phase_rad: FloatArray
    branch_error_index: npt.NDArray[np.int64]
    branch_change_frames: npt.NDArray[np.int64]

    estimated_displacement_mm: FloatArray
    estimated_respiration_mm: FloatArray
    estimated_heartbeat_mm: FloatArray

    frequency_axis_hz: FloatArray
    respiration_spectrum: FloatArray
    heartbeat_spectrum: FloatArray

    estimated_breath_frequency_hz: float
    estimated_heart_frequency_hz: float
    max_true_phase_step_rad: float
    max_recovered_phase_step_rad: float
    phase_rmse_rad: float
    max_phase_error_rad: float
    true_phase_risk_count: int
    actual_branch_change_count: int
    wrong_branch_frame_count: int
    fft_correlation: float


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
    breath_pass: bool
    heart_pass: bool
    overall_pass: bool
    max_true_phase_step_rad: float
    max_recovered_phase_step_rad: float
    phase_rmse_rad: float
    max_phase_error_rad: float
    true_phase_risk_count: int
    actual_branch_change_count: int
    wrong_branch_frame_count: int
    fft_correlation: float
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


def save_radar_config(
    config: RadarConfig,
    plot_config: PlotConfig,
) -> None:
    """儲存單次模擬的完整設定，供日後重現。"""

    file_path: Path = plot_config.output_dir / "radar_config.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    print(f"[已儲存模擬設定] {file_path}")


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
        frame_start_indices: npt.NDArray[np.int64] = np.flatnonzero(
            result.chirp_index_per_measurement == 0
        ).astype(np.int64)
        writer.writerows(
            (
                result.frame_time_s[frame_index],
                result.ground_truth_breath_mm[measurement_index],
                result.ground_truth_heart_mm[measurement_index],
                result.estimated_respiration_mm[frame_index],
                result.estimated_heartbeat_mm[frame_index],
            )
            for frame_index, measurement_index in enumerate(frame_start_indices)
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
    """輸出方法一的批次結果、頻率誤差與相位展開診斷。"""

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
        [record.breath_absolute_error_bpm for record in records], dtype=np.float64
    )
    heart_errors_bpm: FloatArray = np.array(
        [record.heart_absolute_error_bpm for record in records], dtype=np.float64
    )
    range_errors_m: FloatArray = np.array(
        [record.range_absolute_error_m for record in records], dtype=np.float64
    )
    phase_rmse_rad: FloatArray = np.array(
        [record.phase_rmse_rad for record in records], dtype=np.float64
    )
    max_phase_errors_rad: FloatArray = np.array(
        [record.max_phase_error_rad for record in records], dtype=np.float64
    )
    true_phase_risk_counts: FloatArray = np.array(
        [record.true_phase_risk_count for record in records], dtype=np.float64
    )
    branch_change_counts: FloatArray = np.array(
        [record.actual_branch_change_count for record in records], dtype=np.float64
    )
    wrong_branch_frames: FloatArray = np.array(
        [record.wrong_branch_frame_count for record in records], dtype=np.float64
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
    phase_recovery_rate: float = float(
        np.mean(max_phase_errors_rad < np.pi) * 100.0
    )

    report: str = "\n".join(
        [
            "FMCW Vital-Sign Batch Statistics (Method 1 Only + Chirp)",
            "=" * 52,
            f"Case count: {len(records)}",
            f"BPM resolution: {records[0].bpm_resolution:.6f} BPM",
            "",
            "Method 1 Phase Unwrap",
            f"  Recovery success rate:   {phase_recovery_rate:.2f}%",
            f"  Mean phase RMSE:         {np.mean(phase_rmse_rad):.3f} rad",
            f"  Max phase error:         {np.max(max_phase_errors_rad):.3f} rad",
            f"  Mean true phase risks:   {np.mean(true_phase_risk_counts):.2f}",
            f"  Mean branch changes:     {np.mean(branch_change_counts):.2f}",
            f"  Mean wrong branch frames:{np.mean(wrong_branch_frames):.2f}",
            "",
            "Respiration",
            f"  Mean absolute error: {np.mean(breath_errors_bpm):.3f} BPM",
            f"  Max absolute error:  {np.max(breath_errors_bpm):.3f} BPM",
            f"  Success rate:        {breath_success_rate:.2f}%",
            "",
            "Heartbeat",
            f"  Mean absolute error: {np.mean(heart_errors_bpm):.3f} BPM",
            f"  Max absolute error:  {np.max(heart_errors_bpm):.3f} BPM",
            f"  Success rate:        {heart_success_rate:.2f}%",
            "",
            "Range",
            f"  Mean absolute error: {np.mean(range_errors_m):.4f} m",
            f"  Max absolute error:  {np.max(range_errors_m):.4f} m",
            "",
            f"Overall BPM success rate: {overall_success_rate:.2f}%",
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
@lru_cache(maxsize=None)
def calculate_filter_cutoffs(
    sampling_rate_hz: float,
    search_low_hz: float,
    search_high_hz: float,
    order: int = 2,
    max_passband_attenuation_db: float = 0.2,
) -> tuple[float, float]:
    """
    自動求出比搜尋範圍更寬的 Butterworth bandpass 截止頻率。

    求解時納入 filtfilt 的雙向濾波響應，讓搜尋範圍的低、高邊界
    都符合 max_passband_attenuation_db 指定的最大振幅衰減。
    """

    nyquist_hz: float = sampling_rate_hz / 2.0
    cutoff_attenuation_db: float = 20.0 * np.log10(2.0)

    if sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz 必須大於 0。")
    if not 0.0 < search_low_hz < search_high_hz < nyquist_hz:
        raise ValueError(
            "搜尋範圍必須滿足 "
            f"0 < low < high < Nyquist frequency ({nyquist_hz} Hz)。"
        )
    if order <= 0:
        raise ValueError("Butterworth filter order 必須大於 0。")
    if not 0.0 < max_passband_attenuation_db < cutoff_attenuation_db:
        raise ValueError(
            "max_passband_attenuation_db 必須大於 0，且小於 "
            f"filtfilt 截止點的 {cutoff_attenuation_db:.3f} dB。"
        )

    minimum_frequency_hz: float = nyquist_hz * 1.0e-9

    def sigmoid(value: FloatArray) -> FloatArray:
        return 1.0 / (1.0 + np.exp(-value))

    def optimizer_values_to_cutoffs(
        optimizer_values: FloatArray,
    ) -> tuple[float, float]:
        fractions: FloatArray = sigmoid(optimizer_values)
        low_cut_hz: float = minimum_frequency_hz + (
            search_low_hz - minimum_frequency_hz
        ) * float(fractions[0])
        high_cut_hz: float = search_high_hz + (
            nyquist_hz - minimum_frequency_hz - search_high_hz
        ) * float(fractions[1])
        return low_cut_hz, high_cut_hz

    def edge_attenuation_error(optimizer_values: FloatArray) -> FloatArray:
        low_cut_hz, high_cut_hz = optimizer_values_to_cutoffs(optimizer_values)
        second_order_sections: FloatArray = butter(
            order,
            [low_cut_hz, high_cut_hz],
            btype="bandpass",
            fs=sampling_rate_hz,
            output="sos",
        )
        _, edge_response = sosfreqz(
            second_order_sections,
            worN=np.asarray([search_low_hz, search_high_hz]),
            fs=sampling_rate_hz,
        )

        # filtfilt 的振幅響應為單向濾波器 |H| 的平方。
        edge_magnitude: FloatArray = np.maximum(
            np.abs(edge_response),
            np.finfo(np.float64).tiny,
        )
        filtfilt_attenuation_db: FloatArray = -40.0 * np.log10(edge_magnitude)
        return filtfilt_attenuation_db - max_passband_attenuation_db

    initial_guesses: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (2.0, -2.0),
        (-2.0, 2.0),
        (2.0, 0.0),
        (0.0, -2.0),
    )
    solutions = [
        least_squares(
            edge_attenuation_error,
            x0=np.asarray(initial_guess, dtype=np.float64),
            bounds=(-30.0, 30.0),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=500,
        )
        for initial_guess in initial_guesses
    ]
    solution = min(
        solutions,
        key=lambda candidate: float(
            np.max(np.abs(edge_attenuation_error(candidate.x)))
        ),
    )

    edge_errors_db: FloatArray = edge_attenuation_error(solution.x)
    if not solution.success or not np.all(np.abs(edge_errors_db) <= 1.0e-6):
        raise RuntimeError(
            "無法依指定搜尋範圍與最大通帶衰減求出 Butterworth 截止頻率。"
        )

    return optimizer_values_to_cutoffs(solution.x)


def bandpass_filter(
    signal: FloatArray,
    sampling_rate_hz: float,
    low_cut_hz: float,
    high_cut_hz: float,
    order: int = 2,
) -> FloatArray:
    """使用 SOS Butterworth bandpass 進行零相位雙向濾波。"""

    nyquist_hz: float = sampling_rate_hz / 2.0

    if low_cut_hz <= 0.0:
        raise ValueError("low_cut_hz 必須大於 0。")

    if high_cut_hz >= nyquist_hz:
        raise ValueError(
            f"high_cut_hz={high_cut_hz} 必須小於 Nyquist frequency={nyquist_hz}。"
        )

    second_order_sections: FloatArray = butter(
        order,
        [low_cut_hz, high_cut_hz],
        btype="bandpass",
        fs=sampling_rate_hz,
        output="sos",
    )
    filtered: FloatArray = sosfiltfilt(second_order_sections, signal)
    return filtered


def estimate_peak_frequency(
    signal: FloatArray,
    sampling_rate_hz: float,
    search_low_hz: float,
    search_high_hz: float,
    require_strict_local_peak: bool = False,
) -> tuple[float, FloatArray, FloatArray]:
    """
    使用 rFFT 在指定頻率範圍內選擇頻譜 peak。

    若 require_strict_local_peak=True，依振幅由大到小檢查候選點，
    並選擇第一個嚴格大於完整頻譜左右相鄰 bin 的局部 peak；
    否則維持原本行為，直接選擇搜尋區域中的最大值。

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
        require_strict_local_peak=require_strict_local_peak,
    )

    return estimated_frequency_hz, frequency_axis, spectrum


def estimate_peak_from_spectrum(
    frequency_axis_hz: FloatArray,
    magnitude_spectrum: FloatArray,
    search_low_hz: float,
    search_high_hz: float,
    require_strict_local_peak: bool = False,
) -> float:
    """回傳搜尋範圍最大值，或依振幅排序後的第一個嚴格局部 peak。"""

    valid_mask: npt.NDArray[np.bool_] = (frequency_axis_hz >= search_low_hz) & (
        frequency_axis_hz <= search_high_hz
    )

    valid_indices: npt.NDArray[np.int64] = np.where(valid_mask)[0]

    if valid_indices.size == 0:
        raise ValueError("指定頻率搜尋範圍內沒有 FFT bin。")

    if require_strict_local_peak:
        descending_order: npt.NDArray[np.int64] = np.argsort(
            magnitude_spectrum[valid_indices]
        )[::-1].astype(np.int64)
        sorted_candidate_indices: npt.NDArray[np.int64] = valid_indices[
            descending_order
        ]

        last_spectrum_index: int = magnitude_spectrum.size - 1
        for candidate_index_value in sorted_candidate_indices:
            candidate_index: int = int(candidate_index_value)
            if candidate_index == 0 or candidate_index == last_spectrum_index:
                continue

            candidate_magnitude: float = float(
                magnitude_spectrum[candidate_index]
            )
            left_magnitude: float = float(
                magnitude_spectrum[candidate_index - 1]
            )
            right_magnitude: float = float(
                magnitude_spectrum[candidate_index + 1]
            )
            if (
                candidate_magnitude > left_magnitude
                and candidate_magnitude > right_magnitude
            ):
                return float(frequency_axis_hz[candidate_index])

    local_peak_index: int = int(np.argmax(magnitude_spectrum[valid_indices]))
    peak_index: int = int(valid_indices[local_peak_index])

    return float(frequency_axis_hz[peak_index])


# -------------------------- Simulation and analysis ----------------------- #
def simulate_and_process(config: RadarConfig) -> VitalSignResult:
    """模擬 FMCW 雷達生命徵象訊號，並進行 Range FFT、相位解調與頻率估測。"""

    rng: np.random.Generator = np.random.default_rng(config.random_seed)

    # ---------------- 建立 Fast Time / 非均勻 Chirp 量測時間 ---------------- #
    fast_time: FloatArray = (
        np.arange(config.num_fast_time_samples, dtype=np.float64)
        * config.adc_sample_period
    )

    chirps_per_frame: int = config.chirps_per_loop * config.num_loops
    if chirps_per_frame <= 0:
        raise ValueError("每個 Frame 的 Chirp 數量必須大於零。")

    frame_time: FloatArray = (
        np.arange(config.frame_length, dtype=np.float64) * config.frame_periodicity
    )
    chirp_time_offset: FloatArray = (
        np.arange(chirps_per_frame, dtype=np.float64) * config.chirp_period
    )
    if chirp_time_offset[-1] >= config.frame_periodicity:
        raise ValueError("Frame 內最後一個 Chirp 不得超出或落在下一個 Frame。")

    # 排列順序：
    # Frame 0 Chirp 0, ..., Frame 0 Chirp N-1, Frame 1 Chirp 0, ...
    measurement_time: FloatArray = (
        frame_time[:, np.newaxis] + chirp_time_offset[np.newaxis, :]
    ).reshape(-1)
    measurement_count: int = config.frame_length * chirps_per_frame
    frame_index_per_measurement: npt.NDArray[np.int64] = np.repeat(
        np.arange(config.frame_length, dtype=np.int64),
        chirps_per_frame,
    )
    chirp_index_per_measurement: npt.NDArray[np.int64] = np.tile(
        np.arange(chirps_per_frame, dtype=np.int64),
        config.frame_length,
    )
    if measurement_time.size != measurement_count:
        raise RuntimeError("Chirp 量測時間軸的長度不正確。")
    if np.any(np.diff(measurement_time) <= 0.0):
        raise RuntimeError("Chirp 量測時間軸必須嚴格遞增。")

    # ------------------------- 原始呼吸 / 心跳位移 -------------------------- #
    ground_truth_breath_m: FloatArray = config.breath_amplitude_m * np.sin(
        2.0 * np.pi * config.breath_frequency_hz * measurement_time
    )

    ground_truth_heart_m: FloatArray = config.heart_amplitude_m * np.sin(
        2.0 * np.pi * config.heart_frequency_hz * measurement_time
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

    # 直接建立 [ADC sample, measurement] 的二維矩陣。measurement 已按實際
    # 時間排序，包含所有 Frame 內的 Chirp。
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
    expected_if_shape: tuple[int, int] = (
        config.num_fast_time_samples,
        measurement_count,
    )
    if if_signal.shape != expected_if_shape:
        raise RuntimeError(
            f"IF 訊號形狀應為 {expected_if_shape}，實際為 {if_signal.shape}。"
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
    true_vibration_phase_rad: FloatArray = (
        4.0 * np.pi * vibration_m / config.wavelength_m
    )
    true_phase_step_rad: FloatArray = np.diff(true_vibration_phase_rad)
    max_true_phase_step_rad: float = float(np.max(np.abs(true_phase_step_rad)))

    # 方法一：僅使用 NumPy 的局部相位展開。
    extracted_phase_rad: FloatArray = np.unwrap(wrapped_phase_rad)

    two_pi: float = 2.0 * float(np.pi)
    raw_phase_error_rad: FloatArray = (
        extracted_phase_rad - true_vibration_phase_rad
    )

    # 去除整體固定的 2*pi offset，只統計相對於全域分支的選錯情況。
    global_branch_offset: int = int(
        np.rint(np.median(raw_phase_error_rad) / two_pi)
    )
    aligned_phase_error_rad: FloatArray = (
        extracted_phase_rad
        - global_branch_offset * two_pi
        - true_vibration_phase_rad
    )
    branch_error_index: npt.NDArray[np.int64] = np.rint(
        aligned_phase_error_rad / two_pi
    ).astype(np.int64)
    branch_change_index: npt.NDArray[np.int64] = np.where(
        np.diff(branch_error_index) != 0
    )[0].astype(np.int64)
    # diff[i] 代表 i -> i+1 發生改變，所以實際由 frame i+1 開始。
    branch_change_frames: npt.NDArray[np.int64] = branch_change_index + 1
    actual_branch_change_count: int = int(
        branch_change_index.size
    )
    wrong_branch_frame_count: int = int(
        np.count_nonzero(branch_error_index != 0)
    )

    phase_offset_rad: float = float(
        np.median(extracted_phase_rad - true_vibration_phase_rad)
    )
    initial_phase_offset_rad: float = float(
        extracted_phase_rad[0] - true_vibration_phase_rad[0]
    )
    recovered_phase_rad: FloatArray = (
        extracted_phase_rad - initial_phase_offset_rad
    )
    recovery_error_rad: FloatArray = (
        extracted_phase_rad - phase_offset_rad - true_vibration_phase_rad
    )
    phase_rmse_rad: float = float(np.sqrt(np.mean(np.square(recovery_error_rad))))
    max_phase_error_rad: float = float(np.max(np.abs(recovery_error_rad)))
    max_recovered_phase_step_rad: float = float(
        np.max(np.abs(np.diff(extracted_phase_rad)))
    )
    true_phase_risk_count: int = int(
        np.count_nonzero(np.abs(true_phase_step_rad) >= np.pi)
    )
    # 896 個 Chirp 是「密集七點、長時間空白」的非均勻取樣，不能直接進行
    # FFT 或 Butterworth 濾波。以每個 Frame 第一個 Chirp 的時間點建立
    # 12.5 Hz 均勻序列；完整 896 點仍保留給相位解纏與分支診斷。
    frame_start_indices: npt.NDArray[np.int64] = np.arange(
        0,
        measurement_count,
        chirps_per_frame,
        dtype=np.int64,
    )
    uniformly_sampled_true_phase_rad: FloatArray = true_vibration_phase_rad[
        frame_start_indices
    ]
    uniformly_sampled_extracted_phase_rad: FloatArray = extracted_phase_rad[
        frame_start_indices
    ]
    true_phase_spectrum: FloatArray = np.abs(
        np.fft.rfft(
            uniformly_sampled_true_phase_rad
            - np.mean(uniformly_sampled_true_phase_rad)
        )
    )
    recovered_phase_spectrum: FloatArray = np.abs(
        np.fft.rfft(
            uniformly_sampled_extracted_phase_rad
            - np.mean(uniformly_sampled_extracted_phase_rad)
        )
    )
    if np.isclose(np.std(true_phase_spectrum), 0.0) or np.isclose(
        np.std(recovered_phase_spectrum), 0.0
    ):
        fft_correlation: float = float("nan")
    else:
        fft_correlation = float(
            np.corrcoef(true_phase_spectrum, recovered_phase_spectrum)[0, 1]
        )

    phase_vibration_rad: FloatArray = extracted_phase_rad - np.mean(extracted_phase_rad)

    # Phase -> displacement:
    # phase = 4*pi*displacement/lambda
    estimated_displacement_m: FloatArray = (
        phase_vibration_rad * config.wavelength_m / (4.0 * np.pi)
    )
    estimated_displacement_mm: FloatArray = estimated_displacement_m * 1000.0

    # ---------------------------- 呼吸與心跳濾波 ---------------------------- #
    breath_filter_low_hz, breath_filter_high_hz = calculate_filter_cutoffs(
        sampling_rate_hz=config.frame_sampling_rate,
        search_low_hz=config.breath_cut_search_low_hz,
        search_high_hz=config.breath_cut_search_high_hz,
        order=config.filter_order,
        max_passband_attenuation_db=config.filter_max_passband_attenuation_db,
    )
    heart_filter_low_hz, heart_filter_high_hz = calculate_filter_cutoffs(
        sampling_rate_hz=config.frame_sampling_rate,
        search_low_hz=config.heart_cut_search_low_hz,
        search_high_hz=config.heart_cut_search_high_hz,
        order=config.filter_order,
        max_passband_attenuation_db=config.filter_max_passband_attenuation_db,
    )

    respiration_phase_rad: FloatArray = bandpass_filter(
        signal=uniformly_sampled_extracted_phase_rad,
        sampling_rate_hz=config.frame_sampling_rate,
        low_cut_hz=breath_filter_low_hz,
        high_cut_hz=breath_filter_high_hz,
        order=config.filter_order,
    )

    heartbeat_phase_rad: FloatArray = bandpass_filter(
        signal=uniformly_sampled_extracted_phase_rad,
        sampling_rate_hz=config.frame_sampling_rate,
        low_cut_hz=heart_filter_low_hz,
        high_cut_hz=heart_filter_high_hz,
        order=config.filter_order,
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
        require_strict_local_peak=True,
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
        require_strict_local_peak=True,
    )
    # 確認呼吸與心跳 FFT frequency axis 相同
    if not np.allclose(frequency_axis_hz, heartbeat_frequency_axis_hz):
        raise RuntimeError("呼吸與心跳的頻率軸不一致。")

    return VitalSignResult(
        time_s=measurement_time,
        frame_time_s=frame_time,
        frame_index_per_measurement=frame_index_per_measurement,
        chirp_index_per_measurement=chirp_index_per_measurement,
        range_axis_m=range_axis_m,
        range_profile=range_profile,
        target_range_bin=target_range_bin,
        ground_truth_breath_mm=ground_truth_breath_m * 1000.0,
        ground_truth_heart_mm=ground_truth_heart_m * 1000.0,
        ground_truth_total_mm=vibration_m * 1000.0,
        true_vibration_phase_rad=true_vibration_phase_rad,
        recovered_phase_rad=recovered_phase_rad,
        branch_error_index=branch_error_index,
        branch_change_frames=branch_change_frames,
        estimated_displacement_mm=estimated_displacement_mm,
        estimated_respiration_mm=respiration_mm,
        estimated_heartbeat_mm=heartbeat_mm,
        frequency_axis_hz=frequency_axis_hz,
        respiration_spectrum=respiration_spectrum,
        heartbeat_spectrum=heartbeat_spectrum,
        estimated_breath_frequency_hz=estimated_breath_frequency_hz,
        estimated_heart_frequency_hz=estimated_heart_frequency_hz,
        max_true_phase_step_rad=max_true_phase_step_rad,
        max_recovered_phase_step_rad=max_recovered_phase_step_rad,
        phase_rmse_rad=phase_rmse_rad,
        max_phase_error_rad=max_phase_error_rad,
        true_phase_risk_count=true_phase_risk_count,
        actual_branch_change_count=actual_branch_change_count,
        wrong_branch_frame_count=wrong_branch_frame_count,
        fft_correlation=fft_correlation,
    )


# --------------------------------- Plotting -------------------------------- #
def plot_range_time_3d(
    result: VitalSignResult,
    plot_config: PlotConfig,
) -> None:
    """繪製可觀察目標距離、跨 frame 穩定度與雜訊底的 3D Range-Time 圖。"""

    positive_range_bin_count: int = result.range_profile.shape[0] // 2
    positive_range_axis_m: FloatArray = result.range_axis_m[
        :positive_range_bin_count
    ]
    positive_range_profile: ComplexArray = result.range_profile[
        :positive_range_bin_count,
        :,
    ]

    range_magnitude: FloatArray = np.abs(positive_range_profile)
    maximum_magnitude: float = float(np.max(range_magnitude))
    normalized_magnitude_db: FloatArray = 20.0 * np.log10(
        np.maximum(range_magnitude / maximum_magnitude, 1.0e-6)
    )

    target_range_m: float = float(result.range_axis_m[result.target_range_bin])

    range_grid_m: FloatArray
    frame_time_grid_s: FloatArray
    range_grid_m, frame_time_grid_s = np.meshgrid(
        positive_range_axis_m,
        result.time_s,
    )

    fig = plt.figure(figsize=(14, 9))
    range_time_ax = fig.add_subplot(1, 1, 1, projection="3d")
    surface = range_time_ax.plot_surface(
        range_grid_m,
        frame_time_grid_s,
        normalized_magnitude_db.T,
        cmap="viridis",
        vmin=-60.0,
        vmax=0.0,
        rcount=result.time_s.size,
        ccount=positive_range_axis_m.size,
        linewidth=0.0,
        antialiased=True,
    )

    target_bin_magnitude_db: FloatArray = normalized_magnitude_db[
        result.target_range_bin,
        :,
    ]
    range_time_ax.plot(
        np.full(result.time_s.shape, target_range_m),
        result.time_s,
        target_bin_magnitude_db,
        color="red",
        linewidth=2.2,
        label=f"Detected target: {target_range_m:.2f} m",
    )

    range_time_ax.set_title(
        "Complete Positive-Range 3D Range-Time Intensity "
        f"({positive_range_axis_m.size} Bins × "
        f"{result.time_s.size} Chirp Measurements)\n"
        "Peak ridge = detected target, lower surface = noise floor",
        fontsize=14,
    )
    range_time_ax.set_xlabel("Range (m)")
    range_time_ax.set_ylabel("Slow time / frame time (s)")
    range_time_ax.set_zlabel("Normalized Range FFT magnitude (dB)")
    range_time_ax.set_zlim(-60.0, 0.0)
    range_time_ax.view_init(elev=30.0, azim=-125.0)
    range_time_ax.set_box_aspect((1.25, 1.5, 0.8))
    range_time_ax.legend(loc="upper right")

    magnitude_colorbar = fig.colorbar(
        surface,
        ax=range_time_ax,
        shrink=0.68,
        pad=0.08,
    )
    magnitude_colorbar.set_label("Normalized Range FFT magnitude (dB)")

    fig.suptitle(
        "Target Range Information Extracted from the Final IF Signal",
        fontsize=15,
        fontweight="bold",
    )

    save_or_show_figure(
        figure=fig,
        file_path=plot_config.output_dir / "04_range_time_3d.png",
        should_save=plot_config.save_range_time_3d,
        should_show=plot_config.show_figures,
        dpi=plot_config.dpi,
    )


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
        result.frame_time_s,
        result.ground_truth_breath_mm[result.chirp_index_per_measurement == 0],
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Ground Truth Breath",
    )

    filtered_ax.plot(
        result.frame_time_s,
        result.estimated_respiration_mm,
        color="tab:blue",
        linewidth=1.5,
        label="Estimated Breath",
    )

    filtered_ax.plot(
        result.frame_time_s,
        result.ground_truth_heart_mm[result.chirp_index_per_measurement == 0],
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Ground Truth Heart",
    )

    filtered_ax.plot(
        result.frame_time_s,
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


def plot_phase_branch_diagnostics(
    result: VitalSignResult,
    plot_config: PlotConfig,
) -> None:
    """繪製真實/恢復相位，並標出恢復結果切換 ``2*pi`` 分支的位置。"""

    measurement_axis: npt.NDArray[np.int64] = np.arange(
        result.time_s.size,
        dtype=np.int64,
    )
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        measurement_axis,
        result.true_vibration_phase_rad,
        color="tab:blue",
        linewidth=2.0,
        label="True Phase",
    )
    axis.plot(
        measurement_axis,
        result.recovered_phase_rad,
        color="tab:orange",
        linewidth=1.5,
        label="Recovered Phase",
    )
    wrong_branch_mask: npt.NDArray[np.bool_] = result.branch_error_index != 0
    fill_frame_values: list[float] = measurement_axis.astype(np.float64).tolist()
    fill_true_phase_values: list[float] = result.true_vibration_phase_rad.tolist()
    fill_recovered_phase_values: list[float] = result.recovered_phase_rad.tolist()
    fill_wrong_branch_mask: list[bool] = wrong_branch_mask.tolist()
    axis.fill_between(
        fill_frame_values,
        fill_true_phase_values,
        fill_recovered_phase_values,
        where=fill_wrong_branch_mask,
        color="red",
        alpha=0.10,
        label="Wrong Branch Measurements",
    )

    for change_number, measurement_number in enumerate(result.branch_change_frames):
        measurement: int = int(measurement_number)
        branch_step: int = int(
            result.branch_error_index[measurement]
            - result.branch_error_index[measurement - 1]
        )
        branch_step_label: str = f"{branch_step:+d}×2π"
        axis.axvline(
            measurement,
            color="red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="Branch Change" if change_number == 0 else None,
        )
        axis.annotate(
            f"Measurement {measurement}: {branch_step_label}",
            xy=(measurement, result.recovered_phase_rad[measurement]),
            xytext=(8, 18),
            textcoords="offset points",
            color="red",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "red", "alpha": 0.7},
        )

    axis.set_title("True Phase vs. Recovered Phase with Branch Changes")
    axis.set_xlabel("Measurement (Frame-major Chirp order)")
    axis.set_ylabel("Phase (rad)")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="best")

    save_or_show_figure(
        figure=fig,
        file_path=plot_config.output_dir / "03_phase_branch_diagnostics.png",
        should_save=plot_config.save_phase_branch_diagnostics,
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
    print("\n" + "=" * 70)
    print("FMCW Radar Vital Sign Estimation Result")
    print("=" * 70)
    print(f"Seed               : {config.random_seed}")
    print("Unwrap Method      : Method 1: numpy.unwrap")
    print()
    print(f"Max Δφ True        : {result.max_true_phase_step_rad:.2f} rad")
    print(f"Max Δφ Recover     : {result.max_recovered_phase_step_rad:.2f} rad")
    print()
    print(f"RMSE Phase         : {result.phase_rmse_rad:.2f} rad")
    print(f"Max Error          : {result.max_phase_error_rad:.2f} rad")
    print()
    print(f"True Phase Risk Count : {result.true_phase_risk_count}")
    print(f"Actual Branch Change Count : {result.actual_branch_change_count}")
    print(f"Wrong Branch Frame Count   : {result.wrong_branch_frame_count}")
    print("Branch Change Measurements")
    if result.branch_change_frames.size == 0:
        print("  None")
    else:
        for measurement_number in result.branch_change_frames:
            measurement: int = int(measurement_number)
            frame: int = int(result.frame_index_per_measurement[measurement])
            chirp: int = int(result.chirp_index_per_measurement[measurement])
            print(
                f"  Measurement {measurement} "
                f"(Frame {frame}, Chirp {chirp})"
            )
    print()
    print(f"FFT Corr           : {result.fft_correlation:.4f}")
    print()
    print(f"Breath GT          : {original_breath_bpm:.2f} BPM")
    print(f"Breath Recover     : {estimated_breath_bpm:.2f} BPM")
    print()
    print(f"Heart GT           : {original_heart_bpm:.2f} BPM")
    print(f"Heart Recover      : {estimated_heart_bpm:.2f} BPM")
    print("=" * 70 + "\n")


def main() -> None:
    num_runs: int = 10000
    output_dir: Path = Path("output_Method1_Chirp")
    batch_records: list[BatchRunRecord] = []

    radar_config = RadarConfig(
        random_seed=42,
    )
    specific_configs: tuple[RadarConfig, ...] = ()
    run_configs: tuple[RadarConfig, ...] = specific_configs or tuple(
        radar_config.randomized_for_run(run_index) for run_index in range(num_runs)
    )

    for run_index, run_config in enumerate(run_configs):
        run_number: int = run_index + 1

        print(f"\n開始第 {run_number}/{len(run_configs)} 次模擬")
        print(
            f"載波頻率:{run_config.fc / 1.0e9:.3f} GHz, \n"
            "隨機參數 "
            f"距離:{run_config.distance_m:.3f} m, \n"
            f"速度:{run_config.velocity_mps:.3f} m/s, \n"
            f"呼吸幅度:{run_config.breath_amplitude_m * 1000:.3f} mm, \n"
            f"呼吸頻率:{run_config.breath_frequency_bpm:.3f} BPM, \n"
            f"心跳幅度:{run_config.heart_amplitude_m * 1000:.3f} mm, \n"
            f"心跳頻率:{run_config.heart_frequency_bpm:.3f} BPM, \n"
            f"SNR:{run_config.snr_db:.3f} dB, \n"
            f"是否加入雜訊:{run_config.add_noise}"
        )

        result: VitalSignResult = simulate_and_process(run_config)

        print_result_summary(run_config, result)

        estimated_breath_bpm: float = result.estimated_breath_frequency_hz * 60.0
        estimated_heart_bpm: float = result.estimated_heart_frequency_hz * 60.0
        breath_absolute_error_bpm: float = abs(
            estimated_breath_bpm - run_config.breath_frequency_bpm
        )
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
                breath_absolute_error_bpm=breath_absolute_error_bpm,
                estimated_heart_bpm=estimated_heart_bpm,
                heart_absolute_error_bpm=heart_absolute_error_bpm,
                breath_pass=breath_absolute_error_bpm <= bpm_resolution,
                heart_pass=heart_absolute_error_bpm <= bpm_resolution,
                overall_pass=(
                    breath_absolute_error_bpm <= bpm_resolution
                    and heart_absolute_error_bpm <= bpm_resolution
                ),
                max_true_phase_step_rad=result.max_true_phase_step_rad,
                max_recovered_phase_step_rad=(result.max_recovered_phase_step_rad),
                phase_rmse_rad=result.phase_rmse_rad,
                max_phase_error_rad=result.max_phase_error_rad,
                true_phase_risk_count=result.true_phase_risk_count,
                actual_branch_change_count=result.actual_branch_change_count,
                wrong_branch_frame_count=result.wrong_branch_frame_count,
                fft_correlation=result.fft_correlation,
                bpm_resolution=bpm_resolution,
                target_range_bin=result.target_range_bin,
                estimated_range_m=estimated_range_m,
                range_absolute_error_m=abs(estimated_range_m - run_config.distance_m),
            )
        )

        if run_index == 0:
            plot_config = PlotConfig(
                output_dir=output_dir,
                show_figures=False,
                save_fmcw_waveform=True,
                save_vital_sign_summary=True,
                save_range_time_3d=True,
            )
            save_waveform_viewer_config(
                config=run_config,
                plot_config=plot_config,
            )
            save_radar_config(
                config=run_config,
                plot_config=plot_config,
            )
            # save_first_run_data(
            #     result=result,
            #     output_dir=output_dir,
            # )

            # 僅儲存第一次測試的 FMCW、Range-Time 與生命徵象圖。
            plot_transmitted_fmcw_waveform(
                config=run_config,
                plot_config=plot_config,
            )
            plot_vital_sign_summary(
                config=run_config,
                result=result,
                plot_config=plot_config,
            )
            plot_phase_branch_diagnostics(
                result=result,
                plot_config=plot_config,
            )
            plot_range_time_3d(
                result=result,
                plot_config=plot_config,
            )
    print(f"\n方法一完成 Pass/Fail：{len(batch_records)} 次")

    save_batch_results(
        records=batch_records,
        output_dir=output_dir,
        success_tolerance_bpm=batch_records[0].bpm_resolution,
    )


if __name__ == "__main__":
    main()
