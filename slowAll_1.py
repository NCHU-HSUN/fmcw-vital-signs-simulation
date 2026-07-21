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

    # ---------------------- 方法 3：全域動態規劃 --------------------- #
    global_dp_branch_radius: int = 8
    global_dp_velocity_weight: float = 1.0
    global_dp_acceleration_weight: float = 4.0
    global_dp_drift_weight: float = 0.05

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

    # 流程 A（M1 -> M2 -> M3）與流程 B（M1 -> M3）比較
    save_flow_comparison: bool = True


@dataclass(frozen=True)
class PhaseFlowResult:
    """單一 unwrap 流程的相位、分支與生命徵象評估結果。"""

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
    recovery_succeeded: bool
    max_recovered_phase_step_rad: float
    phase_rmse_rad: float
    max_phase_error_rad: float
    actual_branch_change_count: int
    wrong_branch_frame_count: int
    fft_correlation: float


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
    used_kalman_unwrap: bool
    kalman_recovery_succeeded: bool
    used_global_dp_unwrap: bool
    global_dp_recovery_succeeded: bool
    max_true_phase_step_rad: float
    max_recovered_phase_step_rad: float
    phase_rmse_rad: float
    max_phase_error_rad: float
    true_phase_risk_count: int
    actual_branch_change_count: int
    wrong_branch_frame_count: int
    fft_correlation: float
    primary_flow_recovery_succeeded: bool
    direct_dp_flow: PhaseFlowResult


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
    used_kalman_unwrap: bool
    kalman_recovery_succeeded: bool
    used_global_dp_unwrap: bool
    global_dp_recovery_succeeded: bool
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
    flow_a_recovery_succeeded: bool
    flow_b_recovery_succeeded: bool
    flow_b_phase_rmse_rad: float
    flow_b_max_phase_error_rad: float
    flow_b_actual_branch_change_count: int
    flow_b_wrong_branch_frame_count: int
    flow_b_estimated_breath_bpm: float
    flow_b_estimated_heart_bpm: float
    flow_b_breath_absolute_error_bpm: float
    flow_b_heart_absolute_error_bpm: float
    flow_b_breath_pass: bool
    flow_b_heart_pass: bool
    flow_b_overall_pass: bool


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
    flow_b_breath_errors_bpm: FloatArray = np.array(
        [record.flow_b_breath_absolute_error_bpm for record in records],
        dtype=np.float64,
    )
    flow_b_heart_errors_bpm: FloatArray = np.array(
        [record.flow_b_heart_absolute_error_bpm for record in records],
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
    method_1_mask: npt.NDArray[np.bool_] = np.array(
        [not record.used_kalman_unwrap for record in records], dtype=np.bool_
    )
    method_1_failure_mask: npt.NDArray[np.bool_] = ~method_1_mask
    method_2_success_mask: npt.NDArray[np.bool_] = np.array(
        [
            record.used_kalman_unwrap and record.kalman_recovery_succeeded
            for record in records
        ],
        dtype=np.bool_,
    )
    method_2_failure_mask: npt.NDArray[np.bool_] = np.array(
        [
            record.used_kalman_unwrap and not record.kalman_recovery_succeeded
            for record in records
        ],
        dtype=np.bool_,
    )
    method_3_success_mask: npt.NDArray[np.bool_] = np.array(
        [
            record.used_global_dp_unwrap and record.global_dp_recovery_succeeded
            for record in records
        ],
        dtype=np.bool_,
    )
    method_3_failure_mask: npt.NDArray[np.bool_] = np.array(
        [
            record.used_global_dp_unwrap and not record.global_dp_recovery_succeeded
            for record in records
        ],
        dtype=np.bool_,
    )
    method_2_effective_mask: npt.NDArray[np.bool_] = (
        method_1_mask | method_2_success_mask
    )
    method_3_effective_mask: npt.NDArray[np.bool_] = (
        method_2_effective_mask | method_3_success_mask
    )
    breath_pass_mask: npt.NDArray[np.bool_] = np.array(
        [record.breath_pass for record in records], dtype=np.bool_
    )
    heart_pass_mask: npt.NDArray[np.bool_] = np.array(
        [record.heart_pass for record in records], dtype=np.bool_
    )

    def subset_success_rate(
        pass_mask: npt.NDArray[np.bool_], subset_mask: npt.NDArray[np.bool_]
    ) -> float:
        if not np.any(subset_mask):
            return float("nan")
        return float(np.mean(pass_mask[subset_mask]) * 100.0)

    method_1_breath_rate: float = subset_success_rate(breath_pass_mask, method_1_mask)
    method_1_heart_rate: float = subset_success_rate(heart_pass_mask, method_1_mask)
    method_1_overall_rate: float = subset_success_rate(
        breath_pass_mask & heart_pass_mask, method_1_mask
    )
    method_2_breath_rate: float = subset_success_rate(
        breath_pass_mask, method_2_effective_mask
    )
    method_2_heart_rate: float = subset_success_rate(
        heart_pass_mask, method_2_effective_mask
    )
    method_2_overall_rate: float = subset_success_rate(
        breath_pass_mask & heart_pass_mask, method_2_effective_mask
    )
    method_3_breath_rate: float = subset_success_rate(
        breath_pass_mask, method_3_effective_mask
    )
    method_3_heart_rate: float = subset_success_rate(
        heart_pass_mask, method_3_effective_mask
    )
    method_3_overall_rate: float = subset_success_rate(
        breath_pass_mask & heart_pass_mask, method_3_effective_mask
    )

    def rate_change(before: float, after: float) -> str:
        if np.isnan(before):
            return f"N/A -> {after:.2f}%"
        return f"{before:.2f}% -> {after:.2f}% ({after - before:+.2f} pp)"

    def masked_mean(values: FloatArray) -> float:
        if not np.any(method_1_failure_mask):
            return float("nan")
        return float(np.mean(values[method_1_failure_mask]))

    flow_a_recovery_mask: npt.NDArray[np.bool_] = np.array(
        [record.flow_a_recovery_succeeded for record in records], dtype=np.bool_
    )
    flow_b_recovery_mask: npt.NDArray[np.bool_] = np.array(
        [record.flow_b_recovery_succeeded for record in records], dtype=np.bool_
    )
    flow_a_rmse_rad: FloatArray = np.array(
        [record.phase_rmse_rad for record in records], dtype=np.float64
    )
    flow_b_rmse_rad: FloatArray = np.array(
        [record.flow_b_phase_rmse_rad for record in records], dtype=np.float64
    )
    flow_a_wrong_frames: FloatArray = np.array(
        [record.wrong_branch_frame_count for record in records], dtype=np.float64
    )
    flow_b_wrong_frames: FloatArray = np.array(
        [record.flow_b_wrong_branch_frame_count for record in records],
        dtype=np.float64,
    )
    flow_a_branch_changes: FloatArray = np.array(
        [record.actual_branch_change_count for record in records], dtype=np.float64
    )
    flow_b_branch_changes: FloatArray = np.array(
        [record.flow_b_actual_branch_change_count for record in records],
        dtype=np.float64,
    )
    flow_b_breath_pass_mask: npt.NDArray[np.bool_] = np.array(
        [record.flow_b_breath_pass for record in records], dtype=np.bool_
    )
    flow_b_heart_pass_mask: npt.NDArray[np.bool_] = np.array(
        [record.flow_b_heart_pass for record in records], dtype=np.bool_
    )
    flow_a_overall_pass_mask: npt.NDArray[np.bool_] = (
        breath_pass_mask & heart_pass_mask
    )
    flow_b_overall_pass_mask: npt.NDArray[np.bool_] = np.array(
        [record.flow_b_overall_pass for record in records], dtype=np.bool_
    )
    rmse_tolerance_rad: float = 1.0e-9
    flow_a_lower_rmse_mask: npt.NDArray[np.bool_] = (
        flow_a_rmse_rad < flow_b_rmse_rad - rmse_tolerance_rad
    ) & method_1_failure_mask
    flow_b_lower_rmse_mask: npt.NDArray[np.bool_] = (
        flow_b_rmse_rad < flow_a_rmse_rad - rmse_tolerance_rad
    ) & method_1_failure_mask
    equal_rmse_mask: npt.NDArray[np.bool_] = (
        ~(flow_a_lower_rmse_mask | flow_b_lower_rmse_mask)
        & method_1_failure_mask
    )

    report: str = "\n".join(
        [
            "FMCW Vital-Sign Batch Statistics",
            "=" * 40,
            f"Case count: {len(records)}",
            f"BPM resolution: {records[0].bpm_resolution:.6f} BPM",
            "",
            "Unwrap Method Comparison",
            f"  Method 1 valid cases:                 {np.count_nonzero(method_1_mask)}",
            f"  Method 2 added valid cases:          {np.count_nonzero(method_2_success_mask)}",
            f"  Method 2 recovery failures analyzed: {np.count_nonzero(method_2_failure_mask)}",
            f"  Method 3 added valid cases:          {np.count_nonzero(method_3_success_mask)}",
            f"  Method 3 recovery failures analyzed: {np.count_nonzero(method_3_failure_mask)}",
            f"  Valid cases after Method 3:          {np.count_nonzero(method_3_effective_mask)}",
            f"  Cases completing Pass/Fail:          {len(records)}",
            (
                "  Respiration success (M1 -> M2): "
                f"{rate_change(method_1_breath_rate, method_2_breath_rate)}"
            ),
            (
                "  Respiration success (M2 -> M3): "
                f"{rate_change(method_2_breath_rate, method_3_breath_rate)}"
            ),
            (
                "  Heartbeat success (M1 -> M2):   "
                f"{rate_change(method_1_heart_rate, method_2_heart_rate)}"
            ),
            (
                "  Heartbeat success (M2 -> M3):   "
                f"{rate_change(method_2_heart_rate, method_3_heart_rate)}"
            ),
            (
                "  Overall success (M1 -> M2):     "
                f"{rate_change(method_1_overall_rate, method_2_overall_rate)}"
            ),
            (
                "  Overall success (M2 -> M3):     "
                f"{rate_change(method_2_overall_rate, method_3_overall_rate)}"
            ),
            "",
            "Flow Comparison (Method 1 Failure Cases)",
            "  Flow A: Method 1 -> Method 2 -> Method 3",
            "  Flow B: Method 1 -> Method 3",
            f"  Compared cases: {np.count_nonzero(method_1_failure_mask)}",
            (
                "  Recovery success A/B: "
                f"{subset_success_rate(flow_a_recovery_mask, method_1_failure_mask):.2f}% / "
                f"{subset_success_rate(flow_b_recovery_mask, method_1_failure_mask):.2f}%"
            ),
            (
                "  Mean phase RMSE A/B: "
                f"{masked_mean(flow_a_rmse_rad):.3f} / "
                f"{masked_mean(flow_b_rmse_rad):.3f} rad"
            ),
            (
                "  Mean wrong frames A/B: "
                f"{masked_mean(flow_a_wrong_frames):.2f} / "
                f"{masked_mean(flow_b_wrong_frames):.2f}"
            ),
            (
                "  Mean branch changes A/B: "
                f"{masked_mean(flow_a_branch_changes):.2f} / "
                f"{masked_mean(flow_b_branch_changes):.2f}"
            ),
            (
                "  Lower phase RMSE (A/B/Tie): "
                f"{np.count_nonzero(flow_a_lower_rmse_mask)} / "
                f"{np.count_nonzero(flow_b_lower_rmse_mask)} / "
                f"{np.count_nonzero(equal_rmse_mask)}"
            ),
            (
                "  Recovery only A/B: "
                f"{np.count_nonzero(flow_a_recovery_mask & ~flow_b_recovery_mask & method_1_failure_mask)} / "
                f"{np.count_nonzero(flow_b_recovery_mask & ~flow_a_recovery_mask & method_1_failure_mask)}"
            ),
            (
                "  Respiration BPM success A/B: "
                f"{subset_success_rate(breath_pass_mask, method_1_failure_mask):.2f}% / "
                f"{subset_success_rate(flow_b_breath_pass_mask, method_1_failure_mask):.2f}%"
            ),
            (
                "  Heartbeat BPM success A/B: "
                f"{subset_success_rate(heart_pass_mask, method_1_failure_mask):.2f}% / "
                f"{subset_success_rate(flow_b_heart_pass_mask, method_1_failure_mask):.2f}%"
            ),
            (
                "  Overall BPM success A/B: "
                f"{subset_success_rate(flow_a_overall_pass_mask, method_1_failure_mask):.2f}% / "
                f"{subset_success_rate(flow_b_overall_pass_mask, method_1_failure_mask):.2f}%"
            ),
            (
                "  Mean respiration error A/B: "
                f"{masked_mean(breath_errors_bpm):.3f} / "
                f"{masked_mean(flow_b_breath_errors_bpm):.3f} BPM"
            ),
            (
                "  Mean heartbeat error A/B: "
                f"{masked_mean(heart_errors_bpm):.3f} / "
                f"{masked_mean(flow_b_heart_errors_bpm):.3f} BPM"
            ),
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


def kalman_assisted_unwrap(wrapped_phase_rad: FloatArray) -> FloatArray:
    """利用相位與相位速度 Kalman 模型選擇每筆量測的 2*pi 分支。"""

    if wrapped_phase_rad.size == 0:
        return wrapped_phase_rad.copy()

    transition: FloatArray = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    observation: FloatArray = np.array([[1.0, 0.0]], dtype=np.float64)
    process_covariance: FloatArray = np.array(
        [[0.05, 0.0], [0.0, 0.50]], dtype=np.float64
    )
    measurement_variance: float = 0.05
    two_pi: float = 2.0 * float(np.pi)
    state: FloatArray = np.array([wrapped_phase_rad[0], 0.0], dtype=np.float64)
    covariance: FloatArray = np.diag([measurement_variance, 4.0]).astype(np.float64)
    unwrapped_phase_rad: FloatArray = np.empty_like(wrapped_phase_rad)
    unwrapped_phase_rad[0] = state[0]

    for index in range(1, wrapped_phase_rad.size):
        predicted_state: FloatArray = transition @ state
        predicted_covariance: FloatArray = (
            transition @ covariance @ transition.T + process_covariance
        )
        predicted_phase_rad: float = float(predicted_state[0])
        wrapped_measurement_rad: float = float(wrapped_phase_rad[index])
        phase_branch_ratio: float = (
            predicted_phase_rad - wrapped_measurement_rad
        ) / two_pi
        branch_offset: int = round(phase_branch_ratio)
        measurement_rad: float = wrapped_measurement_rad + branch_offset * two_pi
        innovation_rad: float = measurement_rad - predicted_phase_rad
        innovation_variance: float = float(
            (observation @ predicted_covariance @ observation.T)[0, 0]
            + measurement_variance
        )
        kalman_gain: FloatArray = (
            predicted_covariance @ observation.T / innovation_variance
        )
        state = predicted_state + kalman_gain[:, 0] * innovation_rad
        covariance = (
            np.eye(2, dtype=np.float64) - kalman_gain @ observation
        ) @ predicted_covariance
        unwrapped_phase_rad[index] = measurement_rad

    return unwrapped_phase_rad


def global_dynamic_programming_unwrap(
    wrapped_phase_rad: FloatArray,
    *,
    branch_radius: int = 8,
    velocity_weight: float = 1.0,
    acceleration_weight: float = 4.0,
    drift_weight: float = 0.05,
    reference_phase_rad: float | None = None,
) -> FloatArray:
    """以整段訊號的最小成本路徑選擇每點的 ``2*pi*k`` 分支。

    動態規劃狀態保留連續兩點的分支，因此能同時計算相位速度與
    相位加速度成本。速度與加速度不受整條路徑共同加減 ``2*pi``
    影響；回溯後再選擇最佳的共同偏移，使長時間平均相位最接近
    ``reference_phase_rad``。
    """

    sample_count: int = wrapped_phase_rad.size
    if sample_count == 0:
        return wrapped_phase_rad.copy()
    if branch_radius < 1:
        raise ValueError("branch_radius 必須至少為 1。")
    if velocity_weight < 0.0 or acceleration_weight < 0.0 or drift_weight < 0.0:
        raise ValueError("動態規劃的成本權重不可為負數。")

    two_pi: float = 2.0 * float(np.pi)
    reference_phase: float = (
        float(wrapped_phase_rad[0])
        if reference_phase_rad is None
        else float(reference_phase_rad)
    )

    if sample_count == 1:
        branch_offset: int = round(
            (reference_phase - float(wrapped_phase_rad[0])) / two_pi
        )
        return wrapped_phase_rad + branch_offset * two_pi

    branch_numbers: npt.NDArray[np.int64] = np.arange(
        -branch_radius,
        branch_radius + 1,
        dtype=np.int64,
    )
    branch_count: int = branch_numbers.size
    candidate_phase_rad: FloatArray = (
        wrapped_phase_rad[:, np.newaxis]
        + two_pi * branch_numbers[np.newaxis, :]
    )

    # 固定第一點 k=0 只是在去除全域 2*pi 不定性；回溯後會再補回
    # 最符合長時間平均位置限制的共同分支偏移。
    zero_branch_index: int = branch_radius
    path_cost: FloatArray = np.full(
        (branch_count, branch_count),
        np.inf,
        dtype=np.float64,
    )
    first_velocity_rad: FloatArray = (
        candidate_phase_rad[1] - candidate_phase_rad[0, zero_branch_index]
    )
    path_cost[zero_branch_index, :] = velocity_weight * np.square(
        first_velocity_rad
    )

    predecessor_tables: list[npt.NDArray[np.int64]] = []
    for sample_index in range(2, sample_count):
        next_cost: FloatArray = np.full_like(path_cost, np.inf)
        predecessor: npt.NDArray[np.int64] = np.full(
            (branch_count, branch_count),
            -1,
            dtype=np.int64,
        )

        for previous_branch in range(branch_count):
            velocity_rad: FloatArray = (
                candidate_phase_rad[sample_index]
                - candidate_phase_rad[sample_index - 1, previous_branch]
            )
            acceleration_rad: FloatArray = (
                candidate_phase_rad[sample_index][np.newaxis, :]
                - 2.0 * candidate_phase_rad[sample_index - 1, previous_branch]
                + candidate_phase_rad[sample_index - 2, :, np.newaxis]
            )
            transition_cost: FloatArray = (
                path_cost[:, previous_branch, np.newaxis]
                + velocity_weight * np.square(velocity_rad)[np.newaxis, :]
                + acceleration_weight * np.square(acceleration_rad)
            )
            best_predecessor: npt.NDArray[np.int64] = np.argmin(
                transition_cost,
                axis=0,
            ).astype(np.int64)
            next_cost[previous_branch, :] = transition_cost[
                best_predecessor,
                np.arange(branch_count),
            ]
            predecessor[previous_branch, :] = best_predecessor

        path_cost = next_cost
        predecessor_tables.append(predecessor)

    # 漂移項是整段訊號的終端成本。先對每組終點保留下來的最佳
    # 動態路徑計算平均相位，再連同速度/加速度成本選出全域終點。
    terminal_cost: FloatArray = path_cost.copy()
    if drift_weight > 0.0:
        for previous_branch in range(branch_count):
            for current_branch in range(branch_count):
                if not np.isfinite(path_cost[previous_branch, current_branch]):
                    continue
                terminal_path: npt.NDArray[np.int64] = np.empty(
                    sample_count,
                    dtype=np.int64,
                )
                terminal_path[-2] = previous_branch
                terminal_path[-1] = current_branch
                for sample_index in range(sample_count - 1, 1, -1):
                    terminal_path[sample_index - 2] = predecessor_tables[
                        sample_index - 2
                    ][
                        terminal_path[sample_index - 1],
                        terminal_path[sample_index],
                    ]
                terminal_phase_rad: FloatArray = candidate_phase_rad[
                    np.arange(sample_count),
                    terminal_path,
                ]
                common_branch_offset: int = round(
                    (reference_phase - float(np.mean(terminal_phase_rad))) / two_pi
                )
                mean_phase_rad: float = float(np.mean(terminal_phase_rad)) + (
                    common_branch_offset * two_pi
                )
                terminal_cost[previous_branch, current_branch] += (
                    drift_weight * (mean_phase_rad - reference_phase) ** 2
                )

    previous_branch: int
    current_branch: int
    previous_branch, current_branch = (
        int(index)
        for index in np.unravel_index(np.argmin(terminal_cost), terminal_cost.shape)
    )
    best_path: npt.NDArray[np.int64] = np.empty(sample_count, dtype=np.int64)
    best_path[-2] = previous_branch
    best_path[-1] = current_branch

    for sample_index in range(sample_count - 1, 1, -1):
        best_path[sample_index - 2] = predecessor_tables[sample_index - 2][
            best_path[sample_index - 1],
            best_path[sample_index],
        ]

    unwrapped_phase_rad: FloatArray = candidate_phase_rad[
        np.arange(sample_count),
        best_path,
    ]

    if drift_weight > 0.0:
        common_branch_offset: int = round(
            (reference_phase - float(np.mean(unwrapped_phase_rad))) / two_pi
        )
        unwrapped_phase_rad = unwrapped_phase_rad + common_branch_offset * two_pi

    return unwrapped_phase_rad


def analyze_phase_flow(
    extracted_phase_rad: FloatArray,
    true_vibration_phase_rad: FloatArray,
    config: RadarConfig,
) -> PhaseFlowResult:
    """以完全相同的指標與後處理評估一條 unwrap 流程。"""

    two_pi: float = 2.0 * float(np.pi)
    raw_phase_error_rad: FloatArray = (
        extracted_phase_rad - true_vibration_phase_rad
    )
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
    branch_change_frames: npt.NDArray[np.int64] = branch_change_index + 1

    phase_offset_rad: float = float(np.median(raw_phase_error_rad))
    recovery_error_rad: FloatArray = (
        extracted_phase_rad - phase_offset_rad - true_vibration_phase_rad
    )
    phase_rmse_rad: float = float(np.sqrt(np.mean(np.square(recovery_error_rad))))
    max_phase_error_rad: float = float(np.max(np.abs(recovery_error_rad)))
    initial_phase_offset_rad: float = float(raw_phase_error_rad[0])
    recovered_phase_rad: FloatArray = extracted_phase_rad - initial_phase_offset_rad
    max_recovered_phase_step_rad: float = float(
        np.max(np.abs(np.diff(extracted_phase_rad)))
    )

    true_phase_spectrum: FloatArray = np.abs(
        np.fft.rfft(true_vibration_phase_rad - np.mean(true_vibration_phase_rad))
    )
    recovered_phase_spectrum: FloatArray = np.abs(
        np.fft.rfft(extracted_phase_rad - np.mean(extracted_phase_rad))
    )
    if np.isclose(np.std(true_phase_spectrum), 0.0) or np.isclose(
        np.std(recovered_phase_spectrum), 0.0
    ):
        fft_correlation: float = float("nan")
    else:
        fft_correlation = float(
            np.corrcoef(true_phase_spectrum, recovered_phase_spectrum)[0, 1]
        )

    phase_vibration_rad: FloatArray = extracted_phase_rad - np.mean(
        extracted_phase_rad
    )
    estimated_displacement_mm: FloatArray = (
        phase_vibration_rad * config.wavelength_m / (4.0 * np.pi) * 1000.0
    )
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
    if not np.allclose(frequency_axis_hz, heartbeat_frequency_axis_hz):
        raise RuntimeError("呼吸與心跳的頻率軸不一致。")

    return PhaseFlowResult(
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
        recovery_succeeded=max_phase_error_rad < np.pi,
        max_recovered_phase_step_rad=max_recovered_phase_step_rad,
        phase_rmse_rad=phase_rmse_rad,
        max_phase_error_rad=max_phase_error_rad,
        actual_branch_change_count=int(branch_change_index.size),
        wrong_branch_frame_count=int(np.count_nonzero(branch_error_index != 0)),
        fft_correlation=fft_correlation,
    )


# -------------------------- Simulation and analysis ----------------------- #
def simulate_and_process(config: RadarConfig) -> VitalSignResult:
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
    true_vibration_phase_rad: FloatArray = (
        4.0 * np.pi * vibration_m / config.wavelength_m
    )
    true_phase_step_rad: FloatArray = np.diff(true_vibration_phase_rad)
    max_true_phase_step_rad: float = float(np.max(np.abs(true_phase_step_rad)))

    used_kalman_unwrap: bool = max_true_phase_step_rad >= np.pi * 0.9
    kalman_recovery_succeeded: bool = True
    used_global_dp_unwrap: bool = False
    global_dp_recovery_succeeded: bool = False
    global_dp_phase_rad: FloatArray | None = None
    if used_kalman_unwrap:
        print(
            "[方法 1 無效] 真實相鄰相位變化達到或超過 π；"
            "改用方法 2 Kalman-assisted unwrap。"
        )
        kalman_phase_rad: FloatArray = kalman_assisted_unwrap(wrapped_phase_rad)
        kalman_phase_offset_rad: float = float(
            np.median(kalman_phase_rad - true_vibration_phase_rad)
        )
        kalman_error_rad: FloatArray = (
            kalman_phase_rad - kalman_phase_offset_rad - true_vibration_phase_rad
        )
        kalman_recovery_succeeded = (
            float(np.max(np.abs(kalman_error_rad))) < np.pi
        )

        if kalman_recovery_succeeded:
            extracted_phase_rad = kalman_phase_rad
            print("[方法 2 恢復成功] 繼續後續估算。")
        else:
            print(
                "[方法 2 恢復失敗] 改用方法 3 全域動態規劃，"
                "以整段速度、加速度與平均位置成本重新選路。"
            )
            used_global_dp_unwrap = True
            global_dp_phase_rad = global_dynamic_programming_unwrap(
                wrapped_phase_rad,
                branch_radius=config.global_dp_branch_radius,
                velocity_weight=config.global_dp_velocity_weight,
                acceleration_weight=config.global_dp_acceleration_weight,
                drift_weight=config.global_dp_drift_weight,
                reference_phase_rad=float(wrapped_phase_rad[0]),
            )
            extracted_phase_rad = global_dp_phase_rad
    else:
        extracted_phase_rad = np.unwrap(wrapped_phase_rad)

    # 流程 B：方法一失敗後不經 Kalman，直接以方法三恢復。
    if used_kalman_unwrap:
        if global_dp_phase_rad is None:
            global_dp_phase_rad = global_dynamic_programming_unwrap(
                wrapped_phase_rad,
                branch_radius=config.global_dp_branch_radius,
                velocity_weight=config.global_dp_velocity_weight,
                acceleration_weight=config.global_dp_acceleration_weight,
                drift_weight=config.global_dp_drift_weight,
                reference_phase_rad=float(wrapped_phase_rad[0]),
            )
        direct_dp_phase_rad: FloatArray = global_dp_phase_rad
    else:
        direct_dp_phase_rad = extracted_phase_rad

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
    true_phase_spectrum: FloatArray = np.abs(
        np.fft.rfft(true_vibration_phase_rad - np.mean(true_vibration_phase_rad))
    )
    recovered_phase_spectrum: FloatArray = np.abs(
        np.fft.rfft(extracted_phase_rad - np.mean(extracted_phase_rad))
    )
    if np.isclose(np.std(true_phase_spectrum), 0.0) or np.isclose(
        np.std(recovered_phase_spectrum), 0.0
    ):
        fft_correlation: float = float("nan")
    else:
        fft_correlation = float(
            np.corrcoef(true_phase_spectrum, recovered_phase_spectrum)[0, 1]
        )

    if used_global_dp_unwrap:
        global_dp_recovery_succeeded = max_phase_error_rad < np.pi
        if global_dp_recovery_succeeded:
            print("[方法 3 恢復成功] 已用全域最低成本路徑繼續後續估算。")
        else:
            print(
                "[方法 3 恢復失敗] 仍保留全域最低成本結果，"
                "繼續後續估算與 Pass/Fail 分析。"
            )

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

    direct_dp_flow: PhaseFlowResult = analyze_phase_flow(
        extracted_phase_rad=direct_dp_phase_rad,
        true_vibration_phase_rad=true_vibration_phase_rad,
        config=config,
    )
    primary_flow_recovery_succeeded: bool = max_phase_error_rad < np.pi
    if used_kalman_unwrap:
        direct_dp_status: str = (
            "成功" if direct_dp_flow.recovery_succeeded else "失敗"
        )
        print(
            "[流程 B：M1 -> M3] 方法一失敗後直接使用方法三："
            f"{direct_dp_status}。"
        )

    return VitalSignResult(
        time_s=frame_time,
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
        used_kalman_unwrap=used_kalman_unwrap,
        kalman_recovery_succeeded=kalman_recovery_succeeded,
        used_global_dp_unwrap=used_global_dp_unwrap,
        global_dp_recovery_succeeded=global_dp_recovery_succeeded,
        max_true_phase_step_rad=max_true_phase_step_rad,
        max_recovered_phase_step_rad=max_recovered_phase_step_rad,
        phase_rmse_rad=phase_rmse_rad,
        max_phase_error_rad=max_phase_error_rad,
        true_phase_risk_count=true_phase_risk_count,
        actual_branch_change_count=actual_branch_change_count,
        wrong_branch_frame_count=wrong_branch_frame_count,
        fft_correlation=fft_correlation,
        primary_flow_recovery_succeeded=primary_flow_recovery_succeeded,
        direct_dp_flow=direct_dp_flow,
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


def plot_phase_branch_diagnostics(
    result: VitalSignResult,
    plot_config: PlotConfig,
) -> None:
    """繪製真實/恢復相位，並標出恢復結果切換 ``2*pi`` 分支的位置。"""

    frame_axis: npt.NDArray[np.int64] = np.arange(
        result.time_s.size,
        dtype=np.int64,
    )
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        frame_axis,
        result.true_vibration_phase_rad,
        color="tab:blue",
        linewidth=2.0,
        label="True Phase",
    )
    axis.plot(
        frame_axis,
        result.recovered_phase_rad,
        color="tab:orange",
        linewidth=1.5,
        label="Recovered Phase",
    )
    wrong_branch_mask: npt.NDArray[np.bool_] = result.branch_error_index != 0
    fill_frame_values: list[float] = frame_axis.astype(np.float64).tolist()
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
        label="Wrong Branch Frames",
    )

    for change_number, frame_number in enumerate(result.branch_change_frames):
        frame: int = int(frame_number)
        branch_step: int = int(
            result.branch_error_index[frame] - result.branch_error_index[frame - 1]
        )
        branch_step_label: str = f"{branch_step:+d}×2π"
        axis.axvline(
            frame,
            color="red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="Branch Change" if change_number == 0 else None,
        )
        axis.annotate(
            f"Frame {frame}: {branch_step_label}",
            xy=(frame, result.recovered_phase_rad[frame]),
            xytext=(8, 18),
            textcoords="offset points",
            color="red",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "red", "alpha": 0.7},
        )

    axis.set_title("True Phase vs. Recovered Phase with Branch Changes")
    axis.set_xlabel("Frame")
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


def plot_unwrap_flow_comparison(
    result: VitalSignResult,
    plot_config: PlotConfig,
) -> None:
    """比較 M1->M2->M3 與 M1->M3 兩條流程的相位及分支誤差。"""

    frame_axis: npt.NDArray[np.int64] = np.arange(
        result.time_s.size, dtype=np.int64
    )
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

    axes[0].plot(
        frame_axis,
        result.true_vibration_phase_rad,
        color="black",
        linewidth=2.0,
        label="True Phase",
    )
    axes[0].plot(
        frame_axis,
        result.recovered_phase_rad,
        color="tab:blue",
        linewidth=1.5,
        label="Flow A: M1 -> M2 -> M3",
    )
    axes[0].plot(
        frame_axis,
        result.direct_dp_flow.recovered_phase_rad,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label="Flow B: M1 -> M3",
    )
    axes[0].set_title(
        "Unwrap Flow Comparison "
        f"(RMSE A/B = {result.phase_rmse_rad:.3f}/"
        f"{result.direct_dp_flow.phase_rmse_rad:.3f} rad)"
    )
    axes[0].set_ylabel("Phase (rad)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="best")

    axes[1].step(
        frame_axis,
        result.branch_error_index,
        where="post",
        color="tab:blue",
        linewidth=1.5,
        label=(
            "Flow A branch error "
            f"({result.wrong_branch_frame_count} wrong frames)"
        ),
    )
    axes[1].step(
        frame_axis,
        result.direct_dp_flow.branch_error_index,
        where="post",
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label=(
            "Flow B branch error "
            f"({result.direct_dp_flow.wrong_branch_frame_count} wrong frames)"
        ),
    )
    axes[1].axhline(0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Branch Error Index")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="best")

    save_or_show_figure(
        figure=fig,
        file_path=plot_config.output_dir / "04_unwrap_flow_comparison.png",
        should_save=plot_config.save_flow_comparison,
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
    unwrap_method: str
    if result.used_global_dp_unwrap:
        unwrap_method = "Method 3: Global dynamic programming"
    elif result.used_kalman_unwrap:
        unwrap_method = "Method 2: Kalman-assisted unwrap"
    else:
        unwrap_method = "Method 1: numpy.unwrap"

    print("\n" + "=" * 70)
    print("FMCW Radar Vital Sign Estimation Result")
    print("=" * 70)
    print(f"Seed               : {config.random_seed}")
    print(f"Unwrap Method      : {unwrap_method}")
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
    print("Branch Change Frames")
    if result.branch_change_frames.size == 0:
        print("  None")
    else:
        for frame_number in result.branch_change_frames:
            print(f"  Frame {int(frame_number)}")
    print()
    print(f"FFT Corr           : {result.fft_correlation:.4f}")
    print()
    print(f"Breath GT          : {original_breath_bpm:.2f} BPM")
    print(f"Breath Recover     : {estimated_breath_bpm:.2f} BPM")
    print()
    print(f"Heart GT           : {original_heart_bpm:.2f} BPM")
    print(f"Heart Recover      : {estimated_heart_bpm:.2f} BPM")
    if result.used_kalman_unwrap:
        direct_dp_breath_bpm: float = (
            result.direct_dp_flow.estimated_breath_frequency_hz * 60.0
        )
        direct_dp_heart_bpm: float = (
            result.direct_dp_flow.estimated_heart_frequency_hz * 60.0
        )
        print()
        print("Flow Comparison")
        print("  A: Method 1 -> Method 2 -> Method 3")
        print("  B: Method 1 -> Method 3")
        print(
            "  Recovery A/B       : "
            f"{result.primary_flow_recovery_succeeded} / "
            f"{result.direct_dp_flow.recovery_succeeded}"
        )
        print(
            "  Phase RMSE A/B     : "
            f"{result.phase_rmse_rad:.3f} / "
            f"{result.direct_dp_flow.phase_rmse_rad:.3f} rad"
        )
        print(
            "  Wrong Frames A/B   : "
            f"{result.wrong_branch_frame_count} / "
            f"{result.direct_dp_flow.wrong_branch_frame_count}"
        )
        print(
            "  Breath BPM A/B     : "
            f"{estimated_breath_bpm:.2f} / {direct_dp_breath_bpm:.2f}"
        )
        print(
            "  Heart BPM A/B      : "
            f"{estimated_heart_bpm:.2f} / {direct_dp_heart_bpm:.2f}"
        )
    print("=" * 70 + "\n")


def main() -> None:
    num_runs: int = 10000
    output_dir: Path = Path("output")
    batch_records: list[BatchRunRecord] = []

    radar_config = RadarConfig(
        distance_m=1.0,
        random_seed=42,
        add_noise=False,
    )
    specific_configs: tuple[RadarConfig, ...] = ()
    run_configs: tuple[RadarConfig, ...] = specific_configs or tuple(
        radar_config.randomized_for_run(run_index) for run_index in range(num_runs)
    )

    for run_index, run_config in enumerate(run_configs):
        run_number: int = run_index + 1

        print(f"\n開始第 {run_number}/{len(run_configs)} 次模擬")

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
        flow_b_estimated_breath_bpm: float = (
            result.direct_dp_flow.estimated_breath_frequency_hz * 60.0
        )
        flow_b_estimated_heart_bpm: float = (
            result.direct_dp_flow.estimated_heart_frequency_hz * 60.0
        )
        flow_b_breath_absolute_error_bpm: float = abs(
            flow_b_estimated_breath_bpm - run_config.breath_frequency_bpm
        )
        flow_b_heart_absolute_error_bpm: float = abs(
            flow_b_estimated_heart_bpm - run_config.heart_frequency_bpm
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
                used_kalman_unwrap=result.used_kalman_unwrap,
                kalman_recovery_succeeded=(result.kalman_recovery_succeeded),
                used_global_dp_unwrap=result.used_global_dp_unwrap,
                global_dp_recovery_succeeded=(
                    result.global_dp_recovery_succeeded
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
                flow_a_recovery_succeeded=(
                    result.primary_flow_recovery_succeeded
                ),
                flow_b_recovery_succeeded=(
                    result.direct_dp_flow.recovery_succeeded
                ),
                flow_b_phase_rmse_rad=result.direct_dp_flow.phase_rmse_rad,
                flow_b_max_phase_error_rad=(
                    result.direct_dp_flow.max_phase_error_rad
                ),
                flow_b_actual_branch_change_count=(
                    result.direct_dp_flow.actual_branch_change_count
                ),
                flow_b_wrong_branch_frame_count=(
                    result.direct_dp_flow.wrong_branch_frame_count
                ),
                flow_b_estimated_breath_bpm=flow_b_estimated_breath_bpm,
                flow_b_estimated_heart_bpm=flow_b_estimated_heart_bpm,
                flow_b_breath_absolute_error_bpm=(
                    flow_b_breath_absolute_error_bpm
                ),
                flow_b_heart_absolute_error_bpm=(
                    flow_b_heart_absolute_error_bpm
                ),
                flow_b_breath_pass=(
                    flow_b_breath_absolute_error_bpm <= bpm_resolution
                ),
                flow_b_heart_pass=(
                    flow_b_heart_absolute_error_bpm <= bpm_resolution
                ),
                flow_b_overall_pass=(
                    flow_b_breath_absolute_error_bpm <= bpm_resolution
                    and flow_b_heart_absolute_error_bpm <= bpm_resolution
                ),
            )
        )

        if run_index == 0:
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
            save_radar_config(
                config=run_config,
                plot_config=plot_config,
            )
            # save_first_run_data(
            #     result=result,
            #     output_dir=output_dir,
            # )

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
            plot_phase_branch_diagnostics(
                result=result,
                plot_config=plot_config,
            )
            plot_unwrap_flow_comparison(
                result=result,
                plot_config=plot_config,
            )

    method_1_valid_count: int = sum(
        not record.used_kalman_unwrap for record in batch_records
    )
    method_2_added_count: int = sum(
        record.used_kalman_unwrap and record.kalman_recovery_succeeded
        for record in batch_records
    )
    method_2_failure_count: int = sum(
        record.used_global_dp_unwrap
        for record in batch_records
    )
    method_3_added_count: int = sum(
        record.used_global_dp_unwrap and record.global_dp_recovery_succeeded
        for record in batch_records
    )
    method_3_failure_count: int = sum(
        record.used_global_dp_unwrap and not record.global_dp_recovery_succeeded
        for record in batch_records
    )
    print("\nUnwrap 方法比較：")
    print(f"  方法 1 原本有效：       {method_1_valid_count} 次")
    print(f"  方法 2 增加有效：       {method_2_added_count} 次")
    print(f"  方法 2 恢復失敗：       {method_2_failure_count} 次（轉交方法 3）")
    print(f"  方法 3 增加有效：       {method_3_added_count} 次")
    print(f"  方法 3 恢復失敗：       {method_3_failure_count} 次（仍納入分析）")
    print(
        "  加入方法 3 後總有效：  "
        f"{method_1_valid_count + method_2_added_count + method_3_added_count} 次"
    )
    print(f"  完成 Pass/Fail：         {len(batch_records)} 次")

    save_batch_results(
        records=batch_records,
        output_dir=output_dir,
        success_tolerance_bpm=batch_records[0].bpm_resolution,
    )


if __name__ == "__main__":
    main()
