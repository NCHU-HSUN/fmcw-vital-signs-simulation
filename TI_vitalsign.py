"""
Python port of the vital-sign processing flow in vitalsign.c.

Source:
https://github.com/NCHU-HSUN/vital_signs_xwrl64xx/blob/681fdd65aa33a2c86f45e94f2c66622fa4022b7b/vital_signs_xwrl64xx/vitalsign.c

The embedded-memory reads and TI DSP FFT calls are replaced by NumPy arrays
and NumPy FFTs.  Run this file directly to generate synthetic FMCW radar data
and estimate breathing and heart rates.

Dependencies:
    pip install numpy matplotlib

Examples:
    python vitalsign.py
    python vitalsign.py --breath-bpm 18 --heart-bpm 72 --seed 7 --plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, cast

import numpy as np
from numpy.typing import NDArray


# Constants ported from vitalsign.h.
REFRESH_RATE = 32
VS_TOTAL_FRAME = 128
VS_FFT_SIZE = 512
VS_NUM_ANGLE_FFT = 16
VS_NUM_ANGLE_SEL_BIN = 9
VS_NUM_RANGE_SEL_BIN = 5
VS_NUM_VIRTUAL_CHANNEL = 6
PHASE_FFT_SIZE = 512

HEART_INDEX_START = 68
HEART_INDEX_END = 128
HEART_RATE_DECISION_THRESH = 3
BREATH_INDEX_START = 3
BREATH_INDEX_END = 50
SPECTRUM_MULTIPLICATION_FACTOR = 0.882
HEART_RATE_JUMP_LIMIT = 12
VITALS_MASK_LOOP_NO = 7

# The original C conversion is:
# rate [BPM] = FFT bin * 0.882.
# Therefore 0.882 = sample_rate / 512 * 60.
SAMPLE_RATE_HZ = SPECTRUM_MULTIPLICATION_FACTOR * PHASE_FFT_SIZE / 60.0


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


class ResultAxesProtocol(Protocol):
    """The subset of Matplotlib Axes used by ``plot_result``."""

    def plot(
        self,
        x: FloatArray,
        y: FloatArray,
        *,
        label: str | None = None,
    ) -> object: ...

    def axvline(
        self,
        x: float,
        *,
        color: str,
        linestyle: str,
        label: str,
    ) -> object: ...

    def set_title(self, title: str) -> object: ...

    def set_xlabel(self, label: str) -> object: ...

    def set_ylabel(self, label: str) -> object: ...

    def legend(self) -> object: ...

    def grid(self, *, visible: bool, alpha: float) -> None: ...


class ResultFigureProtocol(Protocol):
    """The subset of Matplotlib Figure used by ``plot_result``."""

    def add_subplot(
        self,
        rows: int,
        columns: int,
        index: int,
    ) -> ResultAxesProtocol: ...

    def savefig(
        self,
        file_path: Path,
        *,
        dpi: int,
        bbox_inches: str,
    ) -> None: ...


@dataclass(frozen=True)
class AntennaGeometry:
    """Six virtual antenna locations, expressed in half-wavelength units."""

    rows: NDArray[np.int_]
    cols: NDArray[np.int_]

    @classmethod
    def default(cls) -> "AntennaGeometry":
        # A compact 2 x 3 virtual array. Replace these positions with the
        # device's vsActiveAntennaGeometryCfg values for real radar data.
        return cls(
            rows=np.asarray([0, 0, 0, 1, 1, 1], dtype=int),
            cols=np.asarray([0, 1, 2, 0, 1, 2], dtype=int),
        )

    def __post_init__(self) -> None:
        if self.rows.shape != (VS_NUM_VIRTUAL_CHANNEL,):
            raise ValueError("rows must contain six virtual antenna positions")
        if self.cols.shape != (VS_NUM_VIRTUAL_CHANNEL,):
            raise ValueError("cols must contain six virtual antenna positions")


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the synthetic complex FMCW input."""

    breath_bpm: float = 18.0
    heart_bpm: float = 72.0
    sample_rate_hz: float = SAMPLE_RATE_HZ
    frames: int = VS_TOTAL_FRAME
    wavelength_m: float = 0.005  # Approximately 60 GHz.
    breath_displacement_m: float = 0.0015
    heart_displacement_m: float = 0.00025
    heart_second_harmonic_ratio: float = 0.45
    target_range_bin: int = 2
    target_amplitude: float = 6.0
    noise_std: float = 0.06
    clutter_amplitude: float = 2.0
    azimuth_spatial_frequency: float = 0.11
    elevation_spatial_frequency: float = -0.08
    seed: int = 7


@dataclass
class VitalSignResult:
    breathing_rate_bpm: float
    heart_rate_bpm: float
    breathing_bin: int
    heart_bin: int
    breathing_deviation: float
    angle_peak_row: int
    angle_peak_col: int
    phase_difference: FloatArray
    frequency_bpm: FloatArray
    breathing_spectrum: FloatArray
    heart_harmonic_product: FloatArray


def compute_phase_unwrap(
    phase: float,
    phase_previous: float,
    correction_cumulative: float,
) -> tuple[float, float]:
    """Direct scalar port of MmwDemo_computePhaseUnwrap."""

    difference = phase - phase_previous
    if difference > np.pi:
        modulo_factor = 1.0
    elif difference < -np.pi:
        modulo_factor = -1.0
    else:
        modulo_factor = 0.0

    difference_modulo = difference - modulo_factor * 2.0 * np.pi
    if difference_modulo == -np.pi and difference > 0:
        difference_modulo = np.pi

    correction = difference_modulo - difference
    if (0.0 < correction < np.pi) or (-np.pi < correction < 0.0):
        correction = 0.0

    correction_cumulative += correction
    return phase + correction_cumulative, correction_cumulative


def unwrap_phase_c_style(wrapped_phase: FloatArray) -> FloatArray:
    """Apply the C unwrapping routine to one phase history."""

    if wrapped_phase.ndim != 1 or wrapped_phase.size == 0:
        raise ValueError("wrapped_phase must be a non-empty one-dimensional array")

    output = np.empty_like(wrapped_phase, dtype=float)
    output[0] = wrapped_phase[0]
    correction_cumulative = 0.0
    previous = float(wrapped_phase[0])

    for index in range(1, wrapped_phase.size):
        output[index], correction_cumulative = compute_phase_unwrap(
            float(wrapped_phase[index]),
            previous,
            correction_cumulative,
        )
        previous = float(wrapped_phase[index])

    return output


def compute_magnitude_squared(values: ComplexArray) -> FloatArray:
    """Port of MmwDemo_computeMagnitudeSquared."""

    return np.real(values) ** 2 + np.imag(values) ** 2


def compute_my_deviation(values: FloatArray) -> float:
    """Port of MmwDemo_computeMyDeviation (population variance)."""

    if values.size < 1:
        return -1.0
    return float(np.mean(values**2) - np.mean(values) ** 2)


def _three_bin_peak(spectrum: FloatArray, start: int, end: int) -> int:
    """Return the center of the strongest three-bin neighborhood."""

    candidates = np.arange(start, end)
    scores = (
        spectrum[candidates - 1]
        + spectrum[candidates]
        + spectrum[candidates + 1]
    )
    return int(candidates[int(np.argmax(scores))])


def _five_bin_vote_peak(histogram: FloatArray, start: int, end: int) -> int:
    candidates = np.arange(start, end)
    scores = (
        histogram[candidates - 2]
        + histogram[candidates - 1]
        + histogram[candidates]
        + histogram[candidates + 1]
        + histogram[candidates + 2]
    )
    return int(candidates[int(np.argmax(scores))])


def generate_simulated_radar_data(
    config: SimulationConfig,
    geometry: Optional[AntennaGeometry] = None,
) -> tuple[ComplexArray, FloatArray, FloatArray]:
    """
    Generate [frame, range bin, virtual antenna] complex radar samples.

    The target phase follows 4*pi*displacement/wavelength. Static clutter,
    receiver noise, weaker adjacent-range echoes, and a heart harmonic are
    included so the same processing stages used by the C algorithm can run.
    """

    if config.frames != VS_TOTAL_FRAME:
        raise ValueError(f"this port expects exactly {VS_TOTAL_FRAME} frames")
    if not 0 <= config.target_range_bin < VS_NUM_RANGE_SEL_BIN:
        raise ValueError("target_range_bin must be between 0 and 4")
    if config.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    geometry = geometry or AntennaGeometry.default()
    rng = np.random.default_rng(config.seed)
    time_s = np.arange(config.frames, dtype=float) / config.sample_rate_hz

    breath_hz = config.breath_bpm / 60.0
    heart_hz = config.heart_bpm / 60.0
    breath_displacement = config.breath_displacement_m * np.sin(
        2.0 * np.pi * breath_hz * time_s
    )
    heart_displacement = config.heart_displacement_m * (
        np.sin(2.0 * np.pi * heart_hz * time_s + 0.35)
        + config.heart_second_harmonic_ratio
        * np.sin(2.0 * np.pi * 2.0 * heart_hz * time_s + 0.8)
    )
    displacement = breath_displacement + heart_displacement
    vital_phase = 4.0 * np.pi * displacement / config.wavelength_m

    spatial_phase = 2.0 * np.pi * (
        geometry.cols * config.azimuth_spatial_frequency
        + geometry.rows * config.elevation_spatial_frequency
    )

    shape = (
        config.frames,
        VS_NUM_RANGE_SEL_BIN,
        VS_NUM_VIRTUAL_CHANNEL,
    )
    noise = config.noise_std * (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    )

    # A stable, independent complex clutter value for every range/antenna cell.
    clutter_phase = rng.uniform(-np.pi, np.pi, size=shape[1:])
    clutter_gain = config.clutter_amplitude * rng.uniform(0.7, 1.3, size=shape[1:])
    data = noise + clutter_gain[None, :, :] * np.exp(1j * clutter_phase)[None, :, :]

    # Main target plus smaller energy leakage into neighboring range bins.
    range_gain = np.zeros(VS_NUM_RANGE_SEL_BIN, dtype=float)
    range_gain[config.target_range_bin] = 1.0
    if config.target_range_bin > 0:
        range_gain[config.target_range_bin - 1] = 0.32
    if config.target_range_bin + 1 < VS_NUM_RANGE_SEL_BIN:
        range_gain[config.target_range_bin + 1] = 0.28

    target = (
        config.target_amplitude
        * range_gain[None, :, None]
        * np.exp(1j * vital_phase)[:, None, None]
        * np.exp(1j * spatial_phase)[None, None, :]
    )
    data += target
    return data.astype(np.complex128), breath_displacement, heart_displacement


def preprocess_angle_fft(
    radar_data: ComplexArray,
    geometry: Optional[AntennaGeometry] = None,
) -> tuple[ComplexArray, tuple[int, int]]:
    """
    Port the mean removal, 2-D angle FFT, peak search, and 3x3 selection.

    Returns a complex cube shaped [128 frames, 5 ranges, 9 angle cells].
    """

    expected_shape = (
        VS_TOTAL_FRAME,
        VS_NUM_RANGE_SEL_BIN,
        VS_NUM_VIRTUAL_CHANNEL,
    )
    if radar_data.shape != expected_shape:
        raise ValueError(f"radar_data shape must be {expected_shape}")

    geometry = geometry or AntennaGeometry.default()
    mean_removed = radar_data - np.mean(radar_data, axis=0, keepdims=True)
    angle_cube = np.zeros(
        (
            VS_TOTAL_FRAME,
            VS_NUM_RANGE_SEL_BIN,
            VS_NUM_ANGLE_FFT,
            VS_NUM_ANGLE_FFT,
        ),
        dtype=np.complex128,
    )

    for frame_index in range(VS_TOTAL_FRAME):
        for range_index in range(VS_NUM_RANGE_SEL_BIN):
            sparse_array = np.zeros(
                (VS_NUM_ANGLE_FFT, VS_NUM_ANGLE_FFT),
                dtype=np.complex128,
            )
            sparse_array[geometry.rows, geometry.cols] = mean_removed[
                frame_index, range_index
            ]
            angle_cube[frame_index, range_index] = np.fft.fft2(
                sparse_array,
                s=(VS_NUM_ANGLE_FFT, VS_NUM_ANGLE_FFT),
            )

    accumulated_power = np.sum(np.abs(angle_cube) ** 2, axis=(0, 1))
    peak_row, peak_col = np.unravel_index(
        int(np.argmax(accumulated_power)),
        accumulated_power.shape,
    )

    row_indices = np.mod(
        np.asarray([peak_row - 1, peak_row, peak_row + 1]),
        VS_NUM_ANGLE_FFT,
    )
    col_indices = np.mod(
        np.asarray([peak_col - 1, peak_col, peak_col + 1]),
        VS_NUM_ANGLE_FFT,
    )

    selected = np.empty(
        (
            VS_TOTAL_FRAME,
            VS_NUM_RANGE_SEL_BIN,
            VS_NUM_ANGLE_SEL_BIN,
        ),
        dtype=np.complex128,
    )
    cell = 0
    for row in row_indices:
        for col in col_indices:
            selected[:, :, cell] = angle_cube[:, :, row, col]
            cell += 1

    return selected, (int(peak_row), int(peak_col))


class VitalSignProcessor:
    """Stateful Python equivalent of MmwDemo_computeVitalSignProcessing."""

    def __init__(self) -> None:
        self.loop_count = 0
        self.previous_heart_peaks = np.zeros(4, dtype=int)

    def process(
        self,
        selected_angle_data: ComplexArray,
        angle_peak: tuple[int, int] = (0, 0),
        indicate_no_target: bool = False,
    ) -> VitalSignResult:
        expected_shape = (
            VS_TOTAL_FRAME,
            VS_NUM_RANGE_SEL_BIN,
            VS_NUM_ANGLE_SEL_BIN,
        )
        if selected_angle_data.shape != expected_shape:
            raise ValueError(f"selected_angle_data shape must be {expected_shape}")

        count = VS_NUM_RANGE_SEL_BIN * VS_NUM_ANGLE_SEL_BIN
        breath_peak_bins = np.zeros(count, dtype=int)
        primary_heart_bins = np.zeros(count, dtype=int)
        secondary_heart_bins = np.zeros(count, dtype=int)
        breath_storage = np.zeros(PHASE_FFT_SIZE, dtype=float)
        heart_storage = np.zeros(PHASE_FFT_SIZE // 2, dtype=float)

        representative_phase_difference = np.zeros(VS_TOTAL_FRAME - 1)
        representative_breath_spectrum = np.zeros(PHASE_FFT_SIZE)
        representative_heart_product = np.zeros(PHASE_FFT_SIZE // 2)
        breath_deviation_source = np.zeros(VS_TOTAL_FRAME - 1)

        point_index = 0
        for angle_index in range(VS_NUM_ANGLE_SEL_BIN):
            for range_index in range(VS_NUM_RANGE_SEL_BIN):
                wrapped_phase = np.angle(
                    selected_angle_data[:, range_index, angle_index]
                )
                unwrapped_phase = unwrap_phase_c_style(wrapped_phase)
                phase_difference = np.diff(unwrapped_phase)

                spectrum_complex = np.fft.fft(
                    phase_difference,
                    n=PHASE_FFT_SIZE,
                )
                magnitude_squared = compute_magnitude_squared(spectrum_complex)
                breath_peak = _three_bin_peak(
                    magnitude_squared,
                    BREATH_INDEX_START,
                    BREATH_INDEX_END,
                )

                harmonic_product = (
                    magnitude_squared[: PHASE_FFT_SIZE // 2]
                    * magnitude_squared[
                        0:PHASE_FFT_SIZE:2
                    ][: PHASE_FFT_SIZE // 2]
                )
                heart_peak = _three_bin_peak(
                    harmonic_product,
                    HEART_INDEX_START,
                    HEART_INDEX_END,
                )

                suppressed = harmonic_product.copy()
                suppressed[heart_peak - 1 : heart_peak + 2] = 0.0
                secondary_peak = _three_bin_peak(
                    suppressed,
                    HEART_INDEX_START,
                    HEART_INDEX_END,
                )

                breath_peak_bins[point_index] = breath_peak
                primary_heart_bins[point_index] = heart_peak
                secondary_heart_bins[point_index] = secondary_peak
                breath_storage += magnitude_squared
                heart_storage += harmonic_product

                if angle_index == 5 and range_index == 3:
                    breath_deviation_source = phase_difference
                if angle_index == 4 and range_index == 2:
                    representative_phase_difference = phase_difference
                    representative_breath_spectrum = magnitude_squared
                    representative_heart_product = harmonic_product

                point_index += 1

        # C code votes over all 45 range/angle points for breathing.
        breath_histogram = np.bincount(
            breath_peak_bins,
            minlength=PHASE_FFT_SIZE,
        ).astype(float)
        breathing_bin = _three_bin_peak(
            breath_histogram,
            BREATH_INDEX_START,
            BREATH_INDEX_END,
        )

        # C code discards the first and last range bins before heart voting.
        primary_matrix = primary_heart_bins.reshape(
            VS_NUM_ANGLE_SEL_BIN,
            VS_NUM_RANGE_SEL_BIN,
        )
        secondary_matrix = secondary_heart_bins.reshape(
            VS_NUM_ANGLE_SEL_BIN,
            VS_NUM_RANGE_SEL_BIN,
        )
        heart_candidates = np.concatenate(
            [
                primary_matrix[:, 1:4].ravel(),
                secondary_matrix[:, 1:4].ravel(),
            ]
        )
        heart_histogram = np.bincount(
            heart_candidates,
            minlength=PHASE_FFT_SIZE,
        ).astype(float)
        heart_histogram_bin = _five_bin_vote_peak(
            heart_histogram,
            HEART_INDEX_START,
            HEART_INDEX_END,
        )

        # Five strongest accumulated HPS peaks, then correlation with history.
        accumulated_temp = heart_storage.copy()
        present_peaks = np.zeros(5, dtype=int)
        for index in range(5):
            peak = _three_bin_peak(
                accumulated_temp,
                HEART_INDEX_START,
                HEART_INDEX_END,
            )
            present_peaks[index] = peak
            accumulated_temp[peak - 1 : peak + 2] = 0.0

        previous = int(self.previous_heart_peaks[3])
        differences = np.abs(present_peaks - previous)
        closest_index = int(np.argmin(differences))
        if differences[closest_index] < HEART_RATE_DECISION_THRESH:
            heart_bin = int(present_peaks[closest_index])
        else:
            heart_bin = int(heart_histogram_bin)

        newest_previous = int(self.previous_heart_peaks[0])
        if (
            self.loop_count > VITALS_MASK_LOOP_NO
            and abs(heart_bin - newest_previous) > HEART_RATE_JUMP_LIMIT
        ):
            heart_bin = newest_previous + int(
                np.sign(heart_bin - newest_previous) * HEART_RATE_JUMP_LIMIT
            )

        if self.loop_count > 4:
            self.previous_heart_peaks[1:] = self.previous_heart_peaks[:-1]
            self.previous_heart_peaks[0] = heart_bin
        elif self.loop_count == 0:
            self.previous_heart_peaks[:] = 0

        breathing_rate = breathing_bin * SPECTRUM_MULTIPLICATION_FACTOR
        heart_rate = heart_bin * SPECTRUM_MULTIPLICATION_FACTOR
        breathing_deviation = compute_my_deviation(
            breath_deviation_source[-40:]
        )

        if indicate_no_target:
            breathing_rate = 0.0
            heart_rate = 0.0
            breathing_deviation = 0.0

        self.loop_count += 1
        frequency_bpm = (
            np.arange(PHASE_FFT_SIZE // 2, dtype=float)
            * SPECTRUM_MULTIPLICATION_FACTOR
        )
        return VitalSignResult(
            breathing_rate_bpm=float(breathing_rate),
            heart_rate_bpm=float(heart_rate),
            breathing_bin=int(breathing_bin),
            heart_bin=int(heart_bin),
            breathing_deviation=float(breathing_deviation),
            angle_peak_row=angle_peak[0],
            angle_peak_col=angle_peak[1],
            phase_difference=representative_phase_difference,
            frequency_bpm=frequency_bpm,
            breathing_spectrum=representative_breath_spectrum[
                : PHASE_FFT_SIZE // 2
            ],
            heart_harmonic_product=representative_heart_product,
        )


def run_simulation(config: SimulationConfig) -> tuple[VitalSignResult, FloatArray, FloatArray]:
    """Generate synthetic data, preprocess it, and estimate vital signs."""

    geometry = AntennaGeometry.default()
    radar_data, breath_displacement, heart_displacement = (
        generate_simulated_radar_data(config, geometry)
    )
    selected_data, angle_peak = preprocess_angle_fft(radar_data, geometry)
    processor = VitalSignProcessor()
    result = processor.process(selected_data, angle_peak=angle_peak)
    return result, breath_displacement, heart_displacement


def plot_result(
    config: SimulationConfig,
    result: VitalSignResult,
    breath_displacement: FloatArray,
    heart_displacement: FloatArray,
    output_dir: Path = Path("output_Ti"),
) -> Path:
    """Save the simulated motion and spectra to a PNG file."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
    except ImportError as error:
        raise SystemExit(
            "Plotting requires matplotlib: pip install matplotlib"
        ) from error

    time_s = np.arange(config.frames) / config.sample_rate_hz
    raw_figure: Figure = plt.figure(
        figsize=(10, 9),
        constrained_layout=True,
    )
    figure = cast(ResultFigureProtocol, raw_figure)
    displacement_axis = figure.add_subplot(3, 1, 1)
    breathing_axis = figure.add_subplot(3, 1, 2)
    heart_axis = figure.add_subplot(3, 1, 3)

    displacement_axis.plot(
        time_s,
        breath_displacement * 1e3,
        label="Breathing",
    )
    displacement_axis.plot(
        time_s,
        heart_displacement * 1e3,
        label="Heartbeat",
    )
    displacement_axis.set_title("Simulated chest displacement")
    displacement_axis.set_xlabel("Time (s)")
    displacement_axis.set_ylabel("Displacement (mm)")
    displacement_axis.legend()
    displacement_axis.grid(visible=True, alpha=0.3)

    breath_band = slice(BREATH_INDEX_START, BREATH_INDEX_END)
    breathing_axis.plot(
        result.frequency_bpm[breath_band],
        result.breathing_spectrum[breath_band],
    )
    breathing_axis.axvline(
        result.breathing_rate_bpm,
        color="tab:red",
        linestyle="--",
        label=f"Estimate: {result.breathing_rate_bpm:.2f} BPM",
    )
    breathing_axis.set_title("Breathing spectrum")
    breathing_axis.set_xlabel("Rate (BPM)")
    breathing_axis.set_ylabel("Magnitude squared")
    breathing_axis.legend()
    breathing_axis.grid(visible=True, alpha=0.3)

    heart_band = slice(HEART_INDEX_START, HEART_INDEX_END)
    heart_axis.plot(
        result.frequency_bpm[heart_band],
        result.heart_harmonic_product[heart_band],
    )
    heart_axis.axvline(
        result.heart_rate_bpm,
        color="tab:red",
        linestyle="--",
        label=f"Estimate: {result.heart_rate_bpm:.2f} BPM",
    )
    heart_axis.set_title("Heart harmonic spectrum product")
    heart_axis.set_xlabel("Rate (BPM)")
    heart_axis.set_ylabel("Spectrum product")
    heart_axis.legend()
    heart_axis.grid(visible=True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vital_sign_result.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(raw_figure)
    return output_path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate and process xWRL64xx FMCW vital-sign data."
    )
    parser.add_argument("--breath-bpm", type=float, default=18.0)
    parser.add_argument("--heart-bpm", type=float, default=72.0)
    parser.add_argument("--noise-std", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save simulated displacement and spectra to a PNG file.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    config = SimulationConfig(
        breath_bpm=arguments.breath_bpm,
        heart_bpm=arguments.heart_bpm,
        noise_std=arguments.noise_std,
        seed=arguments.seed,
    )
    result, breath_displacement, heart_displacement = run_simulation(config)

    print("=" * 62)
    print("Ti vital-sign simulation")
    print("=" * 62)
    print(f"Sampling rate       : {config.sample_rate_hz:.4f} Hz")
    print(f"Input breathing     : {config.breath_bpm:.2f} BPM")
    print(f"Estimated breathing : {result.breathing_rate_bpm:.2f} BPM")
    print(f"Input heart rate    : {config.heart_bpm:.2f} BPM")
    print(f"Estimated heart rate: {result.heart_rate_bpm:.2f} BPM")
    print(
        "Angle FFT peak      : "
        f"row {result.angle_peak_row}, column {result.angle_peak_col}"
    )
    print(f"Breathing variance  : {result.breathing_deviation:.6f}")
    print("=" * 62)

    if arguments.plot:
        output_path = plot_result(
            config,
            result,
            breath_displacement,
            heart_displacement,
        )
        print(f"Plot saved           : {output_path}")


if __name__ == "__main__":
    main()
