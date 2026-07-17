# FMCW Vital Signs Simulation

This project simulates an FMCW radar that estimates respiration and heart rates from slow-time phase data.

## Processing Pipeline

```text
FMCW IF signal
  -> Range FFT
  -> target range-bin selection
  -> phase extraction and unwrapping
  -> displacement estimation
  -> respiration and heartbeat bandpass filtering
  -> FFT peak detection
  -> respiration and heart-rate estimation
```

The primary simulation is [`slowtime3.py`](slowtime3.py). Its default configuration simulates a target at 1 m, respiration at 15 BPM, and heartbeat at 180 BPM.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python slowtime3.py
```

The script prints the ground-truth and estimated respiration and heartbeat frequencies. It runs without opening plot windows by default.

## Output

The generated figures are saved in `output/`:

| File | Contents |
| --- | --- |
| `01_fmcw_transmit_waveform.png` | FMCW transmit waveform, instantaneous frequency, and chirp structure. |
| `02_vital_sign_summary_4x1.png` | Range profile, estimated displacement, filtered vital-sign signals, and respiration/heartbeat spectra. |

## Configuration

Edit the `RadarConfig` and `PlotConfig` instances in `main()` within [`slowtime3.py`](slowtime3.py) to change the target distance, vital-sign rates, noise setting, output directory, or figure display behavior.