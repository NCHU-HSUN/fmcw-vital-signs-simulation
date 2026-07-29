# FMCW Vital Signs Simulation

This project simulates an FMCW radar that estimates respiration and heart rates
from slow-time phase data. It includes a browser-based Plotly Dash dashboard for
live frame playback in GitHub Codespaces.

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

The primary Method 1 simulation is
[`slowAll_1_Method1.py`](slowAll_1_Method1.py). The live dashboard reuses the
same simulation and processing functions.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the live dashboard

Start the dashboard on port 8050:

```bash
.venv/bin/python dashboard.py
```

In GitHub Codespaces, open the **PORTS** panel and click **Open in Browser** for
port `8050`. Codespaces normally detects and forwards this port automatically.
Keep the port visibility set to **Private** unless the dashboard intentionally
needs to be shared.

The dashboard provides:

- Start, pause, and reset controls.
- The current frame's Range Profile.
- A Range-Time heatmap that grows frame by frame.
- A rotatable and zoomable 3D Range-Time surface.
- Detected range, configured SNR, and full-capture respiration/heartbeat
  estimates.

The simulation produces all 128 frames once when the server starts. The browser
then plays those frames in sequence. Change the browser update interval with:

```bash
.venv/bin/python dashboard.py --interval-ms 500
```

The default interval is `250 ms`. The server listens on `0.0.0.0:8050`; use
`--host` or `--port` to override either value.

## Run the batch simulation

```bash
MPLBACKEND=Agg .venv/bin/python slowAll_1_Method1.py
```

The batch script prints the ground-truth and estimated respiration and heartbeat
frequencies, saves the first run's figures, and writes batch statistics.

## Output

The generated figures are saved in `output/`:

| File | Contents |
| --- | --- |
| `01_fmcw_transmit_waveform.png` | FMCW transmit waveform, instantaneous frequency, and chirp structure. |
| `02_vital_sign_summary_4x1.png` | Range profile, estimated displacement, filtered vital-sign signals, and respiration/heartbeat spectra. |
| `03_phase_branch_diagnostics.png` | True and recovered phase with branch diagnostics. |
| `04_range_time_3d.png` | Complete positive-range 3D Range-Time intensity. |

## Configuration

Edit `RadarConfig` and `PlotConfig` in
[`slowAll_1_Method1.py`](slowAll_1_Method1.py) to change the radar simulation,
noise, output directory, or saved figures. Dashboard playback configuration is
defined in [`dashboard.py`](dashboard.py).
