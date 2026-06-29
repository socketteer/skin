# Artificial Skin

Pressure-sensing artificial skin with resistive touch detection, live visualization with geodesic contour mapping, and audio synthesis.

Built by [socketteer](https://github.com/socketteer) and [Claude Opus 4.6](https://www.anthropic.com/claude).

## What it does

- **Touch localization**: Detects where on the skin surface a touch occurs using triangulation from multiple resistance probes
- **Touch extent/shape**: Infers the size, shape, and orientation of contact using geodesic contour intersection analysis
- **Pressure estimation**: Estimates touch pressure from total signal strength relative to calibration baseline
- **Live visualization**: Real-time heatmap with geodesic contour overlay, confidence gradients, and fading touch trail
- **Audio synthesis**: Optional 4-oscillator synth where each probe drives a voice — position, size, and shape of touch naturally emerge from the harmonic relationships

## Hardware

Two sheets of silver conductive fabric separated by a squishy insulating layer, with resistance probes at known positions. When pressed, the sheets make contact and create conductive paths whose resistance encodes distance to each probe.

### Supported configurations

- **3-probe triangle** (11×15 cm): Probes at top-center and bottom corners
- **4-probe rectangle** (15×23 cm): Probes at all four corners — better coverage, no degenerate zones

## Quick start

```bash
# Calibrate (touch each grid point when prompted)
python calibration.py --config skin_4probe_15x23.json

# Live visualization
python skin_live.py --config skin_4probe_15x23.json --skip-quickcal

# With audio synthesis
python skin_live.py --config skin_4probe_15x23.json --skip-quickcal --audio

# Quick startup calibration (compensates for probe drift)
python skin_live.py --config skin_4probe_15x23.json
```

## Calibration modes

```bash
# Full grid calibration
python calibration.py --config <config>

# Corner-focused (denser sampling in corner regions)
python calibration.py --config <config> --corners

# Palm press (coarse grid, for touch-size analysis)
python calibration.py --config <config> --palm

# Point touch (coarse grid, use a stylus)
python calibration.py --config <config> --point

# Pressure levels
python calibration.py --config <config> --pressure soft|medium|hard

# Shape calibration (pen lines or circular objects)
python calibration.py --config <config> --shape pen|cap|cylinder
```

## Visualization options

```bash
--no-trail        # Disable fading trail dots
--no-geodesic     # Disable geodesic contour overlay
--audio           # Enable audio synthesis
--cal file1 file2 # Override calibration files
--skip-quickcal   # Skip startup quick-calibration
```

## How it works

### Position estimation
Probe readings are converted to ratios and mapped through log-ratio feature space to an RBF interpolator trained on calibration data. Multiple calibration runs are merged with outlier filtering.

### Geodesic contours
Each probe's calibration data defines a scalar field over the skin surface. For a live touch, the iso-contour at each probe's reading value represents the locus of points at the detected distance from that probe. The intersection of all probes' "outside" regions constrains the feasible touch area.

### Shape inference
A self-consistency analysis compares observed probe ratios against what a point touch would produce at each location. Points where the ratios diverge indicate extended contact. A compensation vector field shows the direction other touched points must exist to explain the observed ratios.

### Audio synthesis
Four sine oscillators, one per probe. Each probe's reading maps to frequency (log scale, sqrt-compressed for mid-range sensitivity). Per-probe detuning creates beating patterns that encode which probes are co-activated. Left-side probes route to the left audio channel, right-side to right, giving natural stereo spatialization.

## Dependencies

- Python 3.10+
- numpy, scipy, matplotlib
- pyserial
- sounddevice (optional, for audio)
- anthropic (optional, for AI integration)

## File structure

```
├── skin_live.py          # Live visualization + audio
├── calibration.py        # Calibration data collection
├── skin_config.py        # Configuration loader
├── configs/              # Skin configuration files
│   ├── skin_3probe_11x15.json
│   └── skin_4probe_15x23.json
└── calibration_*.json    # Calibration data files
```
