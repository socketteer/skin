#!/usr/bin/env python3
"""
skin_live.py — Live calibrated touch heatmap for Princess skin

Reads three probe pairs from Arduino, estimates touch location
using calibrated RBF interpolation, displays live heatmap with
trail history. Outputs x, y, pressure stream for Princess.

Usage:
    python skin_live.py [--port /dev/cu.usbmodem14101] [--session]
"""

import serial
import subprocess
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
import json
import time
import argparse
import threading
import queue
import anthropic
from datetime import datetime
from skin_config import load_config
try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# ── CONFIG (overwritten by config file in main()) ──
SKIN_W = 11
SKIN_H = 15
BAUD = 9600
PROBES = {}
PROBE_NAMES = []     # sorted probe names, set from config
CAL_FILES = {}
MIN_TOTAL = 60

TRAIL_LENGTH = 200   # max number of historical points to keep
TRAIL_MAX_AGE = 10.0 # seconds before trail points fully fade out

# ── LOAD CALIBRATION ──
def compute_features(ratios):
    """Compute RBF input features from probe ratios.

    For N probes, uses N-1 log-ratios as features. This works for any probe count:
    - 3 probes: log(r2/r3), log(r1/(r2+r3)) → 2D features
    - 4 probes: log(r1/r2), log(r3/r4), log((r1+r4)/(r2+r3)) → 3D features
    - N probes: first N-1 log(ri/r_last) → (N-1)D features
    """
    n = ratios.shape[1] if ratios.ndim == 2 else len(ratios)
    if n == 3:
        if ratios.ndim == 1:
            ratios = ratios.reshape(1, -1)
        return np.column_stack([
            np.log(ratios[:, 1] / ratios[:, 2]),
            np.log(ratios[:, 0] / (ratios[:, 1] + ratios[:, 2]))
        ])
    else:
        # General case: log(ri / r_last) for i in 0..N-2
        if ratios.ndim == 1:
            ratios = ratios.reshape(1, -1)
        r_last = ratios[:, -1]
        return np.column_stack([
            np.log(np.clip(ratios[:, i], 1e-6, None) / np.clip(r_last, 1e-6, None))
            for i in range(n - 1)
        ])

def load_calibration_data():
    """Load and merge all calibration files. Returns dict of (x,y) -> list of ratio tuples.
    Filters outlier measurements where total signal is far from the median at that location."""
    # First pass: collect all readings with totals
    raw_points = {}  # (x,y) -> list of (ratios_tuple, total)
    n_probes = len(PROBE_NAMES)

    for cal_key in CAL_FILES:
        filename = CAL_FILES[cal_key]
        try:
            with open(filename) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[WARN] {filename} not found, skipping")
            continue

        count = 0
        for point in data['points']:
            avg = point['averages']
            readings = [avg.get(p, 0) for p in PROBE_NAMES]
            total = sum(readings)
            if total < MIN_TOTAL:
                continue
            ratios = tuple(r / total for r in readings)
            key = (point['target']['x'], point['target']['y'])
            if key not in raw_points:
                raw_points[key] = []
            raw_points[key].append((ratios, total))
            count += 1
        print(f"[CAL] Loaded {count} points from {filename}")

    # Second pass: filter outliers at locations with multiple measurements
    all_points = {}
    outliers_removed = 0
    for key, measurements in raw_points.items():
        if len(measurements) <= 1:
            all_points[key] = [m[0] for m in measurements]
            continue

        totals = [m[1] for m in measurements]
        median_total = np.median(totals)
        # Keep measurements within 20% of median total
        filtered = []
        for ratios, total in measurements:
            if abs(total - median_total) / median_total < 0.20:
                filtered.append(ratios)
            else:
                outliers_removed += 1
        if filtered:
            all_points[key] = filtered
        else:
            # If all got filtered, keep the one closest to median
            best = min(measurements, key=lambda m: abs(m[1] - median_total))
            all_points[key] = [best[0]]

    if outliers_removed > 0:
        print(f"[CAL] Filtered {outliers_removed} outlier measurements")

    return all_points

def load_calibration_totals():
    """Load raw totals from calibration data for pressure baseline.
    Uses same outlier filtering as load_calibration_data."""
    all_totals = {}

    for cal_key in CAL_FILES:
        filename = CAL_FILES[cal_key]
        try:
            with open(filename) as f:
                data = json.load(f)
        except FileNotFoundError:
            continue

        for point in data['points']:
            avg = point['averages']
            readings = [avg.get(p, 0) for p in PROBE_NAMES]
            total = sum(readings)
            if total < MIN_TOTAL:
                continue
            key = (point['target']['x'], point['target']['y'])
            if key not in all_totals:
                all_totals[key] = []
            all_totals[key].append(total)

    # Filter outliers
    for key in all_totals:
        totals = all_totals[key]
        if len(totals) > 1:
            med = np.median(totals)
            all_totals[key] = [t for t in totals if abs(t - med) / med < 0.20] or [min(totals, key=lambda t: abs(t - med))]

    return all_totals

def load_calibration(cfg=None):
    """Load all calibration files, merge, build RBF interpolators."""
    all_points = load_calibration_data()

    if not all_points:
        print("[CAL] No calibration data found!")
        return None, None, None, None

    # Average duplicates at same location
    cal_list = []
    for (x, y), ratio_list in all_points.items():
        avg_r = np.mean(ratio_list, axis=0)
        cal_list.append((x, y, *avg_r))

    cal = np.array(cal_list)
    n_probes = len(PROBE_NAMES)
    ratios = cal[:, 2:2+n_probes]
    features = compute_features(ratios)

    interp_x = RBFInterpolator(features, cal[:, 0],
                                kernel='thin_plate_spline',
                                smoothing=0.1)
    interp_y = RBFInterpolator(features, cal[:, 1],
                                kernel='thin_plate_spline',
                                smoothing=0.1)

    # Build baseline total model: expected total at each position
    # Used for pressure estimation (actual_total / expected_total)
    all_totals = load_calibration_totals()
    baseline_total = None
    if all_totals:
        bt_list = []
        for (x, y), tot_list in all_totals.items():
            avg_t = np.mean(tot_list)
            if (x, y) in all_points:
                avg_r = np.mean(all_points[(x, y)], axis=0)
                bt_list.append((avg_r, avg_t))
        if bt_list:
            bt_ratios = np.array([r for r, t in bt_list])
            bt_totals = np.array([t for r, t in bt_list])
            bt_features = compute_features(bt_ratios)
            baseline_total = RBFInterpolator(bt_features, bt_totals,
                                              kernel='thin_plate_spline',
                                              smoothing=1.0)
            print(f"[CAL] Pressure baseline: mean total={np.mean(bt_totals):.0f}")

    # Build per-probe baseline ratio RBFs (for shape estimation)
    baseline_ratios = None
    if n_probes == 4:
        baseline_ratios = []
        for i in range(n_probes):
            rbf = RBFInterpolator(features, ratios[:, i],
                                   kernel='thin_plate_spline', smoothing=1.0)
            baseline_ratios.append(rbf)
        print(f"[CAL] Shape baseline: {n_probes} per-probe ratio models")

    # Palm threshold from config (total signal above this = palm)
    palm_threshold = cfg.get('palm_total_threshold', None) if cfg else None

    print(f"[CAL] {len(cal_list)} unique locations ({features.shape[1]}D feature space)")
    if palm_threshold:
        print(f"[CAL] Palm detection: total > {palm_threshold}")

    return interp_x, interp_y, palm_threshold, baseline_total, baseline_ratios

def ratio_flatness(r1, r2, r3):
    mean = (r1 + r2 + r3) / 3
    return ((r1-mean)**2 + (r2-mean)**2 + (r3-mean)**2) ** 0.5

# ── QUICK CALIBRATION ──
def get_quickcal_points():
    """Generate quick-cal touch points: center + near each probe."""
    cx, cy = SKIN_W / 2.0, SKIN_H / 2.0
    points = {'center': (round(cx, 1), round(cy, 1))}
    for name, (px, py) in PROBES.items():
        # point ~2cm inward from each probe
        dx = cx - px
        dy = cy - py
        dist = (dx**2 + dy**2)**0.5
        if dist > 0:
            scale = 2.0 / dist
            nx = px + dx * scale
            ny = py + dy * scale
        else:
            nx, ny = px, py
        points[f'near_{name}'] = (round(nx, 1), round(ny, 1))
    return points

def get_expected_ratios(cal_files, quickcal_points):
    """Get expected ratios at quick-cal points from stored calibration data."""
    all_points = load_calibration_data()

    expected = {}
    for name, (tx, ty) in quickcal_points.items():
        # Find closest calibration point
        best_key, best_dist = None, 999
        for key in all_points:
            d = ((key[0] - tx)**2 + (key[1] - ty)**2)**0.5
            if d < best_dist:
                best_dist = d
                best_key = key
        if best_key and best_dist < 2.0:
            avg_r = np.mean(all_points[best_key], axis=0)
            expected[name] = avg_r
    return expected

def draw_quickcal_prompt(fig, ax, point_idx, points_list, completed):
    """Draw the quick-cal map showing current target and completed points."""
    ax.clear()
    ax.set_xlim(-0.5, SKIN_W + 0.5)
    ax.set_ylim(-0.5, SKIN_H + 0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('black')

    skin_rect = patches.Rectangle((0, 0), SKIN_W, SKIN_H,
                                   fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(skin_rect)

    for name, (px, py) in PROBES.items():
        ax.plot(px, py, 'c^', markersize=10, alpha=0.5)
        ax.text(px + 0.3, py + 0.3, name, color='cyan', fontsize=8)

    # completed points
    for i in range(point_idx):
        _, (cx, cy) = points_list[i]
        ax.plot(cx, cy, 'g.', markersize=12)

    # remaining points
    for i in range(point_idx + 1, len(points_list)):
        _, (cx, cy) = points_list[i]
        ax.plot(cx, cy, 'w.', markersize=6, alpha=0.2)

    # current target
    cur_name, (tx, ty) = points_list[point_idx]
    ax.plot(tx, ty, 'ro', markersize=20, fillstyle='none', markeredgewidth=2)
    ax.plot(tx, ty, 'r+', markersize=14, markeredgewidth=2)

    ax.set_title(f"QUICK CAL: Touch ({tx}, {ty}) [{cur_name}]  "
                 f"[{point_idx + 1}/{len(points_list)}]",
                 color='white', fontsize=12)
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='gray')
    plt.tight_layout()

def draw_quickcal_recording(fig, ax, tx, ty, name, vals):
    """Update title during recording."""
    probe_str = '  '.join(f'{p}:{vals[i]}' for i, p in enumerate(PROBE_NAMES))
    ax.set_title(f"RECORDING ({tx}, {ty}) [{name}]  {probe_str}",
                 color='red', fontsize=11)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def run_quickcal(ser, cal_files):
    """Run quick startup calibration with visual map. Returns per-probe gain multipliers."""
    quickcal_points = get_quickcal_points()
    expected = get_expected_ratios(cal_files, quickcal_points)
    if not expected:
        print("[QUICKCAL] No expected ratios found, skipping")
        return np.ones(len(PROBE_NAMES))

    # set up quickcal plot
    plt.ion()
    qc_fig, qc_ax = plt.subplots(figsize=(5, 7))
    qc_fig.show()
    qc_fig.canvas.draw()
    qc_fig.canvas.flush_events()

    points_list = [(name, xy) for name, xy in quickcal_points.items()
                   if name in expected]

    print("\n── QUICK CALIBRATION ──")
    print("Touch 4 points to check probe drift.")
    print("Hold each touch for ~1 second, then release.\n")

    quickcal_data = []

    for idx, (name, (tx, ty)) in enumerate(points_list):
        draw_quickcal_prompt(qc_fig, qc_ax, idx, points_list, quickcal_data)
        # force the draw to complete before waiting for touch
        qc_fig.canvas.draw()
        qc_fig.canvas.flush_events()
        print(f"  Touch ({tx}, {ty}) [{name}] — press and hold...")

        # wait for touch
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line or ':' not in line:
                qc_fig.canvas.flush_events()
                continue
            full = read_full_sample(parse_line(line))
            if full is None:
                continue
            raw = [full[p] for p in PROBE_NAMES]
            if sum(raw) > MIN_TOTAL:
                break

        # collect samples
        subprocess.Popen(['afplay', '/System/Library/Sounds/Tink.aiff'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        samples = []
        zero_count = 0
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line or ':' not in line:
                continue
            full = read_full_sample(parse_line(line))
            if full is None:
                continue
            raw = [full[p] for p in PROBE_NAMES]
            total = sum(raw)
            if total > MIN_TOTAL:
                zero_count = 0
                samples.append(raw)
                draw_quickcal_recording(qc_fig, qc_ax, tx, ty, name, raw)
            else:
                zero_count += 1
                if zero_count >= 5:
                    break

        subprocess.Popen(['afplay', '/System/Library/Sounds/Pop.aiff'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if len(samples) < 3:
            print(f"    Too few samples ({len(samples)}), skipping")
            continue

        avg = np.mean(samples, axis=0)
        total_avg = avg.sum()
        live_ratios = avg / total_avg
        exp_ratios = expected[name]

        live_str = ','.join(f'{r:.3f}' for r in live_ratios)
        exp_str = ','.join(f'{r:.3f}' for r in exp_ratios)
        print(f"    {len(samples)} samples — live=({live_str})  expected=({exp_str})")

        quickcal_data.append({
            'name': name,
            'target': {'x': tx, 'y': ty},
            'live_ratios': live_ratios.tolist(),
            'expected_ratios': exp_ratios.tolist(),
            'num_samples': len(samples),
            'raw_avg': avg.tolist(),
        })

    plt.close(qc_fig)

    if not quickcal_data:
        print("[QUICKCAL] No data collected")
        return np.ones(len(PROBE_NAMES))

    # Compute per-probe gain correction: expected / live, averaged across points
    corrections = []
    for qc in quickcal_data:
        live = np.array(qc['live_ratios'])
        exp = np.array(qc['expected_ratios'])
        corrections.append(exp / np.clip(live, 0.01, None))

    gains = np.mean(corrections, axis=0)
    # Normalize so gains multiply to 1 (we only care about relative correction)
    gains = gains / np.power(np.prod(gains), 1/3)

    gains_str = '  '.join(f'{p}={gains[i]:.3f}' for i, p in enumerate(PROBE_NAMES))
    print(f"\n  Probe gains: {gains_str}")

    # Save quickcal record
    record = {
        'timestamp': datetime.now().isoformat(),
        'gains': {p: round(float(gains[i]), 4) for i, p in enumerate(PROBE_NAMES)},
        'points': quickcal_data,
    }
    filename = f"quickcal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"  Saved to {filename}")
    print("── END QUICK CALIBRATION ──\n")

    return gains

# ── GEODESIC MODE ──
def build_probe_fields():
    """Build interpolated per-probe reading fields from all calibration data.
    Returns dict of {probe_name: griddata interpolator inputs} and the grid."""

    # Collect all calibration points with raw readings
    all_pts = {}  # (x,y) -> list of {probe: reading, 'total': total}
    for cal_key in CAL_FILES:
        try:
            with open(CAL_FILES[cal_key]) as f:
                data = json.load(f)
        except FileNotFoundError:
            continue
        for pt in data['points']:
            avg = pt['averages']
            total = sum(avg.get(p, 0) for p in PROBE_NAMES)
            if total < MIN_TOTAL:
                continue
            key = (pt['target']['x'], pt['target']['y'])
            if key not in all_pts:
                all_pts[key] = []
            entry = {p: avg.get(p, 0) for p in PROBE_NAMES}
            entry['total'] = total
            all_pts[key].append(entry)

    if not all_pts:
        return None, None, None, None

    # Average duplicates with outlier filtering
    xs, ys = [], []
    probe_vals = {p: [] for p in PROBE_NAMES}
    for (x, y), readings_list in all_pts.items():
        if len(readings_list) > 1:
            totals = [r['total'] for r in readings_list]
            med = np.median(totals)
            readings_list = [r for r in readings_list if abs(r['total'] - med) / med < 0.20] or readings_list[:1]
        xs.append(x)
        ys.append(y)
        for p in PROBE_NAMES:
            avg_val = np.mean([r[p] for r in readings_list])
            probe_vals[p].append(avg_val)

    xs, ys = np.array(xs), np.array(ys)

    # Build interpolated grids
    grid_x = np.linspace(0, SKIN_W, 80)
    grid_y = np.linspace(0, SKIN_H, 80)
    gx, gy = np.meshgrid(grid_x, grid_y)

    probe_grids = {}
    for p in PROBE_NAMES:
        from scipy.interpolate import griddata
        vals = np.array(probe_vals[p])
        probe_grids[p] = griddata((xs, ys), vals, (gx, gy), method='cubic')

    return probe_grids, gx, gy, (xs, ys)

GEODESIC_COLORS = {
    'P1': '#ff6666',  # red
    'P2': '#66ff66',  # green
    'P3': '#6666ff',  # blue
    'P4': '#ffff66',  # yellow
}

def draw_geodesic_overlay(ax, probe_grids, gx, gy, readings, gains, touch_pos):
    """Draw geodesic contours, confidence gradient for current touch on given axes.
    Returns area_cm2 of intersection region."""
    raw_list = [readings.get(p, 0) for p in PROBE_NAMES]
    if gains is not None:
        corrected = np.array(raw_list, dtype=float) * gains
    else:
        corrected = np.array(raw_list, dtype=float)

    # Per-probe geodesic contours and hatching
    outside_all = np.ones_like(gx, dtype=bool)
    for i, p in enumerate(PROBE_NAMES):
        grid = probe_grids[p]
        if grid is None:
            continue
        val = corrected[i]
        color = GEODESIC_COLORS.get(p, 'white')
        grid_valid = grid[~np.isnan(grid)]
        if len(grid_valid) == 0:
            continue
        grid_min = grid_valid.min()
        grid_max = grid_valid.max()
        outside_all &= ~np.isnan(grid) & (grid <= val)

        # Hatching
        hatch_patterns = {'P1': '////', 'P2': '\\\\\\\\', 'P3': '////', 'P4': '\\\\\\\\'}
        hatch = hatch_patterns.get(p, '///')
        try:
            cs_fill = ax.contourf(gx, gy, grid, levels=[grid_min - 1, val],
                                  colors=[color], hatches=[hatch], alpha=0.05)
            for collection in cs_fill.collections:
                collection.set_edgecolor(color)
                collection.set_linewidth(0.5)
        except Exception:
            pass

        # Neighboring integer geodesics
        for offset in [-3, -2, -1, 1, 2, 3]:
            neighbor_val = round(val) + offset
            if grid_min < neighbor_val < grid_max:
                try:
                    ax.contour(gx, gy, grid, levels=[neighbor_val],
                              colors=[color], linewidths=0.5,
                              alpha=max(0.08, 0.25 - abs(offset) * 0.06),
                              linestyles='-')
                except Exception:
                    pass

        # Main contour line
        try:
            ax.contour(gx, gy, grid, levels=[val],
                      colors=[color], linewidths=2, alpha=0.7)
        except Exception:
            pass

    # Self-consistency confidence gradient
    area_cm2 = 0
    cell_area = (gx[0,1] - gx[0,0]) * (gy[1,0] - gy[0,0])
    has_all_grids = all(probe_grids[p] is not None for p in PROBE_NAMES)
    probe_pos = np.array([PROBES[p] for p in PROBE_NAMES])

    if has_all_grids and np.any(outside_all):
        obs_total = sum(corrected)
        obs_ratios = corrected / obs_total if obs_total > 0 else corrected

        expected_total = np.zeros_like(gx)
        for i, p in enumerate(PROBE_NAMES):
            expected_total += probe_grids[p]
        expected_total = np.clip(expected_total, 1, None)

        expected_ratios = [probe_grids[p] / expected_total for p in PROBE_NAMES]

        comp_x = np.zeros_like(gx)
        comp_y = np.zeros_like(gx)
        for i, p in enumerate(PROBE_NAMES):
            excess = expected_ratios[i] - obs_ratios[i]
            dx_fp = gx - probe_pos[i, 0]
            dy_fp = gy - probe_pos[i, 1]
            dist = np.clip(np.sqrt(dx_fp**2 + dy_fp**2), 0.1, None)
            comp_x += excess * (dx_fp / dist)
            comp_y += excess * (dy_fp / dist)

        comp_mag = np.sqrt(comp_x**2 + comp_y**2)
        comp_mag_masked = np.where(outside_all, comp_mag, np.nan)
        max_comp = np.nanpercentile(comp_mag_masked, 95) if np.any(outside_all) else 0.1
        max_comp = max(max_comp, 0.001)
        self_consistency = np.clip(1.0 - comp_mag / max_comp, 0, 1)
        self_consistency = np.where(outside_all, self_consistency, 0)

        # Blend in roundness prior from center estimate
        if touch_pos:
            dist_center = np.sqrt((gx - touch_pos[0])**2 + (gy - touch_pos[1])**2)
            region_extent = np.sqrt(np.sum(outside_all) * cell_area / np.pi)
            center_prior = np.exp(-0.5 * (dist_center / max(region_extent * 0.8, 1.0))**2)
            self_consistency = self_consistency * 0.7 + self_consistency * center_prior * 0.3

        area_cm2 = np.sum(outside_all) * cell_area
        self_consistency = np.where(outside_all, self_consistency, 0)

        # Draw confidence gradient
        try:
            max_sc = self_consistency.max()
            if max_sc > 0.01:
                sc_norm = self_consistency / max_sc
                levels = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.01]
                for k in range(len(levels) - 1):
                    band_alpha = 0.04 + 0.16 * (levels[k] + levels[k+1]) / 2
                    try:
                        ax.contourf(gx, gy, sc_norm,
                                   levels=[levels[k], levels[k+1]],
                                   colors=['white'], alpha=band_alpha)
                    except Exception:
                        pass
        except Exception:
            pass

    # Crosshair
    if touch_pos:
        ax.plot(touch_pos[0], touch_pos[1], 'w+', markersize=15,
                markeredgewidth=2, zorder=10)

    return area_cm2

def validate_calibration(interp_x, interp_y, palm_threshold):
    ax.set_ylim(-0.5, SKIN_H + 0.5)
    ax.set_aspect('equal')

    # Skin outline
    skin_rect = patches.Rectangle((0, 0), SKIN_W, SKIN_H,
        fill=False, edgecolor='white', linewidth=1.5, linestyle='--')
    ax.add_patch(skin_rect)

    # Draw probes
    for name, (px, py) in PROBES.items():
        color = GEODESIC_COLORS.get(name, 'cyan')
        ax.plot(px, py, '^', color=color, markersize=10, alpha=0.7)
        ax.text(px, py + 0.8, name, color=color, fontsize=9, ha='center', fontweight='bold')

    # For each probe, draw the iso-contour matching the current reading
    raw_list = [readings.get(p, 0) for p in PROBE_NAMES]
    if gains is not None:
        corrected = np.array(raw_list, dtype=float) * gains
    else:
        corrected = np.array(raw_list, dtype=float)

    # For each probe, the reading tells us the distance to the nearest contact point.
    # The touch is AT that distance — so it's ON the contour, or further away.
    # The region OUTSIDE the contour (further from probe = lower readings) is where
    # the touch could be. The intersection of all 4 "outside" regions is the
    # feasible touch area.

    for i, p in enumerate(PROBE_NAMES):
        grid = probe_grids[p]
        if grid is None:
            continue
        val = corrected[i]
        color = GEODESIC_COLORS.get(p, 'white')
        grid_valid = grid[~np.isnan(grid)]
        if len(grid_valid) == 0:
            continue
        grid_min = grid_valid.min()
        grid_max = grid_valid.max()

        # Fill the OUTSIDE region with hatching (reading <= val, further from probe)
        hatch_patterns = {'P1': '////', 'P2': '\\\\\\\\', 'P3': '////', 'P4': '\\\\\\\\'}
        hatch = hatch_patterns.get(p, '///')
        try:
            # cs_fill = ax.contourf(gx, gy, grid, levels=[grid_min - 1, val],
            #                       colors=[color], hatches=[hatch], alpha=0.05)
            cs_fill = ax.contourf(gx, gy, grid, levels=[val, grid_max],
                                  colors=[color], hatches=[hatch], alpha=0.05)
            for collection in cs_fill.collections:
                collection.set_edgecolor(color)
                collection.set_linewidth(0.5)
        except Exception:
            pass

        # Neighboring geodesics at integer values above and below
        for offset in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
            neighbor_val = round(val) + offset
            if grid_min < neighbor_val < grid_max:
                try:
                    ax.contour(gx, gy, grid, levels=[neighbor_val],
                              colors=[color], linewidths=0.6,
                              alpha=max(0.08, 0.25 - abs(offset) * 0.05),
                              linestyles='-')
                except Exception:
                    pass

        # Main contour line (detected reading)
        try:
            ax.contour(gx, gy, grid, levels=[val],
                      colors=[color], linewidths=2, alpha=0.7)
        except Exception:
            pass



    # Compute and outline the intersection region (outside all 4 geodesics)
    outside_all = np.ones_like(gx, dtype=bool)
    for i, p in enumerate(PROBE_NAMES):
        grid = probe_grids[p]
        if grid is None:
            continue
        val = corrected[i]
        # "Outside" = reading <= val (further from probe)
        outside_all &= ~np.isnan(grid) & (grid <= val)

    # Compensation vector field visualization.
    # For each point in the geodesic region:
    # - Compute what ratios a point touch there would produce
    # - Compare to observed ratios
    # - The DIFFERENCE tells us which probes are over/under-represented
    # - Over-represented probes mean: if this point is touched, other points must
    #   be touched in the direction AWAY from those probes to balance the ratios
    # - This "compensation direction" is a vector we can visualize
    #
    # The compensation vectors should converge on the actual touch shape:
    # points at the edge of the touch need compensation from the interior,
    # points at the center need little compensation.
    area_cm2 = 0
    cell_area = (gx[0,1] - gx[0,0]) * (gy[1,0] - gy[0,0])

    has_all_grids = all(probe_grids[p] is not None for p in PROBE_NAMES)
    probe_pos = np.array([PROBES[p] for p in PROBE_NAMES])

    if has_all_grids and np.any(outside_all):
        obs_total = sum(corrected)
        obs_ratios = corrected / obs_total if obs_total > 0 else corrected

        # Expected ratios at each grid point
        expected_total = np.zeros_like(gx)
        for i, p in enumerate(PROBE_NAMES):
            expected_total += probe_grids[p]
        expected_total = np.clip(expected_total, 1, None)

        expected_ratios = []
        for i, p in enumerate(PROBE_NAMES):
            expected_ratios.append(probe_grids[p] / expected_total)

        # Ratio excess per probe: positive means this probe reads higher than observed
        # If point P is touched, probes where expected > observed are "over-served"
        # Compensation must come from the direction OPPOSITE those probes
        comp_x = np.zeros_like(gx)
        comp_y = np.zeros_like(gx)
        mismatch_mag = np.zeros_like(gx)

        for i, p in enumerate(PROBE_NAMES):
            excess = expected_ratios[i] - obs_ratios[i]
            # Direction FROM this probe (compensation needs to come from opposite side)
            # Use unit vector from probe to each grid point
            dx_from_probe = gx - probe_pos[i, 0]
            dy_from_probe = gy - probe_pos[i, 1]
            dist = np.sqrt(dx_from_probe**2 + dy_from_probe**2)
            dist = np.clip(dist, 0.1, None)

            # If excess > 0: this probe is over-represented, compensation must come
            # from AWAY from this probe (same direction as dx_from_probe)
            # If excess < 0: this probe is under-represented, compensation must come
            # TOWARD this probe (opposite direction)
            comp_x += excess * (dx_from_probe / dist)
            comp_y += excess * (dy_from_probe / dist)
            mismatch_mag += np.abs(excess)

        # Magnitude of compensation needed
        comp_mag = np.sqrt(comp_x**2 + comp_y**2)
        comp_mag_masked = np.where(outside_all, comp_mag, np.nan)

        # Self-consistency: points needing little compensation are near touch center.
        max_comp = np.nanpercentile(comp_mag_masked, 95) if np.any(outside_all) else 0.1
        max_comp = max(max_comp, 0.001)
        self_consistency = np.clip(1.0 - comp_mag / max_comp, 0, 1)
        self_consistency = np.where(outside_all, self_consistency, 0)

        # Blend in distance-from-center as a soft prior toward round touches
        if touch_pos:
            dist_center = np.sqrt((gx - touch_pos[0])**2 + (gy - touch_pos[1])**2)
            # Scale: region extent sets the falloff distance
            region_extent = np.sqrt(np.sum(outside_all) * cell_area / np.pi)
            center_prior = np.exp(-0.5 * (dist_center / max(region_extent * 0.8, 1.0))**2)
            self_consistency = self_consistency * 0.7 + self_consistency * center_prior * 0.3

        area_cm2 = np.sum(outside_all) * cell_area
        self_consistency = np.where(outside_all, self_consistency, 0)

        # Draw geodesic region boundary
        try:
            ax.contour(gx, gy, outside_all.astype(float), levels=[0.5],
                      colors=['white'], linewidths=1.0, alpha=0.4, zorder=7)
            ax.contourf(gx, gy, outside_all.astype(float), levels=[0.5, 1.5],
                       colors=['white'], alpha=0.06)
        except Exception:
            pass

        # Draw self-consistency as a gradient (high = likely touch center)
        try:
            max_sc = self_consistency.max()
            if max_sc > 0.01:
                sc_norm = self_consistency / max_sc
                levels = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.01]
                for k in range(len(levels) - 1):
                    band_alpha = 0.04 + 0.16 * (levels[k] + levels[k+1]) / 2
                    try:
                        ax.contourf(gx, gy, sc_norm,
                                   levels=[levels[k], levels[k+1]],
                                   colors=['white'], alpha=band_alpha)
                    except Exception:
                        pass
        except Exception:
            pass

def validate_calibration(interp_x, interp_y, palm_threshold):
    """Run calibration points through the estimator, print errors"""
    print("\n── CALIBRATION VALIDATION ──")
    print(f"{'Target':>12s}  {'Estimated':>12s}  {'Error':>8s}  {'Type':>6s}")
    print("─" * 50)

    first_cal = next(iter(CAL_FILES.values()), None)
    if not first_cal:
        print("  No calibration files to validate against")
        return
    with open(first_cal) as f:
        data = json.load(f)

    errors = []
    for point in data['points']:
        tx, ty = point['target']['x'], point['target']['y']
        avg = point['averages']

        touch = estimate_touch(avg, interp_x, interp_y, palm_threshold)
        if touch is None:
            print(f"  ({tx:4.1f},{ty:4.1f})  — below threshold —")
            continue

        ex, ey = touch['x'], touch['y']
        err = ((tx - ex)**2 + (ty - ey)**2) ** 0.5
        errors.append(err)

        flag = " ←←←" if err > 2.0 else ""
        print(f"  ({tx:4.1f},{ty:4.1f})  ({ex:4.1f},{ey:4.1f})  "
              f"{err:5.2f}cm  {touch['type']:>6s}{flag}")

    print("─" * 50)
    if errors:
        print(f"  Mean error: {np.mean(errors):.2f} cm")
        print(f"  Max error:  {np.max(errors):.2f} cm")
        print(f"  Median:     {np.median(errors):.2f} cm")
    print("── END VALIDATION ──\n")

def cross_validate(palm_threshold):
    """Leave-one-out cross validation on all calibration data"""
    print("\n── LEAVE-ONE-OUT CROSS VALIDATION ──")

    all_points = load_calibration_data()
    if not all_points:
        print("  No calibration data")
        return

    n_probes = len(PROBE_NAMES)
    cal_list = []
    for (x, y), ratio_list in all_points.items():
        avg_r = np.mean(ratio_list, axis=0)
        cal_list.append((x, y, *avg_r))
    cal = np.array(cal_list)

    ratios = cal[:, 2:2+n_probes]
    features = compute_features(ratios)
    y_max = SKIN_H - 1.0

    errors = []
    for i in range(len(cal)):
        mask = np.ones(len(cal), dtype=bool)
        mask[i] = False
        try:
            ix = RBFInterpolator(features[mask], cal[mask, 0],
                                  kernel='thin_plate_spline', smoothing=0.1)
            iy = RBFInterpolator(features[mask], cal[mask, 1],
                                  kernel='thin_plate_spline', smoothing=0.1)
        except Exception:
            continue

        r = features[i:i+1]
        ex = float(np.clip(ix(r), 0, SKIN_W))
        ey = float(np.clip(iy(r), 0, y_max))
        tx, ty = cal[i, 0], cal[i, 1]

        err = ((tx - ex)**2 + (ty - ey)**2) ** 0.5
        errors.append(err)

        flag = " ←←←" if err > 3.0 else ""
        print(f"  ({tx:4.1f},{ty:4.1f}) → ({ex:4.1f},{ey:4.1f})  err={err:.2f}cm{flag}")

    print("─" * 50)
    if errors:
        print(f"  LOO Mean error: {np.mean(errors):.2f} cm")
        print(f"  LOO Max error:  {np.max(errors):.2f} cm")
        print(f"  LOO Median:     {np.median(errors):.2f} cm")
    print("── END LOO ──\n")

def estimate_touch(readings, interp_x, interp_y, palm_threshold, gains=None, baseline_total=None, baseline_ratios=None):
    """Estimate touch location from probe readings.

    readings: dict of {probe_name: value} or list in PROBE_NAMES order
    """
    if isinstance(readings, dict):
        raw = np.array([readings.get(p, 0) for p in PROBE_NAMES], dtype=float)
    else:
        raw = np.array(readings, dtype=float)

    total = float(raw.sum())
    if total < MIN_TOTAL:
        return None

    # Apply per-probe gain correction from quick-cal
    if gains is not None:
        corrected = raw * gains
        total_c = corrected.sum()
        ratios = corrected / total_c
    else:
        ratios = raw / total

    features = compute_features(ratios)

    x = float(np.clip(interp_x(features), 0.0, SKIN_W))
    y_max = SKIN_H - 1.0  # functional area ends ~1cm below skin edge
    y = float(np.clip(interp_y(features), 0.0, y_max))

    flatness = ratio_flatness(*ratios[:3]) if len(ratios) >= 3 else 0.0

    # Palm detection via total signal strength
    # Palm presses produce higher total (more parallel paths through larger contact)
    touch_type = 'palm' if palm_threshold and total > palm_threshold else 'finger'

    # Pressure estimation: ratio of actual total to expected baseline total
    # at this position. <1 = light touch, ~1 = medium, >1 = hard/palm
    if baseline_total is not None:
        expected_total = float(np.clip(baseline_total(features), MIN_TOTAL, None))
        pressure = total / expected_total  # ~0.85 soft, ~1.0 medium, ~1.02 hard, ~1.15 palm
    else:
        pressure = min(total * 200 / (MIN_TOTAL * 2), 200)

    # Ambiguity detection (3-probe only — 4 probes in corners don't have this issue)
    mirror = None
    if len(PROBE_NAMES) == 3:
        r1 = ratios[0]
        center_x = SKIN_W / 2.0
        if r1 > 0.42:
            r2, r3 = ratios[1], ratios[2]
            p2p3 = r2 / r3
            log_asym = abs(np.log(p2p3))
            ambiguity = min(1.0, (r1 - 0.42) / 0.10)
            min_radius = 2.0 * ambiguity
            asym_radius = log_asym * 15.0
            radius = max(min_radius, min(asym_radius, center_x))

            mirror = {
                'left_x': round(float(np.clip(center_x - radius, 0, SKIN_W)), 1),
                'right_x': round(float(np.clip(center_x + radius, 0, SKIN_W)), 1),
                'y': round(y, 1),
                'hint': 'left' if p2p3 > 1.0 else 'right',
                'radius': round(radius, 1),
            }

    # Shape estimation (4-probe only)
    # Compare observed ratios to expected finger baseline at this position.
    # The DEVIATION from baseline encodes shape/size.
    # For a finger touch, deviation ≈ 0. For extended touches, probes along the
    # contact axis read closer to each other than the baseline predicts.
    shape_ellipse = None
    if len(PROBE_NAMES) == 4:
        # Get expected finger ratios at this position
        if baseline_ratios is not None:
            expected = np.array([float(baseline_ratios[i](features)) for i in range(4)])
            expected = np.clip(expected, 0.01, None)
            expected = expected / expected.sum()  # re-normalize
        else:
            expected = np.array([0.25, 0.25, 0.25, 0.25])

        # Normalized ratios: observed / expected
        # Values > 1 mean this probe reads higher than a finger touch would
        # Values < 1 mean lower
        norm = ratios / expected

        # Symmetry metrics on normalized ratios (baseline-corrected)
        d14 = abs(norm[0] - norm[3])
        d23 = abs(norm[1] - norm[2])
        d12 = abs(norm[0] - norm[1])
        d34 = abs(norm[2] - norm[3])
        h_sym = d14 + d23  # low = horizontally extended
        v_sym = d12 + d34  # low = vertically extended

        # Base angle: 0° = horizontal, 90° = vertical
        base_angle = float(np.degrees(np.arctan2(h_sym, v_sym)))

        # Diagonal tilt from diagonal pair similarity
        slash_sym = abs(norm[1] - norm[3])   # P2-P4 (/ diagonal probes)
        back_sym = abs(norm[0] - norm[2])     # P1-P3 (\ diagonal probes)
        diag_signal = back_sym - slash_sym  # positive = \ leaning, negative = / leaning
        tilt = float(np.clip(diag_signal * 300, -35, 35))

        angle_deg = base_angle + tilt

        # Aspect ratio from h_sym/v_sym
        aspect = float(np.clip(h_sym / (v_sym + 0.005), 0.1, 10.0))

        # Size from pressure
        base_radius = np.clip((pressure - 0.85) * 10.0 + 0.5, 0.3, 5.0) if isinstance(pressure, float) and pressure < 10 else 1.0

        # Ellipse width (along angle direction) and height (perpendicular)
        clamped_aspect = np.clip(aspect, 0.1, 10.0)
        inv_aspect = 1.0 / clamped_aspect
        extent_ratio = max(clamped_aspect, inv_aspect)  # always >= 1
        major = float(np.clip(base_radius * np.sqrt(extent_ratio), 0.3, 8.0))
        minor = float(np.clip(base_radius / np.sqrt(extent_ratio), 0.3, 8.0))

        shape_ellipse = {
            'major': round(major, 1),
            'minor': round(minor, 1),
            'angle': round(angle_deg, 0),
            'aspect': round(float(aspect), 2),
        }

    return {
        'x': round(x, 1),
        'y': round(y, 1),
        'pressure': round(pressure, 3),
        'total': total,
        'type': touch_type,
        'flatness': round(flatness, 3),
        'ratios': tuple(round(float(r), 3) for r in ratios),
        'raw': tuple(int(v) for v in raw),
        'mirror': mirror,
        'shape': shape_ellipse,
    }

# ── SERIAL ──
def parse_line(line):
    """Parse serial line like 'P1:42 P2:30 P3:45' into dict."""
    try:
        parts = line.split()
        vals = {}
        for p in parts:
            k, v = p.split(':')
            vals[k] = int(v)
        return vals
    except:
        return None

# Accumulator for multi-line serial protocols
_serial_acc = {}
_serial_acc_time = 0

def read_full_sample(parsed):
    """Accumulate parsed values until all probes present. Returns full dict or None."""
    global _serial_acc, _serial_acc_time
    if parsed is None:
        return None
    now = time.time()
    # Flush stale accumulator (if >0.2s since last update, readings are from different moments)
    if now - _serial_acc_time > 0.2:
        _serial_acc = {}
    _serial_acc_time = now
    _serial_acc.update(parsed)
    if all(p in _serial_acc for p in PROBE_NAMES):
        result = {p: _serial_acc[p] for p in PROBE_NAMES}
        _serial_acc = {}
        return result
    return None

# ── VISUALIZATION ──
def create_heatmap_colormap():
    colors = [
        (0.0, '#000000'),   # black (no touch)
        (0.15, '#1a0a2e'),  # deep purple
        (0.3, '#3d1f6d'),   # purple
        (0.5, '#e74c3c'),   # red
        (0.7, '#f39c12'),   # orange
        (0.85, '#f1c40f'),  # yellow
        (1.0, '#ffffff'),   # white (max)
    ]
    cmap = LinearSegmentedColormap.from_list('skin_heat',
        [(v, c) for v, c in colors])
    return cmap

def setup_plot():
    plt.ion()
    fig, (ax_map, ax_info) = plt.subplots(1, 2, figsize=(12, 8),
        gridspec_kw={'width_ratios': [3, 1]})

    fig.patch.set_facecolor('black')

    # main map
    ax_map.set_xlim(-0.5, SKIN_W + 0.5)
    ax_map.set_ylim(-0.5, SKIN_H + 0.5)
    ax_map.set_aspect('equal')
    ax_map.set_facecolor('black')
    ax_map.tick_params(colors='gray', labelsize=8)

    skin_rect = patches.Rectangle((0, 0), SKIN_W, SKIN_H,
        fill=False, edgecolor='white', linewidth=1.5, linestyle='--')
    ax_map.add_patch(skin_rect)

    for name, (px, py) in PROBES.items():
        ax_map.plot(px, py, 'c^', markersize=8, alpha=0.5)
        ax_map.text(px, py + 0.5, name, color='cyan',
                    fontsize=7, ha='center', alpha=0.5)

    # info panel
    ax_info.set_facecolor('black')
    ax_info.axis('off')

    return fig, ax_map, ax_info

# ── AUDIO ENGINE ──
class SkinSynth:
    """Real-time 4-oscillator synth driven by probe readings.

    Each probe maps to one voice. Left probes → left channel, right → right.
    Frequency from probe reading, amplitude from total signal.
    """
    SAMPLE_RATE = 44100

    def __init__(self, probe_names, probes):
        self.probe_names = probe_names
        self.probes = probes
        self.n = len(probe_names)

        # Frequency range: map probe readings to a musical range
        self.freq_lo = 110   # A2
        self.freq_hi = 880   # A5 (3 octaves)
        self.reading_lo = 5
        self.reading_hi = 50

        # Per-probe detuning: each voice gets a unique offset so beating
        # patterns encode which probes are active together.
        # Spread evenly: 0, +2, +5, +7 Hz — like intervals in a chord.
        # When two probes dominate, their specific beat frequency is audible.
        n = len(probe_names)
        self.detune = np.array([i * 2.3 for i in range(n)])  # 0, 2.3, 4.6, 6.9 Hz
        self.detune -= self.detune.mean()  # center around zero

        # State: current target frequencies and amplitudes (updated from main thread)
        self.target_freqs = np.zeros(self.n)
        self.target_amps = np.zeros(self.n)
        self.active = False

        # Smooth state for glitch-free audio
        self.phases = np.zeros(self.n)
        self.current_freqs = np.zeros(self.n)
        self.current_amps = np.zeros(self.n)

        # Which probes go to which channel
        # Left-side probes (low x) → left channel, right-side → right channel
        center_x = max(p[0] for p in probes.values()) / 2
        self.left_weight = np.array([
            1.0 - min(probes[p][0] / (center_x * 2), 1.0)
            for p in probe_names
        ])
        self.right_weight = 1.0 - self.left_weight

        self.stream = None

    def _reading_to_freq(self, reading):
        """Map probe reading to frequency. Uses sqrt to spread out the mid-range
        and compress the near-probe extremes."""
        t = np.clip((reading - self.reading_lo) / (self.reading_hi - self.reading_lo), 0, 1)
        # sqrt stretches the lower/mid range where most touches happen
        t = np.sqrt(t)
        return self.freq_lo * (self.freq_hi / self.freq_lo) ** t

    def update(self, readings, active=True):
        """Update target frequencies from probe readings dict."""
        self.active = active
        if not active or readings is None:
            self.target_amps[:] = 0
            return

        for i, p in enumerate(self.probe_names):
            val = readings.get(p, 0)
            self.target_freqs[i] = self._reading_to_freq(val) + self.detune[i]
            self.target_amps[i] = np.clip(val / self.reading_hi, 0, 1) * 0.15

    def _callback(self, outdata, frames, time_info, status):
        """Audio callback — vectorized stereo synthesis with continuous phase."""
        t_step = 1.0 / self.SAMPLE_RATE

        left = np.zeros(frames)
        right = np.zeros(frames)

        for i in range(self.n):
            target_f = self.target_freqs[i] if self.active else self.current_freqs[i]
            target_a = self.target_amps[i] if self.active else 0.0

            # Linearly interpolate frequency across the block for smooth transitions
            freq_ramp = np.linspace(self.current_freqs[i], target_f, frames, endpoint=False)

            # Accumulate phase from frequency ramp (continuous, no discontinuities)
            phase_increments = freq_ramp * t_step
            phases = self.phases[i] + np.cumsum(phase_increments)

            # Amplitude ramp
            amp_ramp = np.linspace(self.current_amps[i], target_a, frames, endpoint=False)

            # Sine wave from continuous phase (no modulo needed for sin)
            samples = amp_ramp * np.sin(2 * np.pi * phases)

            left += samples * self.left_weight[i]
            right += samples * self.right_weight[i]

            # Update state — keep phase continuous (modulo to prevent overflow)
            self.phases[i] = phases[-1] % 1000.0
            self.current_freqs[i] = target_f
            self.current_amps[i] = target_a

        outdata[:, 0] = left
        outdata[:, 1] = right

    def start(self):
        """Start the audio stream."""
        self.stream = sd.OutputStream(
            samplerate=self.SAMPLE_RATE,
            channels=2,
            callback=self._callback,
            blocksize=2048,
            latency='high',
        )
        self.stream.start()

    def stop(self):
        """Stop the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()


# ── UNIFIED DRAW ──
def draw_frame(fig, ax_map, ax_info, probe_grids, gx, gy, cmap,
               touch, touch_active, current_readings, probe_gains,
               trail, now, area_cm2_ref,
               show_trail=True, show_geodesic=True):
    """Draw one frame: trail dots + geodesic overlay if touching + info panel."""
    ax_map.clear()
    ax_map.set_xlim(-0.5, SKIN_W + 0.5)
    ax_map.set_ylim(-0.5, SKIN_H + 0.5)
    ax_map.set_aspect('equal')
    ax_map.set_facecolor('black')

    # Skin outline
    skin_rect = patches.Rectangle((0, 0), SKIN_W, SKIN_H,
        fill=False, edgecolor='white', linewidth=1.5, linestyle='--')
    ax_map.add_patch(skin_rect)

    # Probes
    for name, (px, py) in PROBES.items():
        color = GEODESIC_COLORS.get(name, 'cyan')
        ax_map.plot(px, py, '^', color=color, markersize=8, alpha=0.4)
        ax_map.text(px, py + 0.5, name, color=color,
                    fontsize=7, ha='center', alpha=0.4)

    # Trail dots (fading)
    if trail and show_trail:
        for t in trail:
            age = now - t['time']
            alpha = max(0.03, 1.0 - (age / TRAIL_MAX_AGE))
            p_norm = np.clip((t['pressure'] - 0.7) / 0.5, 0, 1) if isinstance(t['pressure'], float) and t['pressure'] < 10 else 0.5
            color = cmap(p_norm)
            size = 20 + p_norm * 40
            ax_map.scatter(t['x'], t['y'], c=[color], s=size, alpha=alpha,
                          marker='o', edgecolors='none', zorder=3)

    # Geodesic overlay for current touch
    area_cm2 = 0
    if touch_active and current_readings is not None and probe_grids is not None and show_geodesic:
        touch_pos = (touch['x'], touch['y']) if touch else None
        area_cm2 = draw_geodesic_overlay(ax_map, probe_grids, gx, gy,
                                          current_readings, probe_gains, touch_pos)

    # Title
    if touch_active and touch:
        extra = f"  area≈{area_cm2:.1f}cm²" if area_cm2 > 0 else ""
        ax_map.set_title(
            f"({touch['x']:.1f}, {touch['y']:.1f})  "
            f"p={touch['pressure']:.2f}  [{touch['type']}]{extra}",
            color='white', fontsize=12, pad=10)
    else:
        ax_map.set_title("waiting...", color='gray', fontsize=10, pad=10)

    ax_map.tick_params(colors='gray', labelsize=8)

    # Info panel
    ax_info.clear()
    ax_info.set_facecolor('black')
    ax_info.axis('off')

    if touch:
        info_lines = [
            f"x:  {touch['x']:.1f} cm",
            f"y:  {touch['y']:.1f} cm",
            f"",
            f"pressure: {touch['pressure']:.2f}",
            f"type: {touch['type']}",
            f"",
            f"ratios:",
        ] + [f"  {p}: {touch['ratios'][i]:.3f}" for i, p in enumerate(PROBE_NAMES)] + [
            f"",
            f"raw:",
        ] + [f"  {p}: {touch['raw'][i]}" for i, p in enumerate(PROBE_NAMES)] + [
            f"  total: {touch['total']:.0f}",
        ]
        if area_cm2 > 0:
            info_lines += [f"", f"area: {area_cm2:.1f} cm²"]
    else:
        info_lines = [
            f"no touch",
            f"",
            f"trail: {len(trail)} pts",
        ]

    for i, line_text in enumerate(info_lines):
        c = 'white' if touch_active else 'gray'
        ax_info.text(0.05, 0.95 - i * 0.04, line_text,
                     color=c, fontsize=9, fontfamily='monospace',
                     transform=ax_info.transAxes, verticalalignment='top')

    return area_cm2



# ── MAIN ──
def main():
    global SKIN_W, SKIN_H, BAUD, PROBES, PROBE_NAMES, CAL_FILES, MIN_TOTAL

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='Config JSON name')
    parser.add_argument('--port', default=None)
    parser.add_argument('--session', action='store_true',
                        help='Enable Princess API session')
    parser.add_argument('--skip-quickcal', action='store_true',
                        help='Skip startup quick calibration')
    parser.add_argument('--no-trail', action='store_true',
                        help='Disable trail dots')
    parser.add_argument('--no-geodesic', action='store_true',
                        help='Disable geodesic overlay')
    parser.add_argument('--audio', action='store_true',
                        help='Enable audio synthesis from probe readings')
    parser.add_argument('--cal', nargs='+', default=None,
                        help='Override calibration file(s) to use')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    SKIN_W = cfg['skin_w']
    SKIN_H = cfg['skin_h']
    BAUD = cfg['baud']
    PROBES = cfg['probes']
    PROBE_NAMES = sorted(PROBES.keys())
    CAL_FILES = cfg.get('cal_files', {})
    if args.cal:
        CAL_FILES = {f'cli_{i}': f for i, f in enumerate(args.cal)}
    MIN_TOTAL = cfg.get('min_total', 60)
    port = args.port or cfg['port']
    print(f"[CONFIG] {cfg['name']}")
    if args.cal:
        print(f"[CONFIG] Cal files overridden: {args.cal}")

    # Load calibration
    interp_x, interp_y, palm_threshold, baseline_total, baseline_ratios = load_calibration(cfg)

    validate_calibration(interp_x, interp_y, palm_threshold)
    cross_validate(palm_threshold)

    # Connect serial
    try:
        ser = serial.Serial(port, BAUD, timeout=1)
        time.sleep(2)
        print(f"[SKIN] Connected on {port}")
    except:
        print(f"[SKIN] Could not connect on {port}")
        return

    # Quick startup calibration
    if args.skip_quickcal:
        probe_gains = np.ones(len(PROBE_NAMES))
        print("[QUICKCAL] Skipped")
    else:
        probe_gains = run_quickcal(ser, CAL_FILES)

    # Build probe fields for geodesic overlay
    print("[INIT] Building probe fields...")
    probe_grids, gx, gy, cal_xy = build_probe_fields()
    if probe_grids is None:
        print("[WARN] No probe field data — geodesic overlay disabled")

    # Setup visualization
    fig, ax_map, ax_info = setup_plot()
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
    cmap = create_heatmap_colormap()

    trail = deque(maxlen=TRAIL_LENGTH)
    touch_active = False
    last_touch_time = 0
    last_draw_time = 0
    layout_done = False
    session_log = []
    smooth_x, smooth_y = None, None
    current_readings = None
    reading_accum = []
    SMOOTH_ALPHA = 0.4

    # Princess session (optional)
    if args.session:
        try:
            anthropic.Anthropic()
            print("[PRINCESS] API connected")
        except:
            print("[PRINCESS] Could not connect API")

    # Audio synth
    synth = None
    if args.audio and HAS_AUDIO:
        synth = SkinSynth(PROBE_NAMES, PROBES)
        synth.start()
        print("[AUDIO] Synth started")
    elif args.audio:
        print("[AUDIO] sounddevice not installed — audio disabled")

    print("\n[READY] Touch the skin. Ctrl+C to exit.\n")

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line or ':' not in line:
                fig.canvas.flush_events()
                continue

            readings = read_full_sample(parse_line(line))
            if readings is None:
                fig.canvas.flush_events()
                continue
            raw_list = [readings[p] for p in PROBE_NAMES]
            total = sum(raw_list)

            touch = estimate_touch(readings, interp_x, interp_y,
                                   palm_threshold, probe_gains,
                                   baseline_total, baseline_ratios)

            now = time.time()

            if touch:
                if not touch_active:
                    touch_active = True
                    touch_start = now
                    touch_warmup = 3
                    estimate_touch._peak_total = 0
                    reading_accum = []
                    subprocess.Popen(['afplay', '/System/Library/Sounds/Tink.aiff'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"\n── TOUCH START ──")
                    session_log.append({'event': 'touch_start', 'time': now})

                # Skip ramp-up
                if touch_warmup > 0:
                    touch_warmup -= 1
                    smooth_x, smooth_y = touch['x'], touch['y']
                    last_touch_time = now
                    continue

                # Skip ramp-down
                if not hasattr(estimate_touch, '_peak_total'):
                    estimate_touch._peak_total = total
                estimate_touch._peak_total = max(estimate_touch._peak_total, total)
                if total < estimate_touch._peak_total * 0.80:
                    last_touch_time = now
                    continue

                # Smooth position
                smooth_x = SMOOTH_ALPHA * touch['x'] + (1 - SMOOTH_ALPHA) * smooth_x
                smooth_y = SMOOTH_ALPHA * touch['y'] + (1 - SMOOTH_ALPHA) * smooth_y

                # Accumulate readings for geodesic averaging
                reading_accum.append(readings)

                trail.append({
                    'x': round(smooth_x, 1),
                    'y': round(smooth_y, 1),
                    'pressure': touch['pressure'],
                    'type': touch['type'],
                    'time': now,
                })
                last_touch_time = now

                # Print live
                raw_str = ','.join(str(readings[p]) for p in PROBE_NAMES)
                print(f"  {smooth_x:5.1f}, {smooth_y:5.1f}  "
                      f"p={touch['pressure']:.2f}  "
                      f"{touch['type']:6s}  "
                      f"raw=({raw_str})", end='\r')

                session_log.append({
                    'x': touch['x'], 'y': touch['y'],
                    'pressure': touch['pressure'],
                    'type': touch['type'],
                    'raw': raw_list,
                    'time': now,
                })

            else:
                if touch_active and (now - last_touch_time > 0.3):
                    touch_active = False
                    duration = now - touch_start
                    current_readings = None
                    reading_accum = []
                    if synth:
                        synth.update(None, active=False)
                    subprocess.Popen(['afplay', '/System/Library/Sounds/Pop.aiff'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"\n── TOUCH END ({duration:.1f}s) ──")
                    session_log.append({'event': 'touch_end', 'time': now,
                                        'duration': round(duration, 2)})
                    print("---")

            # Prune old trail points
            while trail and (now - trail[0]['time']) > TRAIL_MAX_AGE:
                trail.popleft()

            # Average accumulated readings for geodesic display
            if reading_accum and touch_active:
                current_readings = {}
                for p in PROBE_NAMES:
                    current_readings[p] = np.mean([r[p] for r in reading_accum])

            # Draw (throttled — slower when touching due to geodesic overhead)
            draw_interval = 0.2 if touch_active else 0.1
            if now - last_draw_time >= draw_interval:
                last_draw_time = now
                if touch_active:
                    reading_accum = []

                # Update audio from averaged readings
                if synth and current_readings:
                    synth.update(current_readings, active=touch_active)

                draw_frame(fig, ax_map, ax_info, probe_grids, gx, gy, cmap,
                           touch, touch_active, current_readings, probe_gains,
                           trail, now, 0,
                           show_trail=not args.no_trail,
                           show_geodesic=not args.no_geodesic)

                if not layout_done:
                    plt.tight_layout()
                    layout_done = True

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            else:
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n\n[EXIT] Saving session...")
        if session_log:
            logfile = f"skin_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(logfile, 'w') as f:
                json.dump({
                    'start_time': datetime.now().isoformat(),
                    'num_events': len(session_log),
                    'events': session_log,
                }, f, indent=2)
            print(f"[SAVE] {logfile} ({len(session_log)} events)")

        if synth:
            synth.stop()
        ser.close()
        plt.close()
        print("[DONE]")

if __name__ == '__main__':
    main()
