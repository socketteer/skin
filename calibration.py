import serial
import json
import time
import argparse
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from skin_config import load_config

def beep_start():
    """Short high beep — touch detected, recording started."""
    subprocess.Popen(['afplay', '/System/Library/Sounds/Tink.aiff'],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def beep_done():
    """Lower beep — touch ended, point recorded."""
    subprocess.Popen(['afplay', '/System/Library/Sounds/Pop.aiff'],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── ARGS ──
parser = argparse.ArgumentParser(description='Skin calibration data collector')
parser.add_argument('--config', default=None, help='Config JSON name')
parser.add_argument('--port', default=None, help='Serial port override')
parser.add_argument('--corners', action='store_true',
                    help='Use denser corner grid instead of full grid')
parser.add_argument('--palm', action='store_true',
                    help='Use coarser grid for palm press calibration')
parser.add_argument('--point', action='store_true',
                    help='Use coarser grid for point touch calibration (use a stylus)')
parser.add_argument('--pressure', type=str, default=None,
                    choices=['soft', 'medium', 'hard'],
                    help='Coarse grid pressure test (soft/medium/hard)')
parser.add_argument('--shape', type=str, default=None,
                    choices=['pen', 'cap', 'cylinder'],
                    help='Shape calibration: pen (14cm line), cap (2cm circle), or cylinder (8cm circle)')
args = parser.parse_args()

# ── CONFIG ──
cfg = load_config(args.config)
PORT = args.port or cfg['port']
BAUD = cfg['baud']
SKIN_W = cfg['skin_w']
SKIN_H = cfg['skin_h']
PROBES = cfg['probes']
PROBE_NAMES = sorted(PROBES.keys())

import math

# Shape calibration targets
SHAPE_TARGETS = None
if args.shape == 'pen':
    pen_len = 14.0
    d = pen_len / math.sqrt(2)
    SHAPE_TARGETS = [
        {'name': 'horiz y=4',  'shape': 'line', 'x1': 0.5, 'y1': 4, 'x2': 14.5, 'y2': 4},
        {'name': 'horiz y=8',  'shape': 'line', 'x1': 0.5, 'y1': 8, 'x2': 14.5, 'y2': 8},
        {'name': 'horiz y=12', 'shape': 'line', 'x1': 0.5, 'y1': 12, 'x2': 14.5, 'y2': 12},
        {'name': 'horiz y=16', 'shape': 'line', 'x1': 0.5, 'y1': 16, 'x2': 14.5, 'y2': 16},
        {'name': 'horiz y=20', 'shape': 'line', 'x1': 0.5, 'y1': 20, 'x2': 14.5, 'y2': 20},
        {'name': 'vert x=3',   'shape': 'line', 'x1': 3, 'y1': 4.5, 'x2': 3, 'y2': 18.5},
        {'name': 'vert x=7.5', 'shape': 'line', 'x1': 7.5, 'y1': 4.5, 'x2': 7.5, 'y2': 18.5},
        {'name': 'vert x=12',  'shape': 'line', 'x1': 12, 'y1': 4.5, 'x2': 12, 'y2': 18.5},
        {'name': 'diag /',      'shape': 'line', 'x1': 1, 'y1': 2, 'x2': round(1+d, 1), 'y2': round(2+d, 1)},
        {'name': 'diag \\',     'shape': 'line', 'x1': 14, 'y1': 2, 'x2': round(14-d, 1), 'y2': round(2+d, 1)},
    ]
elif args.shape == 'cap':
    # 2cm circle at coarse grid points
    _full_x = cfg['touch_grid']['x']
    _full_y = cfg['touch_grid']['y']
    _cap_x = _full_x[::2] if len(_full_x) > 4 else _full_x
    _cap_y = _full_y[::2] if len(_full_y) > 4 else _full_y
    SHAPE_TARGETS = [
        {'name': f'cap ({x},{y})', 'shape': 'circle', 'cx': x, 'cy': y, 'r': 1.0}
        for y in _cap_y for x in _cap_x
    ]
elif args.shape == 'cylinder':
    # 8cm circle at coarse grid points (only where it fits: 4cm from edges)
    _full_x = cfg['touch_grid']['x']
    _full_y = cfg['touch_grid']['y']
    sw, sh = cfg['skin_w'], cfg['skin_h']
    _cyl_x = [x for x in _full_x[::2] if 4 <= x <= sw - 4]
    _cyl_y = [y for y in _full_y[::2] if 4 <= y <= sh - 4]
    SHAPE_TARGETS = [
        {'name': f'cyl ({x},{y})', 'shape': 'circle', 'cx': x, 'cy': y, 'r': 4.0}
        for y in _cyl_y for x in _cyl_x
    ]

if args.shape and SHAPE_TARGETS:
    # For shape mode, use center of each shape as the touch sequence point
    TOUCH_SEQUENCE = []
    for st in SHAPE_TARGETS:
        if st['shape'] == 'line':
            cx = (st['x1'] + st['x2']) / 2
            cy = (st['y1'] + st['y2']) / 2
            TOUCH_SEQUENCE.append((cx, cy))
        elif st['shape'] == 'circle':
            TOUCH_SEQUENCE.append((st['cx'], st['cy']))
elif args.palm or args.point or args.pressure:
    # Coarse grid for palm/point: every other point from the full grid
    _full_x = cfg['touch_grid']['x']
    _full_y = cfg['touch_grid']['y']
    _grid_x = _full_x[::2] if len(_full_x) > 4 else _full_x
    _grid_y = _full_y[::2] if len(_full_y) > 4 else _full_y
    _extra = []
elif args.corners and 'corner_grid' in cfg:
    _grid_x = cfg['corner_grid']['x']
    _grid_y = cfg['corner_grid']['y']
    _extra = cfg.get('corner_grid', {}).get('extra', [])
else:
    _grid_x = cfg['touch_grid']['x']
    _grid_y = cfg['touch_grid']['y']
    _extra = cfg.get('touch_grid', {}).get('extra', [])

if not (args.shape and SHAPE_TARGETS):
    TOUCH_SEQUENCE = [(x, y) for y in _grid_y for x in _grid_x]
    TOUCH_SEQUENCE += [(x, y) for x, y in _extra]

TOUCH_ON_THRESHOLD = cfg.get('touch_on_threshold', 20)
TOUCH_OFF_THRESHOLD = cfg.get('touch_off_threshold', 15)

# ── SERIAL ──
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

def parse_serial(line):
    try:
        parts = line.split()
        vals = {}
        for p in parts:
            k, v = p.split(':')
            vals[k] = int(v)
        return vals
    except:
        return None

# Accumulator for multi-line serial protocols where each probe comes on its own line
_serial_acc = {}
_serial_acc_time = 0

def read_full_sample(ser_line_parsed):
    """Accumulate parsed probe values until we have all probes. Returns full dict or None."""
    global _serial_acc, _serial_acc_time
    if ser_line_parsed is None:
        return None
    now = time.time()
    if now - _serial_acc_time > 0.2:
        _serial_acc = {}
    _serial_acc_time = now
    _serial_acc.update(ser_line_parsed)
    if all(p in _serial_acc for p in PROBE_NAMES):
        result = {p: _serial_acc[p] for p in PROBE_NAMES}
        _serial_acc = {}
        return result
    return None

def any_active(vals):
    return any(vals.get(p, 0) > TOUCH_ON_THRESHOLD for p in PROBE_NAMES)

def all_below_off(vals):
    return all(vals.get(p, 0) < TOUCH_OFF_THRESHOLD for p in PROBE_NAMES)

# ── DEBUG: print first few serial lines to check format ──
print("[DEBUG] Reading serial lines to check format...")
debug_count = 0
while debug_count < 30:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if not line:
        continue
    vals = parse_serial(line)
    print(f"  raw: '{line}'  parsed: {vals}")
    debug_count += 1
    full = read_full_sample(vals)
    if full:
        print(f"  FULL SAMPLE: {full}")
        break
print()

# ── VISUALIZATION ──
plt.ion()
fig, ax = plt.subplots(figsize=(5, 7))
fig.show()
fig.canvas.draw()
fig.canvas.flush_events()

def draw_shape_target(ax, st, color='red', alpha=1.0):
    """Draw a shape target (line or circle) on the axes."""
    if st['shape'] == 'line':
        ax.plot([st['x1'], st['x2']], [st['y1'], st['y2']],
                color=color, linewidth=3, alpha=alpha)
    elif st['shape'] == 'circle':
        circle = plt.Circle((st['cx'], st['cy']), st['r'],
                             fill=False, edgecolor=color, linewidth=2, alpha=alpha)
        ax.add_patch(circle)

def draw_prompt(target_idx):
    ax.clear()
    ax.set_xlim(-0.5, SKIN_W + 0.5)
    ax.set_ylim(-0.5, SKIN_H + 0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('black')

    skin_rect = patches.Rectangle((0, 0), SKIN_W, SKIN_H,
                                   fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(skin_rect)

    for name, (px, py) in PROBES.items():
        ax.plot(px, py, 'cs', markersize=8)
        ax.text(px + 0.3, py + 0.3, name, color='cyan', fontsize=7)

    if SHAPE_TARGETS:
        # shape mode: draw completed shapes green, upcoming faint, current red
        for i in range(target_idx):
            draw_shape_target(ax, SHAPE_TARGETS[i], color='green', alpha=0.4)
        for i in range(target_idx + 1, len(SHAPE_TARGETS)):
            draw_shape_target(ax, SHAPE_TARGETS[i], color='white', alpha=0.15)
        draw_shape_target(ax, SHAPE_TARGETS[target_idx], color='red', alpha=1.0)
        st = SHAPE_TARGETS[target_idx]
        title = f"{st['name']}  [{target_idx + 1}/{len(SHAPE_TARGETS)}]  [SHAPE-{args.shape.upper()}]"
    else:
        # point mode
        for i in range(target_idx):
            x, y = TOUCH_SEQUENCE[i]
            ax.plot(x, y, 'g.', markersize=8)
        for i in range(target_idx + 1, len(TOUCH_SEQUENCE)):
            x, y = TOUCH_SEQUENCE[i]
            ax.plot(x, y, 'w.', markersize=4, alpha=0.2)
        tx, ty = TOUCH_SEQUENCE[target_idx]
        ax.plot(tx, ty, 'ro', markersize=16, fillstyle='none', markeredgewidth=2)
        ax.plot(tx, ty, 'r+', markersize=12, markeredgewidth=2)
        title = f"Touch ({tx}, {ty})  [{target_idx + 1}/{len(TOUCH_SEQUENCE)}]  [{mode_str}]"

    ax.set_title(title, color='white', fontsize=12)
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='gray')
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

def draw_recording(target_idx, vals):
    tx, ty = TOUCH_SEQUENCE[target_idx]
    probe_str = '  '.join(f"{p}:{vals.get(p, 0)}" for p in PROBE_NAMES)
    ax.set_title(f"RECORDING at ({tx}, {ty})  {probe_str}",
                 color='red', fontsize=11)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# ── MAIN ──
calibration_data = []
mode_str = (f"SHAPE-{args.shape.upper()}" if args.shape else
            (f"PRESSURE-{args.pressure.upper()}" if args.pressure else
             ("POINT" if args.point else
              ("PALM" if args.palm else
               ("CORNER" if args.corners else "FULL")))))
print(f"=== SKIN CALIBRATION ({mode_str}) ===")
print(f"Config: {cfg['name']}")
print(f"{len(TOUCH_SEQUENCE)} points to touch\n")

for idx, (tx, ty) in enumerate(TOUCH_SEQUENCE):
    draw_prompt(idx)
    if SHAPE_TARGETS:
        st = SHAPE_TARGETS[idx]
        print(f"\n[{idx + 1}/{len(TOUCH_SEQUENCE)}] Press {st['name']} — hold and release")
    else:
        print(f"\n[{idx + 1}/{len(TOUCH_SEQUENCE)}] Touch ({tx}, {ty}) — press when ready")

    # wait for touch to begin
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            fig.canvas.flush_events()
            continue
        vals = read_full_sample(parse_serial(line))
        if vals and any_active(vals):
            break
        fig.canvas.flush_events()

    # record touch
    beep_start()
    print(f"  Recording...")
    buffer = []
    touch_start = time.time()
    zero_count = 0

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        vals = read_full_sample(parse_serial(line))
        if vals is None:
            continue

        if not all_below_off(vals):
            zero_count = 0
            vals['timestamp'] = time.time()
            buffer.append(vals)
            draw_recording(idx, vals)
        else:
            zero_count += 1
            if zero_count >= 5:
                break

        plt.pause(0.01)

    touch_duration = time.time() - touch_start

    # compute stats
    if buffer:
        avg = {p: sum(v[p] for v in buffer) / len(buffer) for p in PROBE_NAMES}
        peak = {p: max(v[p] for v in buffer) for p in PROBE_NAMES}
    else:
        avg = {p: 0 for p in PROBE_NAMES}
        peak = avg

    entry = {
        'target': {'x': tx, 'y': ty},
        'num_samples': len(buffer),
        'duration': round(touch_duration, 2),
        'averages': {k: round(v, 1) for k, v in avg.items()},
        'peaks': peak,
        'raw': buffer,
    }
    # Add shape metadata if in shape mode
    if SHAPE_TARGETS:
        entry['shape'] = SHAPE_TARGETS[idx]
    calibration_data.append(entry)

    beep_done()
    print(f"  Done: {len(buffer)} samples, {touch_duration:.1f}s")
    probe_avg = '  '.join(f"{p}:{avg[p]:.0f}" for p in PROBE_NAMES)
    probe_peak = '  '.join(f"{p}:{peak[p]}" for p in PROBE_NAMES)
    print(f"     avg  {probe_avg}")
    print(f"     peak {probe_peak}")

# ── SAVE ──
suffix = (f"_shape_{args.shape}" if args.shape else
          (f"_pressure_{args.pressure}" if args.pressure else
           ("_point" if args.point else
            ("_palm" if args.palm else
             ("_corners" if args.corners else "")))))
filename = f"calibration{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
output = {
    'skin_dimensions': {'width': SKIN_W, 'height': SKIN_H},
    'probes': PROBES,
    'config': cfg['name'],
    'mode': mode_str.lower(),
    'touch_sequence': [{'x': x, 'y': y} for x, y in TOUCH_SEQUENCE],
    'timestamp': datetime.now().isoformat(),
    'num_points': len(calibration_data),
    'points': calibration_data,
}

with open(filename, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n=== DONE ===")
print(f"Saved {len(calibration_data)} points to {filename}")

ser.close()
plt.ioff()
plt.show()
