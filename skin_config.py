import json
import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
DEFAULT_CONFIG = 'skin_3probe_11x15.json'

def load_config(name=None):
    """Load a skin config JSON. name can be a filename or full path."""
    if name is None:
        name = DEFAULT_CONFIG
    if not os.path.isabs(name) and not name.startswith('configs'):
        name = os.path.join(CONFIG_DIR, name)
    with open(name) as f:
        cfg = json.load(f)
    cfg['probes'] = {k: tuple(v) for k, v in cfg['probes'].items()}
    return cfg
