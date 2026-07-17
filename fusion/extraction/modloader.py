"""Import helper for the four modality codebases.

Each modality package uses generic top-level module names (`config`, `src`,
`model`, `inference`) resolved via its own sys.path hacks, so importing two
modalities naively collides. `load_module` imports one file under a unique
alias with the needed paths prepended, then purges the generic names from
sys.modules so the next modality starts clean. Load order matters only in
that context's zero_shot.py permanently inserts its own root into sys.path —
load it last (PerFrameExtractor does).
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_GENERIC = ("config", "src", "model", "inference")


def _purge_generic():
    for name in list(sys.modules):
        if name in _GENERIC or name.startswith(tuple(g + "." for g in _GENERIC)):
            del sys.modules[name]


def load_module(alias, file_path, extra_sys_path=()):
    """Import `file_path` as module `alias`, with `extra_sys_path` prepended."""
    _purge_generic()
    paths = [str(p) for p in extra_sys_path]
    sys.path[0:0] = paths
    try:
        spec = importlib.util.spec_from_file_location(alias, str(file_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for p in paths:
            try:
                sys.path.remove(p)
            except ValueError:
                pass
        _purge_generic()
