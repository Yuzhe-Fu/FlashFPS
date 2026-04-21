"""Runtime switches for FlashFPS / QuickFPS / FPScache.

These values are read inside `SetAbstraction.forward` in pointnext.py /
pointvector.py. They can be overridden from the command line via EasyConfig
using the dot-notation, e.g.:

    bash script/main_segmentation.sh cfgs/s3dis/pointnext-l.yaml \
        flashfps.useFlashFPS=True flashfps.useFPS_Prune=True \
        flashfps.useFPS_Cache=True flashfps.PruneRate=0.25 \
        flashfps.enable_quick_fps=False

`update_from_cfg(cfg)` is called once at the top of `main(gpu, cfg)` in
`examples/segmentation/main.py` (and similar entry points) to populate these
module-level attributes from `cfg.flashfps.*`. Missing fields fall back to the
defaults defined here.
"""

from __future__ import annotations

useFlashFPS: bool = False
useFPS_Prune: bool = True
useFPS_Cache: bool = True
PruneRate: float = 0.25
enable_quick_fps: bool = False


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)


def update_from_cfg(cfg) -> None:
    """Pull `flashfps.*` out of an EasyConfig (if present) and update globals."""
    global useFlashFPS, useFPS_Prune, useFPS_Cache, PruneRate, enable_quick_fps

    if cfg is None:
        return
    fcfg = cfg.get("flashfps", None) if hasattr(cfg, "get") else None
    if fcfg is None:
        return

    if "useFlashFPS" in fcfg:
        useFlashFPS = _to_bool(fcfg["useFlashFPS"])
    if "useFPS_Prune" in fcfg:
        useFPS_Prune = _to_bool(fcfg["useFPS_Prune"])
    if "useFPS_Cache" in fcfg:
        useFPS_Cache = _to_bool(fcfg["useFPS_Cache"])
    if "PruneRate" in fcfg:
        PruneRate = float(fcfg["PruneRate"])
    if "enable_quick_fps" in fcfg:
        enable_quick_fps = _to_bool(fcfg["enable_quick_fps"])


def snapshot() -> dict:
    """Return a plain-dict copy of current switches (handy for logging)."""
    return {
        "useFlashFPS": useFlashFPS,
        "useFPS_Prune": useFPS_Prune,
        "useFPS_Cache": useFPS_Cache,
        "PruneRate": PruneRate,
        "enable_quick_fps": enable_quick_fps,
    }
