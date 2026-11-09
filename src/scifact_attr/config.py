"""Configuration loading: base.yaml + one condition from conditions.yaml."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _read(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_base(path: str | Path = CONFIG_DIR / "base.yaml") -> Dict[str, Any]:
    return _read(REPO_ROOT / path)


def adapter_dir(cfg: Dict[str, Any], seed: int, smoke: bool = False) -> Path:
    base = REPO_ROOT / cfg["adapter_path"]
    suffix = "_smoke" if smoke else ""
    return base.with_name(f"{base.name}_s{seed}{suffix}")


def load_condition(condition: str, control: bool = False) -> Dict[str, Any]:
    """Return the merged config for a condition such as "c3" or "c3_prime".

    The condition entry is merged on top of base.yaml, so a condition file only has to
    state what it changes. `config["condition"]` and `config["condition_name"]` are set
    for downstream logging, and `config["paths"]` is resolved to absolute paths.
    """
    cfg = load_base()
    conditions = _read(CONFIG_DIR / "conditions.yaml")

    key = condition.lower()
    if key in conditions.get("conditions", {}):
        entry = conditions["conditions"][key]
    elif key in conditions.get("control", {}):
        entry = conditions["control"][key]
    else:
        raise KeyError(
            f"unknown condition {condition!r}; expected one of "
            f"{sorted(conditions.get('conditions', {}))} or {sorted(conditions.get('control', {}))}"
        )

    cfg = _deep_merge(cfg, entry)
    cfg["condition"] = key
    cfg["condition_name"] = entry.get("name", key)
    cfg["paths"] = {k: (REPO_ROOT / v if not Path(v).is_absolute() else Path(v))
                    for k, v in cfg["paths"].items()}
    return cfg
