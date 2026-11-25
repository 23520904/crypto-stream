from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(cfg: dict, key: str, *parts: str) -> Path:
    return (ROOT / cfg["paths"][key]).joinpath(*parts)


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
