from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional


def _make_key(raw_key: str) -> str:
    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()


def get_cache_path(raw_key: str, cache_dir: str = "data/raw", ext: str = "pkl") -> Path:
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    key = _make_key(raw_key)
    return cache_dir_path / f"{key}.{ext}"


def read_cache(raw_key: str, cache_dir: str = "data/raw") -> Optional[Any]:
    path = get_cache_path(raw_key, cache_dir=cache_dir)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def write_cache(raw_key: str, data: Any, cache_dir: str = "data/raw") -> Path:
    path = get_cache_path(raw_key, cache_dir=cache_dir)
    with path.open("wb") as f:
        pickle.dump(data, f)
    return path
