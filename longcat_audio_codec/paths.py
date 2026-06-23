from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

CACHE_ENV = "LONGCAT_AUDIO_CODEC_CACHE"
CKPT_DIR_ENV = "LONGCAT_AUDIO_CODEC_CKPT_DIR"


def checkpoint_dir_from_env() -> Path | None:
    ckpt_dir = os.environ.get(CKPT_DIR_ENV)
    if ckpt_dir:
        return Path(ckpt_dir).expanduser()

    cache_dir = os.environ.get(CACHE_ENV)
    if cache_dir:
        return Path(cache_dir).expanduser() / "ckpts"

    return None


def resolve_checkpoint_path(path: str | os.PathLike[str]) -> str:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return str(raw)

    ckpt_dir = checkpoint_dir_from_env()
    if ckpt_dir is not None:
        return str(ckpt_dir / raw.name if raw.parts[:1] == ("ckpts",) else ckpt_dir / raw)

    return str(raw)


def resolve_resource_path(path: str | os.PathLike[str]) -> str:
    raw = Path(path).expanduser()
    if raw.is_absolute() or raw.exists():
        return str(raw)

    if raw.parts[:1] == ("semantic_tokenizer_general",):
        relative = Path(*raw.parts[1:])
        return str(files("semantic_tokenizer_general").joinpath(*relative.parts))

    return str(raw)
