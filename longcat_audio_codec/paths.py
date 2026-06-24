from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

CKPT_DIR_ENV = "LONGCAT_AUDIO_CODEC_CKPT_DIR"
HF_HOME_ENV = "HF_HOME"
HF_REPO_CACHE = "models--meituan-longcat--LongCat-Audio-Codec"


def _hf_checkpoint_dir(hf_home: Path) -> Path:
    repo_dir = hf_home / "hub" / HF_REPO_CACHE
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        if revision:
            return repo_dir / "snapshots" / revision / "ckpts"

    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            return max(snapshots, key=lambda path: path.stat().st_mtime) / "ckpts"

    raise FileNotFoundError(
        f"{HF_HOME_ENV} is set to {hf_home}, but no cached LongCat-Audio-Codec "
        f"snapshot was found under {repo_dir}."
    )


def checkpoint_dir_from_env() -> Path | None:
    ckpt_dir = os.environ.get(CKPT_DIR_ENV)
    if ckpt_dir:
        return Path(ckpt_dir).expanduser()

    hf_home = os.environ.get(HF_HOME_ENV)
    if hf_home:
        return _hf_checkpoint_dir(Path(hf_home).expanduser())

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
