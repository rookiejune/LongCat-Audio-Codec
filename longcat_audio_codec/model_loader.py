from __future__ import annotations

from importlib.resources import files

from networks.semantic_codec.model_loader import load_decoder, load_encoder


def default_config_path(name: str) -> str:
    config_name = name if name.endswith(".yaml") else f"{name}.yaml"
    return str(files("longcat_audio_codec").joinpath("configs", config_name))


__all__ = [
    "default_config_path",
    "load_decoder",
    "load_encoder",
]

