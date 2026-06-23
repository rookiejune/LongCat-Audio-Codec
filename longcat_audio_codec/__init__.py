from .paths import (
    CACHE_ENV,
    CKPT_DIR_ENV,
    checkpoint_dir_from_env,
    resolve_checkpoint_path,
    resolve_resource_path,
)


def default_config_path(name: str) -> str:
    from .model_loader import default_config_path as _default_config_path

    return _default_config_path(name)


def load_decoder(config_path, device):
    from .model_loader import load_decoder as _load_decoder

    return _load_decoder(config_path, device)


def load_encoder(config_path, device):
    from .model_loader import load_encoder as _load_encoder

    return _load_encoder(config_path, device)


__all__ = [
    "CACHE_ENV",
    "CKPT_DIR_ENV",
    "checkpoint_dir_from_env",
    "default_config_path",
    "load_decoder",
    "load_encoder",
    "resolve_checkpoint_path",
    "resolve_resource_path",
]
