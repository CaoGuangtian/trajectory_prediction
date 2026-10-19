"""PatchTST trajectory forecasting package."""

from .config import PatchTSTConfig, load_patchtst_config

__all__ = ["PatchTSTConfig", "PatchTSTSource", "load_patchtst_config"]


def __getattr__(name: str) -> object:
    if name == "PatchTSTSource":
        from .data import PatchTSTSource

        return PatchTSTSource
    raise AttributeError(name)

