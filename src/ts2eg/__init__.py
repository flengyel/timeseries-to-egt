"""
ts2eg: Time-Series → Evolutionary Games
"""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("ts2eg")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Re-export public API from core
from .core import (
    helmert_Q, projectors, static_game_from_series, var_information_sharing_game,
    estimate_A_from_series, find_ESS, nmf_on_X,
)

# Extensions (optional add-ons)
try:
    from .extensions import (
        weighted_projectors, var_information_sharing_game_seasonal,
        estimate_A_from_series_weighted, iaaft, iaaft_matrix, surrogate_ess_frequency,
    )
except Exception:  # extensions may be absent in minimal setups
    pass

__all__ = [
    "__version__",
    "helmert_Q", "projectors", "static_game_from_series", "var_information_sharing_game",
    "estimate_A_from_series", "find_ESS", "nmf_on_X",
    "weighted_projectors", "var_information_sharing_game_seasonal",
    "estimate_A_from_series_weighted", "iaaft", "iaaft_matrix", "surrogate_ess_frequency",
]