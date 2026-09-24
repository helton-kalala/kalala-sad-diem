from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_gower_numeric(
    x: pd.DataFrame,
    feature_weights: dict[str, float],
    asymmetric_binary: set[str] | None = None,
) -> np.ndarray:
    """Distance de Gower ponderee, avec gestion paire par paire des manquants.

    Les variables doivent deja etre ramenees sur [0, 1]. Pour les binaires
    asymetriques, une double absence n'apporte aucune similarite et est retiree
    du denominateur de la paire.
    """
    asymmetric_binary = asymmetric_binary or set()
    arr = x.to_numpy(dtype=float)
    n, p = arr.shape
    weights = np.array([feature_weights[c] for c in x.columns], dtype=np.float64)
    numerator = np.zeros((n, n), dtype=np.float64)
    denominator = np.zeros((n, n), dtype=np.float64)
    for j, col in enumerate(x.columns):
        v = arr[:, j]
        valid = np.isfinite(v)
        pair_valid = valid[:, None] & valid[None, :]
        if col in asymmetric_binary:
            pair_valid &= ~((v[:, None] == 0) & (v[None, :] == 0))
        delta = np.abs(v[:, None] - v[None, :])
        delta[~pair_valid] = 0.0
        numerator += weights[j] * delta
        denominator += weights[j] * pair_valid
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32)

