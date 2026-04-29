"""Calibration / sharpness metrics.

Lazy import of `uncertainty-toolbox` so it stays a soft dependency.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def calibration_metrics(
    y_pred: jnp.ndarray,
    uncertainty: jnp.ndarray,
    y_true: jnp.ndarray,
    *,
    num_bins: int = 100,
    resolution: int = 99,
) -> dict[str, Any]:
    """Compute accuracy / sharpness / calibration / scoring metrics.

    Lazy-imports `uncertainty-toolbox`; raises a clear error with install
    instructions if missing.
    """
    try:
        import uncertainty_toolbox as utb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "calibration_metrics requires the `uncertainty-toolbox` package. "
            "Install with `pip install jax-ldt[uq-metrics]`."
        ) from exc

    y_pred = np.asarray(y_pred).reshape(-1)
    uncertainty = np.asarray(uncertainty).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    metrics: dict[str, Any] = {}
    metrics["accuracy"] = utb.metrics_accuracy.get_all_accuracy_metrics(y_pred, y_true)
    metrics["sharpness"] = {
        "rms_unc": float(np.sqrt(np.mean(uncertainty**2))),
        "mean_unc": float(np.mean(uncertainty)),
        "median_unc": float(np.median(uncertainty)),
        "75_percentile_unc": float(np.quantile(uncertainty, 0.75)),
        "99_percentile_unc": float(np.quantile(uncertainty, 0.99)),
    }
    metrics["calibration"] = utb.metrics_calibration.get_all_calibration_metrics(
        y_pred, uncertainty, y_true, num_bins=num_bins, resolution=resolution
    )
    metrics["scoring_rule"] = utb.metrics_scoring_rule.get_all_scoring_rule_metrics(
        y_pred, uncertainty, y_true
    )
    return metrics
