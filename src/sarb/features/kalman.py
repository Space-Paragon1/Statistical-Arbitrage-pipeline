"""Online Kalman filter for a time-varying hedge ratio.

Model (random-walk state, linear-Gaussian observation)::

    theta_t = theta_{t-1} + w_t,             w_t ~ N(0, Q),  theta = [alpha, beta]'
    y_t     = [1, x_t] @ theta_t + v_t,      v_t ~ N(0, R)

Timing matters for lookahead:

* ``alpha``/``beta`` in :class:`KalmanResult` are the *filtered* (posterior) state
  ``theta_{t|t}``, which has already seen ``y_t``.
* ``spread`` is the one-step-ahead innovation ``e_t = y_t - H_t theta_{t|t-1}``,
  computed with the *prior* state, so it contains no information from ``y_t``
  in the parameters. ``innovation_var`` is its predicted variance ``S_t`` and
  ``zscore = e_t / sqrt(S_t)``.

When trading on day ``t`` use ``beta.shift(1)`` (the prior), as
:func:`sarb.features.spread.compute_pair_signals` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KalmanConfig:
    delta: float = 1e-4  # state transition covariance scaling (Q = delta * I)
    ve: float = 1e-3  # observation noise variance R
    initial_beta: float = 1.0
    initial_alpha: float = 0.0
    initial_state_var: float = 1.0  # P_0 = initial_state_var * I


@dataclass(frozen=True)
class KalmanResult:
    alpha: pd.Series
    beta: pd.Series
    spread: pd.Series
    measurement_error: pd.Series
    innovation_var: pd.Series

    @property
    def zscore(self) -> pd.Series:
        """Standardised innovation ``e_t / sqrt(S_t)``."""
        return (self.spread / np.sqrt(self.innovation_var)).rename("kalman_z")


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    config: KalmanConfig | None = None,
) -> KalmanResult:
    """Run the filter forward once over ``(y, x)``. O(n), strictly causal."""
    config = config or KalmanConfig()
    n = len(y)
    theta = np.array([config.initial_alpha, config.initial_beta], dtype=np.float64)
    P = np.eye(2) * config.initial_state_var
    Q = np.eye(2) * config.delta
    R = config.ve
    eye_reg = np.eye(2) * 1e-12

    y_vals = np.asarray(y, dtype=np.float64)
    x_vals = np.asarray(x, dtype=np.float64)

    alphas = np.empty(n)
    betas = np.empty(n)
    innov = np.empty(n)
    innov_var = np.empty(n)

    for t in range(n):
        P_pred = P + Q
        H = np.array([1.0, x_vals[t]])
        e = y_vals[t] - H @ theta
        S = float(H @ P_pred @ H + R)
        K = P_pred @ H / S
        theta = theta + K * e
        # Joseph-stabilised covariance update keeps P symmetric positive-definite.
        I_KH = np.eye(2) - np.outer(K, H)
        P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * R + eye_reg

        alphas[t] = theta[0]
        betas[t] = theta[1]
        innov[t] = e
        innov_var[t] = S

    idx = y.index
    spread = pd.Series(innov, index=idx, name="kalman_spread")
    return KalmanResult(
        alpha=pd.Series(alphas, index=idx, name="kalman_alpha"),
        beta=pd.Series(betas, index=idx, name="kalman_beta"),
        spread=spread,
        measurement_error=spread.rename("kalman_error"),
        innovation_var=pd.Series(innov_var, index=idx, name="kalman_S"),
    )


def fit_hedge_ratio_kalman(
    y: pd.Series,
    x: pd.Series,
    config: KalmanConfig | None = None,
) -> tuple[float, float]:
    """Drop-in replacement for ``fit_hedge_ratio``: the final filtered state ``(alpha, beta)``."""
    result = kalman_hedge_ratio(y, x, config)
    return float(result.alpha.iloc[-1]), float(result.beta.iloc[-1])
