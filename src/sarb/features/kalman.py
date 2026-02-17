from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KalmanConfig:
    delta: float = 1e-4       # state transition covariance scaling
    ve: float = 1e-3           # observation noise variance
    initial_beta: float = 1.0
    initial_alpha: float = 0.0


@dataclass(frozen=True)
class KalmanResult:
    alpha: pd.Series
    beta: pd.Series
    spread: pd.Series
    measurement_error: pd.Series


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    config: KalmanConfig = KalmanConfig(),
) -> KalmanResult:
    """
    Online Kalman filter estimating time-varying [alpha, beta] where y = alpha + beta*x + noise.

    State: theta_t = [alpha_t, beta_t]'
    Observation: y_t = H_t @ theta_t + v_t,  where H_t = [1, x_t]
    Transition: theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q)
    """
    n = len(y)
    theta = np.array([config.initial_alpha, config.initial_beta], dtype=np.float64)
    P = np.eye(2) * 1.0
    Q = np.eye(2) * config.delta
    R = config.ve

    y_vals = y.values.astype(np.float64)
    x_vals = x.values.astype(np.float64)

    alphas = np.empty(n)
    betas = np.empty(n)
    spreads = np.empty(n)
    errors = np.empty(n)

    for t in range(n):
        # Predict (random walk transition)
        P_pred = P + Q

        # Observation model: y_t = [1, x_t] @ theta
        H = np.array([1.0, x_vals[t]])
        y_pred = H @ theta
        e = y_vals[t] - y_pred

        # Innovation covariance
        S = H @ P_pred @ H + R

        # Kalman gain
        K = P_pred @ H / S

        # Update
        theta = theta + K * e
        P = P_pred - np.outer(K, H) @ P_pred

        # Regularize to maintain positive-definiteness
        P = (P + P.T) / 2.0 + np.eye(2) * 1e-8

        alphas[t] = theta[0]
        betas[t] = theta[1]
        spreads[t] = e
        errors[t] = e

    idx = y.index
    return KalmanResult(
        alpha=pd.Series(alphas, index=idx, name="kalman_alpha"),
        beta=pd.Series(betas, index=idx, name="kalman_beta"),
        spread=pd.Series(spreads, index=idx, name="kalman_spread"),
        measurement_error=pd.Series(errors, index=idx, name="kalman_error"),
    )


def fit_hedge_ratio_kalman(
    y: pd.Series,
    x: pd.Series,
    config: KalmanConfig = KalmanConfig(),
) -> tuple[float, float]:
    """Drop-in replacement for fit_hedge_ratio. Returns (alpha, beta) as the last Kalman state."""
    result = kalman_hedge_ratio(y, x, config)
    return float(result.alpha.iloc[-1]), float(result.beta.iloc[-1])
