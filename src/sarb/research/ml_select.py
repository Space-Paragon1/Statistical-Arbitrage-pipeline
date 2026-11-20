"""Unsupervised pair prefiltering (OPTICS / DBSCAN), after Sarmento & Horta (2020).

Each ticker's TRAIN return series is standardised *over time*, so the squared
Euclidean distance between two tickers is ``2 T (1 - rho)``: clustering in this
space groups tickers by return correlation. An optional PCA step projects the
series onto the leading common factors first, which denoises the distances.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "scikit-learn is required for ML pair selection. "
            "Install with: pip install 'stat-arb-research-pipeline[ml]'"
        ) from err


@dataclass(frozen=True)
class ClusterPairScore:
    y: str
    x: str
    cluster_id: int
    distance: float


def ticker_features(returns: pd.DataFrame, n_components: int | None = None) -> np.ndarray:
    """(n_tickers, n_features) matrix: per-ticker standardised returns, optionally PCA'd."""
    r = returns.dropna()
    sd = r.std(ddof=0).replace(0.0, np.nan)
    Z = ((r - r.mean()) / sd).fillna(0.0).to_numpy().T / np.sqrt(len(r))
    if n_components is not None and n_components < min(Z.shape):
        _require_sklearn()
        from sklearn.decomposition import PCA

        # Loadings of each ticker on the leading factors of the return panel.
        Z = PCA(n_components=n_components).fit(Z.T).components_.T * np.sqrt(len(r))
    return Z


def _pairs_from_labels(
    X: np.ndarray, tickers: list[str], labels: np.ndarray
) -> list[ClusterPairScore]:
    pairs: list[ClusterPairScore] = []
    for cid in sorted(set(labels) - {-1}):
        members = np.flatnonzero(labels == cid)
        for i, j in itertools.combinations(members, 2):
            pairs.append(
                ClusterPairScore(
                    y=tickers[i],
                    x=tickers[j],
                    cluster_id=int(cid),
                    distance=float(np.linalg.norm(X[i] - X[j])),
                )
            )
    pairs.sort(key=lambda p: p.distance)
    return pairs


def cluster_pairs_optics(
    returns: pd.DataFrame,
    min_samples: int = 2,
    max_eps: float = np.inf,
    n_components: int | None = None,
) -> list[ClusterPairScore]:
    """Cluster tickers with OPTICS; return within-cluster pairs, closest first."""
    _require_sklearn()
    from sklearn.cluster import OPTICS

    X = ticker_features(returns, n_components)
    labels = OPTICS(min_samples=min_samples, max_eps=max_eps).fit_predict(X)
    return _pairs_from_labels(X, list(returns.columns), labels)


def cluster_pairs_dbscan(
    returns: pd.DataFrame,
    eps: float = 1.0,
    min_samples: int = 2,
    n_components: int | None = None,
) -> list[ClusterPairScore]:
    """Same interface using DBSCAN. ``eps`` is in correlation-distance units
    (``sqrt(2 (1 - rho))``), so ``eps=1.0`` links tickers with rho >~ 0.5."""
    _require_sklearn()
    from sklearn.cluster import DBSCAN

    X = ticker_features(returns, n_components)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return _pairs_from_labels(X, list(returns.columns), labels)


def ml_prefilter_pairs(
    prices: pd.DataFrame,
    tickers: list[str],
    train_idx: pd.Index,
    method: str = "optics",
    max_pairs: int = 300,
    **kwargs,
) -> list[tuple[str, str]]:
    """Drop-in replacement for the correlation prefilter in ``scan_pairs``."""
    train_ret = prices.loc[train_idx, tickers].pct_change().dropna()
    if method == "optics":
        scored = cluster_pairs_optics(train_ret, **kwargs)
    elif method == "dbscan":
        scored = cluster_pairs_dbscan(train_ret, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    if not scored:
        # Clustering found no structure: fall back to every combination.
        return list(itertools.combinations(tickers, 2))[:max_pairs]
    return [(p.y, p.x) for p in scored[:max_pairs]]
