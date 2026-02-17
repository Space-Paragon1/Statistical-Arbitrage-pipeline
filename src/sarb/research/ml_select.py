from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import OPTICS, DBSCAN
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError(
        "scikit-learn is required for ML pair selection. "
        "Install with: pip install 'stat-arb-research-pipeline[ml]'"
    )


@dataclass(frozen=True)
class ClusterPairScore:
    y: str
    x: str
    cluster_id: int
    distance: float


def cluster_pairs_optics(
    returns: pd.DataFrame,
    min_samples: int = 2,
    max_eps: float = np.inf,
) -> list[ClusterPairScore]:
    """
    Cluster tickers by normalized return similarity using OPTICS.
    Returns scored pairs within each cluster, ranked by distance.
    """
    X = StandardScaler().fit_transform(returns.T.values)
    tickers = list(returns.columns)

    model = OPTICS(min_samples=min_samples, max_eps=max_eps, metric="euclidean")
    labels = model.fit_predict(X)

    pairs: list[ClusterPairScore] = []
    unique_labels = set(labels)
    unique_labels.discard(-1)

    for cid in unique_labels:
        members = [t for t, lab in zip(tickers, labels) if lab == cid]
        if len(members) < 2:
            continue
        member_idx = [tickers.index(m) for m in members]
        for i_pos, i_tkr in enumerate(members):
            for j_pos, j_tkr in enumerate(members):
                if i_pos >= j_pos:
                    continue
                dist = float(np.linalg.norm(X[member_idx[i_pos]] - X[member_idx[j_pos]]))
                pairs.append(ClusterPairScore(
                    y=i_tkr, x=j_tkr, cluster_id=int(cid), distance=dist,
                ))

    pairs.sort(key=lambda p: p.distance)
    return pairs


def cluster_pairs_dbscan(
    returns: pd.DataFrame,
    eps: float = 1.5,
    min_samples: int = 2,
) -> list[ClusterPairScore]:
    """Same interface using DBSCAN instead of OPTICS."""
    X = StandardScaler().fit_transform(returns.T.values)
    tickers = list(returns.columns)

    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(X)

    pairs: list[ClusterPairScore] = []
    unique_labels = set(labels)
    unique_labels.discard(-1)

    for cid in unique_labels:
        members = [t for t, lab in zip(tickers, labels) if lab == cid]
        if len(members) < 2:
            continue
        member_idx = [tickers.index(m) for m in members]
        for i_pos, i_tkr in enumerate(members):
            for j_pos, j_tkr in enumerate(members):
                if i_pos >= j_pos:
                    continue
                dist = float(np.linalg.norm(X[member_idx[i_pos]] - X[member_idx[j_pos]]))
                pairs.append(ClusterPairScore(
                    y=i_tkr, x=j_tkr, cluster_id=int(cid), distance=dist,
                ))

    pairs.sort(key=lambda p: p.distance)
    return pairs


def ml_prefilter_pairs(
    prices: pd.DataFrame,
    tickers: list[str],
    train_idx: pd.Index,
    method: str = "optics",
    max_pairs: int = 300,
    **kwargs,
) -> list[tuple[str, str]]:
    """
    Drop-in replacement for the correlation prefilter in scan_pairs.
    Returns list of (y, x) ticker pairs ranked by cluster distance.
    """
    train_ret = prices.loc[train_idx, tickers].pct_change().dropna()

    if method == "optics":
        scored = cluster_pairs_optics(train_ret, **kwargs)
    elif method == "dbscan":
        scored = cluster_pairs_dbscan(train_ret, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    if not scored:
        # Fallback: return all combinations if clustering finds nothing
        return list(itertools.combinations(tickers, 2))[:max_pairs]

    return [(p.y, p.x) for p in scored[:max_pairs]]
