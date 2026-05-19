"""
src/regime.py — Market regime identification via t-SNE + HDBSCAN.

fit_regimes(feature_matrix, perplexity, n_clusters) -> RegimeModel
    Discovers regimes in the training feature matrix.
    t-SNE is used for structure discovery only; a KNN trained on the original
    feature space enables fast inference without re-running t-SNE.

classify_regime(feature_vector, regime_model) -> int
    Inference-time regime classification. t-SNE is NOT called here.
    Returns -1 only on unexpected failure; callers should fall back to the
    non-regime model when that happens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

try:
    import hdbscan as _hdbscan
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False


@dataclass
class RegimeModel:
    tsne_embedding:  np.ndarray          # (n, 2) — visualization only
    labels:          np.ndarray          # (n,) int — regime per training row
    cluster_centers: np.ndarray          # (n_regimes, feature_dim) mean feature
    n_regimes:       int
    knn:             KNeighborsClassifier  # trained on original features for inference
    feature_matrix:  np.ndarray          # (n, feature_dim) training features


def _reassign_noise(labels: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    """Reassign HDBSCAN noise points (-1) to the nearest non-noise point's cluster."""
    labels = labels.copy()
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels
    non_noise_idx = np.where(~noise_mask)[0]
    if len(non_noise_idx) == 0:
        labels[:] = 0
        return labels
    for i in np.where(noise_mask)[0]:
        dists = np.linalg.norm(embedding[non_noise_idx] - embedding[i], axis=1)
        labels[i] = labels[non_noise_idx[dists.argmin()]]
    return labels


def fit_regimes(
    feature_matrix: np.ndarray,
    perplexity: int = 30,
    n_clusters: int = 4,
) -> RegimeModel:
    """
    Discover market regimes.

    1. t-SNE (n_components=2, perplexity=perplexity) embeds feature_matrix
    2. HDBSCAN(min_cluster_size=20) clusters the 2D embedding
    3. Noise points (-1) reassigned to nearest non-noise cluster
    4. Labels remapped to 0-indexed integers [0, n_regimes)
    5. KNN (k=5) trained on original feature space for fast inference

    n_clusters is advisory — HDBSCAN decides the actual count.
    """
    if not _HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is not installed. Run: pip install hdbscan")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embedding = tsne.fit_transform(feature_matrix.astype(np.float32))

    min_cs = max(10, len(feature_matrix) // max(n_clusters * 8, 1))
    clusterer = _hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=5)
    raw_labels = clusterer.fit_predict(embedding)

    labels = _reassign_noise(raw_labels, embedding)

    # Remap to 0-indexed
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=int)
    n_regimes = len(unique)

    cluster_centers = np.array([
        feature_matrix[labels == k].mean(axis=0)
        for k in range(n_regimes)
    ])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(feature_matrix, labels)

    return RegimeModel(
        tsne_embedding=embedding,
        labels=labels,
        cluster_centers=cluster_centers,
        n_regimes=n_regimes,
        knn=knn,
        feature_matrix=feature_matrix,
    )


def classify_regime(feature_vector: np.ndarray, regime_model: RegimeModel) -> int:
    """
    Classify a feature vector into a regime using KNN on the original feature space.
    t-SNE is NOT called. Returns -1 only on failure (caller should fall back to
    the non-regime model).
    """
    try:
        fv = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        return int(regime_model.knn.predict(fv)[0])
    except Exception:
        return -1


def between_regime_sharpe_variance(
    labels: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Compute variance of per-regime Sharpe ratios.
    Used by the --regime-analysis sweep to select the best (perplexity, n_clusters).
    Higher variance = regimes are more distinct in predictability.
    """
    sharpes = []
    for k in sorted(set(labels)):
        mask = labels == k
        if mask.sum() < 10:
            continue
        rets = np.sign(preds[mask]) * targets[mask]
        if rets.std() == 0:
            sharpes.append(0.0)
        else:
            sharpes.append((rets.mean() / rets.std()) * np.sqrt(252))
    return float(np.var(sharpes)) if len(sharpes) > 1 else 0.0
