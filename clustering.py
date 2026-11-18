"""Core clustering helpers for the credit-card segmentation app.

Everything here is plain NumPy / pandas / scikit-learn (no Streamlit), so it can be
tested on its own. The pipeline mirrors the lab notebook:

    drop duplicates -> drop CUST_ID -> median imputation -> StandardScaler
    -> K-Means (scikit-learn) and K-Medoids (built-in implementation below)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

ID_COL = "CUST_ID"
K_MIN, K_MAX = 2, 10

# Features shown in the profile tables / "predict a customer" form
KEY_FEATURES = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
                "CASH_ADVANCE", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY",
                "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT"]


# --------------------------------------------------------------------------- data
@dataclass
class Prepared:
    raw: pd.DataFrame          # data after removing duplicate rows (still has CUST_ID)
    ids: pd.Series             # customer ids aligned with X
    X: pd.DataFrame            # imputed numeric features, original units
    X_scaled: np.ndarray       # standardised features (what the models see)
    scaler: StandardScaler
    features: list
    n_rows_in: int
    n_duplicates: int
    missing: pd.Series         # missing values per column *before* imputation


def preprocess(df: pd.DataFrame) -> Prepared:
    """Duplicates -> drop id -> median imputation -> standard scaling."""
    n_in = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    ids = df[ID_COL].astype(str) if ID_COL in df.columns else pd.Series(df.index.astype(str))
    X = df.drop(columns=[ID_COL], errors="ignore").select_dtypes("number")
    X = X.loc[:, X.notna().any()]
    if X.shape[1] < 2:
        raise ValueError("The file needs at least two numeric columns to cluster.")
    missing = X.isna().sum()
    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    scaler = StandardScaler().fit(X_imp)
    return Prepared(raw=df, ids=ids, X=X_imp, X_scaled=scaler.transform(X_imp), scaler=scaler,
                    features=X_imp.columns.tolist(), n_rows_in=n_in,
                    n_duplicates=n_in - len(df), missing=missing)


def skew_outlier_table(X: pd.DataFrame) -> pd.DataFrame:
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    pct = (((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).mean() * 100).round(1)
    return pd.DataFrame({"skewness": X.skew().round(2), "% IQR outliers": pct}) \
        .sort_values("skewness", ascending=False)


# ------------------------------------------------------------------------ K-Means
def fit_kmeans(Xs: np.ndarray, k: int, seed: int = 42, n_init: int = 20):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Xs)
    return km.labels_, km.cluster_centers_, float(km.inertia_)


# ---------------------------------------------------------------------- K-Medoids
def distance_matrix(Xs: np.ndarray) -> np.ndarray:
    """Full Euclidean distance matrix in float32 (~320 MB for 8,950 customers)."""
    D = euclidean_distances(Xs.astype(np.float32))
    np.fill_diagonal(D, 0.0)
    return D


def fit_kmedoids(D: np.ndarray, k: int, seed: int = 42, rows: np.ndarray | None = None,
                 max_iter: int = 300):
    """K-Medoids with k-medoids++ initialisation and alternating (Voronoi) updates.

    Same algorithm family as ``KMedoids(method="alternate", init="k-medoids++")`` from
    scikit-learn-extra, implemented here so the app has no compiled dependency.

    ``rows`` optionally restricts the fit to a subset of the points of ``D`` (used for
    the outlier-robustness test) without copying the big matrix.
    Returns (labels, medoid positions as indices into ``rows``, total cost).
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(D.shape[0]) if rows is None else np.asarray(rows)
    n = len(rows)

    # k-medoids++ initialisation (greedy, like k-means++)
    first = int(rng.integers(n))
    med = [first]
    closest = D[rows[first], rows].astype(np.float64)
    n_trials = 2 + int(np.log(k))
    for _ in range(1, k):
        p = closest ** 2
        cands = rng.choice(n, size=n_trials, p=p / p.sum()) if p.sum() > 0 else rng.integers(n, size=n_trials)
        best = min(cands, key=lambda c: np.minimum(closest, D[rows[c], rows]).sum())
        closest = np.minimum(closest, D[rows[best], rows])
        med.append(int(best))
    med = np.array(med)

    for _ in range(max_iter):
        labels = D[np.ix_(rows, rows[med])].argmin(axis=1)
        new = med.copy()
        for c in range(k):
            idx = np.flatnonzero(labels == c)
            if len(idx):
                new[c] = idx[D[np.ix_(rows[idx], rows[idx])].sum(axis=1).argmin()]
        if np.array_equal(new, med):
            break
        med = new
    labels = D[np.ix_(rows, rows[med])].argmin(axis=1)
    cost = float(D[rows, rows[med][labels]].sum())
    return labels, med, cost


try:  # optional: gives exactly the notebook's K-Medoids results when installed
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn_extra.cluster import KMedoids as _SkExtraKMedoids
    HAS_SKLEARN_EXTRA = True
except Exception:  # not installed / cannot be built on this platform
    _SkExtraKMedoids = None
    HAS_SKLEARN_EXTRA = False

KMEDOIDS_BACKEND = "scikit-learn-extra" if HAS_SKLEARN_EXTRA else "built-in"


def fit_kmedoids_auto(Xs: np.ndarray, D, k: int, seed: int = 42, rows: np.ndarray | None = None,
                      max_iter: int = 300):
    """K-Medoids using scikit-learn-extra if available, else the built-in version.

    ``D`` (distance matrix) is only needed - and only touched - by the built-in version,
    so it may be a callable returning the matrix (built lazily).
    Returns (labels, medoid indices into the fitted rows, total cost).
    """
    if HAS_SKLEARN_EXTRA:
        import warnings
        data = Xs if rows is None else Xs[rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _SkExtraKMedoids(n_clusters=k, method="alternate", init="k-medoids++",
                                 max_iter=max_iter, random_state=seed).fit(data)
        return m.labels_, np.asarray(m.medoid_indices_), float(m.inertia_)
    return fit_kmedoids(D() if callable(D) else D, k, seed, rows, max_iter)


# ------------------------------------------------------------------------ metrics
def evaluate(Xs: np.ndarray, labels: np.ndarray) -> dict:
    return {"Silhouette": float(silhouette_score(Xs, labels)),
            "Davies-Bouldin": float(davies_bouldin_score(Xs, labels)),
            "Calinski-Harabasz": float(calinski_harabasz_score(Xs, labels))}


def elbow_k(ks, values) -> int:
    """Point furthest below the straight line joining the first and last point
    (both axes normalised to 0-1)."""
    ks, v = np.asarray(ks, float), np.asarray(values, float)
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (v - v.min()) / (v.max() - v.min())
    return int(ks[np.argmax((1 - x) - y)])


def match_centres(a: np.ndarray, b: np.ndarray):
    """One-to-one matching of two sets of centres by Euclidean distance."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    i, j = linear_sum_assignment(d)
    return i, j, d[i, j]


def extreme_mask(Xs: np.ndarray, z: float = 5.0) -> np.ndarray:
    return (np.abs(Xs) > z).any(axis=1)


# ------------------------------------------------------------------ cluster names
def name_clusters(z_centres: np.ndarray, features: list) -> list:
    """Readable segment names from cluster centres in standardised (z) units.

    Rule-of-thumb labels for credit-card behaviour; they fall back to 'Segment i'
    if the expected columns are missing (e.g. an uploaded file with other columns).
    """
    f = {name: i for i, name in enumerate(features)}
    need = {"CASH_ADVANCE", "CASH_ADVANCE_FREQUENCY", "PURCHASES", "PURCHASES_FREQUENCY"}
    if not need <= set(f):
        return [f"Segment {i}" for i in range(len(z_centres))]
    names = []
    for i, z in enumerate(z_centres):
        if z[f["CASH_ADVANCE"]] > 0.5 and z[f["CASH_ADVANCE_FREQUENCY"]] > 0.5:
            names.append("Cash-advance users")
        elif z[f["PURCHASES"]] > 1.0:
            names.append("High-spending buyers")
        elif z[f["PURCHASES_FREQUENCY"]] > 0.3:
            names.append("Regular purchasers")
        elif z[f["PURCHASES_FREQUENCY"]] < -0.3:
            names.append("Low-activity customers")
        else:
            names.append("Moderate users")
    counts = {n: names.count(n) for n in set(names)}
    seen = {}
    out = []
    for i, n in enumerate(names):
        if counts[n] > 1:
            seen[n] = seen.get(n, 0) + 1
            n = f"{n} ({seen[n]})"
        out.append(n)
    return out
