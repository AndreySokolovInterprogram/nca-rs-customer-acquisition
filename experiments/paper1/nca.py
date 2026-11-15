"""Методы NCA-RS для статьи №1: профили АВР, кластеризация, TAS, сопоставление, меры качества.

Разделы статьи: 3.7 профили АВР, 3.8 кластеризация, 3.9 сопоставление, 3.10 TAS, 3.11 меры качества M1, M2.
Все векторы L2-нормированы; угловое расстояние d = arccos<x, y>.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.cluster import SpectralBiclustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from sklearn.mixture import BayesianGaussianMixture

K_GRID = [5, 8, 10, 15, 20, 25, 30, 40, 50]
GMM_PCA_DIM = 32


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


# ---------------------------------------------------------------- профили (АВР, раздел 3.7, формула acr)
def acr_profiles(e_posts, e_subs, e_demo, has_subs, q_demo, weights):
    """v_j по АВР: λ*_k = λ_k^0 q_k; нормировка по наблюдаемым компонентам; L2-нормировка.

    e_* — (m, d); has_subs — bool (m,); q_demo = r_j / 8 (0 — демографии нет).
    q_posts = q_subs = 1 (посты есть у всех; подписки — либо есть, либо компонент не наблюдается).
    """
    lp, ls, ld = weights
    w_p = np.full(len(e_posts), lp, dtype=np.float64)
    w_s = np.where(has_subs, ls, 0.0)
    w_d = ld * q_demo
    tot = w_p + w_s + w_d
    v = (w_p / tot)[:, None] * e_posts + (w_s / tot)[:, None] * np.nan_to_num(e_subs) \
        + (w_d / tot)[:, None] * np.nan_to_num(e_demo)
    return l2n(v.astype(np.float32))


# ---------------------------------------------------------------- расстояния и внутренние меры
def angular_dist(X: np.ndarray) -> np.ndarray:
    G = np.clip(X @ X.T, -1.0, 1.0)
    D = np.arccos(G).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    return D


def silhouette_precomputed(D: np.ndarray, labels: np.ndarray, idx: np.ndarray | None = None,
                           per_point: bool = False):
    """Силуэт по матрице расстояний (быстрый, матричный). idx — индексы подвыборки;
    per_point=True — вернуть значения s_i (для бутстреп-интервала при фиксированном разбиении)."""
    if idx is not None:
        D = D[np.ix_(idx, idx)]
        labels = labels[idx]
    uniq, lab = np.unique(labels, return_inverse=True)
    k = len(uniq)
    if k < 2:
        return float("nan")
    oh = np.zeros((len(lab), k), dtype=np.float32)
    oh[np.arange(len(lab)), lab] = 1.0
    cnt = oh.sum(0)
    S = D @ oh  # сумма расстояний до каждого кластера
    own_cnt = cnt[lab] - 1
    a = S[np.arange(len(lab)), lab] / np.clip(own_cnt, 1, None)
    S_other = S / cnt
    S_other[np.arange(len(lab)), lab] = np.inf
    b = S_other.min(1)
    s = (b - a) / np.maximum(a, b)
    s[own_cnt == 0] = 0.0
    return s if per_point else float(np.mean(s))


def ch_terms(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Поточечные вклады в межгрупповую (||μ_c(i) − μ||²) и внутригрупповую (||x_i − μ_c(i)||²) суммы:
    CH = (Σb_i / Σw_i) · (n − k)/(k − 1)."""
    k = labels.max() + 1
    mu = np.zeros((k, X.shape[1]), dtype=np.float64)
    np.add.at(mu, labels, X)
    mu /= np.bincount(labels, minlength=k)[:, None]
    m = X.mean(0)
    b = ((mu - m) ** 2).sum(1)[labels]
    w = ((X - mu[labels]) ** 2).sum(1)
    return b, w, k


def hopkins(X: np.ndarray, rng: np.random.Generator, m: int | None = None) -> float:
    """Статистика Хопкинса: H = Σu / (Σu + Σw); u — от равномерных точек в ограничивающем
    параллелепипеде данных до ближайшей точки данных, w — от выбранных точек данных до
    ближайшей другой точки данных. H > 0,5 — кластерная тенденция."""
    n, d = X.shape
    m = m or min(500, max(10, n // 10))
    lo, hi = X.min(0), X.max(0)
    U = rng.uniform(lo, hi, size=(m, d)).astype(np.float32)
    idx = rng.choice(n, size=m, replace=False)
    sq = (X ** 2).sum(1)

    def nn_dist(Q, exclude=None):
        d2 = (Q ** 2).sum(1)[:, None] + sq[None, :] - 2 * Q @ X.T
        if exclude is not None:
            d2[np.arange(len(Q)), exclude] = np.inf
        return np.sqrt(np.clip(d2.min(1), 0, None))

    u = nn_dist(U)
    w = nn_dist(X[idx], exclude=idx)
    return float(u.sum() / (u.sum() + w.sum()))


def effective_dim(X: np.ndarray, level: float = 0.95) -> int:
    Xc = X - X.mean(0)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    cum = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(cum, level) + 1)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    num = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    return float(num / (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")))


# ---------------------------------------------------------------- кластеризация
def hier_linkage(X: np.ndarray, method: str, D_ang: np.ndarray | None = None) -> np.ndarray:
    """Ward — по евклидову расстоянию (на сфере монотонно связано с угловым);
    average / complete — по угловому расстоянию."""
    if method == "ward":
        G = np.clip(X @ X.T, -1, 1)
        E = np.sqrt(np.clip(2 - 2 * G, 0, None)).astype(np.float64)
        np.fill_diagonal(E, 0)
        return linkage(squareform(E, checks=False), method="ward")
    D = D_ang if D_ang is not None else angular_dist(X)
    return linkage(squareform(D.astype(np.float64), checks=False), method=method)


def cut(Z: np.ndarray, k: int) -> np.ndarray:
    return fcluster(Z, k, criterion="maxclust") - 1


def gmm_fit(X: np.ndarray, k: int, cov: str, seed: int, pca: PCA | None = None):
    Xp = pca.transform(X) if pca is not None else X
    g = BayesianGaussianMixture(n_components=k, covariance_type=cov, random_state=seed,
                                max_iter=500, n_init=1, init_params="k-means++",
                                weight_concentration_prior_type="dirichlet_process")
    g.fit(Xp)
    P = g.predict_proba(Xp)
    return g, P, Xp


def gmm_bic_icl(g, Xp: np.ndarray, P: np.ndarray) -> tuple[float, float, int]:
    """BIC = −2 lnL + p ln n (меньше — лучше); ICL = BIC + 2·EN, EN = −Σ t ln t.
    Число параметров считается по «эффективным» компонентам (вес > 1/n)."""
    n, d = Xp.shape
    eff = int((g.weights_ > 1.0 / n).sum())
    cov_p = {"full": d * (d + 1) / 2, "diag": d, "spherical": 1, "tied": 0}[g.covariance_type]
    p = eff * (d + cov_p) + (eff - 1) + (d * (d + 1) / 2 if g.covariance_type == "tied" else 0)
    ll = g.score(Xp) * n
    bic = -2 * ll + p * np.log(n)
    en = -np.sum(P * np.log(np.clip(P, 1e-300, None)))
    return float(bic), float(bic + 2 * en), eff


def centroids(X: np.ndarray, labels: np.ndarray, k: int, P: np.ndarray | None = None) -> np.ndarray:
    if P is not None:  # взвешенные центроиды байесовской смеси
        mu = (P.T @ X) / np.clip(P.sum(0)[:, None], 1e-12, None)
    else:
        oh = np.zeros((len(labels), k), dtype=np.float32)
        oh[np.arange(len(labels)), labels] = 1
        mu = (oh.T @ X) / np.clip(oh.sum(0)[:, None], 1, None)
    return l2n(mu)


def relabel(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels, return_inverse=True)[1]


def jaccard_stability(labels_orig: np.ndarray, boot_fn, n: int, B: int, rng) -> tuple[np.ndarray, float]:
    """Бутстреп-стабильность Хеннига: для каждого исходного кластера — среднее по повторам
    максимального Жаккара с кластерами разбиения бутстреп-выборки (на её точках);
    плюс средний ARI между исходным и бутстреп-разбиением на тех же точках."""
    k = labels_orig.max() + 1
    J = np.zeros((B, k))
    aris = []
    for b in range(B):
        idx = np.unique(rng.choice(n, size=n, replace=True))
        lb = boot_fn(idx)
        lo = labels_orig[idx]
        aris.append(adjusted_rand_score(lo, lb))
        for a in range(k):
            A = lo == a
            if A.sum() == 0:
                J[b, a] = np.nan
                continue
            best = 0.0
            for c in np.unique(lb[A]):
                C = lb == c
                inter = np.sum(A & C)
                best = max(best, inter / np.sum(A | C))
            J[b, a] = best
    return np.nanmean(J, axis=0), float(np.mean(aris))


# ---------------------------------------------------------------- TAS (раздел 3.10, формула tas)
def cluster_max(G: np.ndarray, labels_u: np.ndarray, k_u: int) -> np.ndarray:
    """M[i, b] = max_{j ∈ T_b} G[i, j]."""
    order = np.argsort(labels_u, kind="stable")
    ls = labels_u[order]
    starts = np.searchsorted(ls, np.arange(k_u))
    present = np.isin(np.arange(k_u), ls)
    out = np.full((G.shape[0], k_u), -np.inf, dtype=np.float32)
    red = np.maximum.reduceat(G[:, order], starts[present], axis=1)
    out[:, present] = red
    return out


def tas_matrix(G: np.ndarray, labels_c: np.ndarray, k_c: int, labels_u: np.ndarray, k_u: int) -> np.ndarray:
    M = cluster_max(G, labels_u, k_u)
    oh = np.zeros((len(labels_c), k_c), dtype=np.float32)
    oh[np.arange(len(labels_c)), labels_c] = 1
    return (oh.T @ M) / np.clip(oh.sum(0)[:, None], 1, None)


# ---------------------------------------------------------------- сопоставление (раздел 3.9, формула match)
MATCH_METHODS = ["threshold", "greedy", "ot", "bicluster", "ipfp"]


def _top_n(weight: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    w = np.where(mask, weight, -np.inf).ravel()
    n = int(min(n, np.isfinite(w).sum()))
    sel = np.zeros(w.shape, dtype=bool)
    if n > 0:
        sel[np.argsort(-w, kind="stable")[:n]] = True
    return sel.reshape(weight.shape)


def match(method: str, S: np.ndarray, TAS: np.ndarray, size_c: np.ndarray, size_u: np.ndarray,
          alpha: float, rho_max: float, valid_c: np.ndarray, seed: int = 42,
          ot_eps: float = 0.05, ipfp_T: float = 0.05, bic_r: int = 4) -> np.ndarray:
    """Возвращает булеву матрицу M (k_C × k_U), |M| ≤ ρ_max k_C k_U.
    Кластеры контента из одного поста (valid_c = False) не сопоставляются."""
    kc, ku = S.shape
    n_max = int(np.floor(rho_max * kc * ku))
    valid = np.repeat(valid_c[:, None], ku, axis=1)
    tau = np.quantile(S[valid], alpha) if valid.any() else np.inf
    above = (S > tau) & valid
    if method == "threshold":  # все пары выше порога; при превышении ρ_max — лучшие по S
        return _top_n(S, above, n_max)
    if method == "greedy":     # пары выше порога по убыванию TAS до ρ_max
        return _top_n(TAS, above, n_max)
    a = size_c * valid_c
    a = a / a.sum()
    b = size_u / size_u.sum()
    if method == "ot":         # энтропийный ОТ (Синкхорн), пары с наибольшим Γ
        import ot
        Gam = np.zeros_like(S, dtype=np.float64)
        Gam[valid_c] = ot.sinkhorn(a[valid_c], b, 1.0 - S[valid_c].astype(np.float64), reg=ot_eps,
                                   numItermax=5000)
        return _top_n(Gam, valid, n_max)
    if method == "ipfp":       # p(b|a) ∝ exp(S/T), согласование маргиналов IPFP (в лог-пространстве)
        from scipy.special import logsumexp
        L = np.where(valid, S.astype(np.float64) / ipfp_T, -np.inf)
        la = np.where(a > 0, np.log(np.clip(a, 1e-300, None)), -np.inf)
        lb = np.log(b)
        u = np.zeros(kc)
        v = np.zeros(ku)
        rows = valid_c
        for _ in range(1000):
            u[rows] = la[rows] - logsumexp(L[rows] + v[None, :], axis=1)
            v = lb - logsumexp(L[rows] + u[rows, None], axis=0)
        logP = np.where(valid, L + u[:, None] + v[None, :], -np.inf)
        return _top_n(np.exp(logP), valid, n_max)
    if method == "bicluster":  # спектральная бикластеризация (Kluger, нормировка scale)
        rr = int(min(bic_r, kc - 1, ku - 1))
        Sp = (S + 1.0) * valid + 1e-6
        bc = SpectralBiclustering(n_clusters=(rr, rr), method="scale", random_state=seed).fit(Sp)
        blocks = []
        for i in range(rr):
            for j in range(rr):
                bm = np.outer(bc.row_labels_ == i, bc.column_labels_ == j) & valid
                if bm.any():
                    blocks.append((S[bm].mean(), bm))
        blocks.sort(key=lambda t: -t[0])
        sel = np.zeros_like(valid)
        for _, bm in blocks:
            if sel.sum() + bm.sum() <= n_max:
                sel |= bm
            else:
                rest = n_max - sel.sum()
                if rest > 0:
                    sel |= _top_n(S, bm & ~sel, rest)
                break
        return sel
    raise ValueError(method)


# ---------------------------------------------------------------- M1: метрики ранжирования
def rank_metrics(pos: np.ndarray, neg: np.ndarray) -> dict[str, np.ndarray]:
    """pos (n,), neg (n, 99). Ожидаемые значения при случайном разрешении ничьих.
    Отсутствующие отрицательные (вариант M1 «по размеру», протокол v2) — NaN: не участвуют."""
    g = (neg > pos[:, None]).sum(1)
    t = (neg == pos[:, None]).sum(1)
    n_neg = (~np.isnan(neg)).sum(1)
    auc = (n_neg - g - t + 0.5 * t) / n_neg
    ranks = np.arange(1, neg.shape[1] + 2)
    rec, mrr, ndcg = [], [], []
    for gi, ti in zip(g, t):
        r = ranks[gi:gi + ti + 1]
        rec.append(np.mean(r <= 10))
        mrr.append(np.mean(1.0 / r))
        ndcg.append(np.mean(np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0)))
    return {"auc": auc, "recall10": np.array(rec), "mrr": np.array(mrr), "ndcg10": np.array(ndcg)}


def bootstrap_ci(values: np.ndarray, B: int, rng, stat=np.mean) -> tuple[float, float, float]:
    n = len(values)
    bs = np.array([stat(values[rng.integers(0, n, n)]) for _ in range(B)])
    return float(stat(values)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_auc_test(auc_a: np.ndarray, auc_b: np.ndarray | float) -> tuple[float, float]:
    """z-тест для AUC по структурным компонентам ДеЛонга в варианте для кластеризованных
    данных (одна положительная пара на пользователя; единица наблюдения — пользователь)."""
    d = auc_a - auc_b
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        return 0.0, 1.0
    z = d.mean() / se
    return float(z), float(2 * norm.sf(abs(z)))


__all__ = [n for n in dir() if not n.startswith("_")]
