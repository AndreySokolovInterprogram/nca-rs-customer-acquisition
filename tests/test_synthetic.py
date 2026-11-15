"""Смоук-тест без данных, GPU и энкодера: импорт всех модулей и методы NCA-RS на случайных векторах.

Запуск из корня репозитория:  python -m pytest -q tests   или   python tests/test_synthetic.py
Все входы — синтетические (numpy.random), реальные данные не используются.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_TMP = Path(tempfile.mkdtemp(prefix="ncars_test_"))
for k, sub in [("PAPER1_MM_WORK", "work"), ("PAPER1_WORK_DIR", "work/main"), ("PAPER1_RESULTS_DIR", "results"),
               ("PAPER1_MM_RESULTS", "results"), ("PAPER1_META_DIR", "work/meta")]:
    os.environ.setdefault(k, str(_TMP / sub))

MODULES = ["config", "nca", "data", "encoders", "mm_media", "mm_embed", "mm_s4_pipeline", "mm_s12_m1_embed",
           "mm_s5_finalize", "s4_protocol", "mm_s6_checks", "s7_tables", "mm_s7_checks_tables", "mm_s13_attrition",
           "mm_s15_m1_diag"]


def sphere(rng, n, d, centers=None, noise=0.3):
    if centers is None:
        X = rng.standard_normal((n, d))
    else:
        X = centers[rng.integers(0, len(centers), n)] + noise * rng.standard_normal((n, d))
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def test_imports():
    for m in MODULES:
        importlib.import_module(f"experiments.paper1.{m}")


def test_acr_profiles():
    from experiments.paper1 import nca
    rng = np.random.default_rng(0)
    m, d = 50, 48
    e_p, e_s, e_d = sphere(rng, m, d), sphere(rng, m, d), sphere(rng, m, d)
    has_subs = rng.random(m) > 0.3
    e_s[~has_subs] = np.nan
    q_demo = rng.integers(0, 9, m) / 8.0
    e_d[q_demo == 0] = np.nan
    V = nca.acr_profiles(e_p, e_s, e_d, has_subs, q_demo, (0.7, 0.2, 0.1))
    assert V.shape == (m, d) and np.isfinite(V).all()
    assert np.allclose(np.linalg.norm(V, axis=1), 1, atol=1e-5)
    # нет подписок и демографии → профиль = посты
    only = (~has_subs) & (q_demo == 0)
    if only.any():
        assert np.allclose(V[only], e_p[only], atol=1e-5)
    # веса (1, 0, 0) → профиль = посты у всех
    assert np.allclose(nca.acr_profiles(e_p, e_s, e_d, has_subs, q_demo, (1, 0, 0)), e_p, atol=1e-5)


def test_clustering_tas_matching():
    from experiments.paper1 import nca, s4_protocol as sp
    rng = np.random.default_rng(1)
    d = 48
    C = rng.standard_normal((6, d)) * 2
    E = sphere(rng, 300, d, C)          # «посты компаний»
    V = sphere(rng, 240, d, C)          # «профили аудитории»
    D_c, D_u = nca.angular_dist(E), nca.angular_dist(V)
    for method in ("ward", "average", "complete"):
        sel = sp.select_k(E, method, D_c)
        assert sel["labels"].shape == (300,) and "max_share" in sel["best"]
    sel_c = sp.select_k(E, "ward", D_c)
    sel_u = sp.select_k(V, "ward", D_u)
    sel_g = sp.select_k(V[:120], "gmm", nca.angular_dist(V[:120]))  # Bayesian GMM на PCA-32
    assert sel_g["P"].shape[0] == 120
    # правило «крупнейший кластер ≤ 50 %»
    rows = [{"sil": 0.9, "max_share": 0.8}, {"sil": 0.5, "max_share": 0.4}]
    sp.MAX_SHARE = 0.5
    assert sp.pick_best(rows)["sil"] == 0.5 and sp.pick_best(rows)["admissible"]
    sp.MAX_SHARE = 1.0
    lab_c, lab_u = sel_c["labels"], sel_u["labels"]
    kc, ku = lab_c.max() + 1, lab_u.max() + 1
    G = E @ V.T
    TAS = nca.tas_matrix(G, lab_c, kc, lab_u, ku)
    # TAS_ab = среднее по постам кластера a максимума сходства с профилями кластера b — проверка циклом
    a, b = 0, 0
    ref = np.mean([G[i, lab_u == b].max() for i in np.flatnonzero(lab_c == a)])
    assert np.isclose(TAS[a, b], ref, atol=1e-5)
    mt = sp.matching_eval(E, V, lab_c, lab_u)
    size_c, size_u, valid = mt["size_c"], mt["size_u"], mt["valid"]
    n_max = int(np.floor(0.10 * kc * ku))
    for m in nca.MATCH_METHODS:
        M = nca.match(m, mt["S"], TAS, size_c, size_u, 0.90, 0.10, valid)
        assert M.shape == (kc, ku) and M.dtype == bool and M.sum() <= max(n_max, 0)
        assert not M[~valid].any()
    # перестановочный тест (как в этапе 3г), уменьшенное B
    obs = float((TAS * nca.match("greedy", mt["S"], TAS, size_c, size_u, 0.9, 0.1, valid)).sum())
    null = []
    for _ in range(50):
        lp = rng.permutation(lab_u)
        Sp = nca.centroids(E, lab_c, kc) @ nca.centroids(V, lp, ku).T
        Tp = nca.tas_matrix(G, lab_c, kc, lp, ku)
        null.append(float((Tp * nca.match("greedy", Sp, Tp, size_c, size_u, 0.9, 0.1, valid)).sum()))
    p = (1 + np.sum(np.array(null) >= obs)) / (len(null) + 1)
    assert 0 < p <= 1


def test_m1_scores_and_metrics():
    import pandas as pd
    from experiments.paper1 import nca, s4_protocol as sp
    rng = np.random.default_rng(2)
    d, n_groups = 48, 120
    E = sphere(rng, 400, d)
    gid = rng.integers(0, n_groups, 400)
    cpd = pd.DataFrame({"post_id": np.arange(400), "gid": gid})
    users = list(range(1000, 1060))
    V = sphere(rng, len(users), d)
    lab_c = nca.relabel(rng.integers(0, 8, 400))
    lab_u = nca.relabel(rng.integers(0, 5, len(users)))
    mt = sp.matching_eval(E, V, lab_c, lab_u)
    groups = np.unique(gid)
    cand = np.stack([rng.choice(groups, 100, replace=False) for _ in users])
    uid = np.array(users)
    s_nca = sp.nca_scores(cpd, lab_c, lab_c.max() + 1, users, lab_u, mt["M"], mt["TAS"], cand, uid)
    s_resp = sp.response_scores(cpd, E, users, V, cand, uid)
    for s in (s_nca, s_resp, sp.norm_rank(s_nca)):
        assert s.shape == cand.shape and np.isfinite(s).all()
    r = nca.rank_metrics(s_resp[:, 0], s_resp[:, 1:])
    assert set(r) == {"auc", "recall10", "mrr", "ndcg10"} and ((r["auc"] >= 0) & (r["auc"] <= 1)).all()
    # константная оценка → AUC = 0,5 ровно
    assert np.allclose(nca.rank_metrics(np.zeros(5), np.zeros((5, 99)))["auc"], 0.5)
    est, lo, hi = nca.bootstrap_ci(r["auc"], 200, rng)
    assert lo <= est <= hi
    z, p = nca.paired_auc_test(r["auc"], 0.5)
    assert 0 <= p <= 1


def test_mad_keyframes():
    from experiments.paper1 import mm_media
    rng = np.random.default_rng(3)
    frames = np.repeat(rng.integers(0, 255, (1, 24, 32, 3), dtype=np.uint8), 40, axis=0)
    frames[10:] = 255 - frames[10:]        # резкая смена сцены на кадре 10
    frames[25:] = rng.integers(0, 255, (1, 24, 32, 3), dtype=np.uint8)
    K, mad = mm_media.keyframes(frames)
    assert K[0] == 0 and 10 in K and 25 in K and len(K) <= mm_media.K_MAX
    many = rng.integers(0, 255, (80, 8, 8, 3), dtype=np.uint8)  # шум: все переходы > θ → обрезка до K_MAX
    assert len(mm_media.keyframes(many)[0]) == mm_media.K_MAX


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
