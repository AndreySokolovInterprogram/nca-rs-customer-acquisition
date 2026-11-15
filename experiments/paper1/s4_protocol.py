"""Этапы 3а–3д: последовательные абляции по таблице «План абляционных экспериментов».

  3a  энкодеры (Ward, жадное сопоставление, базовые веса): M3, M4 (Ward), M5, M1 (AUC)
  3b  веса профиля АВР (лучший энкодер): силуэт T(U), суммарный TAS, AUC M1
  3c  кластеризация (лучшие энкодер и веса): Ward / average / complete / Bayesian GMM
  3d  сопоставление (пять методов, одинаковое ρ): |M|, ρ, суммарный TAS, перестановочный тест,
      кривая «суммарный TAS — разреженность», Уилкоксон + Бенджамини–Хохберг
  3e  M1 для итоговой конфигурации: NCA-RS, модель отклика, B0–B3; ДеЛонг против NCA-RS

Каждый этап пишет results/.../stage_<id>.json и не пересчитывается, если файл уже есть.
Запуск: python -m experiments.paper1.s4_protocol --stage 3a
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from statsmodels.stats.multitest import multipletests

from experiments.paper1 import nca
from experiments.paper1.config import DOMAINS, RESULTS_DIR, SEED, WORK_DIR
from experiments.paper1.data import BASE_WEIGHTS, load_encoder, load_sample, m1_candidates

B_CI = 1000          # бутстреп-интервалы и перестановочный тест
B_STAB = 100         # бутстреп-стабильность Жаккара (переклассеризация на каждом повторе)
B_EFFDIM = 100       # бутстреп эффективной размерности (SVD на каждом повторе)
B_GAP = 20           # опорные наборы gap-статистики
ALPHA, RHO_MAX = 0.90, 0.10
ALPHAS = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
WEIGHTS = {"0.70;0.20;0.10": (0.70, 0.20, 0.10), "1/3;1/3;1/3": (1 / 3, 1 / 3, 1 / 3), "1;0;0": (1.0, 0.0, 0.0)}
N_JOBS = int(os.environ.get("PAPER1_N_JOBS", 4))

LOG_PATH = WORK_DIR / "protocol.log"
# Суффикс варианта для анализа чувствительности (например, "_ward": Ward для постов и профилей в 3d/3e)
SUF = os.environ.get("PAPER1_VARIANT", "")
# Протокол v2: энкодеры (только Qwen3-VL-Embedding-2B) и правило выбора кластеризации — максимум силуэта
# среди разбиений, где крупнейший кластер ≤ MAX_SHARE (v1: без ограничения, 1.0)
ENCS = os.environ.get("PAPER1_ENCODERS", "qwen,jina").split(",")
MAX_SHARE = float(os.environ.get("PAPER1_MAX_CLUSTER_SHARE", 1.0))


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:  # без глобального дескриптора: воркеры joblib сериализуют функции
        f.write(line + "\n")


def save(name: str, obj) -> None:
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def load(name: str):
    p = RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ данные области
def domain_arrays(ed, smp, d, V_all):
    cpd = ed.cp[ed.cp.domain == d].reset_index(drop=True)
    E = ed.E[[ed.E_post[int(p)] for p in cpd.post_id]]
    m1 = smp["m1_users"][smp["m1_users"].domain == d]
    users = sorted(set(smp["corpus_users"][smp["corpus_users"].domain == d].user_id) | set(m1.user_id))
    V = V_all[[ed.user_index[u] for u in users]]
    return cpd, E, users, V


# ------------------------------------------------------------------ кластеризация
def cluster(X, method, k=None, D=None, seed=SEED, cov=None, pca=None, Z=None):
    """Возвращает (labels, P или None). Для иерархических методов можно передать готовое Z."""
    if method in ("ward", "average", "complete"):
        Z = Z if Z is not None else nca.hier_linkage(X, method, D)
        return nca.relabel(nca.cut(Z, k)), None
    g, P, _ = nca.gmm_fit(X, k, cov, seed, pca)
    return nca.relabel(P.argmax(1)), P


def pick_best(rows):
    """Максимум силуэта среди допустимых разбиений (крупнейший кластер ≤ MAX_SHARE); если допустимых нет —
    максимум по всем с пометкой admissible = False (метод тогда не участвует в выборе на этапе 3в)."""
    fin = [r for r in rows if np.isfinite(r["sil"])]
    ok = [r for r in fin if r["max_share"] <= MAX_SHARE + 1e-12]
    best = dict(max(ok or fin, key=lambda r: r["sil"]))
    best["admissible"] = bool(ok)
    return best


def select_k(X, method, D, rng_seed=SEED):
    """Сетка k (и тип ковариации для GMM) с максимизацией углового силуэта."""
    rows = []
    if method in ("ward", "average", "complete"):
        Z = nca.hier_linkage(X, method, D)
        for k in nca.K_GRID:
            lab = nca.relabel(nca.cut(Z, k))
            rows.append({"k_grid": k, "k": int(lab.max() + 1), "sil": nca.silhouette_precomputed(D, lab),
                         "max_share": float(np.bincount(lab).max() / len(lab))})
        best = pick_best(rows)
        lab = nca.relabel(nca.cut(Z, best["k_grid"]))
        return {"grid": rows, "best": best, "labels": lab, "P": None, "Z": Z}
    pca = PCA(n_components=nca.GMM_PCA_DIM, random_state=rng_seed).fit(X)
    fits = {}
    for cov in ("diag", "full"):
        for k in nca.K_GRID:
            g, P, Xp = nca.gmm_fit(X, k, cov, rng_seed, pca)
            lab = nca.relabel(P.argmax(1))
            sil = nca.silhouette_precomputed(D, lab) if lab.max() >= 1 else float("nan")
            bic, icl, eff = nca.gmm_bic_icl(g, Xp, P)
            rows.append({"k_grid": k, "cov": cov, "k": int(lab.max() + 1), "eff_components": eff,
                         "sil": sil, "bic": bic, "icl": icl, "max_share": float(np.bincount(lab).max() / len(lab))})
            fits[(cov, k)] = (lab, P)
    best = pick_best(rows)
    lab, P = fits[(best["cov"], best["k_grid"])]
    # столбцы P для «пустых» компонент убираем, остальные — в порядке relabel
    used = np.unique(P.argmax(1))
    return {"grid": rows, "best": best, "labels": lab, "P": P[:, used], "pca": pca}


def internal_measures(X, D, labels, rng):
    """Силуэт и CH с 95 % бутстреп-интервалами при фиксированном разбиении: ресэмплируются
    поточечные значения s_i и поточечные вклады в суммы CH (повтор точек в выборке не меняет
    расстояний, поэтому прямой бутстреп с переклассеризацией смещал бы силуэт вверх)."""
    n = len(X)
    s = nca.silhouette_precomputed(D, labels, per_point=True)
    b, w, k = nca.ch_terms(X, labels)
    f = (n - k) / (k - 1)
    sil_bs, ch_bs = [], []
    for _ in range(B_CI):
        idx = rng.integers(0, n, n)
        sil_bs.append(s[idx].mean())
        ch_bs.append(b[idx].sum() / w[idx].sum() * f)
    ch = b.sum() / w.sum() * f
    return {"silhouette": [float(s.mean()), float(np.percentile(sil_bs, 2.5)), float(np.percentile(sil_bs, 97.5))],
            "ch": [float(ch), float(np.percentile(ch_bs, 2.5)), float(np.percentile(ch_bs, 97.5))],
            "ch_sklearn": float(calinski_harabasz_score(X, labels))}


def m3_measures(X, rng):
    H = [nca.hopkins(X, rng) for _ in range(B_CI)]
    n = len(X)
    # эффективная размерность — дискретная величина; повторы точек в бутстрепе её занижают,
    # поэтому приводится разброс (2,5–97,5 %) по B_EFFDIM подвыборкам 50 % без возвращения
    ed = [nca.effective_dim(X[rng.choice(n, n // 2, replace=False)]) for _ in range(B_EFFDIM)]
    pca10 = PCA(n_components=10, random_state=SEED).fit_transform(X).astype(np.float32)
    H10 = [nca.hopkins(pca10, rng) for _ in range(200)]
    return {"hopkins": [float(np.mean(H)), float(np.percentile(H, 2.5)), float(np.percentile(H, 97.5))],
            "hopkins_pca10": [float(np.mean(H10)), float(np.percentile(H10, 2.5)), float(np.percentile(H10, 97.5))],
            "eff_dim95": [nca.effective_dim(X), float(np.percentile(ed, 2.5)), float(np.percentile(ed, 97.5))],
            "eff_dim95_interval": "разброс по подвыборкам 50 % без возвращения, не бутстреп"}


# ------------------------------------------------------------------ сопоставление и M1
def matching_eval(E, V, lab_c, lab_u, method="greedy", alpha=ALPHA, rho=RHO_MAX, P_c=None, P_u=None):
    kc, ku = lab_c.max() + 1, lab_u.max() + 1
    mu_c = nca.centroids(E, lab_c, kc, P_c)
    mu_u = nca.centroids(V, lab_u, ku, P_u)
    S = mu_c @ mu_u.T
    G = E @ V.T
    TAS = nca.tas_matrix(G, lab_c, kc, lab_u, ku)
    size_c = np.bincount(lab_c, minlength=kc).astype(float)
    size_u = np.bincount(lab_u, minlength=ku).astype(float)
    valid = size_c > 1
    M = nca.match(method, S, TAS, size_c, size_u, alpha, rho, valid)
    return {"M": M, "S": S, "TAS": TAS, "G": G, "size_c": size_c, "size_u": size_u, "valid": valid}


def nca_scores(cpd, lab_c, kc, users, lab_u, M, TAS, cand, uid):
    """s(u, c) = Σ_{(a,b)∈M} 1[u ∈ T_b] TAS_ab π_{c,a}. Сообщество без постов в выборке → NaN (проверка п. 9а v2).
    Если в cpd есть столбец w (п. 9б v2) — π_{c,a} взвешиваются."""
    if "w" in cpd:
        pi = pd.crosstab(cpd.gid, lab_c, values=cpd.w, aggfunc="sum").fillna(0)
    else:
        pi = pd.crosstab(cpd.gid, lab_c)
    pi = pi.reindex(columns=range(kc), fill_value=0)
    pi = pi.div(pi.sum(1), axis=0)
    W = (M * TAS)  # k_C × k_U
    upos = {u: i for i, u in enumerate(users)}
    out = np.zeros(cand.shape)
    for r, (u, cs) in enumerate(zip(uid, cand)):
        b = lab_u[upos[u]]
        w = W[:, b]
        out[r] = pi.reindex(cs).to_numpy() @ w  # NaN, если у сообщества нет постов в выборке
    return out


def response_scores(cpd, E, users, V, cand, uid):
    """⟨v_u, ē_c⟩; ē_c — среднее (при столбце w — взвешенное, п. 9б v2); нет постов → NaN."""
    w = cpd.w.to_numpy() if "w" in cpd else np.ones(len(cpd))
    g = cpd.gid.to_numpy()
    num = pd.DataFrame(E * w[:, None]).groupby(g).sum()
    cent = nca.l2n(num.to_numpy()), {gg: i for i, gg in enumerate(num.index)}
    Cn = np.vstack([cent[0], np.full((1, E.shape[1]), np.nan)])
    miss = len(cent[0])
    upos = {u: i for i, u in enumerate(users)}
    out = np.zeros(cand.shape)
    for r, (u, cs) in enumerate(zip(uid, cand)):
        out[r] = Cn[[cent[1].get(c, miss) for c in cs]] @ V[upos[u]]
    return out


def m1_auc(scores):
    return nca.rank_metrics(scores[:, 0], scores[:, 1:])


# ------------------------------------------------------------------ 3a: энкодеры
def stage_3a_domain(enc, d, smp, cands):
    rng = np.random.default_rng(SEED)
    ed = load_encoder(enc, smp)
    V_all = ed.profiles(BASE_WEIGHTS)
    cpd, E, users, V = domain_arrays(ed, smp, d, V_all)
    D_c = nca.angular_dist(E)
    D_u = nca.angular_dist(V)
    t = time.time()
    m3 = m3_measures(E, rng)
    sel_c = select_k(E, "ward", D_c)
    sel_u = select_k(V, "ward", D_u)
    m4 = internal_measures(E, D_c, sel_c["labels"], rng)
    mt = matching_eval(E, V, sel_c["labels"], sel_u["labels"])
    c = cands[d]
    sc = nca_scores(cpd, sel_c["labels"], sel_c["labels"].max() + 1, users, sel_u["labels"],
                    mt["M"], mt["TAS"], c["cands"], c["users"])
    rs = response_scores(cpd, E, users, V, c["cands"], c["users"])
    res = {"n_posts": len(E), "n_users": len(V), "m3": m3, "m4_ward_posts": m4,
           "k_posts": sel_c["best"], "k_users": sel_u["best"],
           "sum_tas": float((mt["TAS"] * mt["M"]).sum()), "M_size": int(mt["M"].sum()),
           "auc_nca": float(m1_auc(sc)["auc"].mean()), "auc_response": float(m1_auc(rs)["auc"].mean()),
           "n_m1": len(c["users"]), "sec": round(time.time() - t, 1)}
    np.savez_compressed(WORK_DIR / f"labels_3a_{enc}_{d}.npz", lab_c=sel_c["labels"], lab_u=sel_u["labels"],
                        M=mt["M"], post_ids=cpd.post_id.to_numpy(), users=np.array(users))
    log(f"3a {enc} {d}: {json.dumps({k: res[k] for k in ['k_posts', 'sum_tas', 'auc_nca', 'auc_response', 'sec']}, default=float)}")
    return d, res


def stage_3a_encoder(smp, cands, enc):
    r = load(f"stage_3a_{enc}")
    if r is None:
        load_encoder(enc, smp)  # строит кэш компонентов профилей до запуска параллельных воркеров
        r = dict(Parallel(n_jobs=N_JOBS)(delayed(stage_3a_domain)(enc, d, smp, cands) for d in DOMAINS))
        save(f"stage_3a_{enc}", r)
    return r


def stage_3a(smp, cands):
    out = {enc: stage_3a_encoder(smp, cands, enc) for enc in ENCS}
    if len(ENCS) == 1:  # протокол v2: один энкодер — 3а только характеризует его
        out["best_encoder"] = ENCS[0]
        out["selection_rule"] = "протокол v2: единственный энкодер (п. 6)"
        save("stage_3a", out)
        return out
    # M5: согласие энкодеров на одних и тех же постах и пользователях
    eq, ej = load_encoder("qwen", smp), load_encoder("jina", smp)
    Vq, Vj = eq.profiles(), ej.profiles()
    m5 = {}
    for d in DOMAINS:
        a = np.load(WORK_DIR / f"labels_3a_qwen_{d}.npz")
        b = np.load(WORK_DIR / f"labels_3a_jina_{d}.npz")
        cpd, Eq, users, Vqd = domain_arrays(eq, smp, d, Vq)
        _, Ej, _, Vjd = domain_arrays(ej, smp, d, Vj)
        # совпадение M: Жаккар множеств пар (пост, пользователь), покрытых сопоставлением
        cov = []
        for z in (a, b):
            oc = np.eye(z["lab_c"].max() + 1, dtype=np.float32)[z["lab_c"]]
            ou = np.eye(z["lab_u"].max() + 1, dtype=np.float32)[z["lab_u"]]
            cov.append((oc @ z["M"].astype(np.float32) @ ou.T) > 0)
        inter, union = np.logical_and(*cov).sum(), np.logical_or(*cov).sum()
        m5[d] = {"ari_posts": adjusted_rand_score(a["lab_c"], b["lab_c"]),
                 "ari_users": adjusted_rand_score(a["lab_u"], b["lab_u"]),
                 "cka_posts": nca.linear_cka(Eq, Ej), "cka_users": nca.linear_cka(Vqd, Vjd),
                 "M_overlap_jaccard": float(inter / union) if union else float("nan")}
        log(f"3a M5 {d}: {m5[d]}")
    out["m5"] = m5
    sil = {enc: np.mean([out[enc][d]["m4_ward_posts"]["silhouette"][0] for d in DOMAINS]) for enc in ENCS}
    out["best_encoder"] = max(sil, key=sil.get)
    out["selection_rule"] = "больший средний по областям угловой силуэт Ward постов компаний (M4)"
    save("stage_3a", out)
    return out


# ------------------------------------------------------------------ 3b: веса профиля
def stage_3b_domain(enc, d, smp, cands, wname):
    ed = load_encoder(enc, smp)
    V_all = ed.profiles(WEIGHTS[wname])
    cpd, E, users, V = domain_arrays(ed, smp, d, V_all)
    rng = np.random.default_rng(SEED)
    lab_c = np.load(WORK_DIR / f"labels_3a_{enc}_{d}.npz")["lab_c"]
    D_u = nca.angular_dist(V)
    sel_u = select_k(V, "ward", D_u)
    m4u = internal_measures(V, D_u, sel_u["labels"], rng)
    mt = matching_eval(E, V, lab_c, sel_u["labels"])
    c = cands[d]
    sc = nca_scores(cpd, lab_c, lab_c.max() + 1, users, sel_u["labels"], mt["M"], mt["TAS"], c["cands"], c["users"])
    rs = response_scores(cpd, E, users, V, c["cands"], c["users"])
    res = {"k_users": sel_u["best"], "m4_ward_users": m4u, "sum_tas": float((mt["TAS"] * mt["M"]).sum()),
           "M_size": int(mt["M"].sum()), "auc_nca_per_user": m1_auc(sc)["auc"].tolist(),
           "auc_response_per_user": m1_auc(rs)["auc"].tolist()}
    log(f"3b {wname} {d}: sil_u={m4u['silhouette'][0]:.4f} sumTAS={res['sum_tas']:.3f} "
        f"AUC={np.mean(res['auc_nca_per_user']):.4f} AUCresp={np.mean(res['auc_response_per_user']):.4f}")
    return (wname, d), res


def stage_3b(smp, cands, enc):
    jobs = [(w, d) for w in WEIGHTS for d in DOMAINS]
    r = dict(Parallel(n_jobs=N_JOBS)(delayed(stage_3b_domain)(enc, d, smp, cands, w) for w, d in jobs))
    out = {"encoder": enc, "by_weights": {}}
    rng = np.random.default_rng(SEED)
    for w in WEIGHTS:
        per = {d: r[(w, d)] for d in DOMAINS}
        auc_all = np.concatenate([per[d]["auc_nca_per_user"] for d in DOMAINS])
        aucr_all = np.concatenate([per[d]["auc_response_per_user"] for d in DOMAINS])
        out["by_weights"][w] = {
            "silhouette_users_mean": float(np.mean([per[d]["m4_ward_users"]["silhouette"][0] for d in DOMAINS])),
            "sum_tas_total": float(sum(per[d]["sum_tas"] for d in DOMAINS)),
            "auc_nca": nca.bootstrap_ci(auc_all, B_CI, rng),
            "auc_response": nca.bootstrap_ci(aucr_all, B_CI, rng),
            "per_domain": {d: {k: v for k, v in per[d].items() if not k.endswith("per_user")} |
                           {"auc_nca": float(np.mean(per[d]["auc_nca_per_user"]))} for d in DOMAINS}}
    best = max(out["by_weights"], key=lambda w: out["by_weights"][w]["silhouette_users_mean"])
    out["best_weights"] = best
    out["selection_rule"] = "больший средний силуэт T(U) (M4); M1 — внешний арбитр"
    save("stage_3b", out)
    return out


# ------------------------------------------------------------------ 3c: кластеризация
def gap_stat(X, method, k, labels, rng, cov=None, pca=None, B=B_GAP):
    def log_w(Y, lab):
        w = 0.0
        for a in np.unique(lab):
            Ya = Y[lab == a]
            w += ((Ya - Ya.mean(0)) ** 2).sum()
        return np.log(w)

    lw = log_w(X, labels)
    sil = nca.silhouette_precomputed(nca.angular_dist(X), labels)
    res = {}
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xr = Xc @ Vt.T
    for ref in ("sphere_uniform", "pca_box"):
        lws, sils = [], []
        for _ in range(B):
            if ref == "sphere_uniform":
                R = nca.l2n(rng.standard_normal(X.shape).astype(np.float32))
            else:
                R = rng.uniform(Xr.min(0), Xr.max(0), size=Xr.shape) @ Vt + X.mean(0)
                R = nca.l2n(R.astype(np.float32))
            Dr = nca.angular_dist(R)
            if method == "gmm":
                pr = PCA(n_components=nca.GMM_PCA_DIM, random_state=SEED).fit(R)
                lr, _ = cluster(R, "gmm", k, cov=cov, pca=pr)
            else:
                lr, _ = cluster(R, method, k, D=Dr)
            lws.append(log_w(R, lr))
            sils.append(nca.silhouette_precomputed(Dr, lr))
        lws = np.array(lws)
        res[ref] = {"gap": float(lws.mean() - lw), "s_k": float(lws.std(ddof=1) * np.sqrt(1 + 1 / B)), "B": B,
                    "silhouette_null_mean": float(np.mean(sils)), "silhouette_null_max": float(np.max(sils)),
                    "silhouette_obs": sil}
    return res


def stage_3c_corpus(enc, wname, d, corpus, method, smp):
    ed = load_encoder(enc, smp)
    cpd, E, users, V = domain_arrays(ed, smp, d, ed.profiles(WEIGHTS[wname]))
    X = E if corpus == "posts" else V
    res, lab, P = eval_corpus(X, method)
    np.savez_compressed(WORK_DIR / f"labels_3c_{d}_{corpus}_{method}.npz", labels=lab,
                        P=P if P is not None else np.zeros(0))
    log(f"3c {d} {corpus} {method}: k={res['k']} sil={res['silhouette'][0]:.4f} J={res['jaccard_mean']:.3f} "
        f"gap_pca={res['gap']['pca_box']['gap']:.3f} {res['sec']}s")
    return (d, corpus, method), res


def eval_corpus(X, method):
    """Выбор k, внутренние меры, стабильность и gap для одного корпуса (ядро этапа 3в)."""
    D = nca.angular_dist(X)
    rng = np.random.default_rng(SEED)
    t = time.time()
    sel = select_k(X, method, D)
    lab, k = sel["labels"], int(sel["labels"].max() + 1)
    im = internal_measures(X, D, lab, rng)
    best = sel["best"]
    b_stab, b_gap = B_STAB, B_GAP
    if method == "gmm":
        boot = lambda idx: cluster(X[idx], "gmm", best["k_grid"], cov=best["cov"], pca=sel["pca"])[0]
        t_fit = time.time()
        boot(np.arange(len(X)))
        t_fit = time.time() - t_fit
        # бюджет времени: ~10 мин на стабильность и ~5 мин на gap для одного корпуса
        b_stab = int(min(B_STAB, max(20, 600 // max(t_fit, 1e-3))))
        b_gap = int(min(B_GAP, max(5, 300 // max(t_fit, 1e-3))))
    else:
        Euse = None
        if method == "ward":
            G = np.clip(X @ X.T, -1, 1)
            Euse = np.sqrt(np.clip(2 - 2 * G, 0, None)).astype(np.float64)

        def boot(idx):
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import squareform
            sub = (Euse if method == "ward" else D.astype(np.float64))[np.ix_(idx, idx)]
            np.fill_diagonal(sub, 0)
            Z = linkage(squareform(sub, checks=False), method=method)
            return nca.cut(Z, best["k_grid"])
    jac, ari = nca.jaccard_stability(lab, boot, len(X), b_stab, rng)
    gap = gap_stat(X, method, best["k_grid"], lab, rng, cov=best.get("cov"), pca=sel.get("pca"), B=b_gap)
    res = {"best": best, "grid": sel["grid"], "k": k, **im,
           "bic_icl": [best.get("bic"), best.get("icl")] if method == "gmm" else None,
           "jaccard_per_cluster": jac.tolist(), "jaccard_mean": float(np.nanmean(jac)),
           "share_stable_clusters": float(np.mean(jac >= 0.75)), "boot_ari_mean": ari,
           "gap": gap, "B_stability": b_stab, "B_gap": b_gap, "sec": round(time.time() - t, 1)}
    return res, lab, sel["P"]


def stage_3c(smp, enc, wname):
    jobs = [(d, c, m) for d in DOMAINS for c in ("posts", "users") for m in ("ward", "average", "complete", "gmm")]
    r = dict(Parallel(n_jobs=N_JOBS)(delayed(stage_3c_corpus)(enc, wname, d, c, m, smp) for d, c, m in jobs))
    out = {"encoder": enc, "weights": wname, "results": {f"{d}|{c}|{m}": v for (d, c, m), v in r.items()}}
    best = {}
    for c in ("posts", "users"):
        means = {m: np.mean([r[(d, c, m)]["silhouette"][0] for d in DOMAINS]) for m in ("ward", "average", "complete", "gmm")}
        adm = {m: all(r[(d, c, m)]["best"].get("admissible", True) for d in DOMAINS) for m in means}
        pool = {m: v for m, v in means.items() if adm[m]} or means
        best[c] = max(pool, key=pool.get)
        out[f"mean_silhouette_{c}"] = means
        out[f"admissible_{c}"] = adm
    out["best_method"] = best
    out["selection_rule"] = ("больший средний по областям угловой силуэт (M4)" if MAX_SHARE >= 1 else
                             f"больший средний угловой силуэт среди методов, у которых во всех областях есть разбиение "
                             f"с крупнейшим кластером ≤ {MAX_SHARE:.0%} (протокол v2, п. 7)")
    out["max_cluster_share"] = MAX_SHARE
    save("stage_3c", out)
    return out


# ------------------------------------------------------------------ 3d: сопоставление
def load_labels_3c(d, corpus, method):
    z = np.load(WORK_DIR / f"labels_3c_{d}_{corpus}_{method}.npz")
    P = z["P"] if z["P"].size else None
    return z["labels"], P


def stage_3d_domain(enc, wname, d, smp, mc, mu):
    ed = load_encoder(enc, smp)
    cpd, E, users, V = domain_arrays(ed, smp, d, ed.profiles(WEIGHTS[wname]))
    lab_c, P_c = load_labels_3c(d, "posts", mc)
    lab_u, P_u = load_labels_3c(d, "users", mu)
    base = matching_eval(E, V, lab_c, lab_u, P_c=P_c, P_u=P_u)
    S, TAS, G = base["S"], base["TAS"], base["G"]
    kc, ku = S.shape
    sc, su, valid = base["size_c"], base["size_u"], base["valid"]
    rng = np.random.default_rng(SEED)
    res = {"k_c": int(kc), "k_u": int(ku), "methods": {}, "curve": [], "by_alpha": {}}
    obs = {}
    for m in nca.MATCH_METHODS:
        M = nca.match(m, S, TAS, sc, su, ALPHA, RHO_MAX, valid)
        obs[m] = float((TAS * M).sum())
        res["methods"][m] = {"M_size": int(M.sum()), "rho": float(M.sum() / (kc * ku)), "sum_tas": obs[m],
                             "mean_tas": float(TAS[M].mean()) if M.any() else float("nan"),
                             "pairs": np.argwhere(M).tolist()}
    # перестановочный тест: метки кластеров аудитории перемешиваются, задача решается заново
    null = {m: [] for m in nca.MATCH_METHODS}
    for _ in range(B_CI):
        lp = rng.permutation(lab_u)
        mu_u = nca.centroids(V, lp, ku)
        mu_c = nca.centroids(E, lab_c, kc, P_c)
        Sp = mu_c @ mu_u.T
        Tp = nca.tas_matrix(G, lab_c, kc, lp, ku)
        for m in nca.MATCH_METHODS:
            null[m].append(float((Tp * nca.match(m, Sp, Tp, sc, su, ALPHA, RHO_MAX, valid)).sum()))
    for m in nca.MATCH_METHODS:
        nl = np.array(null[m])
        res["methods"][m]["perm_p"] = float((1 + np.sum(nl >= obs[m])) / (B_CI + 1))
        res["methods"][m]["null_mean"] = float(nl.mean())
        res["methods"][m]["null_q95"] = float(np.percentile(nl, 95))
    # кривая «суммарный TAS — разреженность» и одинаковое ρ = 1 − α для попарных сравнений
    for a in ALPHAS:
        M = nca.match("threshold", S, TAS, sc, su, a, 1.0, valid)
        res["curve"].append({"alpha": a, "rho": float(M.sum() / (kc * ku)), "sum_tas": float((TAS * M).sum()),
                             "M_size": int(M.sum())})
        rho_a = M.sum() / (kc * ku)
        res["by_alpha"][str(a)] = {m: float((TAS * nca.match(m, S, TAS, sc, su, a, rho_a, valid)).sum())
                                   for m in nca.MATCH_METHODS}
    np.savez_compressed(WORK_DIR / f"match_3d{SUF}_{d}.npz", S=S, TAS=TAS, lab_c=lab_c, lab_u=lab_u,
                        post_ids=cpd.post_id.to_numpy(), users=np.array(users))
    log(f"3d {d}: " + ", ".join(f"{m}: |M|={res['methods'][m]['M_size']} sum={res['methods'][m]['sum_tas']:.2f} "
                                f"p={res['methods'][m]['perm_p']:.3f}" for m in nca.MATCH_METHODS))
    return d, res


def stage_3d(smp, enc, wname, mc, mu):
    r = dict(Parallel(n_jobs=N_JOBS)(delayed(stage_3d_domain)(enc, wname, d, smp, mc, mu) for d in DOMAINS))
    out = {"encoder": enc, "weights": wname, "clustering": {"posts": mc, "users": mu},
           "alpha": ALPHA, "rho_max": RHO_MAX, "per_domain": r}
    # Уилкоксон по парам (область × α), одинаковое ρ = 1 − α у всех методов; поправка BH
    pairs, pvals = [], []
    for i, a in enumerate(nca.MATCH_METHODS):
        for b in nca.MATCH_METHODS[i + 1:]:
            xa = [r[d]["by_alpha"][str(al)][a] for d in DOMAINS for al in ALPHAS]
            xb = [r[d]["by_alpha"][str(al)][b] for d in DOMAINS for al in ALPHAS]
            diff = np.array(xa) - np.array(xb)
            p = float(wilcoxon(xa, xb).pvalue) if np.any(diff != 0) else 1.0
            pairs.append({"a": a, "b": b, "median_diff": float(np.median(diff)), "n": len(diff), "p": p})
            pvals.append(p)
    rej, padj, _, _ = multipletests(pvals, method="fdr_bh")
    for row, q, rj in zip(pairs, padj, rej):
        row["p_bh"] = float(q)
        row["reject_0.05"] = bool(rj)
    out["wilcoxon"] = pairs
    tot = {m: sum(r[d]["methods"][m]["sum_tas"] for d in DOMAINS) for m in nca.MATCH_METHODS}
    out["sum_tas_total"] = tot
    out["best_method"] = max(tot, key=tot.get)
    out["selection_rule"] = "наибольший суммарный TAS по четырём областям при одинаковом ρ"
    save(f"stage_3d{SUF}", out)
    plot_curve(r)
    return out


def plot_curve(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.54, 2.6))
    names = {"customer_acquisition": "cust. acq.", "food_delivery": "food deliv.",
             "food_service": "food serv.", "tender_support": "tender"}
    marks = ["o", "s", "^", "D"]
    for (d, mk) in zip(DOMAINS, marks):
        cv = r[d]["curve"]
        ax.plot([c["rho"] for c in cv], [c["sum_tas"] for c in cv], marker=mk, ms=3.5, lw=1.2, label=names[d])
    ax.set_xlabel("Разреженность ρ(M)")
    ax.set_ylabel("Суммарный TAS")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"fig_tas_sparsity{SUF}.pdf")
    fig.savefig(RESULTS_DIR / f"fig_tas_sparsity{SUF}.png", dpi=300)
    pd.DataFrame([{"domain": d, **c} for d in DOMAINS for c in r[d]["curve"]]).to_csv(
        RESULTS_DIR / f"fig_tas_sparsity{SUF}.csv", index=False)


# ------------------------------------------------------------------ 3e: M1
STEM = 6


def toks(s: str) -> set:
    return {w[:STEM] for w in re.findall(r"[a-zа-яё]{3,}", s.lower())}


def norm_rank(s):
    """Нормированный ранг внутри строки (по возрастанию оценки, связи — средним рангом); NaN сохраняется."""
    from scipy.stats import rankdata
    out = np.full(s.shape, np.nan)
    for r in range(len(s)):
        ok = ~np.isnan(s[r])
        x = np.where(np.isinf(s[r][ok]), np.sign(s[r][ok]) * 1e300, s[r][ok])
        rk = rankdata(x, "average")
        out[r, ok] = (rk - 1) / max(1, ok.sum() - 1)
    return out


def m1_scores(ed, smp, d, V_all, lab_c, lab_u, mt, c, rng, info, kw_tok, utop, seeds, cover):
    cpd, E, users, V = domain_arrays(ed, smp, d, V_all)
    pad = c["cands"] < 0
    cs_all = np.where(pad, c["cands"][:, :1], c["cands"])  # заглушка вместо −1, потом → NaN
    sc = {
        "nca": nca_scores(cpd, lab_c, lab_c.max() + 1, users, lab_u, mt["M"], mt["TAS"], cs_all, c["users"]),
        "response": response_scores(cpd, E, users, V, cs_all, c["users"]),
        "B0": rng.random(cs_all.shape),
        "B1": info.reindex(cs_all.ravel()).members_count.fillna(0).to_numpy().reshape(cs_all.shape),
    }
    b2 = np.zeros(cs_all.shape)
    b3 = np.zeros(cs_all.shape)
    for r, (u, cs) in enumerate(zip(c["users"], cs_all)):
        T = utop.get(u, set())
        cover["b2_users_with_subs"] += bool(T)
        vu = V_all[ed.user_index[u]]
        for j, g in enumerate(cs):
            K = kw_tok.get(g, set())
            b2[r, j] = len(T & K) / len(T | K) if (T | K) else 0.0
            sd = [s for s in seeds.get(g, []) if s != u]
            if sd:
                b3[r, j] = float(np.mean(V_all[[ed.user_index[s] for s in sd]] @ vu))
                cover["b3_cands_with_seed"] += int(not pad[r, j])
            else:
                b3[r, j] = -np.inf
    sc["B2"], sc["B3"] = b2, b3
    for k in sc:
        sc[k] = np.where(pad, np.nan, sc[k])
    # протокол v2, п. 8б: гибрид NCA-RS + популярность = среднее нормированных рангов
    sc["hybrid"] = (norm_rank(sc["nca"]) + norm_rank(sc["B1"])) / 2
    cover["n_users"] += len(c["users"])
    cover["n_cands"] += int((~pad).sum())
    return sc


def m1_table(per_user, rng):
    pooled = {k: {m: np.concatenate([x[m] for x in per_user[k]]) for m in ["auc", "recall10", "mrr", "ndcg10"]}
              for k in per_user}
    table = {}
    for k in per_user:
        row = {m: nca.bootstrap_ci(pooled[k][m], B_CI, rng) for m in ["auc", "recall10", "mrr", "ndcg10"]}
        row["vs_0.5"] = nca.paired_auc_test(pooled[k]["auc"], 0.5)
        row["delong_vs_nca"] = nca.paired_auc_test(pooled[k]["auc"], pooled["nca"]["auc"]) if k != "nca" else None
        table[k] = row
    ks = [k for k in table if k != "nca"]
    _, padj, _, _ = multipletests([table[k]["delong_vs_nca"][1] for k in ks], method="fdr_bh")
    for k, q in zip(ks, padj):
        table[k]["delong_vs_nca_p_bh"] = float(q)
    return table, int(len(pooled["nca"]["auc"]))


def m1_context(smp):
    info = smp["pool_info"].drop_duplicates("gid").set_index("gid")
    kw = smp["pool_keywords"]
    kw_tok = kw.groupby("gid").keyword.apply(lambda s: set().union(*[toks(x) for x in s]))
    # B2: топ-20 TF-IDF термов названий и описаний сообществ из подписок пользователя
    subs = smp["user_subs"].merge(smp["sub_communities"], left_on="vk_group_id", right_on="vk_id")
    udoc = subs.groupby("user_id").text.apply(" ".join)
    tf = TfidfVectorizer(token_pattern=r"[a-zа-яё]{3,}", lowercase=True, max_features=200000,
                         preprocessor=lambda s: " ".join(w[:STEM] for w in re.findall(r"[a-zа-яё]{3,}", s.lower())))
    Xtf = tf.fit_transform(udoc.to_numpy())
    vocab = np.array(tf.get_feature_names_out())
    utop = {u: set(vocab[Xtf[i].indices[np.argsort(-Xtf[i].data)[:20]]]) for i, u in enumerate(udoc.index)}
    seeds = smp["seed_users"].groupby("gid").user_id.apply(list)
    return info, kw_tok, utop, seeds


def stage_3e(smp, enc, wname, mc, mu, mm, cands, labels=None, V_all=None, name=None, ed=None):
    """M1. labels: {d: (lab_c, P_c, lab_u, P_u)} — для проверок п. 9 (иначе — из 3в); V_all — профили."""
    ed = ed or load_encoder(enc, smp)
    V_all = ed.profiles(WEIGHTS[wname]) if V_all is None else V_all
    rng = np.random.default_rng(SEED)
    ctx = m1_context(smp)
    variants = {"main": cands}
    if MAX_SHARE < 1:  # протокол v2: вариант (а) «по размеру»
        from experiments.paper1.data import m1_candidates_decile
        variants["size_decile"] = m1_candidates_decile(smp)
    out = {"encoder": enc, "weights": wname, "clustering": {"posts": mc, "users": mu}, "matching": mm,
           "test": "z-тест по пользователям (структурные компоненты ДеЛонга для одной положительной пары "
                   "на пользователя, кластеризованный вариант); BH по сравнениям с NCA-RS", "variants": {}}
    mts = {}
    for d in DOMAINS:
        cpd, E, users, V = domain_arrays(ed, smp, d, V_all)
        if labels is None:
            lab_c, P_c = load_labels_3c(d, "posts", mc)
            lab_u, P_u = load_labels_3c(d, "users", mu)
        else:
            lab_c, P_c, lab_u, P_u = labels[d]
        mts[d] = (lab_c, lab_u, matching_eval(E, V, lab_c, lab_u, method=mm, P_c=P_c, P_u=P_u))
    # протокол v2, решение 10.11: M1 и на подмножестве пользователей с полным мультимодальным профилем
    full_path = WORK_DIR / "sample" / "m1_full_users.pkl"
    m1_full = set(pd.read_pickle(full_path).user_id) if full_path.exists() else None
    for vname, cset in list(variants.items()):
        per_user, res, uorder = {}, {}, []
        cover = {"b2_users_with_subs": 0, "b3_cands_with_seed": 0, "n_users": 0, "n_cands": 0}
        for d in DOMAINS:
            lab_c, lab_u, mt = mts[d]
            sc = m1_scores(ed, smp, d, V_all, lab_c, lab_u, mt, cset[d], rng, *ctx, cover)
            # кандидаты без постов в выборке (только проверка п. 9а) исключаются у всех стратегий одинаково
            miss = np.isnan(sc["nca"]) | np.isnan(sc["response"])
            if miss.any():
                sc = {k: np.where(miss, np.nan, v) for k, v in sc.items()}
                sc["hybrid"] = (norm_rank(sc["nca"]) + norm_rank(sc["B1"])) / 2
            keep = ~miss[:, 0]  # скрытое без постов → пользователь вне M1
            res_drop = int((~keep).sum())
            sc = {k: v[keep] for k, v in sc.items()}
            uorder += list(np.asarray(cset[d]["users"])[keep])
            for k, s in sc.items():
                per_user.setdefault(k, []).append(nca.rank_metrics(s[:, 0], s[:, 1:]))
            res[d] = {k: {m: float(np.mean(v)) for m, v in per_user[k][-1].items()} for k in sc}
            res[d]["users_dropped_hidden_without_posts"] = res_drop
            if "n_neg" in cset[d]:
                res[d]["n_neg_mean"] = float(cset[d]["n_neg"].mean())
                res[d]["n_neg_min"] = int(cset[d]["n_neg"].min())
            log(f"3e[{vname}] {d}: " + ", ".join(f"{k}={res[d][k]['auc']:.3f}" for k in sc))
        table, n_users = m1_table(per_user, rng)
        out["variants"][vname] = {"table": table, "per_domain": res, "coverage": cover, "n_users": n_users}
        if m1_full is not None:
            mask = np.array([u in m1_full for u in uorder])
            sub = {k: [{m: np.concatenate([x[m] for x in v])[mask] for m in v[0]}] for k, v in per_user.items()}
            t_full, n_full = m1_table(sub, np.random.default_rng(SEED))
            out["variants"][f"{vname}@full"] = {"table": t_full, "n_users": n_full,
                                                "note": "только пользователи M1 с полным профилем (все медиа постов 1–5)"}
            log(f"3e[{vname}@full] n={n_full}: " + ", ".join(f"{k}={t_full[k]['auc'][0]:.3f}" for k in t_full))
    # совместимость с s7_tables (v1): основной вариант на верхнем уровне
    out.update({k: out["variants"]["main"][k] for k in ("table", "per_domain", "coverage", "n_users")})
    save(name or f"stage_3e{SUF}", out)
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["3a", "3b", "3c", "3d", "3e", "all"])
    ap.add_argument("--encoder", choices=["qwen", "jina"], help="только для 3a: посчитать один энкодер")
    args = ap.parse_args()
    smp = load_sample()
    cands = m1_candidates(smp)
    if args.stage == "3a" and args.encoder:
        stage_3a_encoder(smp, cands, args.encoder)
        return
    stages = ["3a", "3b", "3c", "3d", "3e"] if args.stage == "all" else [args.stage]
    for st in stages:
        t = time.time()
        log(f"=== stage {st} start")
        if st == "3a":
            load("stage_3a") or stage_3a(smp, cands)
        enc = (load("stage_3a") or {}).get("best_encoder")
        if st == "3b":
            load("stage_3b") or stage_3b(smp, cands, enc)
        wname = (load("stage_3b") or {}).get("best_weights")
        if st == "3c":
            load("stage_3c") or stage_3c(smp, enc, wname)
        bm = (load("stage_3c") or {}).get("best_method", {})
        if SUF == "_ward":
            bm = {"posts": "ward", "users": "ward"}
        if st == "3d":
            load(f"stage_3d{SUF}") or stage_3d(smp, enc, wname, bm["posts"], bm["users"])
        mm = (load(f"stage_3d{SUF}") or {}).get("best_method")
        if st == "3e":
            load(f"stage_3e{SUF}") or stage_3e(smp, enc, wname, bm["posts"], bm["users"], mm, cands)
        log(f"=== stage {st} done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
