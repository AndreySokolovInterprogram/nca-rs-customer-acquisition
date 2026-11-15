"""Протокол v2, M1 10.11 — диагностика (только чтение, результаты этапов не перезаписываются). Запрос 10.11.

Воспроизводит оценки NCA-RS, модели отклика, B1 и гибрида по пользователям тем же кодом (s4_protocol.nca_scores,
response_scores, norm_rank) на тех же метках 3в и сопоставлении 3г; сверяет средние AUC с сохранёнными stage_3e.json.
  1. Санитарная проверка NCA-RS: распределение оценок, число уникальных значений у пользователя, доля пользователей
     с константной оценкой (все кандидаты равны → AUC = 0,5 ровно), доля, у кого кластер профиля не сопоставлен ни с
     одним кластером постов; независимый пересчёт формулы s(u, c) = Σ_{(a,b)∈M} 1[u ∈ T_b] TAS_ab π_{c,a} циклом.
  2. Разложение выбывания M1: выборка → кандидаты → «скрытое без постов в выборке компаний»; число отрицательных с постами.
  3. ДеЛонг (тот же z-тест по пользователям, nca.paired_auc_test): гибрид NCA-RS + B1 против B1 — в целом и по областям,
     все и полные профили, основной вариант и (а). Это сравнение не предрегистрировано (в протоколе — против NCA-RS).
Запуск: PAPER1_WORK_DIR=…/main PAPER1_RESULTS_DIR=…/main PAPER1_MM_RESULTS=… python -m experiments.paper1.mm_s15_m1_diag
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paper1 import nca
from experiments.paper1.config import DOMAINS, WORK_DIR
from experiments.paper1.data import load_encoder, load_sample, m1_candidates, m1_candidates_decile
from experiments.paper1.s4_protocol import (WEIGHTS, domain_arrays, load, load_labels_3c, matching_eval, nca_scores,
                                            norm_rank, response_scores)

R = Path(os.environ["PAPER1_MM_RESULTS"])


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())} if len(x) else None


def main() -> None:
    smp = load_sample()
    e3 = load("stage_3e")
    enc, wname = e3["encoder"], e3["weights"]
    mc, mu, mm = e3["clustering"]["posts"], e3["clustering"]["users"], e3["matching"]
    ed = load_encoder(enc, smp)
    V_all = ed.profiles(WEIGHTS[wname])
    full = set(pd.read_pickle(WORK_DIR / "sample" / "m1_full_users.pkl").user_id)
    info = smp["pool_info"].drop_duplicates("gid").set_index("gid")
    variants = {"main": m1_candidates(smp), "size_decile": m1_candidates_decile(smp)}
    out = {"config": [enc, wname, mc, mu, mm], "sanity": {}, "attrition": {}, "delong_hybrid_vs_B1": {}, "repro": {}}
    pooled = {v: {"all": {}, "full": {}} for v in variants}

    for d in DOMAINS:
        cpd, E, users, V = domain_arrays(ed, smp, d, V_all)
        lab_c, P_c = load_labels_3c(d, "posts", mc)
        lab_u, P_u = load_labels_3c(d, "users", mu)
        assert len(lab_c) == len(E) and len(lab_u) == len(V), "метки 3в не совпадают по размеру с данными"
        mt = matching_eval(E, V, lab_c, lab_u, method=mm, P_c=P_c, P_u=P_u)
        M, TAS = mt["M"], mt["TAS"]
        kc = lab_c.max() + 1
        matched_u = M.any(0)  # кластер профиля сопоставлен хотя бы с одним кластером постов
        upos = {u: i for i, u in enumerate(users)}
        gids_with_posts = set(cpd.gid)
        pool = list(smp["pools"][d])
        m1d = smp["m1_users"][smp["m1_users"].domain == d]
        for vname, cset in variants.items():
            c = cset[d]
            pad = c["cands"] < 0
            cs_all = np.where(pad, c["cands"][:, :1], c["cands"])
            sc = {"nca": nca_scores(cpd, lab_c, kc, users, lab_u, M, TAS, cs_all, c["users"]),
                  "response": response_scores(cpd, E, users, V, cs_all, c["users"]),
                  "B1": info.reindex(cs_all.ravel()).members_count.fillna(0).to_numpy().reshape(cs_all.shape)}
            sc = {k: np.where(pad, np.nan, s) for k, s in sc.items()}
            sc["hybrid"] = (norm_rank(sc["nca"]) + norm_rank(sc["B1"])) / 2
            miss = np.isnan(sc["nca"]) | np.isnan(sc["response"])
            if miss.any():  # как в stage_3e
                sc = {k: np.where(miss, np.nan, v) for k, v in sc.items()}
                sc["hybrid"] = (norm_rank(sc["nca"]) + norm_rank(sc["B1"])) / 2
            keep = ~miss[:, 0]
            uids = np.asarray(c["users"])[keep]
            sc = {k: v[keep] for k, v in sc.items()}
            met = {k: nca.rank_metrics(s[:, 0], s[:, 1:]) for k, s in sc.items()}
            isfull = np.array([u in full for u in uids])
            for k in met:
                pooled[vname]["all"].setdefault(k, []).append(met[k]["auc"])
                pooled[vname]["full"].setdefault(k, []).append(met[k]["auc"][isfull])
            saved = e3["variants"][vname]["per_domain"][d]
            out["repro"][f"{vname}|{d}"] = {k: [round(float(met[k]["auc"].mean()), 6), round(saved[k]["auc"], 6)]
                                            for k in met}
            # ДеЛонг гибрид против B1 по области
            dl = {}
            for who, m_ in (("all", np.ones(len(uids), bool)), ("full", isfull)):
                z, p = nca.paired_auc_test(met["hybrid"]["auc"][m_], met["B1"]["auc"][m_])
                dl[who] = {"n": int(m_.sum()), "auc_hybrid": float(met["hybrid"]["auc"][m_].mean()),
                           "auc_B1": float(met["B1"]["auc"][m_].mean()), "auc_nca": float(met["nca"]["auc"][m_].mean()),
                           "z": float(z), "p": float(p)}
            out["delong_hybrid_vs_B1"][f"{vname}|{d}"] = dl
            if vname != "main":
                continue
            # 1. санитарная проверка NCA-RS
            s = sc["nca"]
            nun = np.array([len(np.unique(r[np.isfinite(r)])) for r in s])
            const = nun == 1
            bu = np.array([lab_u[upos[u]] for u in uids])
            auc = met["nca"]["auc"]
            # независимый пересчёт формулы для 30 пользователей
            pi = pd.crosstab(cpd.gid, lab_c).reindex(columns=range(kc), fill_value=0)
            pi = pi.div(pi.sum(1), axis=0)
            cands_k = cs_all[keep]
            err = 0.0
            for r in range(min(30, len(uids))):
                for j, g in enumerate(cands_k[r]):
                    if not np.isfinite(s[r, j]):
                        continue
                    manual = sum(TAS[a, bu[r]] * pi.at[g, a] for a in range(kc) if M[a, bu[r]])
                    err = max(err, abs(manual - s[r, j]))
            sizes_u = np.bincount(lab_u, minlength=lab_u.max() + 1)
            out["sanity"][d] = {
                "n_users": int(len(uids)), "k_posts": int(kc), "k_profiles": int(len(sizes_u)),
                "M_pairs": int(M.sum()), "profile_clusters_matched": int(matched_u.sum()),
                "share_profiles_in_matched_clusters": float(sizes_u[matched_u].sum() / sizes_u.sum()),
                "share_m1_users_in_unmatched_cluster": float(np.mean(~matched_u[bu])),
                "share_users_constant_score": float(const.mean()),
                "auc_of_constant_users": q(auc[const]),
                "auc_nonconstant_users_mean": float(auc[~const].mean()) if (~const).any() else None,
                "n_nonconstant_users": int((~const).sum()),
                "unique_scores_per_user": q(nun),
                "share_zero_scores": float(np.nanmean(s == 0)),
                "score_all": q(s), "score_hidden": q(s[:, 0]),
                "TAS_matched": q(TAS[M.astype(bool)]) if M.any() else None,
                "largest_profile_cluster_share": float(sizes_u.max() / sizes_u.sum()),
                "manual_formula_max_abs_err_30_users": float(err),
            }
            # 2. выбывание
            hid = m1d.set_index("user_id").hidden_gid.reindex(c["users"])
            out["attrition"][d] = {
                "m1_after_finalize": int(len(m1d)), "with_candidates": int(len(c["users"])),
                "pool_P_d": len(pool), "pool_communities_with_sampled_posts": int(len(set(pool) & gids_with_posts)),
                "hidden_without_sampled_posts": int((~hid.isin(gids_with_posts)).sum()),
                "evaluated": int(keep.sum()), "evaluated_full": int(isfull.sum()),
                "full_after_finalize": int(len(set(m1d.user_id) & full)),
                "valid_negatives_per_user": q((~np.isnan(sc["nca"][:, 1:])).sum(1)),
            }

    for vname in variants:
        for who in ("all", "full"):
            h = np.concatenate(pooled[vname][who]["hybrid"])
            b = np.concatenate(pooled[vname][who]["B1"])
            n_ = np.concatenate(pooled[vname][who]["nca"])
            z, p = nca.paired_auc_test(h, b)
            out["delong_hybrid_vs_B1"][f"{vname}|ALL|{who}"] = {"n": len(h), "auc_hybrid": float(h.mean()),
                                                                 "auc_B1": float(b.mean()), "auc_nca": float(n_.mean()),
                                                                 "z": float(z), "p": float(p)}
    out["repro_max_abs_diff"] = max(abs(a - b) for v in out["repro"].values() for a, b in v.values())
    (R / "m1_diag.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("config", "repro_max_abs_diff", "sanity", "attrition")}, ensure_ascii=False,
                     indent=1, default=float))
    print(json.dumps(out["delong_hybrid_vs_B1"], ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
