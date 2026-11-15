"""Протокол v2, п. 9а и 9б — на итоговой конфигурации основного прогона (энкодер, веса, методы из stage_3a…3d).

  9а  кривые: вложенные подвыборки 25/50/75/100 % постов компаний (поровну по классам) и пользователей корпуса
      (пользователи M1 — всегда); заново выбор k, кластеризация, сопоставление, M1.
  9б  веса реальных долей классов w = доля класса среди эффективных постов P_d / (1/3): взвешенные ē_c, π_{c,a}
      и средний силуэт постов; кластеризация — из 3в без изменений; M1 заново.
Запуск (окружение как у основного прогона): python -m experiments.paper1.mm_s6_checks 9a|9b
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time

import numpy as np
import pandas as pd

from experiments.paper1 import nca
from experiments.paper1.config import DOMAINS, SEED
from experiments.paper1.data import load_encoder, load_sample, m1_candidates
from experiments.paper1.config import META_DIR as OUT
from experiments.paper1.s4_protocol import (WEIGHTS, domain_arrays, load, load_labels_3c, log, matching_eval, save,
                                            select_k, stage_3e)

FRACS = [0.25, 0.5, 0.75, 1.0]


def config():
    enc = load("stage_3a")["best_encoder"]
    w = load("stage_3b")["best_weights"]
    bm = load("stage_3c")["best_method"]
    mm = load("stage_3d")["best_method"]
    return enc, w, bm["posts"], bm["users"], mm


def m1_summary(r):
    out = {}
    for v, x in r["variants"].items():
        out[v] = {k: x["table"][k]["auc"] for k in ("nca", "response", "hybrid", "B1", "B3")}
        out[v]["n_users"] = x["n_users"]
    return out


def check_9a(smp, cands, ed, V_all, enc, w, mc, mu, mm):
    rng = np.random.default_rng(SEED)
    cp = smp["company_posts"].reset_index(drop=True)
    cp["perm"] = 0
    for (d, c), g in cp.groupby(["domain", "cls"]):
        cp.loc[g.index, "perm"] = rng.permutation(len(g))
    cor = smp["corpus_users"].reset_index(drop=True)
    cor["perm"] = 0
    for d, g in cor.groupby("domain"):
        cor.loc[g.index, "perm"] = rng.permutation(len(g))
    size_c = cp.groupby(["domain", "cls"]).size()
    size_u = cor.groupby("domain").size()
    res = {}
    for f in FRACS:
        t = time.time()
        cps = cp[cp.perm < cp.set_index(["domain", "cls"]).index.map(lambda k: round(f * size_c[k]))]
        cos = cor[cor.perm < cor.domain.map(lambda d: round(f * size_u[d]))]
        smp2 = dict(smp, company_posts=cps.drop(columns="perm"), corpus_users=cos.drop(columns="perm"))
        ed2 = dataclasses.replace(ed, cp=cps[["post_id", "gid", "domain"]].reset_index(drop=True))
        labels, per = {}, {}
        for d in DOMAINS:
            cpd, E, users, V = domain_arrays(ed2, smp2, d, V_all)
            Dc, Du = nca.angular_dist(E), nca.angular_dist(V)
            sc, su = select_k(E, mc, Dc), select_k(V, mu, Du)
            mt = matching_eval(E, V, sc["labels"], su["labels"], method=mm, P_c=sc["P"], P_u=su["P"])
            labels[d] = (sc["labels"], sc["P"], su["labels"], su["P"])
            per[d] = {"n_posts": len(E), "n_users": len(V), "k_c": sc["best"]["k"], "k_u": su["best"]["k"],
                      "sil_posts": sc["best"]["sil"], "sil_users": su["best"]["sil"],
                      "max_share_posts": sc["best"]["max_share"], "max_share_users": su["best"]["max_share"],
                      "M_size": int(mt["M"].sum()), "sum_tas": float((mt["TAS"] * mt["M"]).sum())}
        r3e = stage_3e(smp2, enc, w, mc, mu, mm, cands, labels=labels, V_all=V_all, name=f"check_9a_m1_{f}", ed=ed2)
        res[str(f)] = {"per_domain": per, "m1": m1_summary(r3e),
                       "sil_posts_mean": float(np.mean([per[d]["sil_posts"] for d in DOMAINS])),
                       "sil_users_mean": float(np.mean([per[d]["sil_users"] for d in DOMAINS])),
                       "M_size_total": sum(per[d]["M_size"] for d in DOMAINS),
                       "sum_tas_total": float(sum(per[d]["sum_tas"] for d in DOMAINS)), "sec": round(time.time() - t)}
        log(f"9a {f}: sil_c={res[str(f)]['sil_posts_mean']:.3f} sil_u={res[str(f)]['sil_users_mean']:.3f} "
            f"|M|={res[str(f)]['M_size_total']} AUC nca={res[str(f)]['m1']['main']['nca'][0]:.3f}")
    save("check_9a", {"fractions": FRACS, "config": [enc, w, mc, mu, mm], "results": res})


def check_9b(smp, cands, ed, V_all, enc, w, mc, mu, mm):
    share = {d: v["real_share"] for d, v in json.loads((OUT / "sample_stats.json").read_text())["domains"].items()}
    cp = ed.cp.merge(smp["company_posts"][["post_id", "domain", "cls"]].drop_duplicates(["post_id", "domain"]),
                     on=["post_id", "domain"], how="left")
    cp["w"] = [share[d][c] * 3 for d, c in zip(cp.domain, cp.cls)]
    ed3 = dataclasses.replace(ed, cp=cp)
    sil = {}
    for d in DOMAINS:
        cpd, E, users, V = domain_arrays(ed3, smp, d, V_all)
        lab, _ = load_labels_3c(d, "posts", mc)
        s = nca.silhouette_precomputed(nca.angular_dist(E), lab, per_point=True)
        sil[d] = {"unweighted": float(s.mean()), "weighted": float(np.average(s, weights=cpd.w.to_numpy())),
                  "weights": share[d]}
    r3e = stage_3e(smp, enc, w, mc, mu, mm, cands, name="check_9b_m1", ed=ed3, V_all=V_all)
    save("check_9b", {"config": [enc, w, mc, mu, mm], "silhouette_posts": sil, "m1": m1_summary(r3e)})
    log(f"9b: {json.dumps(m1_summary(r3e)['main'], default=float)}")


def check_sens_corpus_full(smp, cands, ed, V_all, enc, w, mc, mu, mm):
    """Решение 10.11 — чувствительность: корпус = только пользователи с полным мультимодальным профилем
    (все медиа постов 1–5 закодированы). Итоговая конфигурация та же; кластеры постов компаний — из 3в (посты
    не меняются); выбор k и кластеризация профилей, сопоставление и M1 — заново. Пользователи M1 — все."""
    from experiments.paper1.config import WORK_DIR
    full = set(pd.read_pickle(WORK_DIR / "sample" / "corpus_full_users.pkl").user_id)
    cor = smp["corpus_users"]
    smp2 = dict(smp, corpus_users=cor[cor.user_id.isin(full)])
    labels, per = {}, {}
    for d in DOMAINS:
        cpd, E, users, V = domain_arrays(ed, smp2, d, V_all)
        lab_c, P_c = load_labels_3c(d, "posts", mc)
        su = select_k(V, mu, nca.angular_dist(V))
        mt = matching_eval(E, V, lab_c, su["labels"], method=mm, P_c=P_c, P_u=su["P"])
        labels[d] = (lab_c, P_c, su["labels"], su["P"])
        per[d] = {"n_corpus_full": int((smp2["corpus_users"].domain == d).sum()), "n_users": len(V),
                  "k_u": su["best"]["k"], "sil_users": su["best"]["sil"], "max_share_users": su["best"]["max_share"],
                  "M_size": int(mt["M"].sum()), "sum_tas": float((mt["TAS"] * mt["M"]).sum())}
        log(f"sens {d}: {per[d]}")
    r3e = stage_3e(smp2, enc, w, mc, mu, mm, cands, labels=labels, V_all=V_all, name="sens_corpus_full_m1", ed=ed)
    save("sens_corpus_full", {"config": [enc, w, mc, mu, mm], "per_domain": per, "m1": m1_summary(r3e)})
    log(f"sens: {json.dumps(m1_summary(r3e)['main'], default=float)}")


def main() -> None:
    what = sys.argv[1]
    smp = load_sample()
    cands = m1_candidates(smp)
    enc, w, mc, mu, mm = config()
    ed = load_encoder(enc, smp)
    V_all = ed.profiles(WEIGHTS[w])
    {"9a": check_9a, "9b": check_9b, "sens": check_sens_corpus_full}[what](smp, cands, ed, V_all, enc, w, mc, mu, mm)


if __name__ == "__main__":
    main()
