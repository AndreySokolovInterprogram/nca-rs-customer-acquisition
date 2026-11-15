"""Протокол v2: сборка результатов конвейера в формат v1 (data.load_sample / load_encoder).

  MM_WORK/main/{sample,emb/qwen}  — основной вариант (ERR ≥ медианы)
  MM_WORK/q3/{sample,emb/qwen}    — вариант п. 9г (ERR ≥ Q3; тот же пул, пользователи и M1)
Пост компании берётся, если его медиа получено и закодировано (замены — следующими по перестановке).
Пользователи корпуса/M1/seed без единого закодированного поста выбывают (число — в media_stats.json).
Запуск: python -m experiments.paper1.mm_s5_finalize
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from experiments.paper1.config import DOMAINS
from experiments.paper1.config import META_DIR as OUT
from experiments.paper1.config import SRC
from experiments.paper1.mm_s4_pipeline import MANIFEST, PARTS, W

COPY = ["pools", "sub_communities", "demo", "pool_info", "pool_keywords", "user_links", "user_subs"]


def load_parts():
    embs, uids = [], []
    for p in sorted(PARTS.glob("p*.npy")):
        embs.append(np.load(p))
        uids += json.loads(p.with_suffix(".json").read_text())
    E = np.concatenate(embs)
    idx = {u: i for i, u in enumerate(uids)}  # при повторе uid (перезапуск порции) берётся последний
    return E, idx


def main() -> None:
    man = pd.DataFrame([json.loads(x) for x in MANIFEST.read_text().splitlines()]).drop_duplicates("uid", keep="last")
    E, idx = load_parts()
    # только векторы постов, которые в манифесте (последняя запись) отмечены закодированными: откаченные записи
    # (18.10: истёкшие ссылки, ограничения VK) оставили в частях старые векторы — они не используются
    emb_ok = set(man.uid[man.get("embedded", pd.Series(False, index=man.index)).fillna(False).astype(bool)])
    idx = {u: i for u, i in idx.items() if u in emb_ok}
    man["has_emb"] = man.uid.isin(idx.keys())
    stats = json.loads((OUT / "sample_stats.json").read_text())["domains"]
    order = pd.read_pickle(SRC / "company_order.pkl")
    order["uid"] = "company:" + order.post_id.astype(str)
    okset = set(man.uid[man.has_emb & (man.status == "ok")])
    order["ok"] = order.uid.isin(okset)
    ms: dict = {"company": {}, "users": {}}

    variants = {}
    for var in ("main", "q3"):
        rows = []
        for d in DOMAINS:
            n = stats[d]["n_main" if var == "main" else "n_q3"]
            for cls in ("text", "image", "video"):
                sub = order[(order.domain == d) & (order.cls == cls)].sort_values("rank")
                if var == "q3":
                    sub = sub[sub.eff_q3]
                tried = sub[sub.uid.isin(man.uid)]
                take = sub[sub.ok].head(n)
                if len(take) < n:
                    raise SystemExit(f"{var} {d}/{cls}: набрано {len(take)} < {n}")
                last = take["rank"].max()
                seen = tried[tried["rank"] <= last]
                ms["company"][f"{var}|{d}|{cls}"] = {"n": n, "looked": len(seen), "replaced": int(len(seen) - n),
                                                     "replace_share": round(1 - n / len(seen), 4)}
                rows.append(take)
        cp = pd.concat(rows)[["post_id", "gid", "domain", "cls", "text", "err", "uid"]]
        variants[var] = cp

    up = pd.read_pickle(SRC / "user_posts_src.pkl")
    up["uid"] = "user:" + up.post_id.astype(str)
    # поправка 6: профиль = 5 последних постов с собственным содержимым (посты 6–10 в профиль не входят)
    up = up.drop_duplicates(["user_id", "post_id"]).copy()
    up["rank"] = up.groupby("user_id").published_at.rank(method="first", ascending=False)
    ms["users"]["posts_rank_gt5_dropped"] = int((up["rank"] > 5).sum())
    up = up[up["rank"] <= 5].drop(columns="rank")
    up_ok = up[up.uid.isin(idx.keys())]
    m = man[man.set == "user"]
    ms["users"]["posts_total"] = int(len(up.drop_duplicates("post_id")))
    ms["users"]["by_class_status"] = m.groupby(["cls", "status"]).size().unstack(fill_value=0).to_dict("index")
    ms["users"]["by_class_modality"] = m[m.has_emb].groupby(["cls", "modality"]).size().unstack(fill_value=0).to_dict("index")
    with_posts = set(up_ok.user_id)
    m1 = pd.read_pickle(SRC / "m1_users.pkl")
    corpus = pd.read_pickle(SRC / "corpus_users.pkl")
    seeds = pd.read_pickle(SRC / "seed_users.pkl")
    ms["users"]["dropped_no_posts"] = {"m1": int((~m1.user_id.isin(with_posts)).sum()),
                                       "corpus": int((~corpus.user_id.isin(with_posts)).sum()),
                                       "seed_rows": int((~seeds.user_id.isin(with_posts)).sum())}
    m1, corpus, seeds = (x[x.user_id.isin(with_posts)] for x in (m1, corpus, seeds))
    # полные мультимодальные профили (все медиа постов 1–5 закодированы) — для M1 «773» и чувствительности по корпусу
    from experiments.paper1.mm_s12_m1_embed import full_users, manifest as man_dict
    mand = man_dict()
    full = {"m1": sorted(full_users(mand, set(m1.user_id))), "corpus": sorted(full_users(mand, set(corpus.user_id)))}
    ms["users"]["full_profiles"] = {k: len(v) for k, v in full.items()}
    ms["users"]["m1_by_domain"] = m1.domain.value_counts().to_dict()
    ms["users"]["corpus_by_domain"] = corpus.domain.value_counts().to_dict()
    ms["company_modality"] = man[(man.set == "company") & man.has_emb].groupby(["cls", "modality"]).size() \
        .unstack(fill_value=0).to_dict("index")
    ms["company_status"] = man[man.set == "company"].groupby(["cls", "status"]).size().unstack(fill_value=0).to_dict("index")
    ms["video_frames"] = man[man.n_frames > 0].groupby("set").n_frames.describe().round(2).to_dict("index")

    sub_ids = [u for u in idx if u.startswith("sub:")]
    demo_ids = [u for u in idx if u.startswith("demo:")]
    for var, cp in variants.items():
        root = W / var
        s, e = root / "sample", root / "emb" / "qwen"
        shutil.rmtree(root, ignore_errors=True)
        s.mkdir(parents=True)
        e.mkdir(parents=True)
        for k in COPY:
            shutil.copy(SRC / f"{k}.pkl", s / f"{k}.pkl")
        cp.drop(columns="uid").to_pickle(s / "company_posts.pkl")
        m1.to_pickle(s / "m1_users.pkl")
        corpus.to_pickle(s / "corpus_users.pkl")
        seeds.to_pickle(s / "seed_users.pkl")
        for k, v in full.items():
            pd.DataFrame({"user_id": v}).to_pickle(s / f"{k}_full_users.pkl")
        up_ok[["user_id", "post_id", "published_at", "text", "cls"]].to_pickle(s / "user_posts.pkl")
        cu = cp.drop_duplicates("post_id")
        sets = {"company_posts": (cu.post_id.to_numpy(), cu.uid.tolist()),
                "user_posts": (up_ok.drop_duplicates("post_id").post_id.to_numpy(),
                               up_ok.drop_duplicates("post_id").uid.tolist()),
                "sub_communities": (np.array([int(u[4:]) for u in sub_ids]), sub_ids),
                "demo": (np.array([int(u[5:]) for u in demo_ids]), demo_ids)}
        for name, (ids, uids) in sets.items():
            np.save(e / f"{name}_ids.npy", ids)
            np.save(e / f"{name}.npy", E[[idx[u] for u in uids]])
        ms[f"{var}_sizes"] = {"company_rows": len(cp), "company_unique": len(cu), "user_posts": len(sets["user_posts"][0]),
                              "sub_communities": len(sub_ids), "demo": len(demo_ids)}
    (OUT / "media_stats.json").write_text(json.dumps(ms, indent=2, ensure_ascii=False, default=int))
    print(json.dumps(ms, indent=2, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
