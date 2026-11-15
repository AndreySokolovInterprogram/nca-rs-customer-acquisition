"""Загрузка выборки и эмбеддингов, сборка компонентов профилей пользователей."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from experiments.paper1 import nca
from experiments.paper1.config import DOMAINS, SEED, WORK_DIR

BASE_WEIGHTS = (0.70, 0.20, 0.10)
N_NEG = 99


@dataclass
class EncoderData:
    encoder: str
    cp: pd.DataFrame                 # посты компаний: post_id, gid, domain (строка на пару пост—область)
    E_post: dict                     # post_id -> строка эмбеддинга
    E: np.ndarray                    # эмбеддинги постов компаний (уникальные post_id)
    user_index: dict                 # user_id -> строка в e_posts/...
    e_posts: np.ndarray
    e_subs: np.ndarray
    e_demo: np.ndarray
    has_subs: np.ndarray
    q_demo: np.ndarray
    sub_vecs: dict = field(default_factory=dict)

    def profiles(self, weights=BASE_WEIGHTS) -> np.ndarray:
        return nca.acr_profiles(self.e_posts, self.e_subs, self.e_demo, self.has_subs, self.q_demo, weights)


def load_sample():
    s = WORK_DIR / "sample"
    return {k: pd.read_pickle(s / f"{k}.pkl") for k in
            ["company_posts", "pools", "m1_users", "corpus_users", "seed_users", "user_posts",
             "user_subs", "sub_communities", "demo", "pool_info", "pool_keywords", "user_links"]}


def load_encoder(encoder: str, smp: dict) -> EncoderData:
    d = WORK_DIR / "emb" / encoder
    ids = {n: np.load(d / f"{n}_ids.npy") for n in ["company_posts", "user_posts", "sub_communities", "demo"]}
    cp = smp["company_posts"][["post_id", "gid", "domain"]].reset_index(drop=True)
    E_post = {int(p): i for i, p in enumerate(ids["company_posts"])}
    E = nca.l2n(np.load(d / "company_posts.npy"))
    users = sorted(set(smp["m1_users"].user_id) | set(smp["corpus_users"].user_id) | set(smp["seed_users"].user_id))
    uidx = {u: i for i, u in enumerate(users)}
    cache = d / "components.npz"
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["users"], np.array(users)):
            return EncoderData(encoder, cp, E_post, E, uidx, z["e_posts"], z["e_subs"], z["e_demo"],
                               z["has_subs"], z["q_demo"])
    emb = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in ids}
    dim = E.shape[1]
    # посты пользователя — среднее эмбеддингов (формула user-posts)
    up = smp["user_posts"]
    prow = {int(p): i for i, p in enumerate(ids["user_posts"])}
    e_posts = np.zeros((len(users), dim), dtype=np.float32)
    cnt = np.zeros(len(users))
    U = np.asarray(emb["user_posts"])
    for u, p in zip(up.user_id.to_numpy(), up.post_id.to_numpy()):
        if u in uidx:
            e_posts[uidx[u]] += U[prow[int(p)]]
            cnt[uidx[u]] += 1
    e_posts /= np.clip(cnt[:, None], 1, None)
    # подписки — среднее эмбеддингов сообществ (формула user-subs)
    crow = {int(v): i for i, v in enumerate(ids["sub_communities"])}
    C = np.asarray(emb["sub_communities"])
    subs = smp["user_subs"]
    e_subs = np.zeros((len(users), dim), dtype=np.float32)
    scnt = np.zeros(len(users))
    for u, v in zip(subs.user_id.to_numpy(), subs.vk_group_id.to_numpy()):
        r = crow.get(int(v))
        if r is not None and u in uidx:
            e_subs[uidx[u]] += C[r]
            scnt[uidx[u]] += 1
    has_subs = scnt > 0
    e_subs = np.where(has_subs[:, None], e_subs / np.clip(scnt[:, None], 1, None), np.nan)
    # демография
    drow = {int(u): i for i, u in enumerate(ids["demo"])}
    Dm = np.asarray(emb["demo"])
    demo = smp["demo"].set_index("user_id")
    e_demo = np.full((len(users), dim), np.nan, dtype=np.float32)
    q_demo = np.zeros(len(users))
    for u, i in uidx.items():
        r = drow.get(int(u))
        if r is not None:
            e_demo[i] = Dm[r]
            q_demo[i] = demo.at[u, "r"] / 8.0
    np.savez(cache, users=np.array(users), e_posts=e_posts, e_subs=e_subs, e_demo=e_demo,
             has_subs=has_subs, q_demo=q_demo)
    return EncoderData(encoder, cp, E_post, E, uidx, e_posts, e_subs, e_demo, has_subs, q_demo)


def m1_candidates(smp: dict) -> dict:
    """Для каждой области: пользователи M1, скрытое сообщество и 99 случайных сообществ пула
    той же области, в которых пользователь не состоит (зерно 42)."""
    rng = np.random.default_rng(SEED)
    links = smp["user_links"].groupby("user_id").gid.apply(set)
    out = {}
    for d in DOMAINS:
        pool = np.array(smp["pools"][d])
        rows = smp["m1_users"][smp["m1_users"].domain == d]
        users, cands = [], []
        for u, c in zip(rows.user_id, rows.hidden_gid):
            avail = np.setdiff1d(pool, list(links.get(u, set())))
            if len(avail) < N_NEG:
                continue
            neg = rng.choice(avail, size=N_NEG, replace=False)
            users.append(int(u))
            cands.append(np.concatenate([[c], neg]))
        out[d] = {"users": np.array(users), "cands": np.array(cands)}
    return out


def m1_candidates_decile(smp: dict) -> dict:
    """Протокол v2, п. 8а: отрицательные — до 99 случайных сообществ пула области из того же дециля
    числа участников (vk_communities.members_count), что и скрытое; пользователь в них не состоит.
    Если доступно меньше 99 — все доступные; недостающие позиции = −1 (метрики их пропускают)."""
    from scipy.stats import rankdata

    rng = np.random.default_rng(SEED + 1)
    links = smp["user_links"].groupby("user_id").gid.apply(set)
    info = smp["pool_info"].drop_duplicates("gid").set_index("gid").members_count
    out = {}
    for d in DOMAINS:
        pool = np.array(smp["pools"][d])
        mem = info.reindex(pool).fillna(0).to_numpy(float)
        dec = np.minimum(9, ((rankdata(mem, "average") - 1) * 10 // len(pool)).astype(int))
        dec_of = dict(zip(pool.tolist(), dec.tolist()))
        rows = smp["m1_users"][smp["m1_users"].domain == d]
        users, cands, n_neg = [], [], []
        for u, c in zip(rows.user_id, rows.hidden_gid):
            same = pool[dec == dec_of[c]]
            avail = np.setdiff1d(same, list(links.get(u, set())) + [c])
            if len(avail) == 0:
                continue
            neg = rng.choice(avail, size=min(N_NEG, len(avail)), replace=False)
            users.append(int(u))
            cands.append(np.concatenate([[c], neg, -np.ones(N_NEG - len(neg), dtype=int)]))
            n_neg.append(len(neg))
        out[d] = {"users": np.array(users), "cands": np.array(cands, dtype=int), "n_neg": np.array(n_neg)}
    return out
