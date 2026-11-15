"""Протокол v2, M1 (решение 10.11): выбывание — пользователи M1 с полным и неполным мультимодальным профилем.

Полный = у всех постов 1–5 с фото/видео закодировано медиа своего класса (список — sample/m1_full_users.pkl из
mm_s5_finalize). Признаки — из выборки (снимок 17.10): область, демография (строка профиля: пол, город, образование,
подписчики, год рождения), посты за год с собственным содержимым (в выборке — до 10 последних), состав постов 1–5,
подписки, связи с одобренными сообществами, давность последнего поста, размер скрытого сообщества.
Тесты: Манн–Уитни (числовые), χ² (категориальные); SMD = разность средних / объединённое SD.
Внимание при чтении: состав постов (фото/видео) связан с неполнотой по построению — неполным может быть только
пользователь, у которого есть пост с медиа.

Запуск: PAPER1_WORK_DIR=…/main PAPER1_MM_RESULTS=… python -m experiments.paper1.mm_s13_attrition
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from experiments.paper1.config import SNAPSHOT_AT, SRC, WORK_DIR

R = Path(os.environ["PAPER1_MM_RESULTS"])
# SRC — исходные таблицы выборки (config.PAPER1_SRC_DIR)


def demo_features(demo: pd.DataFrame) -> pd.DataFrame:
    def one(t: str) -> dict:
        f = dict(re.findall(r"(?:^|\. )([А-ЯЁ][а-яё ]+): (.*?)(?=\. [А-ЯЁ][а-яё ]+: |$)", t or ""))
        by = re.search(r"\d{1,2}\.\d{1,2}\.(\d{4})", f.get("Дата рождения", ""))
        fol = re.match(r"\d+", f.get("Подписчиков", ""))
        return {"female": {"женский": 1.0, "мужской": 0.0}.get(f.get("Пол", "").strip(" ."), np.nan),
                "age": SNAPSHOT_AT.year - int(by.group(1)) if by else np.nan,
                "city_given": float("Город" in f), "edu_given": float("Образование" in f),
                "followers": float(fol.group()) if fol else np.nan}
    return pd.DataFrame([one(t) for t in demo.text], index=demo.user_id)


def features() -> pd.DataFrame:
    m1 = pd.read_pickle(SRC / "m1_users.pkl").set_index("user_id")
    full = set(pd.read_pickle(WORK_DIR / "sample" / "m1_full_users.pkl").user_id)
    kept = set(pd.read_pickle(WORK_DIR / "sample" / "m1_users.pkl").user_id)
    up = pd.read_pickle(SRC / "user_posts_src.pkl").drop_duplicates(["user_id", "post_id"])
    up = up[up.user_id.isin(m1.index)].copy()
    up["rank"] = up.groupby("user_id").published_at.rank(method="first", ascending=False)
    t5 = up[up["rank"] <= 5]
    f = pd.DataFrame(index=m1.index)
    f["domain"] = m1.domain
    f["full"] = f.index.isin(full)
    f["in_m1_sample"] = f.index.isin(kept)
    f["posts_year_upto10"] = up.groupby("user_id").size()
    f["posts_1_5"] = t5.groupby("user_id").size()
    for c in ("image", "video", "text"):
        f[f"n_{c}_1_5"] = t5[t5.cls == c].groupby("user_id").size()
    f = f.fillna({c: 0 for c in f.columns if c.startswith(("n_", "posts_"))})
    for c in ("image", "video", "text"):
        f[f"share_{c}_1_5"] = f[f"n_{c}_1_5"] / f.posts_1_5.clip(lower=1)
    last = up.groupby("user_id").published_at.max()
    f["days_since_last_post"] = (pd.Timestamp(SNAPSHOT_AT) - pd.to_datetime(last, utc=True)).dt.days
    f["n_subs"] = pd.read_pickle(SRC / "user_subs.pkl").groupby("user_id").size().reindex(f.index).fillna(0)
    f["n_approved_links"] = pd.read_pickle(SRC / "user_links.pkl").groupby("user_id").size().reindex(f.index).fillna(0)
    demo = pd.read_pickle(SRC / "demo.pkl")
    f = f.join(demo_features(demo[demo.user_id.isin(f.index)]))
    f["demo_given"] = f.index.isin(demo.user_id[demo.r > 0])
    info = pd.read_pickle(SRC / "pool_info.pkl").drop_duplicates("gid").set_index("gid").members_count
    f["hidden_members"] = info.reindex(m1.hidden_gid).to_numpy()
    f["log10_followers"] = np.log10(1 + f.followers)
    f["log10_hidden_members"] = np.log10(1 + f.hidden_members)
    return f


NUM = [("posts_year_upto10", "постов за год (в выборке ≤ 10)"), ("posts_1_5", "постов в профиле (1–5)"),
       ("share_image_1_5", "доля постов с фото (1–5)"), ("share_video_1_5", "доля постов с видео (1–5)"),
       ("share_text_1_5", "доля текстовых постов (1–5)"), ("n_video_1_5", "видеопостов в профиле"),
       ("days_since_last_post", "дней с последнего поста"), ("female", "доля женщин"), ("age", "возраст, лет"),
       ("city_given", "указан город"), ("edu_given", "указано образование"), ("log10_followers", "log10(1 + подписчиков)"),
       ("n_subs", "подписок (сообществ)"), ("n_approved_links", "связей с одобренными сообществами"),
       ("log10_hidden_members", "log10(1 + участников скрытого сообщества)")]


def compare(f: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    a, b = f[f.full], f[~f.full]
    rows = []
    for c, name in NUM:
        x, y = a[c].dropna(), b[c].dropna()
        sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        p = mannwhitneyu(x, y).pvalue if len(x) and len(y) and x.nunique() + y.nunique() > 2 else np.nan
        rows.append({"признак": name, "полные": round(float(x.mean()), 3), "неполные": round(float(y.mean()), 3),
                     "медиана полн.": round(float(x.median()), 3), "медиана неполн.": round(float(y.median()), 3),
                     "SMD": round(float((x.mean() - y.mean()) / sd), 3) if sd > 0 else np.nan,
                     "p (Манн–Уитни)": float(f"{p:.2g}") if np.isfinite(p) else np.nan,
                     "n полн./неполн.": f"{len(x)}/{len(y)}"})
    ct = pd.crosstab(f.domain, f.full)
    chi = chi2_contingency(ct)
    cat = {"domain_counts": {d: {"full": int(r[True]), "incomplete": int(r[False])} for d, r in ct.iterrows()},
           "domain_chi2_p": float(chi.pvalue)}
    return pd.DataFrame(rows), cat


def main() -> None:
    f = features()
    tab, cat = compare(f)
    vid = f[f.n_video_1_5 > 0]  # только пользователи с видеопостом: сравнение без механической связи «нет медиа»
    tab_v, cat_v = compare(vid) if vid.full.any() and (~vid.full).any() else (None, None)
    out = {"n": {"m1": len(f), "full": int(f.full.sum()), "incomplete": int((~f.full).sum()),
                 "in_m1_sample_after_finalize": int(f.in_m1_sample.sum())},
           "all": {"table": tab.to_dict("records"), **cat},
           "with_video_post": None if tab_v is None else {"n_full": int(vid.full.sum()), "n_incomplete": int((~vid.full).sum()),
                                                            "table": tab_v.to_dict("records"), **cat_v}}
    (R / "attrition_m1.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    tab.to_csv(R / "attrition_m1.csv", index=False)
    if tab_v is not None:
        tab_v.to_csv(R / "attrition_m1_with_video.csv", index=False)
    print(json.dumps(out["n"]), "\n", tab.to_string(index=False), "\n", json.dumps(cat, ensure_ascii=False))


if __name__ == "__main__":
    main()
