"""Сводка: таблицы раздела «Результаты» статьи №1 в Markdown и CSV из JSON этапов.

Порядок и строки — как в sections/04-results.tex. Формат «x [a; b]» — точечная оценка и 95 % бутстреп-интервал.
Запуск: python -m experiments.paper1.s7_tables  → results/.../tables.md и tab_res_*.csv
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.paper1.config import DOMAINS, RESULTS_DIR

SHORT = {"customer_acquisition": "cust. acq.", "food_delivery": "food deliv.",
         "food_service": "food serv.", "tender_support": "tender"}
ENC = {"qwen": "Qwen3-VL-Embedding-2B", "jina": "jina-embeddings-v5-omni-small"}
MATCH = {"threshold": "Пороговый", "greedy": "Жадный", "ot": "Оптимальный транспорт",
         "bicluster": "Бикластеризация", "ipfp": "Мягкое (IPFP)"}
CLU = {"ward": "Ward", "average": "average", "complete": "complete", "gmm": "Bayesian GMM"}
M1 = [("B0", "B0: случайная"), ("B1", "B1: популярность"), ("B2", "B2: ключевые слова"),
      ("B3", "B3: look-alike (proxy-seed)"), ("response", "Модель отклика ⟨v_j, ē_c⟩"), ("nca", "NCA-RS"),
      ("hybrid", "Гибрид NCA-RS + B1 (средний нормир. ранг)")]
M1_VAR = {"main": "1 скрытое против 99 той же области", "size_decile": "вариант (а) «по размеру»: отрицательные из того же дециля числа участников"}


def J(name):
    p = RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ci(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return f"{x[0]:.{nd}f} [{x[1]:.{nd}f}; {x[2]:.{nd}f}]"
    return f"{x:.{nd}f}"


def pct(x):
    return f"{100 * x:.1f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def matching_m1_sections(suf: str, title: str) -> list:
    out = []
    # --- tab:res-matching
    m = J(f"stage_3d{suf}")
    if m:
        rows = []
        for k in MATCH:
            rs = [m["per_domain"][x]["methods"][k] for x in DOMAINS]
            rows.append({"Метод": MATCH[k], "|M|": sum(r["M_size"] for r in rs),
                         "ρ(M) (среднее)": f"{np.mean([r['rho'] for r in rs]):.3f}",
                         "Суммарный TAS": f"{sum(r['sum_tas'] for r in rs):.2f}",
                         "p по областям": " / ".join(f"{r['perm_p']:.3f}" for r in rs),
                         "Нулевое среднее Σ TAS": f"{sum(r['null_mean'] for r in rs):.2f}"})
        t = pd.DataFrame(rows)
        t.to_csv(RESULTS_DIR / f"tab_res_matching{suf}.csv", index=False)
        w = pd.DataFrame([{"Пара": f"{MATCH[r['a']]} — {MATCH[r['b']]}", "медиана разности": f"{r['median_diff']:.3f}",
                           "n": r["n"], "p": f"{r['p']:.4f}", "p (BH)": f"{r['p_bh']:.4f}"} for r in m["wilcoxon"]])
        w.to_csv(RESULTS_DIR / f"tab_res_matching_wilcoxon{suf}.csv", index=False)
        out += [f"### {title}Таблица tab:res-matching — α = {m['alpha']}, ρ_max = {m['rho_max']}; суммы по 4 областям", md(t),
                f"Выбран метод: **{MATCH[m['best_method']]}** ({m['selection_rule']}).",
                "Попарные сравнения (Уилкоксон, пары «область × α», одинаковое ρ = 1 − α; поправка BH):", md(w)]

    # --- tab:res-m1
    e0 = J(f"stage_3e{suf}")
    for vname, e in ((e0.get("variants") or {"main": e0}).items() if e0 else []):
        if "@" in vname:  # подмножества (полные профили M1) — в отчёте mm_s14_m1_report
            continue
        rows = []
        for k, name in [(k, n) for k, n in M1 if k in e["table"]]:
            r = e["table"][k]
            dl = "—" if r["delong_vs_nca"] is None else f"{r['delong_vs_nca'][1]:.2g} (BH {r['delong_vs_nca_p_bh']:.2g})"
            rows.append({"Стратегия": name, "AUC": ci(r["auc"]), "Recall@10": ci(r["recall10"]), "MRR": ci(r["mrr"]),
                         "nDCG@10": ci(r["ndcg10"]), "p (ДеЛонг, против NCA-RS)": dl,
                         "p (AUC = 0,5)": f"{r['vs_0.5'][1]:.2g}"})
        t = pd.DataFrame(rows)
        vs = "" if vname == "main" else f"_{vname}"
        t.to_csv(RESULTS_DIR / f"tab_res_m1{suf}{vs}.csv", index=False)
        per = pd.DataFrame([{"Область": SHORT[x], **{n: f"{e['per_domain'][x][k]['auc']:.3f}" for k, n in M1 if k in e["table"]},
                             **({"отрицательных (сред./мин.)": f"{e['per_domain'][x]['n_neg_mean']:.1f} / {e['per_domain'][x]['n_neg_min']}"}
                                if "n_neg_mean" in e["per_domain"][x] else {})} for x in DOMAINS])
        per.to_csv(RESULTS_DIR / f"tab_res_m1_per_domain{suf}{vs}.csv", index=False)
        out += [f"### {title}Таблица tab:res-m1 — {e['n_users']} пользователей, {M1_VAR[vname]}", md(t),
                "AUC по областям:", md(per), f"Покрытие: {e['coverage']}"]

    return out


def main() -> None:
    out = []
    # --- tab:res-data
    # агрегаты по исходной базе (s1_data_characteristics; в публичную версию не входит) — таблица необязательна
    if (RESULTS_DIR / "tab_res_data.csv").exists():
        d = pd.read_csv(RESULTS_DIR / "tab_res_data.csv").set_index("label")
        cols = DOMAINS + ["TOTAL"]
        rows = [("Сообществ", lambda c: f"{int(d.at[c, 'communities']):,}"),
                ("Постов компаний", lambda c: f"{int(d.at[c, 'posts']):,}"),
                ("  из них с изображением, %", lambda c: pct(d.at[c, "share_photo"])),
                ("  из них с видео, %", lambda c: pct(d.at[c, "share_video"])),
                ("Пользователей", lambda c: f"{int(d.at[c, 'users']):,}"),
                ("  с постами за год, % (от всех)", lambda c: pct(d.at[c, "share_users_posts_year"])),
                ("  с постами за год, % (от тех, у кого посты собирались)", lambda c: pct(d.at[c, "share_posts_year_among_collected"])),
                ("Связей «пользователь — сообщество»", lambda c: f"{int(d.at[c, 'links']):,}")]
        t = pd.DataFrame([{"Показатель": n, **{(SHORT.get(c, "Всего")): f(c) for c in cols}} for n, f in rows])
        t.to_csv(RESULTS_DIR / "tab_res_data_formatted.csv", index=False)
        out += ["### Таблица tab:res-data — характеристики данных", md(t).replace(",", " ")]

    # --- tab:res-encoders
    a = J("stage_3a")
    if a:
        rows = []
        for e in [x for x in ("qwen", "jina") if x in a]:
            r = a[e]
            mean = lambda f: float(np.mean([f(r[x]) for x in DOMAINS]))
            rows.append({"Энкодер": ENC[e],
                         "H (среднее по обл.)": f"{mean(lambda z: z['m3']['hopkins'][0]):.3f}",
                         "H, PCA-10": f"{mean(lambda z: z['m3']['hopkins_pca10'][0]):.3f}",
                         "Эфф. размерность 95 %": f"{mean(lambda z: z['m3']['eff_dim95'][0]):.0f}",
                         "Силуэт (Ward)": f"{mean(lambda z: z['m4_ward_posts']['silhouette'][0]):.3f}",
                         "CH": f"{mean(lambda z: z['m4_ward_posts']['ch'][0]):.1f}",
                         "k (Ward)": "/".join(str(r[x]["k_posts"]["k"]) for x in DOMAINS),
                         "AUC M1 (NCA-RS)": f"{mean(lambda z: z['auc_nca']):.3f}"})
        out += ["### Таблица tab:res-encoders — среднее по четырём областям", md(pd.DataFrame(rows))]
        per = []
        for e in [x for x in ("qwen", "jina") if x in a]:
            for x in DOMAINS:
                z = a[e][x]
                per.append({"Энкодер": ENC[e], "Область": SHORT[x], "H": ci(z["m3"]["hopkins"]),
                            "H, PCA-10": ci(z["m3"]["hopkins_pca10"]), "Эфф. разм. (подвыборки 50 %)": f"{z['m3']['eff_dim95'][0]:.0f} ({z['m3']['eff_dim95'][1]:.0f}–{z['m3']['eff_dim95'][2]:.0f})",
                            "Силуэт Ward": ci(z["m4_ward_posts"]["silhouette"]), "CH": ci(z["m4_ward_posts"]["ch"], 1),
                            "k": z["k_posts"]["k"], "Σ TAS": f"{z['sum_tas']:.2f}", "|M|": z["M_size"],
                            "AUC NCA-RS": f"{z['auc_nca']:.3f}", "AUC отклик": f"{z['auc_response']:.3f}"})
        pt = pd.DataFrame(per)
        pt.to_csv(RESULTS_DIR / "tab_res_encoders_per_domain.csv", index=False)
        out += ["Подробно по областям (95 % бутстреп-интервалы):", md(pt)]
        if "m5" in a:
            m5 = pd.DataFrame([{"Область": SHORT[x], **{k: f"{v:.3f}" for k, v in a["m5"][x].items()}} for x in DOMAINS])
            m5.to_csv(RESULTS_DIR / "tab_res_encoders_m5.csv", index=False)
            out += ["Согласие энкодеров (M5):", md(m5)]
        out += [f"Выбран энкодер: **{ENC[a['best_encoder']]}** ({a['selection_rule']})."]

    # --- tab:res-weights
    b = J("stage_3b")
    if b:
        rows = [{"Веса (посты; подписки; демография)": w, "Силуэт T(U)": f"{v['silhouette_users_mean']:.3f}",
                 "Суммарный TAS": f"{v['sum_tas_total']:.2f}", "AUC M1 (NCA-RS)": ci(v["auc_nca"]),
                 "AUC M1 (модель отклика)": ci(v["auc_response"]),
                 "k_U по областям": "/".join(str(v["per_domain"][x]["k_users"]["k"]) for x in DOMAINS)}
                for w, v in b["by_weights"].items()]
        t = pd.DataFrame(rows)
        t.to_csv(RESULTS_DIR / "tab_res_weights.csv", index=False)
        out += ["### Таблица tab:res-weights", md(t), f"Выбраны веса: **{b['best_weights']}** ({b['selection_rule']})."]

    # --- tab:res-clustering
    c = J("stage_3c")
    if c:
        rows = []
        for corpus, name in (("posts", "Посты"), ("users", "Профили")):
            for m in ("ward", "average", "complete", "gmm"):
                rs = [c["results"][f"{x}|{corpus}|{m}"] for x in DOMAINS]
                bic = "—"
                if m == "gmm":
                    bic = f"{np.mean([r['bic_icl'][0] for r in rs]):.0f} / {np.mean([r['bic_icl'][1] for r in rs]):.0f}"
                rows.append({"Корпус": name, "Метод": CLU[m], "k": "/".join(str(r["k"]) for r in rs),
                             "Силуэт": f"{np.mean([r['silhouette'][0] for r in rs]):.3f}",
                             "CH": f"{np.mean([r['ch'][0] for r in rs]):.1f}", "BIC / ICL": bic,
                             "Стабильность J̄": f"{np.mean([r['jaccard_mean'] for r in rs]):.3f}",
                             "Доля кластеров J≥0,75": f"{np.mean([r['share_stable_clusters'] for r in rs]):.2f}",
                             "ARI бутстреп": f"{np.mean([r['boot_ari_mean'] for r in rs]):.3f}",
                             "Gap (PCA-box)": f"{np.mean([r['gap']['pca_box']['gap'] for r in rs]):.3f}",
                             "Силуэт нуля (PCA-box)": f"{np.mean([r['gap']['pca_box']['silhouette_null_mean'] for r in rs]):.3f}",
                             "Gap (сфера)": f"{np.mean([r['gap']['sphere_uniform']['gap'] for r in rs]):.3f}"})
        # доля крупнейшего кластера — индикатор вырожденных разбиений («один большой + выбросы»)
        from experiments.paper1.config import WORK_DIR
        for row, (corpus, m) in zip(rows, [(cc, mm) for cc in ("posts", "users") for mm in ("ward", "average", "complete", "gmm")]):
            shares = []
            for x in DOMAINS:
                lab = np.load(WORK_DIR / f"labels_3c_{x}_{corpus}_{m}.npz")["labels"]
                shares.append(np.bincount(lab).max() / len(lab))
            row["Доля крупнейшего кластера"] = "/".join(f"{v:.2f}" for v in shares)
        t = pd.DataFrame(rows)
        t.to_csv(RESULTS_DIR / "tab_res_clustering.csv", index=False)
        per = pd.DataFrame([{"Область": SHORT[k.split('|')[0]], "Корпус": k.split('|')[1], "Метод": k.split('|')[2],
                             "k": v["k"], "Силуэт": ci(v["silhouette"]), "CH": ci(v["ch"], 1),
                             "BIC/ICL": "—" if not v["bic_icl"] else f"{v['bic_icl'][0]:.0f} / {v['bic_icl'][1]:.0f}",
                             "J̄": f"{v['jaccard_mean']:.3f}", "gap PCA-box": f"{v['gap']['pca_box']['gap']:.3f} ± {v['gap']['pca_box']['s_k']:.3f}",
                             "gap сфера": f"{v['gap']['sphere_uniform']['gap']:.3f}"} for k, v in c["results"].items()])
        per.to_csv(RESULTS_DIR / "tab_res_clustering_per_domain.csv", index=False)
        out += ["### Таблица tab:res-clustering — среднее по областям; k по областям через «/»", md(t),
                f"Выбрано: посты — **{CLU[c['best_method']['posts']]}**, профили — **{CLU[c['best_method']['users']]}** ({c['selection_rule']}).",
                "Подробно по областям:", md(per)]

    out += matching_m1_sections("", "")
    out += matching_m1_sections("_ward", "Чувствительность (Ward для постов и профилей). ")
    # --- пермские кейсы
    f = J("tab_res_perm_funnel")
    if f:
        out += ["### Таблица tab:res-perm-funnel", md(pd.read_csv(RESULTS_DIR / "tab_res_perm_funnel.csv")),
                f"Уникальных кандидатов в двух кейсах: {f['unique_candidates_total']}, открытых: {f['open_candidates_total']}; "
                f"общих конкурентов: {f['competitors_common']}. {f['note']}"]
    (RESULTS_DIR / "tables.md").write_text("\n\n".join(out) + "\n")
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
