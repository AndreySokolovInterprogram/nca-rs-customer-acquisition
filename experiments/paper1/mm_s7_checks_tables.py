"""Протокол v2: сводка проверок п. 8–9, долей модальностей и сравнения с текстовым прогоном v1 → results/paper1-mm/checks.md.

Запуск: python -m experiments.paper1.mm_s7_checks_tables
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paper1.config import DOMAINS, META_DIR
from experiments.paper1.s7_tables import CLU, MATCH, SHORT, ci, md

META = META_DIR  # sample_stats / media_stats / video_check
R = Path(os.environ.get("PAPER1_MM_RESULTS", META))  # main/, q3/, seed*/ и checks.md
V1 = Path(os.environ.get("PAPER1_V1_RESULTS", "/nonexistent"))  # необязательно: результаты текстового прогона v1 для сравнения
KEYS = [("nca", "NCA-RS"), ("hybrid", "Гибрид NCA-RS + B1"), ("response", "Модель отклика"), ("B1", "B1 популярность"),
        ("B3", "B3 look-alike")]


def J(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    out = []
    # --- выборка и модальности
    ss = J(META / "sample_stats.json")
    ms = J(META / "media_stats.json")
    if ss:
        rows = [{"Область": SHORT[d], "Пул P_d": v["pool"], "Эфф. ТЕКСТ/ИЗОБР./ВИДЕО в P_d":
                 f"{v['eff_text']} / {v['eff_image']} / {v['eff_video']}", "n_d (осн.)": v["n_main"], "n_d (Q3)": v["n_q3"],
                 "Реальные доли Т/И/В": " / ".join(f"{100 * v['real_share'][c]:.1f}" for c in ("text", "image", "video")),
                 "M1": v["m1"], "Корпус": v["users"]} for d, v in ss["domains"].items()]
        out += ["### Выборка (протокол v2)", md(pd.DataFrame(rows)),
                f"Пользователей: корпус + M1 {ss['users_main']}, только seed {ss['users_seed_only']} (seed-строк {ss['seed_rows']}); "
                f"постов пользователей в выборке {ss['user_posts']} ({ss['user_posts_by_class']})."]
    if ms:
        rows = []
        for key, v in ms["company"].items():
            var, d, c = key.split("|")
            rows.append({"Вариант": var, "Область": SHORT[d], "Класс": c, "n": v["n"], "просмотрено": v["looked"],
                         "замен": v["replaced"], "доля замен, %": f"{100 * v['replace_share']:.1f}"})
        out += ["### Посты компаний: замены недоступных медиа", md(pd.DataFrame(rows))]
        out += ["### Модальности на входе энкодера",
                "Посты компаний (закодированы): " + json.dumps(ms["company_modality"], ensure_ascii=False),
                "Статусы постов компаний: " + json.dumps(ms["company_status"], ensure_ascii=False),
                "Посты пользователей, статусы: " + json.dumps(ms["users"]["by_class_status"], ensure_ascii=False),
                "Посты пользователей, модальности: " + json.dumps(ms["users"]["by_class_modality"], ensure_ascii=False),
                "Выбыли без единого закодированного поста: " + json.dumps(ms["users"]["dropped_no_posts"], ensure_ascii=False),
                "Ключевые кадры видео (describe): " + json.dumps(ms["video_frames"], ensure_ascii=False)]
    vc = J(META / "video_check.json")
    if vc:
        out += ["### Проверка п. 4 (первые 200 видеопостов области)",
                md(pd.DataFrame([{"Область": SHORT[d], "недоступно": v["unavailable"], "доля, %": f"{100 * v['share']:.1f}",
                                  "из них пост/видео исчезли в API": v["of_which_post_or_video_gone_in_api"],
                                  "ошибки": json.dumps(v["errors"], ensure_ascii=False)} for d, v in vc.items()]))]

    # --- п. 8: варианты M1
    e = J(R / "main" / "stage_3e.json")
    if e:
        rows = []
        for vname, title in (("main", "основной (99 той же области)"), ("size_decile", "(а) по размеру")):
            t = e["variants"][vname]["table"]
            for k, n in KEYS + [("B0", "B0 случайная"), ("B2", "B2 ключевые слова")]:
                rows.append({"Вариант M1": title, "Стратегия": n, "AUC": ci(t[k]["auc"]), "Recall@10": ci(t[k]["recall10"]),
                             "p против NCA-RS (BH)": "—" if t[k]["delong_vs_nca"] is None else f"{t[k]['delong_vs_nca_p_bh']:.2g}"})
        out += ["### П. 8: M1 — основной вариант, (а) по размеру, (б) гибрид", md(pd.DataFrame(rows))]

    # --- п. 9а
    a = J(R / "main" / "check_9a.json")
    if a:
        rows = [{"Доля выборки": f, "Силуэт постов": f"{v['sil_posts_mean']:.3f}", "Силуэт профилей": f"{v['sil_users_mean']:.3f}",
                 "|M|": v["M_size_total"], "Σ TAS": f"{v['sum_tas_total']:.2f}",
                 "AUC NCA-RS": ci(v["m1"]["main"]["nca"]), "AUC гибрид": ci(v["m1"]["main"]["hybrid"]),
                 "AUC отклик": ci(v["m1"]["main"]["response"]), "M1 польз.": v["m1"]["main"]["n_users"]}
                for f, v in a["results"].items()]
        out += [f"### П. 9а: кривые по доле выборки (конфигурация {a['config']})", md(pd.DataFrame(rows))]
    b = J(R / "main" / "check_9b.json")
    if b:
        rows = [{"Область": SHORT[d], "Силуэт постов": f"{v['unweighted']:.3f}", "взвешенный": f"{v['weighted']:.3f}",
                 "веса Т/И/В (доли)": " / ".join(f"{v['weights'][c]:.3f}" for c in ("text", "image", "video"))}
                for d, v in b["silhouette_posts"].items()]
        m = b["m1"]["main"]
        out += ["### П. 9б: веса реальных долей классов", md(pd.DataFrame(rows)),
                "M1 со взвешенными ē_c и π: " + ", ".join(f"{n} {ci(m[k])}" for k, n in KEYS)]
    # --- п. 9в
    seeds = {}
    for s in (42, 43, 44, 45, 46):
        d = R / ("main" if s == 42 else f"seed{s}")
        c3, e3 = J(d / "stage_3c.json"), J(d / "stage_3e.json")
        if c3 and e3:
            seeds[s] = {"Зерно": s, "Посты": CLU[c3["best_method"]["posts"]], "Профили": CLU[c3["best_method"]["users"]],
                        "k посты": "/".join(str(c3["results"][f"{x}|posts|{c3['best_method']['posts']}"]["k"]) for x in DOMAINS),
                        "k профили": "/".join(str(c3["results"][f"{x}|users|{c3['best_method']['users']}"]["k"]) for x in DOMAINS),
                        **{f"AUC {n}": f"{e3['table'][k]['auc'][0]:.3f}" for k, n in KEYS[:3]}}
    if seeds:
        df = pd.DataFrame(list(seeds.values()))
        stat = {c: f"{np.mean([float(x) for x in df[c]]):.3f} ± {np.std([float(x) for x in df[c]], ddof=1):.3f}"
                for c in df.columns if c.startswith("AUC")} if len(df) > 1 else {}
        out += ["### П. 9в: 5 зёрен (кластеризация + M1)", md(df), "Среднее ± SD: " + json.dumps(stat, ensure_ascii=False)]
    # --- п. 9г
    q3 = R / "q3"
    c3, d3, e3 = J(q3 / "stage_3c.json"), J(q3 / "stage_3d.json"), J(q3 / "stage_3e.json")
    if c3 and e3:
        out += ["### П. 9г: порог ERR «верхняя четверть»",
                f"Кластеризация: посты {CLU[c3['best_method']['posts']]}, профили {CLU[c3['best_method']['users']]}; "
                f"сопоставление {MATCH[d3['best_method']]}, Σ TAS {d3['sum_tas_total'][d3['best_method']]:.2f}.",
                "M1: " + ", ".join(f"{n} {ci(e3['table'][k]['auc'])}" for k, n in KEYS) +
                "; (а) по размеру: " + ", ".join(f"{n} {ci(e3['variants']['size_decile']['table'][k]['auc'])}" for k, n in KEYS)]
    # --- сравнение с текстовым прогоном
    e1 = J(V1 / "stage_3e.json")
    c1, c2 = J(V1 / "stage_3c.json"), J(R / "main" / "stage_3c.json")
    a2 = J(R / "main" / "stage_3a.json")
    a1 = J(V1 / "stage_3a.json")
    if e1 and e:
        rows = []
        for k, n in [("nca", "NCA-RS"), ("response", "Модель отклика"), ("B0", "B0"), ("B1", "B1"), ("B2", "B2"), ("B3", "B3")]:
            rows.append({"Стратегия (AUC M1)": n, "v1 текст (протокол)": ci(e1["table"][k]["auc"]), "v2 мультимодальный": ci(e["table"][k]["auc"])})
        out += ["### Сравнение с текстовым прогоном v1 (2026-10-17)", md(pd.DataFrame(rows))]
        if c1 and c2:
            out += [f"Кластеризация v1: посты {CLU[c1['best_method']['posts']]}, профили {CLU[c1['best_method']['users']]}; "
                    f"v2: посты {CLU[c2['best_method']['posts']]}, профили {CLU[c2['best_method']['users']]}."]
        if a1 and a2:
            q = lambda A, enc, f: np.mean([f(A[enc][x]) for x in DOMAINS])
            out += [f"Qwen3-VL-Embedding-2B, среднее по областям: Хопкинс v1 {q(a1, 'qwen', lambda z: z['m3']['hopkins'][0]):.3f} → v2 "
                    f"{q(a2, 'qwen', lambda z: z['m3']['hopkins'][0]):.3f}; силуэт Ward постов v1 "
                    f"{q(a1, 'qwen', lambda z: z['m4_ward_posts']['silhouette'][0]):.3f} → v2 {q(a2, 'qwen', lambda z: z['m4_ward_posts']['silhouette'][0]):.3f}; "
                    f"эфф. размерность v1 {q(a1, 'qwen', lambda z: z['m3']['eff_dim95'][0]):.0f} → v2 {q(a2, 'qwen', lambda z: z['m3']['eff_dim95'][0]):.0f}."]
    (R / "checks.md").write_text("\n\n".join(out) + "\n")
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
