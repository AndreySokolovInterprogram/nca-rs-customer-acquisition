# NCA-RS: рекомендательная система привлечения новых клиентов

Код экспериментов к статье о NCA-RS для журнала «Компьютерные исследования и моделирование».
NCA-RS сопоставляет кластеры мультимодального контента компаний (тексты, изображения, видео постов) с кластерами
внешних профилей аудитории и ранжирует сообщества-кандидаты для пользователя по оценке согласованности TAS.

> **English.** Code for the experiments of the NCA-RS paper (recommender system for acquiring new customers by
> matching clusters of companies' multimodal content to clusters of external audience profiles). It contains
> multimodal embeddings with Qwen3-VL-Embedding-2B (MAD-based video keyframe selection), adaptive convex
> renormalization (ACR) of user profiles, clustering (Ward / average / complete / Bayesian GMM, largest cluster
> ≤ 50 % rule), five matching methods with the TAS score, a permutation test, the M1 hidden-community ranking
> evaluation against baselines B0–B3, the response model and a hybrid, and ablations. Mapping to the paper:
> §3.4 MAD video compression (`mm_media.py`); §3.5–3.6 encoder choice and company post vectorization
> (`encoders.py`, `mm_embed.py`); §3.7 user profile vectorization, ACR (`nca.py`); §3.8 clustering;
> §3.9 five cluster matching methods; §3.10 the TAS functional; §3.11 quality measures M1, M2 and the
> permutation test; §3.12 baselines B0–B3; §3.13 protocol and ablations (`s4_protocol.py`); §4 results
> (4.2 partition properties, 4.3 matching comparison, 4.4 measure M1; `s7_tables.py`, `mm_s6_checks.py`,
> `mm_s7_checks_tables.py`). No data is included;
> the data were collected through the official VK API. Licensed under MIT.

## Что в репозитории

Код, который использовался в прогоне статьи 10.11.2026 (протокол v2): модули перенесены из рабочего репозитория
без истории и с сохранением структуры пакета `experiments.paper1`. Отличия от рабочей версии: пути, адреса и
параметры окружения вынесены в переменные окружения; убран код получения данных (см. «Данные»).

| Раздел статьи | Что | Файлы |
|---|---|---|
| 3.4 | сжатие видео: ключевые кадры по MAD (2 кадра/с, θ = 10, K ≤ 32) | `mm_media.py` |
| 3.5, 3.6 | выбор энкодера; векторизация постов компаний — Qwen3-VL-Embedding-2B (текст, изображение, видео) | `encoders.py`, `mm_embed.py` (кодирование и хранение эмбеддингов — `mm_s4_pipeline.py`, `mm_s12_m1_embed.py`) |
| 3.7 | векторизация профилей пользователей: адаптивная выпуклая ренормализация (АВР) | `nca.acr_profiles`, `data.EncoderData.profiles` |
| 3.8 | кластеризация: Ward, average, complete, Bayesian GMM; выбор k по угловому силуэту при условии «крупнейший кластер ≤ 50 %»; силуэт, CH, Хопкинс, стабильность Жаккара, gap | `nca.py`, `s4_protocol.select_k / pick_best / eval_corpus` (этап 3c) |
| 3.9 | пять методов сопоставления кластеров (пороговый, жадный, оптимальный транспорт, бикластеризация, IPFP) | `nca.match` |
| 3.10 | функционал TAS | `nca.tas_matrix`, `nca.cluster_max` |
| 3.11 | меры качества M1, M2 и перестановочный тест; бутстреп-интервалы, тест ДеЛонга, поправка BH | `nca.rank_metrics`, `nca.bootstrap_ci`, `nca.paired_auc_test`, `s4_protocol.stage_3d` (перестановочный тест) |
| 3.12 | базовые стратегии B0–B3 (а также модель отклика и гибрид) | `s4_protocol.m1_scores`, `s4_protocol.m1_context` |
| 3.13 | протокол и абляции 3a–3e (энкодер, веса профиля, кластеризация, сопоставление, M1) | `s4_protocol.py`, `mm_chain_m1_20261110.sh` |
| — | сборка выборки и эмбеддингов в формат этапов, полные профили | `mm_s5_finalize.py`, `data.py` |
| 4 | результаты: 4.2 свойства разбиений, 4.3 сравнение методов сопоставления, 4.4 мера M1; проверки 9а–9г (кривые по объёму, веса долей классов, зёрна 43–46, порог Q3), чувствительность по полным профилям | `s7_tables.py`, `mm_s6_checks.py`, `mm_s7_checks_tables.py` |
| 4 | дополнительно: анализ выбывания, диагностика M1 | `mm_s13_attrition.py`, `mm_s15_m1_diag.py` |

`tests/test_synthetic.py` — проверка без данных и GPU: импорт всех модулей и методы на случайных векторах.

## Данные

Данные в репозиторий **не входят**: ни посты, ни профили, ни эмбеддинги, ни идентификаторы пользователей и
сообществ. Данные собраны из открытых сообществ и профилей через официальный API VK отдельным сборщиком;
его код, а также код выборки из базы сборщика и получения медиафайлов, в публичный репозиторий не включены.
Ниже — схема входных таблиц, по которой можно подготовить собственные данные.

### Входные данные

Таблицы — `pandas.DataFrame` в формате pickle (`*.pkl`), каталог `$PAPER1_SRC_DIR` (по умолчанию `work/sample_src`).
Идентификаторы — целые числа (любые внутренние, не обязательно VK). Области (`domain`) —
`customer_acquisition`, `food_delivery`, `food_service`, `tender_support` (см. `config.DOMAINS`).

| Файл | Столбцы | Смысл |
|---|---|---|
| `pools.pkl` | `dict[domain → list[gid]]` | пул сообществ-компаний области |
| `pool_info.pkl` | `gid, vk_id, label, members_count, name, description` | сведения о сообществах пула (B1 — по `members_count`) |
| `pool_keywords.pkl` | `gid, subject, keyword` | ключевые слова сообществ (B2) |
| `company_order.pkl` | `post_id, gid, vk_owner_id, vk_post_id, domain, cls, err, eff_q3, rank, q3_rank, need_main, need_q3, text` | посты компаний в случайном порядке внутри области и класса; `cls ∈ {text, image, video}`; `err` — вовлечённость; `eff_q3` — пост проходит порог Q3 |
| `user_posts_src.pkl` | `user_id, post_id, vk_owner_id, vk_post_id, published_at (UTC), text, has_photo, has_video, cls` | посты пользователей (в профиль — 5 последних) |
| `user_subs.pkl` | `user_id, vk_group_id` | подписки пользователей |
| `sub_communities.pkl` | `vk_id, text` | текст (название + описание) сообществ из подписок |
| `demo.pkl` | `user_id, text, r` | демография строкой; `r` — число заполненных полей из 8 (`q = r/8` в АВР) |
| `corpus_users.pkl` | `user_id, domain` | пользователи корпуса (кластеризация аудитории) |
| `m1_users.pkl` | `user_id, domain, hidden_gid` | пользователи M1 и скрытое сообщество |
| `seed_users.pkl` | `gid, user_id` | известные участники сообществ (B3, look-alike) |
| `user_links.pkl` | `user_id, gid` | все связи «пользователь — сообщество пула» (исключаются из отрицательных) |

`$PAPER1_META_DIR/sample_stats.json` — `{"domains": {domain: {"n_main", "n_q3", "real_share": {"text", "image", "video"}, …}}}`:
число постов каждого класса в основном варианте и в варианте Q3, реальные доли классов (проверка 9б).

Эмбеддинги: `$PAPER1_MM_WORK/emb_parts/pNNNNN.npy` (float32, L2-нормированы, 2048) + `pNNNNN.json` (список `uid`) и
`$PAPER1_MM_WORK/manifest.jsonl` (`uid, set, post_id, key, cls, url, status, modality, n_frames, err, embedded`).
`uid`: `company:<post_id>`, `user:<post_id>`, `sub:<vk_id>`, `demo:<user_id>`. Их строит фаза
`python -m experiments.paper1.mm_s4_pipeline local items.jsonl` из локальных файлов; строка `items.jsonl`:
`{"uid", "set", "post_id", "key", "cls", "text", "image": <путь> | "video": <путь>}`.
`mm_s5_finalize.py` собирает из этого каталоги `work/main` и `work/q3` (`sample/*.pkl`, `emb/qwen/*.npy`) для этапов.

## Требования

- Python 3.12 (прогон: 3.12.3), Linux (прогон — WSL2).
- Построение эмбеддингов: GPU NVIDIA с 8 ГБ памяти (bfloat16), CUDA; модель `Qwen/Qwen3-VL-Embedding-2B`,
  ревизия закреплена в `encoders.py`. Видео декодируется PyAV.
- Этапы 3a–3e и проверки: только CPU (параллельность — `PAPER1_N_JOBS`; в прогоне 2 процесса).
- Версии библиотек — `requirements.txt` (из окружения прогона).

## Запуск

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python tests/test_synthetic.py            # проверка без данных и GPU

# 1. эмбеддинги (GPU): по items.jsonl с локальными файлами и текстами
export PAPER1_MM_WORK=/path/to/work       # там же sample_src/ и meta/sample_stats.json
python -m experiments.paper1.mm_s4_pipeline local items.jsonl

# 2. сборка, этапы 3a–3e, проверки 9а–9г, таблицы (CPU)
PY=python PAPER1_OUT=/path/to/results ./experiments/paper1/mm_chain_m1_20261110.sh
# возобновление с шага: RESUME_FROM=s4_3d …; отдельный этап: PYTHONPATH=. python -m experiments.paper1.s4_protocol --stage 3c
```

Основные параметры: `PAPER1_ENCODERS=qwen`, `PAPER1_MAX_CLUSTER_SHARE=0.5` (правило «крупнейший кластер ≤ 50 %»),
`PAPER1_SEED=42`; α = 0,90, ρ_max = 0,10, B = 1000 (бутстреп и перестановочный тест), веса АВР — сетка в
`s4_protocol.WEIGHTS`. Каждый этап пишет `stage_<id>.json` в `$PAPER1_RESULTS_DIR` и при повторном запуске не
пересчитывается. `mm_s12_m1_embed.py media` в прогоне кодировал медиа, полученные отдельно (с проверкой sha256);
фаза `fallback` кодирует текстом посты, медиа которых получить не удалось.

## Лицензия

MIT, © 2026 Соколов А. В., Шуткин А. С. — см. `LICENSE`. Модель Qwen3-VL-Embedding-2B распространяется
под собственной лицензией (Apache-2.0, см. карточку модели на Hugging Face).
