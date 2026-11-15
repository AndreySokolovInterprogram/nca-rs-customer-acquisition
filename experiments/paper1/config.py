"""Общие настройки экспериментов статьи NCA-RS.

Все случайности — SEED = 42 (другие зёрна — только проверка устойчивости, п. 9в протокола).
Все каталоги задаются переменными окружения (значения по умолчанию — внутри репозитория, папка work/):

  PAPER1_MM_WORK     корень рабочих данных (по умолчанию ./work)
  PAPER1_SRC_DIR     исходные таблицы выборки (по умолчанию $PAPER1_MM_WORK/sample_src) — см. README, «Входные данные»
  PAPER1_META_DIR    сводки выборки и медиа: sample_stats.json, media_stats.json (по умолчанию $PAPER1_MM_WORK/meta)
  PAPER1_WORK_DIR    данные варианта прогона: sample/, emb/qwen/, метки кластеров (по умолчанию $PAPER1_MM_WORK/main)
  PAPER1_RESULTS_DIR результаты этапов (JSON, CSV, рисунки; по умолчанию ./results)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

SEED = int(os.environ.get("PAPER1_SEED", 42))  # 42; другие зёрна — только проверка п. 9в протокола

# Дата заморозки снимка данных (используется для признаков давности в анализе выбывания)
SNAPSHOT_AT = datetime(2026, 10, 17, 11, 40, 41, tzinfo=timezone.utc)

DOMAINS = ["customer_acquisition", "food_delivery", "food_service", "tender_support"]

REPO_DIR = Path(__file__).resolve().parents[2]
PLATFORM_DIR = REPO_DIR  # совместимость с исходным кодом прогона
MM_WORK = Path(os.environ.get("PAPER1_MM_WORK", REPO_DIR / "work"))
SRC = Path(os.environ.get("PAPER1_SRC_DIR", MM_WORK / "sample_src"))
META_DIR = Path(os.environ.get("PAPER1_META_DIR", MM_WORK / "meta"))
RESULTS_DIR = Path(os.environ.get("PAPER1_RESULTS_DIR", REPO_DIR / "results"))
WORK_DIR = Path(os.environ.get("PAPER1_WORK_DIR", MM_WORK / "main"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
