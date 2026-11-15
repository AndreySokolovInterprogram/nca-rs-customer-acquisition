"""Протокол v2, п. 6: мультимодальное кодирование Qwen3-VL-Embedding-2B (sentence-transformers 6).

Элемент: {"text": str | None, "image": путь | None, "frames": [пути] | None, "frame_idx": [int] | None}.
  * ТЕКСТ  → {"text": ...}
  * ИЗОБРАЖЕНИЕ → {"image": PIL, "text"?: ...}
  * ВИДЕО  → {"video": {"array": (K, H, W, 3), "video_metadata": {fps = 2, frames_indices = индексы
    ключевых кадров в потоке 2 кадра/с}}, "text"?: ...}; выборка кадров процессором отключена
    (do_sample_frames = False): кадры уже отобраны по MAD (mm_media.keyframes).
Инструкция — по умолчанию модели. Векторы L2-нормированы, float32.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from experiments.paper1 import encoders
from experiments.paper1.mm_media import FPS


MAX_SEQ_LEN = 16384  # ≥ числа токенов видео при лимите процессора 24 Мпикс (32 кадра 768×768 ≈ 9,2 тыс.); 512 из v1 обрезает токены картинок


def load():
    model = encoders.load("qwen")
    model.max_seq_length = MAX_SEQ_LEN
    return model


def to_input(item: dict) -> dict:
    x = {}
    if item.get("text"):
        x["text"] = item["text"]
    if item.get("frames"):
        arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in item["frames"]])
        idx = item.get("frame_idx") or list(range(len(arr)))
        if len(arr) == 1:  # один ключевой кадр: Qwen3-VL требует ≥ 2 кадров (temporal_patch_size = 2) — дублируем
            arr, idx = np.concatenate([arr, arr]), [idx[0], idx[0] + 1]
        x["video"] = {"array": arr, "video_metadata": {"fps": FPS, "total_num_frames": int(max(idx)) + 1,
                                                       "frames_indices": list(map(int, idx)),
                                                       "duration": (max(idx) + 1) / FPS}}
    elif item.get("image"):
        x["image"] = Image.open(item["image"]).convert("RGB")
    if not x:
        raise ValueError("пустой элемент")
    return x


def encode(model, items: list[dict], batch_size: int = 1) -> np.ndarray:
    inputs = [to_input(it) for it in items]
    emb = model.encode(inputs, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False, processing_kwargs={"video": {"do_sample_frames": False}})
    emb = emb.astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)
