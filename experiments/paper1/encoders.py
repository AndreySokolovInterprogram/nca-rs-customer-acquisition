"""Энкодеры Φ_emb для статьи №1: Qwen3-VL-Embedding-2B и jina-embeddings-v5-omni-small.

Загрузка и вызов — по карточкам моделей на Hugging Face (sentence-transformers):
  * Qwen/Qwen3-VL-Embedding-2B — инструкция по умолчанию «Represent the user's input.»;
  * jinaai/jina-embeddings-v5-omni-small — адаптер задачи `clustering`, encode_document
    (префикс «Document: »), загружается только текстовая часть (modality="text").
Ревизии зафиксированы. Здесь — загрузка модели и текстовый режим (протокол v1); мультимодальное кодирование
(текст, изображение, видео) для прогона статьи — mm_embed.py поверх load("qwen"). В прогоне v2 использовался
только Qwen3-VL-Embedding-2B (PAPER1_ENCODERS=qwen).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ENCODERS = {
    "qwen": {
        "repo": "Qwen/Qwen3-VL-Embedding-2B",
        "revision": "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
        "dim": 2048,
        "license": "apache-2.0",
    },
    "jina": {
        "repo": "jinaai/jina-embeddings-v5-omni-small",
        "revision": "950200abde3a3bcf3d5bdfe5d28fb5bb5e7dfbb0",
        "dim": 1024,
        "license": "cc-by-nc-4.0",
    },
}

MAX_SEQ_LEN = 512


def load(name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    spec = ENCODERS[name]
    if name == "qwen":
        model = SentenceTransformer(spec["repo"], revision=spec["revision"], device="cuda",
                                    model_kwargs={"torch_dtype": torch.bfloat16})
    else:
        model = SentenceTransformer(spec["repo"], revision=spec["revision"], device="cuda",
                                    trust_remote_code=True,
                                    model_kwargs={"default_task": "clustering", "modality": "text",
                                                  "torch_dtype": torch.bfloat16})
    model.max_seq_length = MAX_SEQ_LEN
    return model


JINA_DOC_PREFIX = "Document: "


def encode(model, name: str, texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Только текстовая модальность, без сетевых обращений.

    Qwen: вход — словарь {"text": ...}; строку, целиком состоящую из URL картинки/видео
    (или ссылки YouTube), sentence-transformers иначе считает медиа и скачивает.
    jina: обёртка custom_st.py скачивает ЛЮБУЮ строку, начинающуюся с http(s)://, поэтому
    префикс задачи «Document: » (как в encode_document) приписывается заранее, а prompt
    библиотекой не добавляется: строка никогда не начинается с URL."""
    if name == "jina":
        inputs = [JINA_DOC_PREFIX + t for t in texts]
        assert not any(x.startswith(("http://", "https://")) for x in inputs)
        emb = model.encode(inputs, prompt="", batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    else:
        emb = model.encode([{"text": t} for t in texts], batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    emb = emb.astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def encode_checkpointed(model, name: str, texts: list[str], path: Path, chunk: int = 4000,
                        batch_size: int = 16, log=print) -> np.ndarray:
    """Кодирует по кускам; готовые куски лежат в path.parent/<stem>.partNNN.npy и не пересчитываются."""
    if path.exists():
        return np.load(path)
    parts = []
    # сортировка по длине ускоряет батчи; порядок восстанавливается в конце
    order = np.argsort([len(t) for t in texts], kind="stable")
    for k, start in enumerate(range(0, len(texts), chunk)):
        ppath = path.with_name(f"{path.stem}.part{k:03d}.npy")
        if ppath.exists():
            parts.append(np.load(ppath))
            continue
        t0 = time.time()
        idx = order[start:start + chunk]
        e = encode(model, name, [texts[i] for i in idx], batch_size=batch_size)
        np.save(ppath, e)
        parts.append(e)
        log(f"{path.stem}: {start + len(idx)}/{len(texts)} ({len(idx) / (time.time() - t0):.1f} текст/с)")
    emb = np.empty((len(texts), ENCODERS[name]["dim"]), dtype=np.float32)
    emb[order] = np.concatenate(parts) if parts else emb[:0]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(path, emb)
    for ppath in path.parent.glob(f"{path.stem}.part*.npy"):
        ppath.unlink()
    return emb
