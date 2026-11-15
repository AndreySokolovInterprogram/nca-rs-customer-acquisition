"""Протокол v2, п. 4: ключевые кадры видео по разделу 3.4 статьи (сжатие видео по MAD, на CPU).

Декодирование 2 кадра/с (первые 180 с), масштаб 768 px по длинной стороне, MAD (eq:mad) на сером, θ = 10,
K_max = 32 (первый кадр всегда; при избытке — 31 кадр с наибольшим MAD, в порядке времени).
Картинки перед кодированием уменьшаются до 768 px по длинной стороне (PIL thumbnail) и сохраняются как JPEG q = 90.

В публичную версию НЕ входят функции получения медиа (скачивание картинок и видео по ссылкам из официального
API VK): они относятся к сбору данных. На вход подаются уже полученные локальные файлы.
Файлы — недоверенные данные: обрабатываются только PyAV и PIL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

FPS = 2
LONG_SIDE = 768
THETA = 10.0
K_MAX = 32
MAX_SEC = 180


def decode_frames(path, options: dict | None = None) -> np.ndarray:
    """Кадры первых MAX_SEC секунд, 2 кадра/с, 768 по длинной стороне: массив (F, H, W, 3) uint8.

    Декодирование — PyAV (собственная сборка FFmpeg): статический ffmpeg из imageio-ffmpeg падает (SIGSEGV)
    на MPEG-TS из HLS-потоков VK. Выборка 2 кадра/с как у фильтра fps: на каждую точку сетки k/2 с —
    последний кадр с меткой времени ≤ точки; масштаб — бикубический (PIL), с учётом поворота."""
    import av

    out = []
    with av.open(str(path), options=options or {}, timeout=30) as c:
        if not c.streams.video:
            raise RuntimeError("нет видеопотока")
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        grid, prev = 0.0, None
        for fr in c.decode(st):
            t = float(fr.time) if fr.time is not None else (prev[0] + 1 / 25 if prev else 0.0)
            if t > MAX_SEC:
                break
            if prev is None and t > 0:
                grid = min(grid, t)
            while prev is not None and grid < t and grid <= MAX_SEC - 1e-9 and len(out) < MAX_SEC * FPS:
                out.append(prev[1])
                grid += 1 / FPS
            prev = (t, fr)
            if t >= grid - 1e-9 and len(out) < MAX_SEC * FPS:
                out.append(fr)
                grid += 1 / FPS
        if not out:
            raise RuntimeError("0 кадров")
        imgs = []
        for fr in out:
            im = fr.to_image()
            rot = getattr(fr, "rotation", 0) or 0
            if rot:
                im = im.rotate(rot, expand=True)
            w, h = im.size
            s_ = LONG_SIDE / max(w, h)
            W, H = max(2, int(round(w * s_ / 2)) * 2), max(2, int(round(h * s_ / 2)) * 2)
            imgs.append(np.asarray(im.convert("RGB").resize((W, H), Image.BICUBIC)))
    return np.stack(imgs)


def keyframes(frames: np.ndarray) -> tuple[list[int], np.ndarray]:
    """K = {0} ∪ {t+1 : MAD(f_t, f_{t+1}) > θ}; |K| ≤ K_MAX (по наибольшему MAD)."""
    if len(frames) == 0:
        return [], np.zeros(0)
    gray = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    mad = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2)) if len(frames) > 1 else np.zeros(0)
    cand = [t + 1 for t in np.flatnonzero(mad > THETA)]
    if len(cand) > K_MAX - 1:
        cand = sorted(sorted(cand, key=lambda t: -mad[t - 1])[: K_MAX - 1])
    return [0] + cand, mad


def frames_to_dir(frames: np.ndarray, out_dir: Path, info: dict) -> dict:
    K, mad = keyframes(frames)
    for i, t in enumerate(K):
        Image.fromarray(frames[t]).save(out_dir / f"k{i:02d}.jpg", "JPEG", quality=90)
    info.update({"frames_2fps": int(len(frames)), "keyframes": len(K), "k_idx": [int(t) for t in K],
                 "mad_mean": float(mad.mean()) if len(mad) else 0.0, "mad_gt_theta": int((mad > THETA).sum()),
                 "frame_size": list(frames.shape[1:3][::-1]) if len(frames) else None})
    return info

