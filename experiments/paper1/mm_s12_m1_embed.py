"""Протокол v2, M1: эмбеддинги медиа, докачанных на сервере 08–10.11, и текстовые замены (решение 10.11).

Без сети. Тот же энкодер и те же параметры, что в конвейере (mm_s4_pipeline / mm_s11_m1_images):
  * картинка — оригинал с сервера → PIL thumbnail((768, 768)) (BICUBIC, reducing_gap 2,0) → JPEG q = 90 → Qwen3-VL;
  * видео — сжатый файл (H.264 crf 28, ≤ 480 по короткой стороне) → mm_media.decode_frames (первые 180 с, 2 кадра/с,
    768 по длинной стороне) → keyframes (MAD, θ = 10, K ≤ 32) → Qwen3-VL;
  * текст поста — собственный текст после п. 5 (user_posts_src.text, ≤ 1 500 символов), если есть.
Файлы — недоверенные данные: читаются только PIL и PyAV; перед использованием сверяется sha256 с логом сервера.

Фазы:
  media     — посты 1–5 пользователей M1, у которых на сервере скачана картинка (класс ИЗОБРАЖЕНИЕ) или видео (ВИДЕО),
              а в манифесте ещё нет закодированного медиа этого класса;
  fallback  — посты 1–5 с фото/видео у M1, корпуса и seed, которых всё ещё нет в манифесте: кодируются текстом
              (status = fallback, modality = text), без текста — исключаются (status = excluded).
              reason: «unavailable» (M1: медиа невосстановимо) или «not_downloaded» (корпус/seed: не скачаны — отклонение).
  count     — полные профили M1 по правилу протокола (все медиа постов 1–5 закодированы).
Запуск: python -m experiments.paper1.mm_s12_m1_embed media|fallback|count
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from experiments.paper1 import mm_media
from experiments.paper1.config import SRC
from experiments.paper1.mm_s4_pipeline import MANIFEST, TMP, Pipeline, log

M1DIR = SRC.parent / "m1"
DL = M1DIR / "server_downloads_20261110.jsonl"
DIRS = {"image": M1DIR / "images_night_20261109", "video_full": M1DIR / "videos_20261109"}
SRC_TAG = "server_20261110"
CHUNK = 40
TEXT_CHUNK = 2000


def manifest() -> dict:
    man = {}
    for line in MANIFEST.read_text().splitlines():
        r = json.loads(line)
        man[r["uid"]] = r
    return man


def media_ok(r: dict | None, cls: str) -> bool:
    return r is not None and r["status"] == "ok" and bool(r.get("embedded")) and cls in (r.get("modality") or "")


def top5() -> pd.DataFrame:
    up = pd.read_pickle(SRC / "user_posts_src.pkl").drop_duplicates(["user_id", "post_id"]).copy()
    up["rank"] = up.groupby("user_id").published_at.rank(method="first", ascending=False)
    up = up[up["rank"] <= 5].copy()
    up["uid"] = "user:" + up.post_id.astype(str)
    up["key"] = up.vk_owner_id.astype(str) + "_" + up.vk_post_id.astype(str)
    up["text"] = up.text.fillna("").str.strip()
    return up


def groups() -> dict[str, set]:
    return {g: set(pd.read_pickle(SRC / f"{g}_users.pkl").user_id) for g in ("m1", "corpus", "seed")}


def server_ok() -> dict[tuple[str, str], dict]:
    last = {}
    for line in DL.read_text().splitlines():
        r = json.loads(line)
        last[(r["key"], r["kind"])] = r
    return {k: r for k, r in last.items() if r["status"] == "ok" and k[1] in DIRS}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prepare(it: dict, rec: dict, tmp: Path) -> dict:
    """Вход энкодера и строка манифеста для одного поста (как fetch_item, но из локального файла)."""
    path = DIRS[rec["kind"]] / Path(rec["path"]).name
    row = {"uid": it["uid"], "set": "user", "post_id": int(it["post_id"]), "key": it["key"], "cls": it["cls"],
           "url": f"file:{rec['path']}", "status": "ok", "modality": None, "n_frames": 0, "err": None, "src": SRC_TAG}
    text = it["text"]
    enc = {"text": text} if text else {}
    try:
        if sha256(path) != rec["sha256"]:
            raise RuntimeError("sha256 не совпадает с логом сервера")
        if it["cls"] == "image":
            p = tmp / f"i_{it['post_id']}.jpg"
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((mm_media.LONG_SIDE, mm_media.LONG_SIDE))  # уменьшает только > 768 — как у компаний
                im.save(p, "JPEG", quality=90)
            enc["image"] = p
            row["modality"] = "image+text" if text else "image"
        else:
            out = tmp / f"v_{it['post_id']}"
            out.mkdir(parents=True, exist_ok=True)
            frames = mm_media.decode_frames(path)
            info = mm_media.frames_to_dir(frames, out, {})
            if info["keyframes"] == 0:
                raise RuntimeError("0 кадров")
            enc.update(frames=sorted(out.glob("k*.jpg")), frame_idx=info["k_idx"])
            row.update(modality="video+text" if text else "video", n_frames=info["keyframes"],
                       video_sec=(rec.get("out") or {}).get("duration"), frames_2fps=info["frames_2fps"])
        return {"row": row, "enc": enc}
    except Exception as e:  # noqa: BLE001 — в манифест не пишем: пост остаётся кандидатом fallback
        log(f"m1-embed: {it['key']} {type(e).__name__}: {str(e)[:200]}")
        return {"row": dict(row, status="deferred", err=f"{type(e).__name__}: {e}"[:300]), "enc": None}


def phase_media() -> None:
    man = manifest()
    srv = server_ok()
    up = top5()
    up = up[up.user_id.isin(groups()["m1"]) & up.cls.isin(["image", "video"])]
    kind = {"image": "image", "video": "video_full"}
    todo = [dict(r, rec=srv[(r["key"], kind[r["cls"]])]) for r in up.to_dict("records")
            if (r["key"], kind[r["cls"]]) in srv and not media_ok(man.get(r["uid"]), r["cls"])]
    log(f"m1-embed media: {len(todo)} постов (картинок {sum(t['cls'] == 'image' for t in todo)}, "
        f"видео {sum(t['cls'] == 'video' for t in todo)})")
    pl = Pipeline()
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        tmp = TMP / f"m1new_{i}"
        tmp.mkdir(parents=True, exist_ok=True)
        fetched = [prepare(it, it["rec"], tmp) for it in todo[i:i + CHUNK]]
        pl.embed_and_save(fetched, tmp)
        bad = sum(f["enc"] is None for f in fetched)
        log(f"m1-embed media: {min(i + CHUNK, len(todo))}/{len(todo)}; ошибок подготовки {bad}; "
            f"{time.time() - t0:.0f} с; {pl.guard.resources()}")
    shutil.rmtree(TMP, ignore_errors=True)
    log("m1-embed media: done")


def phase_fallback() -> None:
    man = manifest()
    g = groups()
    up = top5()
    up = up[up.cls.isin(["image", "video"]) & ~up.uid.isin(man.keys())]
    up = up[up.user_id.isin(g["m1"] | g["corpus"] | g["seed"])].drop_duplicates("uid")
    reason = ["unavailable" if u in g["m1"] else "not_downloaded" for u in up.user_id]
    items, rows_ex = [], []
    for r, why in zip(up.to_dict("records"), reason):
        row = {"uid": r["uid"], "set": "user", "post_id": int(r["post_id"]), "key": r["key"], "cls": r["cls"],
               "url": None, "status": "fallback", "modality": "text", "n_frames": 0, "src": SRC_TAG, "reason": why,
               "err": "медиа невосстановимо (M1)" if why == "unavailable" else "медиа не скачано (отклонение 10.11)"}
        if r["text"]:
            items.append({"row": row, "enc": {"text": r["text"]}})
        else:
            rows_ex.append({"row": dict(row, status="excluded", modality=None), "enc": None})
    log(f"m1-embed fallback: текстом {len(items)}, исключается {len(rows_ex)}; "
        f"по причине {pd.Series(reason).value_counts().to_dict()}")
    pl = Pipeline()
    for i in range(0, len(items), TEXT_CHUNK):
        tmp = TMP / f"fb_{i}"
        tmp.mkdir(parents=True, exist_ok=True)
        pl.embed_and_save(items[i:i + TEXT_CHUNK], tmp)
        log(f"m1-embed fallback: {min(i + TEXT_CHUNK, len(items))}/{len(items)}")
    pl.embed_and_save(rows_ex, TMP / "fb_ex")
    shutil.rmtree(TMP, ignore_errors=True)
    log("m1-embed fallback: done")


def full_users(man: dict, users: set) -> set:
    """Полный профиль: у всех постов 1–5 с фото/видео закодировано медиа своего класса."""
    up = top5()
    up = up[up.user_id.isin(users)]
    m = up[up.cls.isin(["image", "video"])]
    ok = pd.Series([media_ok(man.get(u), c) for u, c in zip(m.uid, m.cls)], index=m.index)
    bad = set(m.user_id[~ok])
    return set(users) - bad


def count() -> dict:
    man = manifest()
    g = groups()
    return {k: {"users": len(v), "full": len(full_users(man, v))} for k, v in g.items()}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "media":
        phase_media()
    elif cmd == "fallback":
        phase_fallback()
    print(json.dumps(count(), ensure_ascii=False))
