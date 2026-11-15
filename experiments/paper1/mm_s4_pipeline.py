"""Протокол v2: кодирование медиа и текстов Qwen3-VL-Embedding-2B с хранением только эмбеддингов.

Порция элементов → (видео: ключевые кадры по MAD, mm_media) → эмбеддинги (mm_embed) → удаление временных файлов.
Хранятся только эмбеддинги (emb_parts/pNNNNN.npy + pNNNNN.json со списком uid) и манифест
(manifest.jsonl: uid, id поста, класс, статус, модальность, число кадров).

Публичная версия: из исходного конвейера убраны получение медиа по ссылкам (скачивание с CDN VK) и фазы обхода
выборки, которые его вызывают; оставлены хранилище, кодирование и запись манифеста — без изменений.
Для кодирования локальных файлов добавлена фаза `local` (адаптер публичной версии, в прогоне не использовалась;
логика подготовки — как в mm_s12_m1_embed.prepare, то есть как в прогоне для локальных файлов):

  python -m experiments.paper1.mm_s4_pipeline local items.jsonl

items.jsonl — по строке на элемент: {"uid", "set", "post_id", "key", "cls", "text", "image" | "video"}, где image/video —
путь к локальному файлу (см. README, «Входные данные»).
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.paper1 import mm_media
from experiments.paper1.config import SRC

W = SRC.parent
PARTS = W / "emb_parts"
TMP = W / "tmp"
MANIFEST = W / "manifest.jsonl"
LOG = W / "pipeline.log"
MIN_FREE_GB, MIN_AVAIL_GB = 30, 1.5


class Stop(Exception):
    pass


def log(msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class Store:
    """Манифест + эмбеддинги порциями."""

    def __init__(self):
        PARTS.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict] = {}
        if MANIFEST.exists():
            for line in MANIFEST.read_text().splitlines():
                r = json.loads(line)
                self.done[r["uid"]] = r
        self.part = len(list(PARTS.glob("*.npy")))
        self.lock = threading.Lock()

    def save(self, rows: list[dict], emb: np.ndarray | None, emb_uids: list[str]) -> None:
        with self.lock:
            if emb is not None and len(emb_uids):
                np.save(PARTS / f"p{self.part:05d}.npy", emb.astype(np.float32))
                (PARTS / f"p{self.part:05d}.json").write_text(json.dumps(emb_uids))
                self.part += 1
            with open(MANIFEST, "a") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    self.done[r["uid"]] = r


class Guard:
    @staticmethod
    def resources() -> str:
        """Свободное место в рабочем каталоге и доступная память (в исходном прогоне — диск C: WSL)."""
        W.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(W).free / 2**30
        try:
            avail = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 2**20
        except OSError:
            avail = float("inf")
        if free < MIN_FREE_GB:
            raise Stop(f"свободно {free:.1f} ГБ < {MIN_FREE_GB}")
        if avail < MIN_AVAIL_GB:
            raise Stop(f"доступной памяти {avail:.1f} ГБ < {MIN_AVAIL_GB}")
        return f"диск: {free:.0f} ГБ, RAM avail {avail:.1f} ГБ"


class Pipeline:
    def __init__(self):
        from experiments.paper1 import mm_embed

        self.mm_embed = mm_embed
        self.store = Store()
        self.guard = Guard()
        self.model = None
        self.t0 = time.time()
        self.n_media = 0

    def load_model(self):
        if self.model is None:
            self.model = self.mm_embed.load()

    def embed_and_save(self, fetched: list[dict], tmp: Path) -> None:
        self.load_model()
        ok = [f for f in fetched if f["enc"] is not None]
        groups = {"text": [], "image": [], "video": []}
        for f in ok:
            groups["video" if "frames" in f["enc"] else "image" if "image" in f["enc"] else "text"].append(f)
        embs, uids = [], []
        for g, bs in (("text", 16), ("image", 4), ("video", 1)):
            fs = groups[g]
            for i in range(0, len(fs), bs * 8):
                part = fs[i:i + bs * 8]
                try:
                    e = self.mm_embed.encode(self.model, [f["enc"] for f in part], batch_size=bs)
                    embs.append(e)
                    uids += [f["row"]["uid"] for f in part]
                except Exception as ex:  # noqa: BLE001 — по одному, чтобы найти битый элемент
                    log(f"batch error {type(ex).__name__}: {str(ex)[:200]}; по одному")
                    for f in part:
                        try:
                            embs.append(self.mm_embed.encode(self.model, [f["enc"]], batch_size=1))
                            uids.append(f["row"]["uid"])
                        except Exception as e1:  # noqa: BLE001
                            f["row"].update(status="fail_embed", err=f"{type(e1).__name__}: {e1}"[:300])
        good = set(uids)
        rows = [f["row"] for f in fetched if f["row"]["status"] != "deferred"]  # отложенные — не в манифест
        for r in rows:
            r["embedded"] = r["uid"] in good
        self.store.save(rows, np.concatenate(embs) if embs else None, uids)
        shutil.rmtree(tmp, ignore_errors=True)
        self.n_media += len(fetched)


# ------------------------------------------------------------------ адаптер публичной версии: локальные файлы
def prepare_local(it: dict, tmp: Path) -> dict:
    row = {k: it.get(k) for k in ("uid", "set", "post_id", "key", "cls")}
    row.update(url=None, status="ok", modality=None, n_frames=0, err=None)
    text = (it.get("text") or "").strip()
    enc = {"text": text} if text else {}
    try:
        if it.get("video"):
            out = tmp / f"v_{it['post_id']}"
            out.mkdir(parents=True, exist_ok=True)
            frames = mm_media.decode_frames(it["video"])
            info = mm_media.frames_to_dir(frames, out, {})
            if info["keyframes"] == 0:
                raise RuntimeError("0 кадров")
            enc.update(frames=sorted(out.glob("k*.jpg")), frame_idx=info["k_idx"])
            row.update(modality="video+text" if text else "video", n_frames=info["keyframes"])
        elif it.get("image"):
            p = tmp / f"i_{it['post_id']}.jpg"
            with Image.open(it["image"]) as im:
                im = im.convert("RGB")
                im.thumbnail((mm_media.LONG_SIDE, mm_media.LONG_SIDE))
                im.save(p, "JPEG", quality=90)
            enc["image"] = p
            row["modality"] = "image+text" if text else "image"
        elif text:
            row["modality"] = "text"
        else:
            row["status"] = "empty"
            return {"row": row, "enc": None}
        return {"row": row, "enc": enc}
    except Exception as e:  # noqa: BLE001
        row.update(status="fail_media", err=f"{type(e).__name__}: {e}"[:300])
        return {"row": row, "enc": None}


def phase_local(path: str, chunk: int = 40) -> None:
    items = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    pl = Pipeline()
    items = [it for it in items if it["uid"] not in pl.store.done]
    log(f"local: к обработке {len(items)}")
    for i in range(0, len(items), chunk):
        tmp = TMP / f"local_{i}"
        tmp.mkdir(parents=True, exist_ok=True)
        pl.embed_and_save([prepare_local(it, tmp) for it in items[i:i + chunk]], tmp)
        log(f"local: {min(i + chunk, len(items))}/{len(items)}; {pl.guard.resources()}")
    log("local: done")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "local":
        raise SystemExit("usage: python -m experiments.paper1.mm_s4_pipeline local items.jsonl")
    phase_local(sys.argv[2])
