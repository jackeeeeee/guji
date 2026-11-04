"""Plate recheck / prediction service.

Run locally:
    pip install -r requirements.txt
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                        -> frontend
    GET  /api/datasets            -> built-in datasets (no upload needed)
    GET  /api/results?dataset=green|blue|all&errors_only=true|false
                                     -> precomputed trial results (HyperLPR3,
                                        CPU, 2026-11-03 night run). Each row:
                                        image, predicted, ground truth, time.
    POST /api/predict-live         -> actually run HyperLPR3 on images under
                                     $PLATE_DATA_DIR (see scripts/download_ccpd.sh).
                                     Returns 409 if no dataset images are present.
    GET  /api/image/<filename>    -> sample / dataset image, if present locally.

No LLM / multimodal generative model is used for OCR. The model is
HyperLPR3 (ONNX Runtime, CPU by default), a small plate-recognition model.
"""
import csv
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
# Precomputed trial results always ship with the package; dataset images
# for live prediction may live elsewhere (PLATE_DATA_DIR, e.g. a CCPD checkout).
DATA_DIR = ROOT / "data"
IMAGE_DATA_DIR = Path(os.environ.get("PLATE_DATA_DIR", DATA_DIR))
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="plate-prediction-service")

DATASETS = {
    "green": {"label": "绿牌 8 位（CCPD2020 test）", "file": "precomputed_green300.csv", "type": "green"},
    "blue": {"label": "蓝牌 7 位（CCPD2019 ccpd_fn）", "file": "precomputed_blue100.csv", "type": "blue"},
}


def _load_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _image_url(filename):
    # Only serve a file that actually exists under an allowed directory.
    candidates = [
        DATA_DIR / "samples" / "cases" / "green_correct" / filename,
        DATA_DIR / "samples" / "cases" / "green_wrong" / filename,
        DATA_DIR / "samples" / "cases" / "blue_wrong" / filename,
        IMAGE_DATA_DIR / "CCPD2020" / "ccpd_green" / "test" / filename,
        IMAGE_DATA_DIR / "ccpd2019_partial" / "CCPD2019" / "ccpd_fn" / filename,
    ]
    for p in candidates:
        if p.exists():
            return f"/api/image/{filename}"
    return None


def _summarise(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    full = sum(int(r["full_correct"]) for r in rows)
    cc = sum(int(r["char_correct"]) for r in rows)
    ct = sum(int(r["char_total"]) for r in rows)
    detected = sum(1 for r in rows if r["predicted"])
    times = [float(r["time_sec"]) for r in rows]
    out = {
        "n": n,
        "detected": detected,
        "detected_rate": detected / n,
        "full_correct": full,
        "full_accuracy": full / n,
        "char_accuracy": cc / ct if ct else 0,
        "total_time_sec": round(sum(times), 2),
        "mean_time_sec": round(sum(times) / len(times), 3),
    }
    green = [r for r in rows if r["type"] == "green"]
    if green:
        def _df_ok(r):
            if "df_correct" in r:
                return bool(r["df_correct"])
            return r.get("df_char_correct") == "True"
        df = [r for r in green if r.get("df_correct") is not None or r.get("df_char_correct") != ""]
        green_detected = sum(1 for r in green if r["predicted"])
        out["df_accuracy_all"] = sum(1 for r in df if _df_ok(r)) / len(df) if df else 0
        out["df_accuracy_detected"] = (
            sum(1 for r in df if _df_ok(r) and r["predicted"]) / green_detected
            if green_detected else 0
        )
    return out


def _rows_for(dataset):
    if dataset == "all":
        rows = _load_csv("precomputed_green300.csv") + _load_csv("precomputed_blue100.csv")
    elif dataset in DATASETS:
        rows = _load_csv(DATASETS[dataset]["file"])
    else:
        raise HTTPException(status_code=400, detail="dataset must be green, blue or all")
    out = []
    for r in rows:
        out.append({
            "filename": r["filename"],
            "type": r["type"],
            "ground_truth": r["ground_truth"],
            "predicted": r["predicted"],
            "full_correct": r["full_correct"] == "1",
            "pred_len": int(r["pred_len"]),
            "gt_len": int(r["gt_len"]),
            "char_correct": int(r["char_correct"]),
            "char_total": int(r["char_total"]),
            "confidence": r["confidence"],
            "time_sec": float(r["time_sec"]),
            "df_correct": (r["type"] == "green" and len(r["ground_truth"]) >= 3 and len(r["predicted"]) >= 3 and r["predicted"][2] == r["ground_truth"][2]) if r["predicted"] else False,
            "image_url": _image_url(r["filename"]),
        })
    return out


@app.get("/api/datasets")
def datasets():
    return [
        {"id": "green", "label": DATASETS["green"]["label"], "n": len(_load_csv("precomputed_green300.csv"))},
        {"id": "blue", "label": DATASETS["blue"]["label"], "n": len(_load_csv("precomputed_blue100.csv"))},
        {"id": "all", "label": "全部（绿牌 + 蓝牌）", "n": len(_load_csv("precomputed_green300.csv")) + len(_load_csv("precomputed_blue100.csv"))},
    ]


@app.get("/api/results")
def results(dataset: str = "green", errors_only: bool = False):
    rows = _rows_for(dataset)
    if errors_only:
        rows = [r for r in rows if not r["full_correct"]]
    return {"mode": "precomputed", "dataset": dataset, "summary": _summarise(rows) if not errors_only else _summarise(_rows_for(dataset)), "rows": rows}


@app.post("/api/predict-live")
def predict_live(payload: dict):
    """Actually run HyperLPR3 on dataset images present under PLATE_DATA_DIR."""
    dataset = payload.get("dataset", "green")
    limit = int(payload.get("limit", 50))
    search_dirs = []
    if dataset in ("green", "all"):
        search_dirs.append(IMAGE_DATA_DIR / "CCPD2020" / "ccpd_green" / "test")
    if dataset in ("blue", "all"):
        search_dirs.append(IMAGE_DATA_DIR / "ccpd2019_partial" / "CCPD2019" / "ccpd_fn")
    files = []
    for d in search_dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.jpg"))[:limit])
    if not files:
        raise HTTPException(
            status_code=409,
            detail="No dataset images under PLATE_DATA_DIR. Run scripts/download_ccpd.sh first, or use /api/results (precomputed).",
        )
    # Imported lazily so the demo endpoints work even without the model installed.
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from ccpd_parse import parse_ccpd_filename  # noqa: E402
    from hyperlpr3 import LicensePlateCatcher  # noqa: E402
    import cv2  # noqa: E402

    catcher = LicensePlateCatcher()
    rows = []
    for f in files:
        gt, _idx, _verts, _b, _bl = parse_ccpd_filename(f.name)
        img = cv2.imread(str(f))
        t0 = time.time()
        plates = catcher(img) if img is not None else []
        dt = time.time() - t0
        if plates:
            p = plates[0]
            pred = p[0] if isinstance(p, (list, tuple)) else p.get("plate_code", "") if isinstance(p, dict) else getattr(p, "plate_code", "")
            raw_conf = p[1] if isinstance(p, (list, tuple)) and len(p) > 1 else ""
            conf = float(raw_conf) if raw_conf != "" and raw_conf is not None else ""
        else:
            pred, conf = "", ""
        rows.append({
            "filename": f.name, "type": "green" if "ccpd_green" in str(f) else "blue",
            "ground_truth": gt, "predicted": pred or "",
            "full_correct": (pred or "") == gt,
            "pred_len": len(pred or ""), "gt_len": len(gt),
            "char_correct": sum(1 for a, b in zip(gt, pred or "") if a == b),
            "char_total": len(gt), "confidence": conf, "time_sec": round(dt, 3),
            "image_url": _image_url(f.name),
        })
    return {"mode": "live", "dataset": dataset, "summary": _summarise([{
        "full_correct": "1" if r["full_correct"] else "0", "char_correct": str(r["char_correct"]),
        "char_total": str(r["char_total"]), "predicted": r["predicted"], "type": r["type"],
        "df_correct": (r["type"] == "green" and len(r["ground_truth"]) >= 3 and len(r["predicted"]) >= 3 and r["predicted"][2] == r["ground_truth"][2]) if r["predicted"] else False,
        "time_sec": str(r["time_sec"]),
    } for r in rows]), "rows": rows}


@app.get("/api/image/{filename}")
def image(filename: str):
    # Only a bare filename is accepted; no path traversal.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="bad filename")
    for base in (DATA_DIR / "samples" / "cases", IMAGE_DATA_DIR / "CCPD2020" / "ccpd_green" / "test",
                 IMAGE_DATA_DIR / "ccpd2019_partial" / "CCPD2019" / "ccpd_fn"):
        for sub in (base, base / "green_correct", base / "green_wrong", base / "blue_wrong"):
            p = sub / filename
            if p.exists():
                return FileResponse(p)
    raise HTTPException(status_code=404, detail="image not included in this package; download the dataset to serve it")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
