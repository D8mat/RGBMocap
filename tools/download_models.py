"""Download research weights. SMPL / SMPL-X are deliberately not downloaded."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap.models import BY_KEY, ready, download
from mocap.storage import atomic_json, sha256

# Mirror used by the demo linked from the official GVHMR README:
# https://huggingface.co/spaces/LittleFrog/GVHMR/blob/main/app.py
MIRROR = "https://huggingface.co/camenduru/GVHMR/resolve/main/"
EXPECTED_GVHMR_SHA256 = "4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91"


def fetch(key):
    import requests
    model = BY_KEY[key]
    if ready(model):
        print("Already present:", model.relative, flush=True)
        return
    if key not in {"gvhmr", "hmr2", "vitpose"}:
        download(key)
        return
    model.path.parent.mkdir(parents=True, exist_ok=True)
    part = model.path.with_suffix(model.path.suffix + ".part")
    url = MIRROR + model.relative.replace("=", "%3D")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with part.open("wb") as output:
            for chunk in response.iter_content(8 * 1024 * 1024):
                output.write(chunk)
    if part.stat().st_size < model.minimum_bytes:
        raise RuntimeError(f"Incomplete download: {key}")
    checksum = sha256(part)
    if key == "gvhmr" and checksum != EXPECTED_GVHMR_SHA256:
        raise RuntimeError("GVHMR checkpoint hash mismatch")
    part.replace(model.path)
    atomic_json(model.path.with_suffix(model.path.suffix + ".receipt.json"), {"source": url, "sha256": checksum, "bytes": model.path.stat().st_size})
    print("Downloaded:", model.relative, model.path.stat().st_size, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["gvhmr", "hmr2", "vitpose", "yolo", "pva"])
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, args.models))
