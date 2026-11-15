"""Download immutable originals, with retries, cached hashes and provenance."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

REPO = "djordjebatic/GridCharge"
REVISION = "fd70bf347d6cf8596b7e45d9658cb6d3fd55b7a0"
MONTHS = "january february march april may june july august september october november december".split()
CPS_PAGE = "https://chargeplacescotland.org/monthly-charge-point-performance/"
SEPTEMBER_URL = "https://chargeplacescotland.org/wp-content/uploads/2024/11/Sessions-Sept-24-CLEAN-2-1.xlsx"


def fetch(url, destination, expected=None, previous=None):
    destination = Path(destination)
    if destination.exists() and previous:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != previous["sha256"]:
            raise ValueError(f"Cached original changed: {destination}")
        return previous
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(20, 180))
            response.raise_for_status()
            content = response.content
            if "Content-Length" in response.headers and not response.headers.get("Content-Encoding"):
                assert len(content) == int(response.headers["Content-Length"]), "Truncated HTTP response"
            digest = hashlib.sha256(content).hexdigest()
            if expected:
                observed = digest if len(expected) == 64 else hashlib.sha1(
                    f"blob {len(content)}\0".encode() + content).hexdigest()
                assert observed == expected, f"Repository hash mismatch: {url}"
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite different original: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".part")
            partial.write_bytes(content)
            partial.replace(destination)
            return {"url": url, "sha256": digest, "bytes": len(content),
                    "upstream_hash": expected, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
        except (requests.RequestException, AssertionError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def download_all(root):
    raw = Path(root) / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    meta_path = raw / "repository_pinned.json"
    api = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
    manifest["repository_pinned.json"] = fetch(api, meta_path, previous=manifest.get("repository_pinned.json"))
    info = json.loads(meta_path.read_text())
    entries = {x["rfilename"]: x for x in info["siblings"]}
    wanted = [f"sessions/{month}-{year}.csv" for year in (2023, 2024) for month in MONTHS]
    missing = [path for path in wanted if path not in entries]
    tasks = [path for path in wanted if path in entries]
    tasks += [path for path in entries if path.endswith("/charging_infrastructure.csv")]
    tasks += ["README.md", "process.py"]

    def get(path):
        entry = entries[path]
        expected = (entry.get("lfs") or {}).get("sha256") or entry.get("blobId")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(path)}"
        result = fetch(url, raw / path, expected, manifest.get(path))
        print(f"Verified {path}", flush=True)
        return path, result

    with ThreadPoolExecutor(max_workers=4) as pool:
        for path, result in pool.map(get, tasks):
            manifest[path] = result
            manifest_path.write_text(json.dumps(manifest, indent=2))
    supplements = {"official/monthly-performance.html": CPS_PAGE,
                   "official/september-2024.xlsx": SEPTEMBER_URL}
    for path, url in supplements.items():
        manifest[path] = fetch(url, raw / path, previous=manifest.get(path))
        manifest_path.write_text(json.dumps(manifest, indent=2))
    status = {"revision": REVISION, "missing_gridcharge_files": missing,
              "supplement": {"month": "2024-09", "source": SEPTEMBER_URL},
              "downloaded_gridcharge_months": len([p for p in tasks if p.startswith("sessions/")])}
    (raw / "source_status.json").write_text(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    download_all(Path(__file__).resolve().parents[2] / "gridcharge-data")
