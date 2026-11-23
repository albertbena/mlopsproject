"""Download the monthly NYC TLC yellow taxi files listed in params.yaml into data/raw/.

Source: NYC Taxi & Limousine Commission (TLC) Trip Record Data,
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The raw files are never committed: this script plus `dvc repro` rebuilds them. A file already on disk with
the expected size is not downloaded again.

    python -m taxi.data.download
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

from taxi.config import params, path

CHUNK = 1 << 20


def file_name(month: str) -> str:
    return f"yellow_tripdata_{month}.parquet"


def download(url: str, dest: Path, attempts: int = 4) -> int:
    """Stream url to dest (through a .part file, so a cut download never looks complete). Returns bytes."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                expected = int(r.headers.get("content-length", 0))
                if dest.exists() and expected and dest.stat().st_size == expected:
                    return expected
                size = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        size += len(chunk)
            if expected and size != expected:
                raise IOError(f"got {size} bytes, expected {expected}")
            tmp.replace(dest)
            return size
        except (requests.RequestException, IOError) as err:
            if attempt == attempts:
                raise
            wait = 2 ** attempt
            print(f"  {dest.name}: {err}, retrying in {wait} s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> None:
    cfg = params()["data"]
    out = path(cfg["raw_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for month in cfg["months"]:
        name = file_name(month)
        size = download(f"{cfg['base_url']}/{name}", out / name)
        print(f"{name}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
