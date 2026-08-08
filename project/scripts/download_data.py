"""Download the flight CSV and the FAQ PDF into ./data (idempotent).

Usage:  python scripts/download_data.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python scripts/x.py`

from app import config  # noqa: E402

FILES = {
    config.CSV_PATH: "https://github.com/MLOPS-test/Artifacts/raw/refs/heads/main/datasets/Flights_Schedule_Data_v1.csv",
    config.PDF_PATH: "https://raw.githubusercontent.com/MLOPS-test/Artifacts/refs/heads/main/datasets/Knowledge_Base_for_Airline_Info_and_FAQs.pdf",
}


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dest, url in FILES.items():
        if Path(dest).exists() and Path(dest).stat().st_size > 0:
            print(f"✓ already present: {Path(dest).name}")
            continue
        print(f"↓ downloading {Path(dest).name} …")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved {Path(dest).stat().st_size} bytes")
    print("Done.")


if __name__ == "__main__":
    main()
