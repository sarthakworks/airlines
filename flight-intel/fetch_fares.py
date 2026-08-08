"""Fetch Indian flight-fare datasets from public GitHub mirrors (no auth/Kaggle).

Lists each repo's file tree, downloads every .csv/.xlsx, and profiles it.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "fares")
os.makedirs(OUT, exist_ok=True)

REPOS = [
    "datasciencelovers/Airlines-Flights-Data-Analysis",  # Clean_Dataset (2022, ~300k rows)
    "Avij112/flight-fare-analysis",                       # multi-year 2016-2024 airfare
    "OludolapoAnalyst/Indian_Flight_Data",                # 12-airline itineraries
]


def fetch_repo(repo: str) -> list[str]:
    saved = []
    meta = requests.get(f"https://api.github.com/repos/{repo}", timeout=30).json()
    branch = meta.get("default_branch", "main")
    tree = requests.get(
        f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1", timeout=30
    ).json()
    for node in tree.get("tree", []):
        path = node.get("path", "")
        if node.get("type") == "blob" and path.lower().endswith((".csv", ".xlsx")):
            size = node.get("size", 0)
            if size > 60_000_000:  # skip anything absurdly large
                print(f"  [skip >60MB] {path}")
                continue
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
            fname = f"{repo.split('/')[-1]}__{os.path.basename(path)}"
            dest = os.path.join(OUT, fname)
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(r.content)
                saved.append(dest)
                print(f"  ↓ {fname}  ({len(r.content)//1024} KB)")
            except Exception as e:  # noqa: BLE001
                print(f"  [err] {path}: {e}")
    return saved


def profile(path: str):
    try:
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    except Exception as e:  # noqa: BLE001
        print(f"    (could not read {os.path.basename(path)}: {e})")
        return
    print(f"\n### {os.path.basename(path)}  shape={df.shape}")
    print("    cols:", list(df.columns)[:25])
    with pd.option_context("display.max_columns", 30, "display.width", 200):
        print(df.head(2).to_string(max_colwidth=16))


if __name__ == "__main__":
    all_saved = []
    for repo in REPOS:
        print(f"\n== {repo} ==")
        all_saved += fetch_repo(repo)
    print("\n" + "=" * 60 + "\nPROFILES")
    for p in all_saved:
        if p.endswith(".csv"):
            profile(p)
    print(f"\nSaved {len(all_saved)} files to {OUT}")
