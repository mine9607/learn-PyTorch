from __future__ import annotations

import io
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

IMDB_URL = "https://ai.standford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def download_if_missing(
    url: str = IMDB_URL, dest: Path = Path("aclImdb_v1.tar.gz")
) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def _safe_extractall(tar: tarfile.Tarfile, path: Path) -> None:
    # Prevent path traversal
    for m in tar.getmembers():
        p = (path / m.name).resolve()
        if not str(p).startswith(str(path.resolve())):
            raise RuntimeError(f"Blocked unsafe path: {p}")

    tar.extractall(path)


def extract_if_missing(tar_path: Path, target_dir: Path = Path("aclImdb")) -> Path:
    if target_dir.exists():
        return target_dir
    with tarfile.open(tar_path, "r:gz") as tar:
        _safe_extractall(tar, Path("."))
    return target_dir


def build_imdb_df(basepath: Path = Path("aclImdb")) -> pd.DataFrame:
    labels = {"pos": 1, "neg": 0}
    rows: List[Tuple[str, int]] = []
    for split in ("train", "test"):
        for label, y in labels.items():
            dir_ = basepath / split / label
            files = sorted(os.listdir(dir_))
            for fname in tqdm(files, desc=f"{split}/{label}"):
                with open(dir_ / fname, "r", encoding="utf-8") as f:
                    rows.append((f.read(), y))
    df = pd.DataFrame(rows, columns=["review", "sentiment"])
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


def ensure_movie_csv(csv_path: Path = Path("movie_data.csv")) -> Path:
    if csv_path.exists():
        return csv_path
    tarball = download_if_missing()
    _ = extract_if_missing(tarball, Path("aclImdb"))
    df = build_imdb_df(Path("aclImdb"))
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path
