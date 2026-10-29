"""Configuration, paths and small I/O helpers shared by all pipeline stages."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

AA = list("ARNDCQEGHILKMFPSTWYV")
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "CSO": "C",
}

MODEL_ORDER = ["lr", "rf", "svm", "xgb", "knn"]
MODEL_NAMES = {
    "lr": "Logistic regression",
    "rf": "Random forest",
    "svm": "Support vector machine",
    "xgb": "XGBoost",
    "knn": "k-nearest neighbours",
}


@lru_cache(maxsize=1)
def config() -> dict:
    with open(ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


def path(key: str) -> Path:
    p = ROOT / config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw(*parts: str) -> Path:
    return ROOT / config()["paths"]["raw"] / Path(*parts)


def processed(name: str) -> Path:
    return path("processed") / name


def table(name: str) -> Path:
    return path("tables") / name


def read_processed(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(processed(name), **kw)


def save_csv(df: pd.DataFrame, p: Path, index: bool = False) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index)
    return p


def update_values(section: str, values: dict) -> None:
    """Merge numbers reported in the manuscript into results/values.json."""
    p = ROOT / "results" / "values.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    data[section] = values
    p.write_text(json.dumps(data, indent=2, default=float))


def load_values() -> dict:
    return json.loads((ROOT / "results" / "values.json").read_text())


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


def domain_of(position: int) -> str:
    for name, start, end in config()["domains"]:
        if start <= position <= end:
            return name
    return "Outside model"
