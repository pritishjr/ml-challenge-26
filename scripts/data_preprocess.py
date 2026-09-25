#!/usr/bin/env python3
"""Normalize business name and address fields for the entity-resolution challenge.

This step is intentionally conservative and does not remove any records. It creates a
cleaned representation that keeps the original text alongside normalized versions for
feature extraction and downstream matching.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUTPUT_DIR = DATA_DIR / "normalized"

LEGAL_SUFFIXES = {
    "inc": "inc",
    "incorporated": "inc",
    "llc": "llc",
    "ltd": "ltd",
    "limited": "ltd",
    "corp": "corp",
    "corporation": "corp",
    "co": "co",
    "company": "co",
    "pvt": "pvt",
    "private": "pvt",
    "llp": "llp",
    "partners": "partners",
    "partnership": "partners",
    "assoc": "assoc",
    "association": "assoc",
    "gmbh": "gmbh",
    "intl": "intl",
    "international": "intl",
}

ADDRESS_ALIASES = {
    "street": "st",
    "st": "st",
    "road": "rd",
    "rd": "rd",
    "avenue": "ave",
    "ave": "ave",
    "boulevard": "blvd",
    "blvd": "blvd",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "court": "ct",
    "ct": "ct",
    "place": "pl",
    "pl": "pl",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "highway": "hwy",
    "hwy": "hwy",
    "suite": "ste",
    "ste": "ste",
    "apartment": "apt",
    "apt": "apt",
    "building": "bldg",
    "bldg": "bldg",
    "tower": "tower",
    "district": "district",
    "near": "near",
    "nr": "near",
    "opp": "opp",
    "opposite": "opp",
    "beside": "beside",
    "main": "main",
    "center": "center",
    "centre": "center",
    "city": "city",
    "state": "state",
    "province": "province",
    "country": "country",
}


def normalize_unicode(value: str) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return text.strip()


def tokenize(value: str) -> list[str]:
    if not value:
        return []
    text = normalize_unicode(value).lower()
    text = text.replace("&", " and ")
    text = text.replace("+", " plus ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def normalize_name(value: str) -> str:
    tokens = tokenize(value)
    if not tokens:
        return ""

    norm_tokens: list[str] = []
    for token in tokens:
        if token in LEGAL_SUFFIXES:
            norm_tokens.append(LEGAL_SUFFIXES[token])
        else:
            norm_tokens.append(token)

    return " ".join(norm_tokens)


def normalize_address(value: str) -> str:
    tokens = tokenize(value)
    if not tokens:
        return ""

    norm_tokens: list[str] = []
    for token in tokens:
        token = ADDRESS_ALIASES.get(token, token)
        norm_tokens.append(token)

    return " ".join(norm_tokens)


def normalize_record_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["business_name_raw"] = result["business_name"].fillna("")
    result["business_address_raw"] = result["business_address"].fillna("")
    result["business_name_norm"] = result["business_name_raw"].map(normalize_name)
    result["business_address_norm"] = result["business_address_raw"].map(normalize_address)
    result["name_tokens"] = result["business_name_norm"].map(lambda x: x.split())
    result["address_tokens"] = result["business_address_norm"].map(lambda x: x.split())
    return result


def process_source_file(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, sep="\t")
    normalized = normalize_record_frame(df)
    normalized.to_csv(output_path, sep="\t", index=False)


def process_ground_truth(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, sep="\t")
    df.to_csv(output_path, sep="\t", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Training files
    for source in ("source1", "source2", "source3"):
        in_file = DATA_DIR / "train" / f"train_{source}.tsv"
        out_file = OUTPUT_DIR / f"train_{source}.tsv"
        process_source_file(in_file, out_file)

    # Test files
    for source in ("source1", "source2", "source3"):
        in_file = DATA_DIR / "test" / f"test_{source}.tsv"
        out_file = OUTPUT_DIR / f"test_{source}.tsv"
        process_source_file(in_file, out_file)

    gt_in = DATA_DIR / "train" / "train_ground_truth.tsv"
    gt_out = OUTPUT_DIR / "train_ground_truth.tsv"
    process_ground_truth(gt_in, gt_out)

    print("Normalized files written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

