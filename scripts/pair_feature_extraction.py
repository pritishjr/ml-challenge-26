#!/usr/bin/env python3
"""Extract label-free pairwise features from Step 3 candidate pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from candidate_generation import char_grams, soundex


STRATEGIES = (
    "canonical_name", "token_name", "char_name", "phonetic_name", "sorted_name",
    "canonical_address", "token_address", "char_address",
)
ADDRESS_STOP = {
    "and", "the", "of", "near", "opp", "opposite", "beside", "road", "rd", "street",
    "st", "avenue", "ave", "lane", "ln", "drive", "dr", "building", "bldg", "unit",
    "suite", "ste", "district", "state", "province", "country", "city",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/normalized"))
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-name", default="pair_features.tsv")
    parser.add_argument("--report-name", default="feature_quality_report.json")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Debug limit; zero means all candidates.")
    return parser.parse_args()


def text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def tokens(value: object) -> tuple[str, ...]:
    return tuple(sorted(set(text(value).split())))


def grams(value: object) -> Counter[str]:
    value_text = text(value).replace(" ", "_")
    if not value_text:
        return Counter()
    padded = "__" + value_text + "__"
    return Counter(padded[i:i + width] for width in (2, 3) for i in range(len(padded) - width + 1))


def phonetic_codes(value: object) -> set[str]:
    return {soundex(token) for token in tokens(value) if len(token) >= 3 and soundex(token)}


def normalized_edit(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def token_metrics(left: tuple[str, ...], right: tuple[str, ...], idf: dict[str, float]) -> dict[str, float]:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    intersection = left_set & right_set
    jaccard = len(intersection) / len(union) if union else 0.0
    dice = 2.0 * len(intersection) / (len(left_set) + len(right_set)) if left_set or right_set else 0.0
    left_weight = sum(idf.get(token, 1.0) for token in left_set)
    right_weight = sum(idf.get(token, 1.0) for token in right_set)
    shared_weight = sum(idf.get(token, 1.0) for token in intersection)
    weighted_denominator = left_weight + right_weight - shared_weight
    weighted_jaccard = shared_weight / weighted_denominator if weighted_denominator else 0.0
    distinctive = sum(1.0 for token in intersection if idf.get(token, 1.0) >= 2.0)
    return {
        "jaccard": jaccard,
        "dice": dice,
        "idf_jaccard": weighted_jaccard,
        "shared_idf": shared_weight,
        "distinctive_shared_count": distinctive,
        "shared_token_count": float(len(intersection)),
    }


def char_cosine(left: object, right: object) -> float:
    left_counts, right_counts = grams(left), grams(right)
    if not left_counts or not right_counts:
        return 0.0
    shared = set(left_counts) & set(right_counts)
    numerator = sum(left_counts[key] * right_counts[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def field_features(left: str, right: str, left_idf: dict[str, float], prefix: str) -> dict[str, float]:
    left_tokens, right_tokens = tokens(left), tokens(right)
    metrics = token_metrics(left_tokens, right_tokens, left_idf)
    result = {f"{prefix}_{key}": value for key, value in metrics.items()}
    result[f"{prefix}_char_cosine"] = char_cosine(left, right)
    result[f"{prefix}_edit_ratio"] = normalized_edit(left, right)
    result[f"{prefix}_exact"] = float(bool(left and left == right))
    result[f"{prefix}_missing_left"] = float(not left)
    result[f"{prefix}_missing_right"] = float(not right)
    result[f"{prefix}_sparse_left"] = float(len(left_tokens) <= 1)
    result[f"{prefix}_sparse_right"] = float(len(right_tokens) <= 1)
    result[f"{prefix}_token_count_left"] = float(len(left_tokens))
    result[f"{prefix}_token_count_right"] = float(len(right_tokens))
    result[f"{prefix}_token_count_diff"] = float(abs(len(left_tokens) - len(right_tokens)))
    result[f"{prefix}_length_left"] = float(len(left))
    result[f"{prefix}_length_right"] = float(len(right))
    result[f"{prefix}_length_diff"] = float(abs(len(left) - len(right)))
    return result


def idf_from_records(records: list[dict[str, str]], field: str) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for record in records:
        document_frequency.update(set(tokens(record[field])))
    total = max(1, len(records))
    return {token: math.log((1 + total) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()}


def load_records(input_dir: Path, split: str) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for source in (1, 2, 3):
        path = input_dir / f"{split}_source{source}.tsv"
        frame = pd.read_csv(path, sep="\t", usecols=[
            "entity_id", "business_name_norm", "business_address_norm", "country"
        ], dtype=str, keep_default_na=False)
        for row in frame.itertuples(index=False):
            records[row.entity_id] = {
                "name": text(row.business_name_norm),
                "address": text(row.business_address_norm),
                "country": text(row.country),
                "source": row.entity_id[:2],
            }
        del frame
    return records


def read_candidates(path: Path, max_pairs: int = 0) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if {"source1_entity_id", "candidate_entity_id"}.issubset(frame.columns):
        result = frame.copy()
        if "contributing_strategies" not in result:
            result["contributing_strategies"] = ""
        if "similarity_scores" not in result:
            result["similarity_scores"] = "{}"
    elif {"source1_entity_id", "candidate_entity_ids"}.issubset(frame.columns):
        rows: list[dict[str, str]] = []
        for row in frame.itertuples(index=False):
            for candidate_id in text(row.candidate_entity_ids).split(","):
                if candidate_id:
                    rows.append({"source1_entity_id": row.source1_entity_id,
                                 "candidate_entity_id": candidate_id,
                                 "contributing_strategies": "",
                                 "similarity_scores": "{}"})
        result = pd.DataFrame(rows)
    else:
        raise ValueError("Candidate file must be detailed or challenge-compatible Step 3 output")
    if max_pairs:
        result = result.head(max_pairs)
    return result


def address_parts(value: str) -> tuple[set[str], set[str]]:
    current = tokens(value)
    numeric = {token for token in current if any(character.isdigit() for character in token)}
    places = {token for token in current if token not in ADDRESS_STOP and not any(character.isdigit() for character in token)}
    return numeric, places


def parse_scores(value: str) -> dict[str, float]:
    try:
        parsed = json.loads(value) if value else {}
        return {str(key): float(score) for key, score in parsed.items()}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def pair_features(row: pd.Series, records: dict[str, dict[str, str]],
                  name_idf: dict[str, float], address_idf: dict[str, float]) -> dict[str, object]:
    left = records[row.source1_entity_id]
    right = records[row.candidate_entity_id]
    result: dict[str, object] = {
        "source1_entity_id": row.source1_entity_id,
        "candidate_entity_id": row.candidate_entity_id,
        "source_pair_s1_s2": float(right["source"] == "S2"),
        "source_pair_s1_s3": float(right["source"] == "S3"),
        "country_agreement": float(bool(left["country"] and right["country"] and left["country"] == right["country"])),
        "country_missing_either": float(not left["country"] or not right["country"]),
    }
    result.update(field_features(left["name"], right["name"], name_idf, "name"))
    result.update(field_features(left["address"], right["address"], address_idf, "address"))
    left_numbers, left_places = address_parts(left["address"])
    right_numbers, right_places = address_parts(right["address"])
    result["address_shared_numeric_jaccard"] = len(left_numbers & right_numbers) / len(left_numbers | right_numbers) if left_numbers | right_numbers else 0.0
    result["address_shared_place_jaccard"] = len(left_places & right_places) / len(left_places | right_places) if left_places | right_places else 0.0
    result["name_address_product"] = float(result["name_char_cosine"]) * float(result["address_char_cosine"])
    result["name_address_mean"] = (float(result["name_char_cosine"]) + float(result["address_char_cosine"])) / 2.0
    result["name_high_address_low"] = float(float(result["name_char_cosine"]) > 0.8 and float(result["address_char_cosine"]) < 0.2)
    raw_scores = parse_scores(row.similarity_scores)
    strategies = set(filter(None, text(row.contributing_strategies).split("|")))
    result["blocking_strategy_count"] = float(len(strategies))
    for strategy in STRATEGIES:
        result[f"blocked_by_{strategy}"] = float(strategy in strategies)
        result[f"blocking_score_{strategy}"] = raw_scores.get(strategy, 0.0)
    return result


def add_group_features(features: pd.DataFrame) -> pd.DataFrame:
    primary = features["name_char_cosine"] * 0.6 + features["address_char_cosine"] * 0.4
    features["primary_similarity"] = primary
    grouped = features.groupby("source1_entity_id", sort=False)["primary_similarity"]
    features["group_rank"] = grouped.rank(method="min", ascending=False)
    group_max = grouped.transform("max")
    features["group_gap_from_best"] = group_max - primary
    features["group_mean"] = grouped.transform("mean")
    features["group_std"] = grouped.transform("std").fillna(0.0)
    features["group_zscore"] = ((primary - features["group_mean"]) /
                                  features["group_std"].replace(0.0, np.nan)).fillna(0.0)
    group_next = features.groupby("source1_entity_id", sort=False)["primary_similarity"].transform(
        lambda values: values.sort_values(ascending=False).shift(-1).reindex(values.index).fillna(values.min())
    )
    features["group_gap_to_next"] = primary - group_next
    return features


def quality_report(features: pd.DataFrame, records: dict[str, dict[str, str]], split: str,
                   input_dir: Path) -> dict[str, object]:
    numeric = features.select_dtypes(include=[np.number])
    missing = numeric.isna().mean().sort_values(ascending=False)
    variance = numeric.var()
    near_constant = variance[variance <= 1e-8].index.tolist()
    correlation = numeric.corr().abs()
    collinear: list[dict[str, object]] = []
    for index, column in enumerate(correlation.columns):
        for other in correlation.columns[index + 1:]:
            value = correlation.at[column, other]
            if value >= 0.98:
                collinear.append({"feature_a": column, "feature_b": other, "absolute_correlation": float(value)})
    report: dict[str, object] = {
        "rows": len(features),
        "feature_count": len(features.columns) - 2,
        "missing_fraction": {key: float(value) for key, value in missing.items()},
        "near_constant_features": near_constant,
        "highly_collinear_pairs": collinear[:100],
        "highly_collinear_pair_count": len(collinear),
        "sample_vectors": features.head(3).to_dict(orient="records"),
        "notes": [
            "All features are computed from the two records and Step 3 provenance only; labels are never inputs.",
            "Address decomposition uses numeric-token and non-generic place-token overlap; no country-specific parser is assumed.",
            "Missing fields receive explicit flags and zero overlap metrics rather than silently becoming neutral matches.",
            "Semantic embeddings were omitted because no embedding dependency is required for this pipeline.",
        ],
    }
    if split == "train":
        truth = pd.read_csv(input_dir / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
        labels = dict(zip(truth.source1_entity_id, truth.matched_entity_ids))
        target_ids = features.candidate_entity_id.map(lambda value: value in records)
        match_labels = [float(row.candidate_entity_id in set(text(labels.get(row.source1_entity_id)).split(",")))
                        for row in features.itertuples(index=False)]
        correlations = numeric.assign(match_label=match_labels).corr(numeric_only=True)["match_label"].drop("match_label").abs().sort_values(ascending=False)
        report["label_signal"] = {
            "positive_pairs": int(sum(match_labels)),
            "feature_abs_correlation": {key: float(value) for key, value in correlations.head(20).items()},
            "mean_by_label": features.assign(match_label=match_labels).groupby("match_label")[numeric.columns].mean().to_dict(),
        }
    return report


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.input_dir, args.split)
    target_records = [record for record in records.values() if record["source"] in ("S2", "S3")]
    name_idf = idf_from_records(target_records, "name")
    address_idf = idf_from_records(target_records, "address")
    candidates = read_candidates(args.candidate_file, args.max_pairs)
    rows = [pair_features(row, records, name_idf, address_idf) for row in candidates.itertuples(index=False)]
    features = pd.DataFrame(rows)
    features = add_group_features(features)
    output_path = args.output_dir / args.output_name
    features.to_csv(output_path, sep="\t", index=False)
    report = quality_report(features, records, args.split, args.input_dir)
    report["split"] = args.split
    report["candidate_file"] = str(args.candidate_file)
    (args.output_dir / args.report_name).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "report": str(args.output_dir / args.report_name),
                      "rows": len(features), "feature_count": len(features.columns) - 2}, indent=2))


if __name__ == "__main__":
    main()
