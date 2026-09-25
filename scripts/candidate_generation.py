#!/usr/bin/env python3
"""High-recall blocking for the business entity-resolution challenge.

This script only generates candidate pairs. It does not classify or filter matches.
It writes a challenge-compatible candidate_pairs.tsv plus a detailed pair file with
strategy provenance and raw retrieval scores.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import math
import re
import time
import unicodedata
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


STRATEGIES = (
    "canonical_name",
    "token_name",
    "char_name",
    "phonetic_name",
    "sorted_name",
    "canonical_address",
    "token_address",
    "char_address",
)

GENERIC_NAME_TOKENS = {
    "and", "the", "of", "for", "at", "inc", "llc", "ltd", "corp", "co",
    "pvt", "llp", "company", "limited", "private", "international", "intl",
    "group", "holdings", "services", "solutions", "store", "shop", "hotel",
}


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/normalized"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top character-ngram candidates per S1 record and field.")
    parser.add_argument("--max-postings", type=int, default=5000,
                        help="Maximum postings read from one token/ngram/phonetic key.")
    parser.add_argument("--max-token-candidates", type=int, default=300)
    parser.add_argument("--sorted-window", type=int, default=25)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Debug limit for S1 rows; zero means all rows.")
    parser.add_argument("--target-max-rows", type=int, default=0,
                        help="Debug limit for S2/S3 rows; zero means all rows.")
    parser.add_argument("--report-name", default="blocking_report.json")
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def token_set(value: object) -> tuple[str, ...]:
    return tuple(sorted(set(clean(value).split())))


def canonical_key(value: object) -> str:
    tokens = [token for token in token_set(value) if token not in GENERIC_NAME_TOKENS]
    return " ".join(tokens)


def char_grams(value: object) -> tuple[str, ...]:
    text = clean(value).replace(" ", "_")
    if not text:
        return ()
    padded = "__" + text + "__"
    return tuple(sorted(set(padded[i:i + 3] for i in range(len(padded) - 2))))


def soundex(token: str) -> str:
    """Small dependency-free Soundex implementation for Latin tokens."""
    token = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("ascii")
    token = re.sub(r"[^A-Za-z]", "", token).upper()
    if not token:
        return ""
    groups = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
    lookup = {letter: digit for letters, digit in groups.items() for letter in letters}
    first = token[0]
    previous = lookup.get(first, "")
    digits: list[str] = []
    for letter in token[1:]:
        digit = lookup.get(letter, "")
        if digit and digit != previous:
            digits.append(digit)
        previous = digit
    return (first + "".join(digits) + "000")[:4]


def phonetic_keys(value: object) -> tuple[str, ...]:
    return tuple(sorted(set(code for token in token_set(value) if len(token) >= 3
                            for code in (soundex(token),) if code)))


@dataclass
class TargetRecords:
    ids: list[str]
    names: list[str]
    addresses: list[str]
    countries: list[str]


def load_targets(paths: Iterable[Path], max_rows: int = 0) -> TargetRecords:
    ids: list[str] = []
    names: list[str] = []
    addresses: list[str] = []
    countries: list[str] = []
    for path in paths:
        frame = pd.read_csv(path, sep="\t", usecols=[
            "entity_id", "business_name_norm", "business_address_norm", "country"
        ], dtype=str, keep_default_na=False)
        if max_rows:
            frame = frame.head(max_rows)
        ids.extend(frame["entity_id"].tolist())
        names.extend(frame["business_name_norm"].tolist())
        addresses.extend(frame["business_address_norm"].tolist())
        countries.extend(frame["country"].tolist())
        del frame
    return TargetRecords(ids, names, addresses, countries)


def add_posting(index: dict[str, list[int]], key: str, row: int) -> None:
    if key:
        index.setdefault(key, []).append(row)


def build_postings(values: list[str], key_fn, max_df: int | None = None) -> tuple[dict[str, list[int]], Counter[str]]:
    index: dict[str, list[int]] = {}
    frequencies: Counter[str] = Counter()
    for row, value in enumerate(values):
        keys = key_fn(value)
        for key in keys if isinstance(keys, tuple) else (keys,):
            if key:
                frequencies[key] += 1
                add_posting(index, key, row)
    if max_df is not None:
        index = {key: rows for key, rows in index.items() if len(rows) <= max_df}
    return index, frequencies


def build_char_index(values: list[str], max_df: int) -> tuple[dict[str, list[int]], dict[str, float], list[float]]:
    index, frequencies = build_postings(values, char_grams, max_df)
    total = max(1, len(values))
    weights = {key: math.log((1 + total) / (1 + frequency)) + 1.0
               for key, frequency in frequencies.items() if key in index}
    norms: list[float] = []
    for value in values:
        norms.append(math.sqrt(sum(weights.get(key, 0.0) ** 2 for key in char_grams(value))) or 1.0)
    return index, weights, norms


def limited(rows: list[int], limit: int) -> list[int]:
    return rows if len(rows) <= limit else rows[:limit]


def add_hit(hits: dict[int, dict[str, float]], row: int, strategy: str, score: float = 1.0) -> None:
    hits.setdefault(row, {})[strategy] = max(score, hits.setdefault(row, {}).get(strategy, 0.0))


def exact_hits(value: str, index: dict[str, list[int]], strategy: str, limit: int,
               hits: dict[int, dict[str, float]]) -> int:
    key = canonical_key(value)
    if not key:
        return 0
    rows = limited(index.get(key, []), limit)
    for row in rows:
        add_hit(hits, row, strategy, 1.0)
    return len(rows)


def token_hits(value: str, index: dict[str, list[int]], frequencies: Counter[str],
               strategy: str, max_postings: int, max_candidates: int,
               hits: dict[int, dict[str, float]]) -> int:
    tokens = [token for token in token_set(value) if len(token) >= 2 and token not in GENERIC_NAME_TOKENS]
    tokens = sorted((token for token in tokens if token in index), key=lambda token: frequencies[token])
    counts: Counter[int] = Counter()
    weights: defaultdict[int, float] = defaultdict(float)
    for token in tokens[:8]:
        rows = limited(index[token], max_postings)
        weight = math.log1p(max(1, sum(frequencies.values())) / max(1, frequencies[token]))
        for row in rows:
            counts[row] += 1
            weights[row] += weight
    selected = sorted(counts, key=lambda row: (counts[row], weights[row]), reverse=True)[:max_candidates]
    for row in selected:
        add_hit(hits, row, strategy, weights[row] / max(1, len(tokens)))
    return len(selected)


def char_hits(value: str, index: dict[str, list[int]], weights: dict[str, float], norms: list[float],
              strategy: str, top_k: int, max_postings: int,
              hits: dict[int, dict[str, float]]) -> int:
    grams = char_grams(value)
    if not grams:
        return 0
    query_norm = math.sqrt(sum(weights.get(key, 0.0) ** 2 for key in grams)) or 1.0
    scores: defaultdict[int, float] = defaultdict(float)
    for key in grams:
        weight = weights.get(key, 0.0)
        if not weight:
            continue
        for row in limited(index.get(key, []), max_postings):
            scores[row] += weight * weight
    ranked = sorted(scores, key=lambda row: scores[row] / norms[row], reverse=True)[:top_k]
    for row in ranked:
        add_hit(hits, row, strategy, scores[row] / (query_norm * norms[row]))
    return len(ranked)


def phonetic_hits(value: str, index: dict[str, list[int]], max_postings: int,
                  hits: dict[int, dict[str, float]]) -> int:
    rows: set[int] = set()
    for key in phonetic_keys(value):
        rows.update(limited(index.get(key, []), max_postings))
    for row in rows:
        add_hit(hits, row, "phonetic_name", 1.0)
    return len(rows)


def sorted_hits(value: str, country: str, sorted_keys: list[str], sorted_rows: list[int],
                window: int, hits: dict[int, dict[str, float]]) -> int:
    key = f"{country}\t{canonical_key(value)[:12]}"
    position = bisect_left(sorted_keys, key)
    selected = sorted_rows[max(0, position - window):position + window + 1]
    for row in selected:
        add_hit(hits, row, "sorted_name", 1.0)
    return len(selected)


def quality_report(s1: pd.DataFrame, targets: TargetRecords) -> dict[str, object]:
    all_names = s1["business_name_norm"].tolist() + targets.names
    all_addresses = s1["business_address_norm"].tolist() + targets.addresses
    return {
        "s1_rows": len(s1),
        "target_rows": len(targets.ids),
        "s1_empty_name": sum(not clean(value) for value in s1["business_name_norm"]),
        "target_empty_name": sum(not clean(value) for value in targets.names),
        "s1_empty_address": sum(not clean(value) for value in s1["business_address_norm"]),
        "target_empty_address": sum(not clean(value) for value in targets.addresses),
        "non_ascii_name_values": sum(any(ord(char) > 127 for char in clean(value)) for value in all_names),
        "non_ascii_address_values": sum(any(ord(char) > 127 for char in clean(value)) for value in all_addresses),
        "s1_countries": sorted(set(s1["country"].tolist())),
        "target_countries": sorted(set(targets.countries)),
    }


def parse_truth(value: str) -> set[str]:
    return {item for item in clean(value).split(",") if item}


def validate_training(output_pairs: dict[str, dict[str, dict[str, float]]], path: Path,
                      target_ids: set[str]) -> dict[str, object]:
    truth = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    total = 0
    covered = 0
    for row in truth.itertuples(index=False):
        if row.source1_entity_id not in output_pairs:
            continue
        expected = parse_truth(row.matched_entity_ids) & target_ids
        actual = output_pairs.get(row.source1_entity_id, {})
        total += len(expected)
        covered += sum(entity_id in actual for entity_id in expected)
    candidate_count = sum(len(value) for value in output_pairs.values())
    return {
        "true_pairs": total,
        "true_pairs_in_union": covered,
        "pair_completeness": covered / total if total else 1.0,
        "full_cross_product_pairs": None,
        "candidate_pairs": candidate_count,
    }


def write_outputs(output_dir: Path, rows: list[tuple[str, str, dict[str, float]]],
                  s1_ids: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / "candidate_pairs_detailed.tsv"
    with detailed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "contributing_strategies",
                         "similarity_scores"])
        for source1_id, candidate_id, scores in rows:
            strategies = "|".join(sorted(scores))
            writer.writerow([source1_id, candidate_id, strategies, json.dumps(scores, sort_keys=True)])

    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for source1_id, candidate_id, _ in rows:
        grouped[source1_id].append(candidate_id)
    with (output_dir / "candidate_pairs.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for source1_id in s1_ids:
            writer.writerow([source1_id, ",".join(grouped[source1_id])])


def main() -> None:
    args = cli()
    start = time.perf_counter()
    input_dir = args.input_dir
    s1_path = input_dir / f"{args.split}_source1.tsv"
    s1 = pd.read_csv(s1_path, sep="\t", usecols=[
        "entity_id", "business_name_norm", "business_address_norm", "country"
    ], dtype=str, keep_default_na=False)
    if args.max_rows:
        s1 = s1.head(args.max_rows).copy()
    target_paths = [input_dir / f"{args.split}_source{source}.tsv" for source in (2, 3)]
    targets = load_targets(target_paths, args.target_max_rows)
    parameters = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
    report: dict[str, object] = {"split": args.split, "parameters": parameters,
                                 "quality": quality_report(s1, targets), "strategies": {}}

    name_canonical, _ = build_postings(targets.names, canonical_key, max_df=args.max_postings)
    name_tokens, name_token_df = build_postings(targets.names, token_set, max_df=None)
    name_chars, name_char_weights, name_char_norms = build_char_index(targets.names, args.max_postings)
    name_phonetic, _ = build_postings(targets.names, phonetic_keys, max_df=args.max_postings)
    address_canonical, _ = build_postings(targets.addresses, canonical_key, max_df=args.max_postings)
    address_tokens, address_token_df = build_postings(targets.addresses, token_set, max_df=None)
    address_chars, address_char_weights, address_char_norms = build_char_index(targets.addresses, args.max_postings)

    target_sorted = sorted((f"{targets.countries[row]}\t{canonical_key(targets.names[row])[:12]}", row)
                           for row in range(len(targets.ids)) if canonical_key(targets.names[row]))
    sorted_keys = [item[0] for item in target_sorted]
    sorted_rows = [item[1] for item in target_sorted]

    pair_map: dict[str, dict[str, dict[str, float]]] = {}
    strategy_pairs: Counter[str] = Counter()
    strategy_times: dict[str, float] = defaultdict(float)
    s1_ids = s1["entity_id"].tolist()
    detailed_rows: list[tuple[str, str, dict[str, float]]] = []
    for query in s1.itertuples(index=False):
        hits: dict[int, dict[str, float]] = {}
        operations = (
            ("canonical_name", lambda local_hits: exact_hits(query.business_name_norm, name_canonical, "canonical_name", args.max_postings, local_hits)),
            ("token_name", lambda local_hits: token_hits(query.business_name_norm, name_tokens, name_token_df, "token_name", args.max_postings, args.max_token_candidates, local_hits)),
            ("char_name", lambda local_hits: char_hits(query.business_name_norm, name_chars, name_char_weights, name_char_norms, "char_name", args.top_k, args.max_postings, local_hits)),
            ("phonetic_name", lambda local_hits: phonetic_hits(query.business_name_norm, name_phonetic, args.max_postings, local_hits)),
            ("sorted_name", lambda local_hits: sorted_hits(query.business_name_norm, query.country, sorted_keys, sorted_rows, args.sorted_window, local_hits)),
            ("canonical_address", lambda local_hits: exact_hits(query.business_address_norm, address_canonical, "canonical_address", args.max_postings, local_hits)),
            ("token_address", lambda local_hits: token_hits(query.business_address_norm, address_tokens, address_token_df, "token_address", args.max_postings, args.max_token_candidates, local_hits)),
            ("char_address", lambda local_hits: char_hits(query.business_address_norm, address_chars, address_char_weights, address_char_norms, "char_address", args.top_k, args.max_postings, local_hits)),
        )
        def run_operation(item: tuple[str, object]) -> tuple[str, dict[int, dict[str, float]], float]:
            strategy, operation = item
            local_hits: dict[int, dict[str, float]] = {}
            operation_start = time.perf_counter()
            operation(local_hits)
            return strategy, local_hits, time.perf_counter() - operation_start

        with ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as executor:
            results = list(executor.map(run_operation, operations))
        for strategy, local_hits, elapsed in results:
            strategy_times[strategy] += elapsed
            for target_row, scores in local_hits.items():
                for local_strategy, score in scores.items():
                    add_hit(hits, target_row, local_strategy, score)
            strategy_pairs[strategy] += len(local_hits)
        current: dict[str, dict[str, float]] = {}
        for target_row, scores in hits.items():
            candidate_id = targets.ids[target_row]
            current[candidate_id] = scores
            detailed_rows.append((query.entity_id, candidate_id, scores))
        pair_map[query.entity_id] = current

    write_outputs(args.output_dir, detailed_rows, s1_ids)
    if args.split == "train":
        validation = validate_training(pair_map, input_dir / "train_ground_truth.tsv",
                           set(targets.ids))
        full_space = len(s1) * len(targets.ids)
        validation["full_cross_product_pairs"] = full_space
        validation["reduction_ratio"] = 1.0 - validation["candidate_pairs"] / full_space
        validation["is_sampled_run"] = bool(args.max_rows or args.target_max_rows)
        report["validation"] = validation
    report["candidate_pairs"] = sum(len(value) for value in pair_map.values())
    report["strategy_unique_contributions"] = dict(strategy_pairs)
    report["strategy_seconds"] = {key: round(value, 3) for key, value in strategy_times.items()}
    report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    report["notes"] = [
        "Country is used only by sorted-neighborhood keys; no strategy hard-filters on country.",
        "Empty normalized fields are skipped by key builders and reported under quality.",
        "candidate_pairs.tsv is challenge-compatible; candidate_pairs_detailed.tsv retains provenance and scores.",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
