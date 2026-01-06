#!/usr/bin/env python3
"""
Analyze KG pipeline cost/throughput for a single essay snapshot.

Usage:
  python analyze_costs.py \
    --snapshot "KGs_from_Essays/KG_Essay_001" \
    --token-logs-glob "data/**/llm_usage_*.jsonl" \
    --usd-per-1k-prompt 0.003 \
    --usd-per-1k-completion 0.006

Assumptions:
- The snapshot dir looks like:
    KGs_from_Essays/KG_Essay_001/
      data/...
      Trace_KG_per_essay_stats.json
- Inside `data/` there are one or more JSON / JSONL files that record per‑call token usage.
- Those files match the glob passed in --token-logs-glob.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import glob

# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception as e:
                print(f"[WARN] Failed to parse JSONL line in {path}: {e}", file=sys.stderr)
    return rows


def iter_usage_records(snapshot_data_root: Path, pattern: str) -> List[Dict[str, Any]]:
    """
    Find all JSON / JSONL files under snapshot_data_root matching pattern and return list of usage dicts.
    """
    usage_records: List[Dict[str, Any]] = []
    full_pattern = str(snapshot_data_root / pattern)
    matches = glob.glob(full_pattern, recursive=True)
    for m in matches:
        p = Path(m)
        if not p.is_file():
            continue
        if p.suffix in (".jsonl", ".jl"):
            usage_records.extend(load_jsonl(p))
        elif p.suffix == ".json":
            obj = load_json(p)
            if isinstance(obj, list):
                usage_records.extend(obj)
            elif isinstance(obj, dict):
                usage_records.append(obj)
    return usage_records


def get_int_field(d: Dict[str, Any], *keys: str) -> int:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except Exception:
                try:
                    return int(float(d[k]))
                except Exception:
                    continue
    return 0


def get_step_name(d: Dict[str, Any]) -> str:
    for k in ("step", "phase", "component", "stage"):
        if k in d and d[k]:
            return str(d[k])
    return "unknown"


def group_usage_by_step(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Returns:
      {
        step_name: {
          "prompt_tokens": int,
          "completion_tokens": int,
          "total_tokens": int,
          "n_calls": int,
        },
        ...
      }
    """
    grouped: Dict[str, Dict[str, int]] = {}
    for r in records:
        step = get_step_name(r)
        p = get_int_field(r, "prompt_tokens", "prompt", "input_tokens")
        c = get_int_field(r, "completion_tokens", "completion", "output_tokens")
        if p == 0 and c == 0:
            continue
        g = grouped.setdefault(step, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "n_calls": 0})
        g["prompt_tokens"] += p
        g["completion_tokens"] += c
        g["total_tokens"] += p + c
        g["n_calls"] += 1
    return grouped


def load_per_essay_stats(snapshot_root: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Loads the Trace_KG_per_essay_stats.json that lives in the snapshot root.

    Returns:
      (essay_stats_for_this_essay, raw_json)
    """
    stats_path = snapshot_root / "Trace_KG_per_essay_stats.json"
    if not stats_path.exists():
        print(f"[WARN] No per-essay stats at {stats_path}", file=sys.stderr)
        return {}, None

    raw = load_json(stats_path)
    if not isinstance(raw, dict) or not raw:
        return {}, raw

    # For a snapshot of a single essay, there should typically be exactly one key, e.g. "1"
    if len(raw) == 1:
        essay_idx_str = next(iter(raw.keys()))
        essay_stats = raw[essay_idx_str]
        return essay_stats, raw
    else:
        # Fallback: just pick the first
        essay_idx_str = sorted(raw.keys(), key=lambda x: int(x))[0]
        return raw[essay_idx_str], raw


# ------------------------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="Path to KG_Essay_XXX snapshot dir")
    ap.add_argument(
        "--token-logs-glob",
        default="data/**/llm_usage_*.jsonl",
        help="Glob pattern under snapshot's data/ to find token usage logs (default: data/**/llm_usage_*.jsonl)",
    )
    ap.add_argument("--usd-per-1k-prompt", type=float, default=0.003, help="$/1K prompt tokens")
    ap.add_argument("--usd-per-1k-completion", type=float, default=0.006, help="$/1K completion tokens")
    args = ap.parse_args()

    snapshot_root = Path(args.snapshot).resolve()
    data_root = snapshot_root / "data"

    if not data_root.exists():
        print(f"[FATAL] {data_root} does not exist (snapshot missing data dir?)", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Snapshot root: {snapshot_root}")
    print(f"[info] Data root:     {data_root}")

    # 1) Load per-essay timing stats
    essay_stats, raw_stats = load_per_essay_stats(snapshot_root)
    total_time = essay_stats.get("seconds_total", None)

    # 2) Load all usage records under data/ matching pattern
    usage_records = iter_usage_records(data_root, args.token_logs_glob)
    print(f"[info] Found {len(usage_records)} raw usage records under pattern {args.token_logs_glob}")

    grouped = group_usage_by_step(usage_records)

    # 3) Aggregate totals and print per-step table
    total_prompt = sum(v["prompt_tokens"] for v in grouped.values())
    total_completion = sum(v["completion_tokens"] for v in grouped.values())
    total_tokens = total_prompt + total_completion

    # If we don't know total_time from stats, compute a rough sum of step times, else use essay_stats.
    if total_time is None:
        steps = essay_stats.get("steps", {}) if isinstance(essay_stats, dict) else {}
        total_time = sum(s.get("seconds", 0.0) for s in steps.values())

    print()
    print("=== Token / Cost / Throughput Summary for this Essay ===")
    print(f"Total prompt tokens:     {total_prompt:,}")
    print(f"Total completion tokens: {total_completion:,}")
    print(f"Total tokens:            {total_tokens:,}")
    print(f"Total time (s):          {total_time:.1f}" if total_time is not None else "Total time (s):  N/A")

    # Cost estimate
    cost_prompt = (total_prompt / 1000.0) * args.usd_per_1k_prompt
    cost_completion = (total_completion / 1000.0) * args.usd_per_1k_completion
    total_cost = cost_prompt + cost_completion

    print(f"Estimated cost ($):      {total_cost:.4f}")
    print(f"  prompt:                {cost_prompt:.4f}")
    print(f"  completion:            {cost_completion:.4f}")

    if total_time and total_time > 0:
        throughput = total_tokens / total_time
        print(f"Throughput (tokens/s):   {throughput:,.1f}")
    else:
        print("Throughput (tokens/s):   N/A (no time available)")

    print()
    print("Per-step aggregates (from token logs):")
    print("{:<30} {:>12} {:>12} {:>12} {:>10}".format("Step", "Prompt", "Completion", "Total", "Calls"))
    print("-" * 80)
    for step, v in sorted(grouped.items(), key=lambda kv: kv[0]):
        print(
            "{:<30} {:>12} {:>12} {:>12} {:>10}".format(
                step,
                f"{v['prompt_tokens']:,}",
                f"{v['completion_tokens']:,}",
                f"{v['total_tokens']:,}",
                v["n_calls"],
            )
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()