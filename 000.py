



#*######################### Start ##########################
#region:#?   KG Bundle Export


# """
# KG Bundle Export

# Collect everything needed from the whole KG pipeline into ONE file:
# - Classes + entities (from final Class Res overall_summary)
# - Chunks (for evidence and document structure)
# - Relations (from final Rel Res overall_summary, including qualifiers, canonical fields, etc.)
# - Relation schema (canonical_rel_schema, rel_cls_schema, rel_cls_group_schema)

# Output:
#   SGCE-KG/data/KG/kg_bundle.json

# This file is intended as the single input for downstream KG-construction code.
# """

# import json
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Optional


# # -----------------------
# # Paths / Config
# # -----------------------

# # Final classes + entities after iterative Class Res
# CLS_RES_OVERALL_DIR = Path(
#     "SGCE-KG/data/Classes/Cls_Res_IterativeRuns/overall_summary"
#     )
# CLASSES_AND_ENTITIES_PATH = CLS_RES_OVERALL_DIR / "classes_and_entities.json"

# # Chunks (sentence-level)
# CHUNKS_PATH = Path(
#     "SGCE-KG/data/Chunks/chunks_sentence.jsonl"
# )

# # Final relations + relation schemas after iterative Rel Res
# REL_RES_OVERALL_DIR = Path(
#     "SGCE-KG/data/Relations/Rel Res_IterativeRuns/overall_summary"
# )
# RELATIONS_RESOLVED_JSONL = REL_RES_OVERALL_DIR / "relations_resolved.jsonl"
# RELATIONS_RESOLVED_JSON = REL_RES_OVERALL_DIR / "relations_resolved.json"

# CANONICAL_REL_SCHEMA_PATH = REL_RES_OVERALL_DIR / "canonical_rel_schema.json"
# REL_CLS_SCHEMA_PATH = REL_RES_OVERALL_DIR / "rel_cls_schema.json"
# REL_CLS_GROUP_SCHEMA_PATH = REL_RES_OVERALL_DIR / "rel_cls_group_schema.json"

# # Where to write the final bundle
# KG_BUNDLE_OUT = Path(
#     "SGCE-KG/data/KG/kg_bundle.json"
# )


# # -----------------------
# # Small helpers
# # -----------------------

# def _load_json(path: Path, default: Any) -> Any:
#     if not path.exists():
#         return default
#     try:
#         return json.loads(path.read_text(encoding="utf-8"))
#     except Exception:
#         return default

# def _safe_json_load_line(line: str) -> Optional[Dict]:
#     line = line.strip()
#     if not line:
#         return None
#     try:
#         return json.loads(line)
#     except Exception:
#         return None


# # -----------------------
# # Core export function
# # -----------------------

# def build_kg_bundle() -> None:
#     """
#     Build a single JSON "kg_bundle.json" that contains:

#     {
#       "metadata": {...},
#       "classes": [...],
#       "entities_map": {...},
#       "chunks": { chunk_id: { ...chunk fields... }, ... },
#       "relations": [...],
#       "canonical_rel_schema": [... or {}],
#       "rel_cls_schema": [... or {}],
#       "rel_cls_group_schema": [... or {}]
#     }

#     Relations are taken from the final Rel Res output and are expected to already
#     contain:
#       - relation_id
#       - subject_entity_id / object_entity_id
#       - relation_name (raw)
#       - canonical_rel_name / canonical_rel_desc (if filled by Rel Res)
#       - rel_cls, rel_cls_group (if filled)
#       - qualifiers dict
#       - chunk_id
#       - evidence_excerpt
#       - any other fields kept by the pipeline (rel_desc, rel_hint_type, etc.)
#     """

#     # 1) Load final classes + entities
#     classes_and_entities = _load_json(CLASSES_AND_ENTITIES_PATH, default={"classes": [], "entities_map": {}})
#     classes: List[Dict[str, Any]] = classes_and_entities.get("classes", []) or []
#     entities_map: Dict[str, Dict[str, Any]] = classes_and_entities.get("entities_map", {}) or {}

#     print(f"[KG bundle] Loaded {len(classes)} classes and {len(entities_map)} entities from {CLASSES_AND_ENTITIES_PATH}")

#     # 2) Load chunks
#     chunks_by_id: Dict[str, Dict[str, Any]] = {}
#     if CHUNKS_PATH.exists():
#         with CHUNKS_PATH.open("r", encoding="utf-8") as fh:
#             for line in fh:
#                 obj = _safe_json_load_line(line)
#                 if not obj:
#                     continue
#                 cid = obj.get("id")
#                 if cid:
#                     chunks_by_id[cid] = obj
#         print(f"[KG bundle] Loaded {len(chunks_by_id)} chunks from {CHUNKS_PATH}")
#     else:
#         print(f"[KG bundle] WARNING: chunks file not found at {CHUNKS_PATH}, 'chunks' will be empty.")

#     # 3) Load final relations (prefer jsonl; fallback to json)
#     relations: List[Dict[str, Any]] = []
#     if RELATIONS_RESOLVED_JSONL.exists():
#         with RELATIONS_RESOLVED_JSONL.open("r", encoding="utf-8") as fh:
#             for line in fh:
#                 obj = _safe_json_load_line(line)
#                 if obj:
#                     relations.append(obj)
#         print(f"[KG bundle] Loaded {len(relations)} relations from {RELATIONS_RESOLVED_JSONL}")
#     elif RELATIONS_RESOLVED_JSON.exists():
#         relations = _load_json(RELATIONS_RESOLVED_JSON, default=[])
#         if not isinstance(relations, list):
#             relations = []
#         print(f"[KG bundle] Loaded {len(relations)} relations from {RELATIONS_RESOLVED_JSON}")
#     else:
#         print(f"[KG bundle] WARNING: no final relations file found in {REL_RES_OVERALL_DIR}, 'relations' will be empty.")

#     # 4) Load relation schemas (canonical / class / group)
#     canonical_rel_schema = _load_json(CANONICAL_REL_SCHEMA_PATH, default=[])
#     rel_cls_schema = _load_json(REL_CLS_SCHEMA_PATH, default=[])
#     rel_cls_group_schema = _load_json(REL_CLS_GROUP_SCHEMA_PATH, default=[])

#     print(f"[KG bundle] canonical_rel_schema size: {len(canonical_rel_schema) if isinstance(canonical_rel_schema, list) else 'dict'}")
#     print(f"[KG bundle] rel_cls_schema size: {len(rel_cls_schema) if isinstance(rel_cls_schema, list) else 'dict'}")
#     print(f"[KG bundle] rel_cls_group_schema size: {len(rel_cls_group_schema) if isinstance(rel_cls_group_schema, list) else 'dict'}")

#     # 5) Assemble bundle
#     bundle = {
#         "metadata": {
#             "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#             "source_paths": {
#                 "classes_and_entities": str(CLASSES_AND_ENTITIES_PATH),
#                 "chunks": str(CHUNKS_PATH),
#                 "relations_resolved_jsonl": str(RELATIONS_RESOLVED_JSONL),
#                 "relations_resolved_json": str(RELATIONS_RESOLVED_JSON),
#                 "canonical_rel_schema": str(CANONICAL_REL_SCHEMA_PATH),
#                 "rel_cls_schema": str(REL_CLS_SCHEMA_PATH),
#                 "rel_cls_group_schema": str(REL_CLS_GROUP_SCHEMA_PATH),
#             },
#         },
#         "classes": classes,
#         "entities_map": entities_map,
#         "chunks": chunks_by_id,
#         "relations": relations,
#         "canonical_rel_schema": canonical_rel_schema,
#         "rel_cls_schema": rel_cls_schema,
#         "rel_cls_group_schema": rel_cls_group_schema,
#     }

#     # 6) Write out
#     KG_BUNDLE_OUT.parent.mkdir(parents=True, exist_ok=True)
#     KG_BUNDLE_OUT.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
#     print(f"[KG bundle] Written KG bundle to {KG_BUNDLE_OUT}")


# # After defining everything, you can call:
# build_kg_bundle()

#endregion#?   KG Bundle Export
#*#########################  End  ##########################






#*######################### Start ##########################
#region:#?   Cypher Queries - V6



MATCH (n)
DETACH DELETE n;





LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CALL (row) {
  WITH row
  // skip empty rows
  WITH row
  WHERE row.entity_id IS NOT NULL AND row.entity_id <> ''

  MERGE (e:Entity {entity_id: row.entity_id})
  SET
    e.name                 = row.entity_name,
    e.description          = row.entity_description,
    e.type_hint            = row.entity_type_hint,
    e.confidence           = CASE row.entity_confidence
                               WHEN "" THEN NULL
                               ELSE toFloat(row.entity_confidence)
                             END,
    e.resolution_context   = row.entity_resolution_context,
    e.flag                 = row.entity_flag,
    e.class_id             = row.class_id,
    e.class_label          = row.class_label,
    e.class_group          = row.class_group,
    // chunk_ids is a JSON list string
    e.chunk_ids            = apoc.convert.fromJsonList(row.chunk_ids),
    // node_properties is a JSON list string
    e.node_properties      = apoc.convert.fromJsonList(row.node_properties)
} IN TRANSACTIONS OF 1000 ROWS;












LOAD CSV WITH HEADERS FROM 'file:///rels_fixed_no_raw.csv' AS row
CALL (row) {
  WITH row
  // skip rows missing endpoints
  WITH row
  WHERE row.start_id IS NOT NULL AND row.start_id <> ''
    AND row.end_id   IS NOT NULL AND row.end_id   <> ''

  MATCH (s:Entity {entity_id: row.start_id})
  MATCH (o:Entity {entity_id: row.end_id})

  MERGE (s)-[r:RELATION {relation_id: row.relation_id}]->(o)
  WITH r, row, apoc.convert.fromJsonMap(row.qualifiers) AS qmap
  SET
    r.raw_name           = row.raw_relation_name,
    r.canonical_name     = row.canonical_rel_name,
    r.canonical_desc     = row.canonical_rel_desc,
    r.rel_cls            = row.rel_cls,
    r.rel_cls_group      = row.rel_cls_group,
    r.rel_hint_type      = row.rel_hint_type,
    r.confidence         = CASE row.confidence
                             WHEN "" THEN NULL
                             ELSE toFloat(row.confidence)
                           END,
    r.description        = row.rel_desc,
    r.resolution_context = row.resolution_context,
    r.justification      = row.justification,
    r.remark             = row.remark,
    r.evidence_excerpt   = row.evidence_excerpt,
    r.chunk_id           = row.chunk_id,
    // keep the pretty JSON text for inspection
    r.qualifiers_json    = row.qualifiers
  WITH r, qmap
  FOREACH (k IN keys(qmap) |
    SET r['qual_' + k] = toString(qmap[k])
  )
} IN TRANSACTIONS OF 1000 ROWS;








//#?######################### Start ##########################
//#region:#?   See classes as colors



// 1) create a sanitized label string in property `_viz_label`
MATCH (e:Entity)
WHERE e.class_group IS NOT NULL AND e.class_group <> ''
WITH e,
     replace(
       replace(
         replace(
           replace(
             replace(
               replace(
                 replace(
                   replace(
                     replace(
                       replace(e.class_group, ' ', '_'),
                     '-', '_'),
                   '/', '_'),
                 '.', '_'),
               ',', '_'),
             '"', '_'),
           "'", '_'),
         '(', '_'),
       ')', '_'),
     ':', '_') AS lab0
// ensure label begins with a letter - prefix 'C_' otherwise
WITH e,
     CASE WHEN lab0 =~ '^[A-Za-z].*' THEN lab0 ELSE 'C_' + lab0 END AS vizlabel
SET e._viz_label = vizlabel
RETURN count(e) AS nodes_updated;



// 2) add labels from _viz_label (one label per node)
MATCH (e:Entity)
WHERE e._viz_label IS NOT NULL AND e._viz_label <> ''
CALL apoc.create.addLabels(e, [e._viz_label]) YIELD node
RETURN count(node) AS labels_added;


// 3) remove Entity label from nodes that have a viz label (visualization-only)
MATCH (e:Entity)
WHERE e._viz_label IS NOT NULL AND e._viz_label <> ''
REMOVE e:Entity
RETURN count(e) AS entity_labels_removed;





//#endregion#? See classes as colors
//#?#########################  End  ##########################






//MATCH (n)
//RETURN n





MATCH (n)-[r]->(m)
// WHERE n.seen_in_chunks = ["Ch_000018"]
RETURN n, r, m






#endregion#? Cypher Queries
#*#########################  End  ##########################









#?######################### Start ##########################
#region:#?   Analyze_costs


import json
import glob
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Optional: import Trace_KG so we're sure we're in the right project
import TKG_Main  # noqa: F401  # not used directly, just to ensure import works

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

    # Typically only one key, e.g. "1"
    if len(raw) == 1:
        essay_idx_str = next(iter(raw.keys()))
        essay_stats = raw[essay_idx_str]
        return essay_stats, raw
    else:
        # Fallback: pick the first numerically sorted essay index
        essay_idx_str = sorted(raw.keys(), key=lambda x: int(x))[0]
        return raw[essay_idx_str], raw


# ------------------------------------------------------------------------------------
# MAIN: configure and run for Essay 1
# ------------------------------------------------------------------------------------

# 1) CONFIGURE THESE PATHS / PRICES

# Path to the snapshot for essay 1
SNAPSHOT_ROOT = Path("KGs_from_Essays00/KG_Essay_001").resolve()
# Glob pattern for token usage logs under SNAPSHOT_ROOT/data
TOKEN_LOGS_GLOB = "data/**/llm_usage_*.jsonl"   # adjust if your filenames differ

# Pricing assumptions (change to match your model/pricing)
USD_PER_1K_PROMPT = 0.003
USD_PER_1K_COMPLETION = 0.006

# ------------------------------------------------------------------------------------
# RUN ANALYSIS
# ------------------------------------------------------------------------------------

data_root = SNAPSHOT_ROOT / "KGs_from_Essays00/KG_Essay_001"
if not data_root.exists():
    raise FileNotFoundError(f"Snapshot data dir not found: {data_root}")

print(f"[info] Snapshot root: {SNAPSHOT_ROOT}")
print(f"[info] Data root:     {data_root}")

# 1) Load per-essay timing stats
essay_stats, raw_stats = load_per_essay_stats(SNAPSHOT_ROOT)
total_time = essay_stats.get("seconds_total", None)

# 2) Load usage records
usage_records = iter_usage_records(data_root, TOKEN_LOGS_GLOB)
print(f"[info] Found {len(usage_records)} raw usage records under pattern {TOKEN_LOGS_GLOB}")

grouped = group_usage_by_step(usage_records)

# 3) Aggregate totals
total_prompt = sum(v["prompt_tokens"] for v in grouped.values())
total_completion = sum(v["completion_tokens"] for v in grouped.values())
total_tokens = total_prompt + total_completion

# If total_time is missing, approximate by sum of step times
if total_time is None and isinstance(essay_stats, dict):
    steps = essay_stats.get("steps", {}) or {}
    total_time = sum(s.get("seconds", 0.0) for s in steps.values())

print()
print("=== Token / Cost / Throughput Summary for this Essay ===")
print(f"Total prompt tokens:     {total_prompt:,}")
print(f"Total completion tokens: {total_completion:,}")
print(f"Total tokens:            {total_tokens:,}")
if total_time is not None:
    print(f"Total time (s):          {total_time:.1f}")
else:
    print("Total time (s):          N/A")

# Cost estimate
cost_prompt = (total_prompt / 1000.0) * USD_PER_1K_PROMPT
cost_completion = (total_completion / 1000.0) * USD_PER_1K_COMPLETION
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

print("\nDone.")

#endregion#?  Analyze_costs
#?#########################  End  ##########################

















#?######################### Start ##########################
#region:#?   review the structure of /mine_evaluation_dataset.json


import json
from pathlib import Path
from typing import Any, Dict

DATASET_JSON_PATH = Path("Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset.json")

def summarize_value(v: Any, max_len: int = 120) -> str:
    """Return a short, human-readable summary of a value."""
    if isinstance(v, dict):
        return f"<dict with keys: {list(v.keys())[:5]}{' ...' if len(v) > 5 else ''}>"
    if isinstance(v, list):
        return f"<list len={len(v)} first_item_type={type(v[0]).__name__ if v else 'None'}>"
    s = str(v)
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s

def inspect_dataset(path: Path, show_items: int = 3) -> None:
    print(f"Loading dataset from: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n=== Top-level structure ===")
    print(f"type: {type(data).__name__}")

    if isinstance(data, list):
        print(f"length: {len(data)}")
        items = data
    elif isinstance(data, dict):
        print(f"keys: {list(data.keys())}")
        # If it’s a dict with a list under a key like 'data'/'items'/..., pick that
        for key in ("data", "items", "essays", "documents"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                print(f"Using data[{key}] as items list (len={len(items)})")
                break
        else:
            print("Top-level dict but no obvious list of items; showing the dict itself.")
            items = [data]
    else:
        print("Unexpected top-level type; cannot inspect further.")
        return

    print("\n=== Keys present in first few items ===")
    for i, item in enumerate(items[:show_items]):
        print(f"\n--- Item {i} ---")
        if not isinstance(item, dict):
            print(f"type: {type(item).__name__}, value: {summarize_value(item)}")
            continue
        print("keys:", list(item.keys()))
        for k, v in item.items():
            print(f"  {k!r}: {summarize_value(v)}")

    # Try to detect likely fields
    print("\n=== Heuristic detection of important fields ===")
    sample = items[0] if items else {}
    if isinstance(sample, dict):
        candidate_id_keys = [k for k in sample.keys() if k.lower() in ("id", "essay_id", "doc_id", "uid")]
        candidate_query_keys = [k for k in sample.keys() if "query" in k.lower()]
        candidate_answer_keys = [k for k in sample.keys() if "answer" in k.lower()]
        candidate_kg_keys = [k for k in sample.keys() if "kg" in k.lower()]

        print("Possible ID fields:      ", candidate_id_keys or "None found")
        print("Possible query fields:   ", candidate_query_keys or "None found")
        print("Possible answer fields:  ", candidate_answer_keys or "None found")
        print("Possible KG fields:      ", candidate_kg_keys or "None found")

        # Show example values for likely ID / query / answer fields
        if candidate_id_keys:
            k = candidate_id_keys[0]
            print(f"\nExample ID values for key '{k}':")
            for i, item in enumerate(items[:min(show_items, 5)]):
                print(f"  item[{i}][{k!r}] = {summarize_value(item.get(k))}")

        if candidate_query_keys:
            k = candidate_query_keys[0]
            print(f"\nExample QUERY values for key '{k}':")
            for i, item in enumerate(items[:min(show_items, 5)]):
                print(f"  item[{i}][{k!r}] = {summarize_value(item.get(k))}")

        if candidate_answer_keys:
            k = candidate_answer_keys[0]
            print(f"\nExample ANSWER values for key '{k}':")
            for i, item in enumerate(items[:min(show_items, 5)]):
                print(f"  item[{i}][{k!r}] = {summarize_value(item.get(k))}")

        if candidate_kg_keys:
            k = candidate_kg_keys[0]
            print(f"\nExample KG fields for key '{k}':")
            for i, item in enumerate(items[:min(show_items, 5)]):
                print(f"  item[{i}][{k!r}] = {summarize_value(item.get(k))}")
    else:
        print("First item is not a dict; cannot detect fields heuristically.")

# Run the inspection
inspect_dataset(DATASET_JSON_PATH)


#endregion#? review the structure of /mine_evaluation_dataset.json
#?#########################  End  ##########################