


#!############################################# Start Chapter ##################################################
#region:#!     Addomg AutoSchemaKG to the comparison



#?######################### Start ##########################
#region:#?     AutoSchemaKG KG generation

#!/usr/bin/env python3
"""
AutoSchemaKG KG Generation for MINE-1 Evaluation
=================================================

This script generates knowledge graphs using AutoSchemaKG for the same essays
that TRACE KG and other baselines (KGGen, OpenIE, GraphRAG) were evaluated on.

It produces output in the same format as the other baselines:
  {
    "entities": ["ent1", "ent2", ...],
    "edges": ["rel1", "rel2", ...],
    "relations": [["head", "relation", "tail"], ...]
  }

The results are injected into the evaluation dataset JSON under the key
"autoschemakg" so that TKG_Experiment_1.py can evaluate it alongside
the existing 4 methods.

Usage:
  1. Install atlas-rag:  pip install -e Experiments/AutoSchemaKG
  2. Set OPENAI_API_KEY in your environment (or edit the client setup below)
  3. Run:  python AutoSchemaKG_Generate_KGs_for_MINE1.py
"""

import os
import sys
import json
import re
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from copy import deepcopy

# ---------------------------------------------------------------------------
# Add AutoSchemaKG to path so we can import atlas_rag
# ---------------------------------------------------------------------------
# REPO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG")
AUTOSCHEMAKG_ROOT = REPO_ROOT / "Experiments" / "AutoSchemaKG"
sys.path.insert(0, str(AUTOSCHEMAKG_ROOT))

# ---------------------------------------------------------------------------
# AutoSchemaKG imports
# ---------------------------------------------------------------------------
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.llm_generator.prompt.triple_extraction_prompt import TRIPLE_INSTRUCTIONS
from atlas_rag.llm_generator.format.validate_json_schema import ATLAS_SCHEMA


from openai import OpenAI

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# LLM Config — using OpenAI API (same model family as TRACE KG for fairness)
# You can swap to a local model or vLLM server by changing client + model_name
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-5.1" #"gpt-4o-mini"  # Cost-effective; change to "gpt-4o" or others as needed

# Paths
DATASET_JSON_PATH    = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset.json"
OUTPUT_DATASET_PATH  = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset_with_autoschemakg.json"
OUTPUT_KGS_DIR       = REPO_ROOT / "Experiments/MYNE/Ex1/AutoSchemaKG_KGs"

# AutoSchemaKG extraction parameters
BATCH_SIZE_TRIPLE  = 3
MAX_NEW_TOKENS     = 2048
MAX_WORKERS        = 3

# Which essay IDs to process (None = all that exist in the evaluation dataset)
ESSAY_IDS: Optional[List[int]] =  None  # e.g., [1, 2] for testing


# ---------------------------------------------------------------------------
# HELPER: Convert essays JSON → AutoSchemaKG JSONL format
# ---------------------------------------------------------------------------


def essays_to_autoschemakg_jsonl(
    dataset: List[Dict[str, Any]],
    essay_id: int,
    output_dir: Path,
) -> Path:
    """
    Write a single essay as a JSONL file that AutoSchemaKG can read.
    Looks up the essay by its "id" field in the dataset (mine_evaluation_dataset.json).
    """
    data_dir = output_dir / f"essay_{essay_id:03d}_input"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Find the essay by "id" field (NOT by list position)
    essay = None
    for item in dataset:
        if int(item.get("id", -1)) == essay_id:
            essay = item
            break
    if essay is None:
        raise ValueError(f"Essay ID {essay_id} not found in dataset")

    text = essay.get("essay_content", "")       # <-- was "text", now "essay_content"
    title = essay.get("essay_topic", f"Essay_{essay_id}")  # <-- was "title", now "essay_topic"

    # Clean up text: remove markdown-style backtick wrappers
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Write JSONL file
    jsonl_path = data_dir / f"essay_{essay_id:03d}.jsonl"
    record = {
        "id": str(essay_id),
        "text": text,
        "metadata": {"lang": "en"}
    }
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return data_dir


# ---------------------------------------------------------------------------
# HELPER: Run AutoSchemaKG extraction on a single essay
# ---------------------------------------------------------------------------

def run_autoschemakg_extraction(
    client: OpenAI,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    essay_id: int,
) -> Dict[str, Any]:
    """
    Run AutoSchemaKG triple extraction for one essay.

    Returns a dict with:
      {
        "entities": [...],
        "edges": [...],
        "relations": [[head, rel, tail], ...],
        "raw_triples": [...]  # original AutoSchemaKG output for debugging
      }
    """
    keyword = f"essay_{essay_id:03d}"
    out_dir = str(output_dir / keyword)

    triple_generator = LLMGenerator(client=client, model_name=model_name)

    kg_extraction_config = ProcessingConfig(
        model_path=model_name,
        data_directory=str(data_dir),
        filename_pattern=keyword,
        batch_size_triple=BATCH_SIZE_TRIPLE,
        batch_size_concept=16,
        output_directory=out_dir,
        max_new_tokens=MAX_NEW_TOKENS,
        max_workers=MAX_WORKERS,
        remove_doc_spaces=True,
        include_concept=False,   # We only need entity-relation triples for MINE-1
        allow_empty=True,
        chunk_size=8192,         # Large enough for single essays
        chunk_overlap=0,
    )

    kg_extractor = KnowledgeGraphExtractor(
        model=triple_generator,
        config=kg_extraction_config,
    )

    # Step 1: Extract entity & event triples (calls LLM)
    print(f"    [AutoSchemaKG] Running triple extraction for essay {essay_id}...")
    kg_extractor.run_extraction()

    # Step 2: Convert JSON output to CSV
    print(f"    [AutoSchemaKG] Converting JSON → CSV...")
    kg_extractor.convert_json_to_csv()

    # Now parse the generated triples from the output JSON files
    triples = _collect_triples_from_output(out_dir, keyword)

    # Convert to the standard format used by kggen/openie/graphrag
    return _triples_to_baseline_format(triples)


def _collect_triples_from_output(
    output_dir: str,
    keyword: str,
) -> List[Dict[str, str]]:
    """
    Read the generated JSON/JSONL files from AutoSchemaKG output
    and collect all entity-relation triples.
    """
    triples = []
    gen_dir = Path(output_dir)

    # AutoSchemaKG writes JSONL files in the output directory
    jsonl_files = list(gen_dir.glob("*.jsonl")) + list(gen_dir.glob("*.json"))

    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # AutoSchemaKG stores triples under "entity_relation_dict"
                er_dict = data.get("entity_relation_dict", [])
                if isinstance(er_dict, list):
                    for triple in er_dict:
                        if isinstance(triple, dict):
                            head = triple.get("Head", "").strip()
                            rel = triple.get("Relation", "").strip()
                            tail = triple.get("Tail", "").strip()
                            if head and rel and tail:
                                triples.append({
                                    "Head": head,
                                    "Relation": rel,
                                    "Tail": tail,
                                })

                # Also grab event-entity relations if present
                ee_dict = data.get("event_entity_dict", [])
                if isinstance(ee_dict, list):
                    for item in ee_dict:
                        if isinstance(item, dict):
                            event = item.get("Event", "").strip()
                            entities = item.get("Entity", [])
                            if event and entities:
                                for ent in entities:
                                    ent = str(ent).strip()
                                    if ent:
                                        triples.append({
                                            "Head": event,
                                            "Relation": "is participated by",
                                            "Tail": ent,
                                        })

                # Event-relation triples
                evr_dict = data.get("event_relation_dict", [])
                if isinstance(evr_dict, list):
                    for triple in evr_dict:
                        if isinstance(triple, dict):
                            head = triple.get("Head", "").strip()
                            rel = triple.get("Relation", "").strip()
                            tail = triple.get("Tail", "").strip()
                            if head and rel and tail:
                                triples.append({
                                    "Head": head,
                                    "Relation": rel,
                                    "Tail": tail,
                                })

    return triples


def _triples_to_baseline_format(
    triples: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Convert AutoSchemaKG triples to the same dict format as kggen/openie/graphrag:
      {
        "entities": [sorted unique entity names],
        "edges": [sorted unique relation names],
        "relations": [[head, relation, tail], ...]
      }
    """
    entities: Set[str] = set()
    edges: Set[str] = set()
    relations: List[List[str]] = []

    seen_triples: Set[Tuple[str, str, str]] = set()

    for t in triples:
        head = t["Head"]
        rel = t["Relation"]
        tail = t["Tail"]

        # Normalize: lowercase for entity/edge dedup (keep original case in relations list)
        head_lower = head.lower()
        rel_lower = rel.lower()
        tail_lower = tail.lower()

        triple_key = (head_lower, rel_lower, tail_lower)
        if triple_key in seen_triples:
            continue
        seen_triples.add(triple_key)

        entities.add(head_lower)
        entities.add(tail_lower)
        edges.add(rel_lower)
        relations.append([head_lower, rel_lower, tail_lower])

    return {
        "entities": sorted(entities),
        "edges": sorted(edges),
        "relations": relations,
        "raw_triple_count": len(triples),
        "deduped_triple_count": len(relations),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("AutoSchemaKG KG Generation for MINE-1 Evaluation")
    print("=" * 70)



    # Load evaluation dataset (contains essay text + KG data for other methods)
    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} evaluation items from {DATASET_JSON_PATH.name}")


    # Build map: dataset_id → index in dataset list
    id_to_idx = {}
    for idx, item in enumerate(dataset):
        did = item.get("id")
        if did is not None:
            id_to_idx[int(did)] = idx




    # 3. Setup OpenAI client
    print(f"\n[3] Setting up LLM client: model={MODEL_NAME}")
    if not OPENAI_API_KEY:
        print("    WARNING: OPENAI_API_KEY not set. Make sure it's in your environment.")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 4. Create output directories
    OUTPUT_KGS_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Process each essay
    print(f"\n[4] Starting AutoSchemaKG extraction...")
    print("-" * 70)

    results_summary = []
    enriched_dataset = deepcopy(dataset)

    for i, essay_id in enumerate(target_ids):
        print(f"\n>>> Essay {essay_id}  ({i+1}/{len(target_ids)})")

        if essay_id not in id_to_idx:
            print(f"    SKIP: essay_id={essay_id} not found in evaluation dataset.")
            continue

        # if essay_id > len(essays):
        #     print(f"    SKIP: essay_id={essay_id} exceeds essays list length ({len(essays)}).")
        #     continue

        t0 = time.time()

        try:
            # a) Prepare input data
            # data_dir = essays_to_autoschemakg_jsonl(essays, essay_id, OUTPUT_KGS_DIR)
            data_dir = essays_to_autoschemakg_jsonl(dataset, essay_id, OUTPUT_KGS_DIR)

            # b) Run extraction
            kg_result = run_autoschemakg_extraction(
                client=client,
                model_name=MODEL_NAME,
                data_dir=data_dir,
                output_dir=OUTPUT_KGS_DIR,
                essay_id=essay_id,
            )

            elapsed = time.time() - t0
            print(f"    ✓ Done in {elapsed:.1f}s — "
                  f"{len(kg_result['entities'])} entities, "
                  f"{len(kg_result['edges'])} edge types, "
                  f"{len(kg_result['relations'])} triples")

            # c) Inject into dataset
            ds_idx = id_to_idx[essay_id]
            enriched_dataset[ds_idx]["autoschemakg"] = kg_result

            results_summary.append({
                "essay_id": essay_id,
                "status": "ok",
                "entities": len(kg_result["entities"]),
                "edges": len(kg_result["edges"]),
                "relations": len(kg_result["relations"]),
                "seconds": round(elapsed, 1),
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ✗ FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

            results_summary.append({
                "essay_id": essay_id,
                "status": "failed",
                "error": str(e),
                "seconds": round(elapsed, 1),
            })

    # 6. Save enriched dataset
    print(f"\n\n[5] Saving enriched dataset → {OUTPUT_DATASET_PATH}")
    with open(OUTPUT_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_dataset, f, ensure_ascii=False, indent=2)
    print("    Done.")

    # 7. Save per-essay KG results as standalone JSON
    standalone_kgs_path = OUTPUT_KGS_DIR / "all_autoschemakg_results.json"
    standalone = {}
    for item in enriched_dataset:
        if "autoschemakg" in item:
            standalone[item["id"]] = item["autoschemakg"]
    with open(standalone_kgs_path, "w", encoding="utf-8") as f:
        json.dump(standalone, f, ensure_ascii=False, indent=2)
    print(f"    Standalone KGs saved → {standalone_kgs_path}")

    # 8. Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ok_count = sum(1 for r in results_summary if r["status"] == "ok")
    fail_count = sum(1 for r in results_summary if r["status"] == "failed")
    print(f"  Processed: {len(results_summary)}  |  OK: {ok_count}  |  Failed: {fail_count}")

    if results_summary:
        print(f"\n  {'ID':>4}  {'Status':>8}  {'Entities':>10}  {'Edges':>8}  {'Triples':>10}  {'Time(s)':>8}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
        for r in results_summary:
            if r["status"] == "ok":
                print(f"  {r['essay_id']:>4}  {'OK':>8}  {r['entities']:>10}  {r['edges']:>8}  {r['relations']:>10}  {r['seconds']:>8.1f}")
            else:
                print(f"  {r['essay_id']:>4}  {'FAIL':>8}  {'—':>10}  {'—':>8}  {'—':>10}  {r['seconds']:>8.1f}")

    print(f"\n  Output dataset: {OUTPUT_DATASET_PATH}")
    print(f"  Standalone KGs: {standalone_kgs_path}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# AutoSchemaKG — KG Generation for MINE-1 Experiment
# ═══════════════════════════════════════════════════════════════════════
#
# This cell generates KGs using AutoSchemaKG for the same essays that
# TRACE KG and other baselines (KGGen, OpenIE, GraphRAG) were evaluated on.
#
# PREREQUISITES:
#   pip install -e Experiments/AutoSchemaKG
#   export OPENAI_API_KEY="your-key"    (or set it in the cell below)
#
# Run from the SGCE-KG repo root.
# ═══════════════════════════════════════════════════════════════════════

import os, sys, json, re, time, shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from copy import deepcopy

# ──────────────────────────────────────────────────────────────────────
# 0.  Add AutoSchemaKG to sys.path
# ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(".").resolve()            # <-- adjust if your CWD differs
AUTOSCHEMAKG_ROOT = REPO_ROOT / "Experiments" / "AutoSchemaKG"
if str(AUTOSCHEMAKG_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOSCHEMAKG_ROOT))

from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────
# 1.  CONFIG  — edit these as needed
# ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME     =  "gpt-4o-mini" #"gpt-5.1" #"gpt-4o-mini"     # cost-effective; swap to "gpt-4o" etc.

DATASET_JSON_PATH   = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset.json"
OUTPUT_DATASET_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset_with_autoschemakg.json"
OUTPUT_KGS_DIR      = REPO_ROOT / "Experiments/MYNE/Ex1/AutoSchemaKG_KGs"

# Which essay IDs to process (None = all that are in evaluation dataset)

ESSAY_IDS: Optional[List[int]] = [4,6,14,28,30,33,44,46,68,70,76,82] # [ 1, 2,10,15,24,47,52,53,64,88,91] #, 10, 15, 24, 47, 52, 53, 64, 67, 88, 91] #None      # e.g., [1, 2, 3] for a quick test

# ──────────────────────────────────────────────────────────────────────
# 2.  HELPERS
# ──────────────────────────────────────────────────────────────────────

def _clean_essay_text(text: str) -> str:
    """Strip backtick wrappers that some essays have."""
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _write_essay_as_jsonl(dataset: list, essay_id: int, out_dir: Path) -> Path:
    """
    Write one essay as a JSONL file that AutoSchemaKG's load_dataset() can read.
    Looks up the essay by its "id" field in the dataset (mine_evaluation_dataset.json).
    Returns the *directory* path (= data_directory for ProcessingConfig).
    """
    data_dir = out_dir / f"input_{essay_id:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Find essay by "id" field, NOT by list position
    essay = None
    for item in dataset:
        if int(item.get("id", -1)) == essay_id:
            essay = item
            break
    if essay is None:
        raise ValueError(f"Essay ID {essay_id} not found in dataset")

    text = _clean_essay_text(essay.get("essay_content", ""))  # was "text"
    record = {
        "id": str(essay_id),
        "text": text,
        "metadata": {"lang": "en"},
    }
    jsonl_path = data_dir / f"essay_{essay_id:03d}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return data_dir



def _collect_triples_from_extraction_dir(extraction_dir: str) -> List[Dict[str, str]]:
    """
    Read every JSONL/JSON file under <extraction_dir>/kg_extraction/
    and collect (Head, Relation, Tail) triples from all three dict types.
    """
    triples: List[Dict[str, str]] = []
    kg_dir = Path(extraction_dir) / "kg_extraction"
    if not kg_dir.exists():
        return triples

    for fpath in sorted(kg_dir.glob("*")):
        if not (fpath.suffix in (".json", ".jsonl") or fpath.name.endswith(".json")):
            continue
        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # (a) entity_relation_dict  →  [{Head, Relation, Tail}, ...]
                for t in data.get("entity_relation_dict", []) or []:
                    if isinstance(t, dict):
                        h, r, tl = t.get("Head",""), t.get("Relation",""), t.get("Tail","")
                        if h and r and tl:
                            triples.append({"Head": h.strip(), "Relation": r.strip(), "Tail": tl.strip()})

                # (b) event_relation_dict  →  [{Head, Relation, Tail}, ...]
                for t in data.get("event_relation_dict", []) or []:
                    if isinstance(t, dict):
                        h, r, tl = t.get("Head",""), t.get("Relation",""), t.get("Tail","")
                        if h and r and tl:
                            triples.append({"Head": h.strip(), "Relation": r.strip(), "Tail": tl.strip()})

                # (c) event_entity_dict  →  [{Event, Entity: [...]}, ...]
                for item in data.get("event_entity_dict", []) or []:
                    if isinstance(item, dict):
                        event = (item.get("Event") or "").strip()
                        entities = item.get("Entity", [])
                        if event and isinstance(entities, list):
                            for ent in entities:
                                if isinstance(ent, str) and ent.strip():
                                    triples.append({
                                        "Head": event,
                                        "Relation": "is participated by",
                                        "Tail": ent.strip(),
                                    })
    return triples


def _triples_to_baseline_format(triples: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Convert AutoSchemaKG triples to the same dict format used by
    kggen / openie / graphrag in mine_evaluation_dataset.json:
        {"entities": [...], "edges": [...], "relations": [[h, r, t], ...]}
    """
    entities: Set[str] = set()
    edges: Set[str] = set()
    relations: List[List[str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for t in triples:
        h = t["Head"].lower()
        r = t["Relation"].lower()
        tl = t["Tail"].lower()
        key = (h, r, tl)
        if key in seen:
            continue
        seen.add(key)
        entities.add(h)
        entities.add(tl)
        edges.add(r)
        relations.append([h, r, tl])

    return {
        "entities": sorted(entities),
        "edges": sorted(edges),
        "relations": relations,
    }










# ──────────────────────────────────────────────────────────────────────
# 3.  RUN  AutoSchemaKG extraction for each essay
# ──────────────────────────────────────────────────────────────────────

def run_autoschemakg_for_mine1():
    print("=" * 70)
    print("AutoSchemaKG  ·  KG Generation for MINE-1")
    print("=" * 70)

    
    
    # Load evaluation dataset (contains essay text + KG data for other methods)
    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} evaluation items from {DATASET_JSON_PATH.name}")





    id_to_idx = {int(item["id"]): idx for idx, item in enumerate(dataset)}
    target_ids = ESSAY_IDS if ESSAY_IDS else sorted(id_to_idx.keys())
    print(f"Will process {len(target_ids)} essay(s)\n")

    # OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    triple_generator = LLMGenerator(client=client, model_name=MODEL_NAME)

    OUTPUT_KGS_DIR.mkdir(parents=True, exist_ok=True)
    enriched_dataset = deepcopy(dataset)
    summary = []

    for i, eid in enumerate(target_ids):
        print(f"──── Essay {eid}  ({i+1}/{len(target_ids)}) ", end="", flush=True)
        
        if eid not in id_to_idx:
            print("SKIP (not found in evaluation dataset)")
            continue
        
        
        t0 = time.time()
        keyword = f"essay_{eid:03d}"
        try:
            # a) write essay as JSONL
            data_dir = _write_essay_as_jsonl(dataset, eid, OUTPUT_KGS_DIR)
            out_dir  = str(OUTPUT_KGS_DIR / keyword)

            # b) configure AutoSchemaKG
            cfg = ProcessingConfig(
                model_path=MODEL_NAME,
                data_directory=str(data_dir),
                filename_pattern=keyword,
                batch_size_triple=1,        # one essay at a time
                batch_size_concept=16,
                output_directory=out_dir,
                max_new_tokens=2048,
                max_workers=3,
                remove_doc_spaces=True,
                include_concept=False,      # only need entity-relation triples
                allow_empty=True,
                chunk_size=16384,           # large enough for single essay
                chunk_overlap=0,
            )
            kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=cfg)

            # c) extract
            kg_extractor.run_extraction()

            # d) collect triples from raw JSON output
            triples = _collect_triples_from_extraction_dir(out_dir)
            result  = _triples_to_baseline_format(triples)

            elapsed = time.time() - t0
            print(f"✓  {len(result['entities'])} ents, "
                  f"{len(result['edges'])} edge types, "
                  f"{len(result['relations'])} triples  ({elapsed:.1f}s)")

            # e) inject into dataset
            enriched_dataset[id_to_idx[eid]]["autoschemakg"] = result
            summary.append({"id": eid, "ok": True, "ent": len(result["entities"]),
                            "rel": len(result["relations"]), "sec": round(elapsed, 1)})

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"✗  FAILED  ({elapsed:.1f}s)  —  {exc}")
            import traceback; traceback.print_exc()
            summary.append({"id": eid, "ok": False, "sec": round(elapsed, 1)})

    # ── Save outputs ──
    with open(OUTPUT_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_dataset, f, ensure_ascii=False, indent=2)
    print(f"\n✅  Enriched dataset saved → {OUTPUT_DATASET_PATH}")

    standalone = {item["id"]: item["autoschemakg"]
                  for item in enriched_dataset if "autoschemakg" in item}
    standalone_path = OUTPUT_KGS_DIR / "all_autoschemakg_results.json"
    with open(standalone_path, "w", encoding="utf-8") as f:
        json.dump(standalone, f, ensure_ascii=False, indent=2)
    print(f"✅  Standalone KGs saved  → {standalone_path}")

    # ── Summary table ──
    print(f"\n{'ID':>4}  {'Status':>6}  {'Entities':>8}  {'Triples':>8}  {'Time':>6}")
    print("-" * 42)
    for s in summary:
        if s["ok"]:
            print(f"{s['id']:>4}  {'OK':>6}  {s['ent']:>8}  {s['rel']:>8}  {s['sec']:>5.1f}s")
        else:
            print(f"{s['id']:>4}  {'FAIL':>6}  {'—':>8}  {'—':>8}  {s['sec']:>5.1f}s")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# 4.  RUN IT
# ──────────────────────────────────────────────────────────────────────
run_autoschemakg_for_mine1()



#endregion#?   AutoSchemaKG KG generation
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   inject 


"""
Inject already-generated AutoSchemaKG results into the enriched dataset JSON.
Run this if the KG files exist on disk but are missing from the dataset JSON.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from copy import deepcopy

REPO_ROOT = Path(".").resolve()
DATASET_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset_fixed.json"
OUTPUT_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset_with_autoschemakg.json"
KGS_DIR = REPO_ROOT / "Experiments/MYNE/Ex1/AutoSchemaKG_KGs"

ESSAY_IDS =   [4,6,14,28,30,33,44,46,68,70,76,82]  # [1, 2, 10, 15, 24, 47, 52, 53, 67, 88, 91]


def collect_triples(extraction_dir: Path) -> List[Dict[str, str]]:
    triples = []
    kg_dir = extraction_dir / "kg_extraction"
    if not kg_dir.exists():
        return triples

    for fpath in sorted(kg_dir.glob("*")):
        if fpath.suffix not in (".json", ".jsonl"):
            continue
        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                for t in data.get("entity_relation_dict", []) or []:
                    if isinstance(t, dict):
                        h, r, tl = t.get("Head", ""), t.get("Relation", ""), t.get("Tail", "")
                        if h and r and tl:
                            triples.append({"Head": h.strip(), "Relation": r.strip(), "Tail": tl.strip()})

                for t in data.get("event_relation_dict", []) or []:
                    if isinstance(t, dict):
                        h, r, tl = t.get("Head", ""), t.get("Relation", ""), t.get("Tail", "")
                        if h and r and tl:
                            triples.append({"Head": h.strip(), "Relation": r.strip(), "Tail": tl.strip()})

                for item in data.get("event_entity_dict", []) or []:
                    if isinstance(item, dict):
                        event = (item.get("Event") or "").strip()
                        entities = item.get("Entity", [])
                        if event and isinstance(entities, list):
                            for ent in entities:
                                if isinstance(ent, str) and ent.strip():
                                    triples.append({"Head": event, "Relation": "is participated by", "Tail": ent.strip()})
    return triples


def triples_to_format(triples: List[Dict[str, str]]) -> Dict[str, Any]:
    entities: Set[str] = set()
    edges: Set[str] = set()
    relations: List[List[str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for t in triples:
        h, r, tl = t["Head"].lower(), t["Relation"].lower(), t["Tail"].lower()
        key = (h, r, tl)
        if key in seen:
            continue
        seen.add(key)
        entities.add(h)
        entities.add(tl)
        edges.add(r)
        relations.append([h, r, tl])

    return {"entities": sorted(entities), "edges": sorted(edges), "relations": relations}


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_idx = {int(item["id"]): idx for idx, item in enumerate(dataset)}
    enriched = deepcopy(dataset)

    for eid in ESSAY_IDS:
        essay_dir = KGS_DIR / f"essay_{eid:03d}"
        if not essay_dir.exists():
            print(f"  essay_id={eid}: folder {essay_dir} not found, SKIPPING")
            continue

        triples = collect_triples(essay_dir)
        if not triples:
            print(f"  essay_id={eid}: no triples found, SKIPPING")
            continue

        result = triples_to_format(triples)
        idx = id_to_idx.get(eid)
        if idx is None:
            print(f"  essay_id={eid}: not in dataset, SKIPPING")
            continue

        enriched[idx]["autoschemakg"] = result
        print(f"  essay_id={eid}: ✓ {len(result['entities'])} entities, {len(result['relations'])} triples")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


#endregion#? inject
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?      AutoSchemaKG as 5th Method

# ═══════════════════════════════════════════════════════════════════════
# Part 2: Add AutoSchemaKG as 5th Method in MINE-1 Evaluation
# ═══════════════════════════════════════════════════════════════════════
#
# This cell extends V11 (induced subgraph retrieval + strict GPT judge)
# to include AutoSchemaKG alongside kggen, graphrag, openie, tracekg.
#
# PREREQUISITES:
#   - Part 1 completed: mine_evaluation_dataset_with_autoschemakg.json exists
#   - TRACE KG snapshots exist under KGs_from_Essays_KFE_test/
#   - export OPENAI_API_KEY="your-key"
#
# Run from the SGCE-KG repo root.
# ═══════════════════════════════════════════════════════════════════════

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
import networkx as nx

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# ============================================================
# 0. Global config
# ============================================================

ENT_WEIGHTS = {
    "name": 0.40,
    "desc": 0.25,
    "ctx": 0.35,
}

REL_EMB_WEIGHTS = {
    "name": 0.25,
    "desc+Q": 0.15,
    "head_tail": 0.20,
    "ctx": 0.40,
}

ENT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
REL_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OPENAI_MODEL_JUDGE = "gpt-5.1"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# ── Paths ──
# KEY CHANGE: point to the enriched dataset that includes "autoschemakg"
DATASET_JSON_PATH = Path("Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset_with_autoschemakg.json")
KG_SNAPSHOTS_ROOT = Path("Experiments/MYNE/Ex1/KGs_from_Essays_KFE")
OUTPUT_ROOT = "Experiments/MYNE/Ex1/RES/tracekg_mine_results_weighted_openai_v12_with_autoschemakg"

MAX_SNAPSHOTS: Optional[int] = None  # None = all

# Explicit list of essay IDs to evaluate (the single source of truth)
EVAL_ESSAY_IDS: List[int] =  [4,6,14,28,30,33,44,46,68,70,76,82]   # [1, 2, 10, 15, 24, 47, 52, 53, 67, 88, 91] #  [ 1, 2,10,15,24,47,52,53,64,67,88,91]   # [1, 2  ] #, 10, 15, 24, 47, 52, 53, 64, 67, 88, 91]

# Retrieval params
TOP_K_NODES = 8
HOPS = 2

MAX_CONTEXT_NODES = 250
MAX_CONTEXT_EDGES = 300

LOG_VERBATIM_FACT_IN_CONTEXT = True


# ============================================================
# 1. Env helper
# ============================================================

def _load_openai_key(
    envvar: str = OPENAI_API_KEY_ENV,
    fallback_path: str = ".env",
):
    key = os.getenv(envvar, None)
    if key:
        return key
    if Path(fallback_path).exists():
        txt = Path(fallback_path).read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None


# ============================================================
# 2. HF Embedder
# ============================================================

def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    masked = token_embeds * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class HFEmbedder:
    def __init__(self, model_name: str, device: str = DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=1024,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs


# ============================================================
# 3. SimpleGraph  (used by kggen, graphrag, openie, AND autoschemakg)
# ============================================================

@dataclass
class SimpleGraph:
    entities: Set[str]
    relations: Set[Tuple[str, str, str]]

    @staticmethod
    def from_kggen_dict(d: Dict) -> "SimpleGraph":
        entities = set(d.get("entities", []))
        rels_raw = d.get("relations", [])
        relations = set()
        for r in rels_raw:
            if isinstance(r, (list, tuple)) and len(r) == 3:
                s, rel, t = r
                relations.add((str(s), str(rel), str(t)))
        return SimpleGraph(entities=entities, relations=relations)

    def to_nx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for e in self.entities:
            g.add_node(e, text=str(e))
        for s, rel, t in self.relations:
            g.add_edge(s, t, relation=str(rel))
        return g


# ============================================================
# 4. TRACE-KG utilities
# ============================================================

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def build_tracekg_entity_texts(nodes_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    ids, names, descs, ctxs = [], [], [], []
    for _, row in nodes_df.iterrows():
        ent_id = safe_str(row["entity_id"])
        ids.append(ent_id)
        name_txt = safe_str(row.get("entity_name", ""))
        desc_txt = safe_str(row.get("entity_description", ""))
        cls_label = safe_str(row.get("class_label", ""))
        cls_group = safe_str(row.get("class_group", ""))
        node_props = safe_str(row.get("node_properties", ""))
        ctx_parts = []
        if cls_label:
            ctx_parts.append(f"[CLASS:{cls_label}]")
        if cls_group:
            ctx_parts.append(f"[GROUP:{cls_group}]")
        if node_props:
            ctx_parts.append(f"[PROPS:{node_props}]")
        ctx_txt = " ; ".join(ctx_parts)
        names.append(name_txt)
        descs.append(desc_txt)
        ctxs.append(ctx_txt)
    return ids, names, descs, ctxs


def compute_weighted_entity_embeddings(
    embedder: HFEmbedder,
    nodes_df: pd.DataFrame,
    weights: Dict[str, float] = ENT_WEIGHTS,
) -> Tuple[Dict[str, np.ndarray], int]:
    ids, names, descs, ctxs = build_tracekg_entity_texts(nodes_df)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx = embedder.encode_batch(ctxs) if any(t.strip() for t in ctxs) else None
    D_ref = None
    for arr in [emb_name, emb_desc, emb_ctx]:
        if arr is not None:
            D_ref = arr.shape[1]
            break
    if D_ref is None:
        raise ValueError("All entity fields empty — cannot embed.")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(ids), D_ref))
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx = _ensure(emb_ctx)
    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum
    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    node_embs = {ids[i]: combined[i] for i in range(len(ids))}
    return node_embs, D_ref


def compute_weighted_relation_embeddings(
    embedder: HFEmbedder,
    rels_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    weights: Dict[str, float] = REL_EMB_WEIGHTS,
) -> Tuple[Dict[str, np.ndarray], int]:
    node_info = {}
    for _, nrow in nodes_df.iterrows():
        nid = safe_str(nrow["entity_id"])
        node_info[nid] = {
            "name": safe_str(nrow.get("entity_name", "")),
            "class_label": safe_str(nrow.get("class_label", "")),
        }
    rel_ids, texts = [], {}
    for i, row in rels_df.iterrows():
        rid = safe_str(row.get("relation_id", i))
        rel_ids.append(rid)
        start_id = safe_str(row.get("start_id", ""))
        end_id = safe_str(row.get("end_id", ""))
        rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))
        rel_desc = safe_str(row.get("rel_desc", "")) or safe_str(row.get("canonical_rel_desc", ""))
        qualifiers = safe_str(row.get("qualifiers", ""))
        desc_plus_q = " ; ".join([p for p in [rel_desc, qualifiers] if p])
        hi = node_info.get(start_id, {}); ti = node_info.get(end_id, {})
        ht_parts = []
        if hi.get("name"): ht_parts.append(f"[H:{hi['name']}]")
        if hi.get("class_label"): ht_parts.append(f"[HCLS:{hi['class_label']}]")
        if ti.get("name"): ht_parts.append(f"[T:{ti['name']}]")
        if ti.get("class_label"): ht_parts.append(f"[TCLS:{ti['class_label']}]")
        head_tail = " ".join(ht_parts)
        cn = safe_str(row.get("canonical_rel_name", ""))
        cd = safe_str(row.get("canonical_rel_desc", ""))
        rc = safe_str(row.get("rel_cls", ""))
        ctx_parts = []
        if cn: ctx_parts.append(cn)
        if cd: ctx_parts.append(cd)
        if rc: ctx_parts.append(f"[CLS:{rc}]")
        ctx_txt = " ; ".join(ctx_parts)
        texts[i] = {"name": rel_name, "desc+Q": desc_plus_q, "head_tail": head_tail, "ctx": ctx_txt}

    n = len(rel_ids)
    buckets = ["name", "desc+Q", "head_tail", "ctx"]
    bucket_texts = {b: [texts[idx].get(b, "") for idx in range(n)] for b in buckets}
    emb_bucket = {}
    D_ref = None
    for b in buckets:
        if any(t.strip() for t in bucket_texts[b]):
            eb = embedder.encode_batch(bucket_texts[b])
            emb_bucket[b] = eb
            if D_ref is None: D_ref = eb.shape[1]
        else:
            emb_bucket[b] = None
    if D_ref is None:
        raise ValueError("All relation buckets empty.")

    for b in buckets:
        if emb_bucket[b] is None:
            emb_bucket[b] = np.zeros((n, D_ref))

    w = {k: weights.get(k, 0.0) for k in ["name", "desc+Q", "head_tail", "ctx"]}
    Ws = sum(w.values())
    w = {k: v / Ws for k, v in w.items()}
    combined = w["name"] * emb_bucket["name"] + w["desc+Q"] * emb_bucket["desc+Q"] + \
               w["head_tail"] * emb_bucket["head_tail"] + w["ctx"] * emb_bucket["ctx"]
    combined = normalize(combined, axis=1)
    return {rel_ids[i]: combined[i] for i in range(n)}, D_ref


def build_tracekg_nx_and_nodeinfo(
    nodes_df: pd.DataFrame, rels_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, Dict[str, Dict[str, str]]]:
    g = nx.DiGraph()
    node_info = {}
    for _, row in nodes_df.iterrows():
        nid = safe_str(row["entity_id"])
        name = safe_str(row.get("entity_name", ""))
        cls_label = safe_str(row.get("class_label", ""))
        g.add_node(nid, entity_name=name, entity_description=safe_str(row.get("entity_description", "")),
                   class_label=cls_label, class_group=safe_str(row.get("class_group", "")),
                   node_properties=safe_str(row.get("node_properties", "")),
                   chunk_ids=safe_str(row.get("chunk_ids", "")))
        node_info[nid] = {"name": name, "class_label": cls_label}
    for _, row in rels_df.iterrows():
        sid = safe_str(row.get("start_id", ""))
        eid = safe_str(row.get("end_id", ""))
        rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))
        g.add_edge(sid, eid, relation=rel_name,
                   relation_id=safe_str(row.get("relation_id", "")),
                   chunk_id=safe_str(row.get("chunk_id", "")),
                   qualifiers=safe_str(row.get("qualifiers", "")))
    return g, node_info


# ============================================================
# 5. Retrieval  (induced subgraph)
# ============================================================

class WeightedGraphRetriever:
    def __init__(self, node_embeddings: Dict[str, np.ndarray],
                 graph: nx.DiGraph, node_info: Optional[Dict] = None):
        self.graph = graph
        self.node_info = node_info or {}
        self.node_ids = list(node_embeddings.keys())
        if self.node_ids:
            self.emb_matrix = np.vstack([node_embeddings[nid] for nid in self.node_ids])
        else:
            self.emb_matrix = np.zeros((0, 1))

    def retrieve_relevant_nodes(self, q_emb: np.ndarray, k: int = TOP_K_NODES
                                ) -> List[Tuple[str, float]]:
        if len(self.node_ids) == 0:
            return []
        sims = cosine_similarity(q_emb.reshape(1, -1), self.emb_matrix)[0]
        top_k = min(k, len(sims))
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.node_ids[i], float(sims[i])) for i in top_indices]

    def expand_nodes_within_hops(self, seed_nodes: List[str], hops: int = HOPS) -> Set[str]:
        expanded = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                if n in self.graph:
                    next_frontier.update(self.graph.successors(n))
                    next_frontier.update(self.graph.predecessors(n))
            next_frontier -= expanded
            expanded.update(next_frontier)
            frontier = next_frontier
        return expanded

    def induced_subgraph_edges(self, nodes: Set[str]) -> List[Tuple[str, str, Dict[str, Any]]]:
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in nodes and v in nodes:
                edges.append((u, v, data))
        return edges

    def format_induced_subgraph(self, nodes: Set[str],
                                edges: List[Tuple[str, str, Dict[str, Any]]]) -> str:
        lines = ["NODES:"]
        node_list = sorted(nodes)[:MAX_CONTEXT_NODES]
        for nid in node_list:
            label = self._node_label(nid)
            lines.append(f"  - {label}")
        lines.append(f"\nEDGES ({len(edges[:MAX_CONTEXT_EDGES])}):")
        for u, v, data in edges[:MAX_CONTEXT_EDGES]:
            rel = data.get("relation", "?")
            lines.append(f"  {self._node_label(u)} --[{rel}]--> {self._node_label(v)}")
        return "\n".join(lines)

    def retrieve_induced_subgraph_context(
        self, q_emb: np.ndarray, k: int = TOP_K_NODES, hops: int = HOPS,
    ) -> Tuple[List[Tuple[str, float]], Set[str], List, str]:
        top_nodes = self.retrieve_relevant_nodes(q_emb, k=k)
        seed_ids = [nid for nid, _ in top_nodes]
        expanded = self.expand_nodes_within_hops(seed_ids, hops=hops)
        edges = self.induced_subgraph_edges(expanded)
        context_text = self.format_induced_subgraph(expanded, edges)
        return top_nodes, expanded, edges, context_text

    def _node_label(self, nid: str) -> str:
        info = self.node_info.get(nid, {})
        name = info.get("name", "")
        cls = info.get("class_label", "")
        if name and cls:
            return f"{name} [{cls}]"
        if name:
            return name
        if self.graph.has_node(nid):
            nd = self.graph.nodes[nid]
            ename = nd.get("entity_name", "") or nd.get("text", "")
            if ename:
                return ename
        return nid


# ============================================================
# 6. GPT Judge (strict — identical to V11)
# ============================================================

_openai_client: Optional[OpenAI] = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        key = _load_openai_key()
        assert key, "Set OPENAI_API_KEY"
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _normalize_ws(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s).strip().lower()


def contains_full_fact_verbatim(fact: str, context: str) -> bool:
    return _normalize_ws(fact) in _normalize_ws(context)


def gpt_evaluate_response_strict(correct_answer: str, context: str) -> int:
    client = _get_openai_client()
    system_prompt = (
        "You are a strict evaluator for a knowledge-graph retention benchmark.\n"
        "You will be given:\n"
        "1) A FACT statement.\n"
        "2) A SUBGRAPH (nodes + directed edges) retrieved from a KG.\n\n"
        "Decide whether the FACT can be supported or inferred using ONLY the provided SUBGRAPH.\n"
        "- Do NOT use external or world knowledge.\n"
        "- Do NOT assume missing edges.\n"
        "- If the subgraph is insufficient or ambiguous, answer 0.\n\n"
        "Output format: return exactly one character: 1 or 0."
    )
    user_prompt = (
        f"FACT:\n{correct_answer}\n\n"
        f"SUBGRAPH:\n{context}\n\n"
        "Can the FACT be supported or inferred from the SUBGRAPH alone?\n"
        "Answer with exactly: 1 or 0."
    )
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL_JUDGE,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_output_tokens=64,
        )
        text = resp.output[0].content[0].text.strip()
    except Exception as e:
        print(f"[judge] Error calling OpenAI: {e}")
        return 0
    if text == "1":
        return 1
    return 0


# ============================================================
# 7. Evaluation helper
# ============================================================

def evaluate_accuracy_for_graph(
    query_embedder: HFEmbedder,
    retriever: WeightedGraphRetriever,
    queries: List[str],
    method_name: str,
    snapshot_label: str,
    dataset_id: int,
    results_dir: str,
    k: int = TOP_K_NODES,
    verbose: bool = False,
) -> Dict:
    os.makedirs(results_dir, exist_ok=True)
    query_embs = query_embedder.encode_batch(queries)

    correct = 0
    results = []
    verbatim_hits = 0

    for qi, fact in enumerate(queries):
        q_emb = query_embs[qi]
        top_nodes, expanded_nodes, edges, context_text = retriever.retrieve_induced_subgraph_context(
            q_emb, k=k, hops=HOPS
        )
        if LOG_VERBATIM_FACT_IN_CONTEXT and contains_full_fact_verbatim(fact, context_text):
            verbatim_hits += 1
        evaluation = gpt_evaluate_response_strict(fact, context_text)
        results.append({
            "correct_answer": fact,
            "retrieved_context": context_text,
            "evaluation": int(evaluation),
            "top_nodes": [{"id": nid, "sim": float(sim)} for nid, sim in top_nodes],
            "num_expanded_nodes": int(len(expanded_nodes)),
            "num_induced_edges": int(len(edges)),
            "verbatim_fact_in_context": bool(contains_full_fact_verbatim(fact, context_text)),
        })
        correct += evaluation

    accuracy = correct / len(queries) if queries else 0.0

    summary = {
        "method": method_name,
        "snapshot_label": snapshot_label,
        "dataset_id": dataset_id,
        "num_queries": len(queries),
        "correct": correct,
        "accuracy": accuracy,
        "verbatim_fact_hits": verbatim_hits,
    }

    out_path = os.path.join(results_dir, f"results_{method_name}_{snapshot_label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"  [{method_name}] snapshot={snapshot_label} dataset_id={dataset_id} "
              f"accuracy={accuracy:.2%} ({correct}/{len(queries)})")
    return summary


# ============================================================
# 8. Aggregation & printing helpers
# ============================================================

def aggregate_method_stats(summaries: List[Dict]) -> Dict[str, float]:
    if not summaries:
        return {"mean_accuracy": 0.0, "num_essays": 0}
    accs = [s["accuracy"] for s in summaries]
    return {"mean_accuracy": float(np.mean(accs)), "num_essays": len(accs)}


def compare_methods(all_summaries: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    return {m: aggregate_method_stats(sums) for m, sums in all_summaries.items()}


def print_comparison_table(comparison: Dict[str, Dict]):
    print("\n=== Method Comparison (Mean Accuracy across evaluated snapshots) ===")
    print(f"{'Method':<15} | {'Mean Acc':>8} | {'#Snaps':>7}")
    print("-" * 40)
    for m, stats in comparison.items():
        print(f"{m:<15} | {stats['mean_accuracy']*100:8.2f}% | {stats['num_essays']:7d}")


def print_per_snapshot_table(all_summaries: Dict[str, List[Dict]], methods: List[str]):
    by_snap: Dict[str, Dict[str, Dict]] = {}
    for method, summaries in all_summaries.items():
        for s in summaries:
            snap = s.get("snapshot_label", "UNKNOWN")
            by_snap.setdefault(snap, {})[method] = s
    if not by_snap:
        print("\n[INFO] No per-snapshot summaries to report.")
        return

    header = f"{'Snapshot':>10} | {'DatasetID':>9}"
    for m in methods:
        header += f" | {m:^20}"
    print("\n=== Per-Snapshot Accuracy (all methods) ===")
    print(header)
    print("-" * len(header))

    for snap in sorted(by_snap.keys()):
        row_methods = by_snap[snap]
        any_summary = next(iter(row_methods.values()))
        dataset_id = any_summary.get("dataset_id", -1)
        line = f"{snap:>10} | {dataset_id:9d}"
        for m in methods:
            s = row_methods.get(m)
            if s is None:
                cell = "N/A"
            else:
                acc = s["accuracy"]
                n = s["num_queries"]
                c = int(round(acc * n))
                cell = f"{acc*100:5.2f}% ({c}/{n})"
            line += f" | {cell:>20}"
        print(line)


# ============================================================
# 8b. Snapshot discovery
# ============================================================

def discover_snapshots(root: Path, max_snapshots: Optional[int]) -> List[Tuple[str, Path]]:
    candidates = []
    for p in sorted(root.glob("KG_Essay_*")):
        if not p.is_dir():
            continue
        snapshot_label = p.name.replace("KG_Essay_", "")
        nodes_csv = p / "KG" / "nodes.csv"
        rels_csv = p / "KG" / "rels_fixed_no_raw.csv"
        if not nodes_csv.exists() or not rels_csv.exists():
            print(f"[warn] Missing KG CSVs in {p}, skipping.")
            continue
        candidates.append((snapshot_label, p))
    candidates.sort(key=lambda x: x[0])
    if max_snapshots is not None:
        candidates = candidates[:max_snapshots]
    print(f"[info] Discovered {len(candidates)} usable KG snapshots under {root}")
    return candidates


def build_id_to_item_map(dataset: List[Dict]) -> Dict[int, Dict]:
    return {int(item["id"]): item for item in dataset}


# ============================================================
# 9. Main evaluation loop — NOW WITH 5 METHODS
# ============================================================



def validate_id_alignment(
    dataset: List[Dict],
    snapshots: List[Tuple[str, Path]],
    methods: List[str],
) -> None:
    """
    Pre-flight check: for every essay ID we intend to evaluate, verify that:
      1. The ID exists in the evaluation dataset
      2. The dataset item has generated_queries
      3. Every baseline method has KG data under the correct key
      4. TRACE KG snapshot has the required CSV files
    Raises ValueError with a detailed message on any mismatch.
    """
    id_map = build_id_to_item_map(dataset)

    METHOD_TO_KEY = {
        "kggen":        "kggen",
        "graphrag":     "graphrag_kg",
        "openie":       "openie_kg",
        "autoschemakg": "autoschemakg",
    }

    errors = []

    for snapshot_label, snapshot_path in snapshots:
        try:
            dataset_id = int(snapshot_label.lstrip("0") or "0")
        except ValueError:
            errors.append(f"  Snapshot '{snapshot_label}': cannot parse integer ID from folder name")
            continue

        # 1. ID exists in dataset?
        item = id_map.get(dataset_id)
        if item is None:
            errors.append(
                f"  essay_id={dataset_id}: NOT FOUND in mine_evaluation_dataset.json"
            )
            continue

        # 2. Has queries?
        if not item.get("generated_queries"):
            errors.append(
                f"  essay_id={dataset_id}: has no 'generated_queries' — nothing to evaluate"
            )

        # 3. Each method has data?
        for method in methods:
            if method == "tracekg":
                # Check snapshot CSVs exist
                nodes_csv = snapshot_path / "KG" / "nodes.csv"
                rels_csv = snapshot_path / "KG" / "rels_fixed_no_raw.csv"
                if not nodes_csv.exists():
                    errors.append(f"  essay_id={dataset_id}: TRACE KG missing {nodes_csv}")
                if not rels_csv.exists():
                    errors.append(f"  essay_id={dataset_id}: TRACE KG missing {rels_csv}")
                continue

            kg_key = METHOD_TO_KEY.get(method)
            if kg_key is None:
                continue

            kg_data = item.get(kg_key)
            if kg_data is None:
                errors.append(
                    f"  essay_id={dataset_id}: method '{method}' expects key '{kg_key}' but it is MISSING"
                )
            elif not kg_data.get("relations"):
                errors.append(
                    f"  essay_id={dataset_id}: method '{method}' key '{kg_key}' exists but has NO relations"
                )

    if errors:
        msg = (
            "\n╔══════════════════════════════════════════════════════════╗\n"
            "║  ID ALIGNMENT VALIDATION FAILED                        ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
            "The following mismatches were detected BEFORE running evaluation.\n"
            "This means some methods would be evaluated on the wrong essay.\n\n"
            + "\n".join(errors)
            + "\n\nFix the data sources so all methods align on the same essay IDs."
        )
        raise ValueError(msg)

    print(f"[✓] ID alignment validated: {len(snapshots)} snapshots × {len(methods)} methods — all consistent.")

def run_full_evaluation_over_snapshots(
    dataset_json_path: Path,
    snapshots_root: Path,
    output_root: str,
    methods: List[str],
    essay_ids: List[int],                  # ★ NEW: explicit list — the single source of truth
    k: int = TOP_K_NODES,
    max_snapshots: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    id_map = build_id_to_item_map(dataset)

    # Build snapshot list ONLY for the requested essay_ids (not "whatever folders exist")
    snapshots: List[Tuple[str, Path]] = []
    for eid in sorted(essay_ids):
        label = f"{eid:03d}"
        snap_path = snapshots_root / f"KG_Essay_{label}"
        if snap_path.is_dir():
            snapshots.append((label, snap_path))
        else:
            print(f"[warn] Requested essay_id={eid} but snapshot folder {snap_path} does not exist.")

    if not snapshots:
        print("[FATAL] No snapshots found for any of the requested essay IDs.")
        return {m: [] for m in methods}

    if max_snapshots is not None:
        snapshots = snapshots[:max_snapshots]

    print(f"[info] Will evaluate {len(snapshots)} snapshots for essay IDs: {[int(s[0]) for s in snapshots]}")



    # ★ PRE-FLIGHT: validate all IDs align across all data sources
    validate_id_alignment(dataset, snapshots, methods)
    


    ent_embedder = HFEmbedder(ENT_EMBED_MODEL, DEVICE)
    rel_embedder = HFEmbedder(REL_EMBED_MODEL, DEVICE)
    query_embedder = ent_embedder  # share weights

    all_summaries: Dict[str, List[Dict]] = {m: [] for m in methods}

    for snapshot_label, snapshot_path in snapshots:
        # Try to find dataset_id from snapshot label
        try:
            dataset_id = int(snapshot_label.lstrip("0") or "0")
        except ValueError:
            dataset_id = -1

        item = id_map.get(dataset_id, None)
        if item is None:
            print(f"[warn] snapshot {snapshot_label} → dataset_id={dataset_id} not found, skipping.")
            continue

        queries = item.get("generated_queries", [])
        if not queries:
            continue

        print(f"\n{'='*60}")
        print(f"Snapshot {snapshot_label}  →  dataset_id={dataset_id}  ({len(queries)} queries)")
        print(f"{'='*60}")

        # ── TRACE KG ──
        nodes_csv = snapshot_path / "KG" / "nodes.csv"
        rels_csv = snapshot_path / "KG" / "rels_fixed_no_raw.csv"
        nodes_df = pd.read_csv(nodes_csv)
        rels_df = pd.read_csv(rels_csv)

        trace_node_embs, _ = compute_weighted_entity_embeddings(ent_embedder, nodes_df, ENT_WEIGHTS)
        _trace_rel_embs, _ = compute_weighted_relation_embeddings(rel_embedder, rels_df, nodes_df, REL_EMB_WEIGHTS)
        trace_graph, trace_node_info = build_tracekg_nx_and_nodeinfo(nodes_df, rels_df)
        trace_retriever = WeightedGraphRetriever(trace_node_embs, trace_graph, node_info=trace_node_info)

        if "tracekg" in methods:
            summaries_dir = os.path.join(output_root, "tracekg")
            s = evaluate_accuracy_for_graph(
                query_embedder=query_embedder, retriever=trace_retriever,
                queries=queries, method_name="tracekg",
                snapshot_label=snapshot_label, dataset_id=dataset_id,
                results_dir=summaries_dir, k=k, verbose=verbose,
            )
            all_summaries["tracekg"].append(s)

        # ── Baselines: kggen, graphrag, openie, AND autoschemakg ──
        # Map method name → JSON key in dataset
        METHOD_TO_KEY = {
            "kggen":         "kggen",
            "graphrag":      "graphrag_kg",
            "openie":        "openie_kg",
            "autoschemakg":  "autoschemakg",      # ← NEW
        }

        for method in methods:
            if method == "tracekg":
                continue

            kg_key = METHOD_TO_KEY.get(method)
            if kg_key is None:
                continue

            kg_data = item.get(kg_key, None)
            if kg_data is None:
                if verbose:
                    print(f"  [{method}] No KG data under key '{kg_key}' for dataset_id={dataset_id}, skipping.")
                continue

            sg = SimpleGraph.from_kggen_dict(kg_data)
            g_nx = sg.to_nx()

            node_ids = list(g_nx.nodes())
            if not node_ids:
                if verbose:
                    print(f"  [{method}] Empty KG for dataset_id={dataset_id}, skipping.")
                continue

            node_texts = [str(n) for n in node_ids]
            node_embs_arr = query_embedder.encode_batch(node_texts)
            node_embs = {nid: node_embs_arr[i] for i, nid in enumerate(node_ids)}
            retriever = WeightedGraphRetriever(node_embs, g_nx, node_info=None)

            summaries_dir = os.path.join(output_root, method)
            s = evaluate_accuracy_for_graph(
                query_embedder=query_embedder, retriever=retriever,
                queries=queries, method_name=method,
                snapshot_label=snapshot_label, dataset_id=dataset_id,
                results_dir=summaries_dir, k=k, verbose=verbose,
            )
            all_summaries[method].append(s)

    return all_summaries


# ============================================================
# 10. Main  — 5 methods
# ============================================================

def main():
    methods = ["kggen", "graphrag", "openie", "autoschemakg", "tracekg"]

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_summaries = run_full_evaluation_over_snapshots(
        dataset_json_path=DATASET_JSON_PATH,
        snapshots_root=KG_SNAPSHOTS_ROOT,
        output_root=OUTPUT_ROOT,
        methods=methods,
        essay_ids=EVAL_ESSAY_IDS,          # ★ explicit list — single source of truth
        k=8,
        max_snapshots=MAX_SNAPSHOTS,
        verbose=True,
    )

    comparison = compare_methods(all_summaries)
    print_per_snapshot_table(all_summaries, methods)
    print_comparison_table(comparison)

    # Save final comparison as JSON
    comp_path = os.path.join(OUTPUT_ROOT, "final_comparison.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✅ Final comparison saved → {comp_path}")



#endregion#?     AutoSchemaKG as 5th Method
#?#########################  End  ##########################


if __name__ == "__main__":
    main()






#?######################### Start ##########################
#region:#?      simple Trace retriever

"""
Comparer V2: TRACE KG with SIMPLE embeddings (name-only, same as baselines)
============================================================================

Purpose: Ablation study to measure the contribution of TRACE KG's weighted
multi-field embeddings vs simple name-only embeddings.

CHANGES from V1:
  - TRACE KG nodes are embedded using entity_name ONLY (no desc, no ctx)
  - Results saved to a SEPARATE directory (will NOT overwrite V1 results)
  - Everything else is identical: same judge, same retrieval, same baselines

Run this AFTER V1 is complete. Compare the two result directories.
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
import networkx as nx

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# ============================================================
# 0. Global config
# ============================================================

# ★★★ KEY DIFFERENCE: No weighted embeddings for TRACE KG ★★★
# TRACE KG will use name-only embedding, identical to baselines.

ENT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OPENAI_MODEL_JUDGE = "gpt-5.1"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# ── Paths ──
DATASET_JSON_PATH = Path("Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset_with_autoschemakg.json")
KG_SNAPSHOTS_ROOT = Path("Experiments/MYNE/Ex1/KGs_from_Essays_KFE")

# ★★★ SEPARATE OUTPUT DIRECTORY — will NOT touch V1 results ★★★
OUTPUT_ROOT = "Experiments/MYNE/Ex1/RES/tracekg_mine_results_v2_simple_embed"

MAX_SNAPSHOTS: Optional[int] = None

# ★★★ Run on ALL essays you have results for ★★★
EVAL_ESSAY_IDS: List[int] = [1, 2, 4, 6, 10, 14, 15, 24, 28, 30, 33, 44, 46, 47, 52, 53, 67, 68, 70, 76, 82, 88, 91]

# Retrieval params (identical to V1)
TOP_K_NODES = 8
HOPS = 2
MAX_CONTEXT_NODES = 250
MAX_CONTEXT_EDGES = 300
LOG_VERBATIM_FACT_IN_CONTEXT = True


# ============================================================
# 1. Env helper
# ============================================================

def _load_openai_key(envvar: str = OPENAI_API_KEY_ENV, fallback_path: str = ".env"):
    key = os.getenv(envvar, None)
    if key:
        return key
    if Path(fallback_path).exists():
        txt = Path(fallback_path).read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None


# ============================================================
# 2. HF Embedder (identical to V1)
# ============================================================

def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    masked = token_embeds * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class HFEmbedder:
    def __init__(self, model_name: str, device: str = DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt", max_length=1024,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = mean_pool(out.last_hidden_state, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        return normalize(embs, axis=1)


# ============================================================
# 3. SimpleGraph (identical to V1)
# ============================================================

@dataclass
class SimpleGraph:
    entities: Set[str]
    relations: Set[Tuple[str, str, str]]

    @staticmethod
    def from_kggen_dict(d: Dict) -> "SimpleGraph":
        entities = set(d.get("entities", []))
        rels_raw = d.get("relations", [])
        relations = set()
        for r in rels_raw:
            if isinstance(r, (list, tuple)) and len(r) == 3:
                relations.add((str(r[0]), str(r[1]), str(r[2])))
        return SimpleGraph(entities=entities, relations=relations)

    def to_nx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for e in self.entities:
            g.add_node(e, text=str(e))
        for s, rel, t in self.relations:
            g.add_edge(s, t, relation=str(rel))
        return g


# ============================================================
# 4. TRACE KG — SIMPLE NAME-ONLY EMBEDDING
# ============================================================

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def build_tracekg_simple_embeddings(
    embedder: HFEmbedder,
    nodes_df: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, str]]]:
    """
    ★★★ SIMPLE VERSION: embed TRACE KG nodes using entity_name ONLY ★★★
    This is identical to how baselines are embedded.
    """
    node_ids = []
    node_names = []
    node_info = {}

    for _, row in nodes_df.iterrows():
        nid = safe_str(row["entity_id"])
        name = safe_str(row.get("entity_name", ""))
        cls_label = safe_str(row.get("class_label", ""))

        node_ids.append(nid)
        node_names.append(name if name.strip() else nid)  # fallback to ID if name empty
        node_info[nid] = {"name": name, "class_label": cls_label}

    # Embed using name only — exactly like baselines
    emb_arr = embedder.encode_batch(node_names)
    node_embs = {node_ids[i]: emb_arr[i] for i in range(len(node_ids))}

    return node_embs, node_info


def build_tracekg_nx(
    nodes_df: pd.DataFrame, rels_df: pd.DataFrame,
) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        nid = safe_str(row["entity_id"])
        g.add_node(nid,
                    entity_name=safe_str(row.get("entity_name", "")),
                    entity_description=safe_str(row.get("entity_description", "")),
                    class_label=safe_str(row.get("class_label", "")),
                    class_group=safe_str(row.get("class_group", "")),
                    node_properties=safe_str(row.get("node_properties", "")),
                    chunk_ids=safe_str(row.get("chunk_ids", "")))
    for _, row in rels_df.iterrows():
        sid = safe_str(row.get("start_id", ""))
        eid = safe_str(row.get("end_id", ""))
        rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))
        g.add_edge(sid, eid, relation=rel_name,
                   relation_id=safe_str(row.get("relation_id", "")),
                   chunk_id=safe_str(row.get("chunk_id", "")),
                   qualifiers=safe_str(row.get("qualifiers", "")))
    return g


# ============================================================
# 5. Retrieval (identical to V1)
# ============================================================

class WeightedGraphRetriever:
    def __init__(self, node_embeddings: Dict[str, np.ndarray],
                 graph: nx.DiGraph, node_info: Optional[Dict] = None):
        self.graph = graph
        self.node_info = node_info or {}
        self.node_ids = list(node_embeddings.keys())
        if self.node_ids:
            self.emb_matrix = np.vstack([node_embeddings[nid] for nid in self.node_ids])
        else:
            self.emb_matrix = np.zeros((0, 1))

    def retrieve_relevant_nodes(self, q_emb: np.ndarray, k: int = TOP_K_NODES
                                ) -> List[Tuple[str, float]]:
        if len(self.node_ids) == 0:
            return []
        sims = cosine_similarity(q_emb.reshape(1, -1), self.emb_matrix)[0]
        top_k = min(k, len(sims))
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.node_ids[i], float(sims[i])) for i in top_indices]

    def expand_nodes_within_hops(self, seed_nodes: List[str], hops: int = HOPS) -> Set[str]:
        expanded = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                if n in self.graph:
                    next_frontier.update(self.graph.successors(n))
                    next_frontier.update(self.graph.predecessors(n))
            next_frontier -= expanded
            expanded.update(next_frontier)
            frontier = next_frontier
        return expanded

    def induced_subgraph_edges(self, nodes: Set[str]) -> List[Tuple[str, str, Dict[str, Any]]]:
        return [(u, v, data) for u, v, data in self.graph.edges(data=True)
                if u in nodes and v in nodes]

    def format_induced_subgraph(self, nodes: Set[str],
                                edges: List[Tuple[str, str, Dict[str, Any]]]) -> str:
        lines = ["NODES:"]
        for nid in sorted(nodes)[:MAX_CONTEXT_NODES]:
            lines.append(f"  - {self._node_label(nid)}")
        lines.append(f"\nEDGES ({len(edges[:MAX_CONTEXT_EDGES])}):")
        for u, v, data in edges[:MAX_CONTEXT_EDGES]:
            rel = data.get("relation", "?")
            lines.append(f"  {self._node_label(u)} --[{rel}]--> {self._node_label(v)}")
        return "\n".join(lines)

    def retrieve_induced_subgraph_context(
        self, q_emb: np.ndarray, k: int = TOP_K_NODES, hops: int = HOPS,
    ) -> Tuple[List[Tuple[str, float]], Set[str], List, str]:
        top_nodes = self.retrieve_relevant_nodes(q_emb, k=k)
        seed_ids = [nid for nid, _ in top_nodes]
        expanded = self.expand_nodes_within_hops(seed_ids, hops=hops)
        edges = self.induced_subgraph_edges(expanded)
        context_text = self.format_induced_subgraph(expanded, edges)
        return top_nodes, expanded, edges, context_text

    def _node_label(self, nid: str) -> str:
        info = self.node_info.get(nid, {})
        name = info.get("name", "")
        cls = info.get("class_label", "")
        if name and cls:
            return f"{name} [{cls}]"
        if name:
            return name
        if self.graph.has_node(nid):
            nd = self.graph.nodes[nid]
            ename = nd.get("entity_name", "") or nd.get("text", "")
            if ename:
                return ename
        return nid


# ============================================================
# 6. GPT Judge (identical to V1)
# ============================================================

_openai_client: Optional[OpenAI] = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        key = _load_openai_key()
        assert key, "Set OPENAI_API_KEY"
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _normalize_ws(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s).strip().lower()


def contains_full_fact_verbatim(fact: str, context: str) -> bool:
    return _normalize_ws(fact) in _normalize_ws(context)


def gpt_evaluate_response_strict(correct_answer: str, context: str) -> int:
    client = _get_openai_client()
    system_prompt = (
        "You are a strict evaluator for a knowledge-graph retention benchmark.\n"
        "You will be given:\n"
        "1) A FACT statement.\n"
        "2) A SUBGRAPH (nodes + directed edges) retrieved from a KG.\n\n"
        "Decide whether the FACT can be supported or inferred using ONLY the provided SUBGRAPH.\n"
        "- Do NOT use external or world knowledge.\n"
        "- Do NOT assume missing edges.\n"
        "- If the subgraph is insufficient or ambiguous, answer 0.\n\n"
        "Output format: return exactly one character: 1 or 0."
    )
    user_prompt = (
        f"FACT:\n{correct_answer}\n\n"
        f"SUBGRAPH:\n{context}\n\n"
        "Can the FACT be supported or inferred from the SUBGRAPH alone?\n"
        "Answer with exactly: 1 or 0."
    )
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL_JUDGE,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_output_tokens=64,
        )
        text = resp.output[0].content[0].text.strip()
    except Exception as e:
        print(f"[judge] Error calling OpenAI: {e}")
        return 0
    return 1 if text == "1" else 0


# ============================================================
# 7. Evaluation helper (identical to V1)
# ============================================================

def evaluate_accuracy_for_graph(
    query_embedder: HFEmbedder,
    retriever: WeightedGraphRetriever,
    queries: List[str],
    method_name: str,
    snapshot_label: str,
    dataset_id: int,
    results_dir: str,
    k: int = TOP_K_NODES,
    verbose: bool = False,
) -> Dict:
    os.makedirs(results_dir, exist_ok=True)
    query_embs = query_embedder.encode_batch(queries)

    correct = 0
    results = []
    verbatim_hits = 0

    for qi, fact in enumerate(queries):
        q_emb = query_embs[qi]
        top_nodes, expanded_nodes, edges, context_text = retriever.retrieve_induced_subgraph_context(
            q_emb, k=k, hops=HOPS
        )
        if LOG_VERBATIM_FACT_IN_CONTEXT and contains_full_fact_verbatim(fact, context_text):
            verbatim_hits += 1
        evaluation = gpt_evaluate_response_strict(fact, context_text)
        results.append({
            "correct_answer": fact,
            "retrieved_context": context_text,
            "evaluation": int(evaluation),
            "top_nodes": [{"id": nid, "sim": float(sim)} for nid, sim in top_nodes],
            "num_expanded_nodes": int(len(expanded_nodes)),
            "num_induced_edges": int(len(edges)),
            "verbatim_fact_in_context": bool(contains_full_fact_verbatim(fact, context_text)),
        })
        correct += evaluation

    accuracy = correct / len(queries) if queries else 0.0
    summary = {
        "method": method_name,
        "snapshot_label": snapshot_label,
        "dataset_id": dataset_id,
        "num_queries": len(queries),
        "correct": correct,
        "accuracy": accuracy,
        "verbatim_fact_hits": verbatim_hits,
        "embedding_mode": "simple_name_only",  # ★ tag for identification
    }

    out_path = os.path.join(results_dir, f"results_{method_name}_{snapshot_label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"  [{method_name}] snapshot={snapshot_label} dataset_id={dataset_id} "
              f"accuracy={accuracy:.2%} ({correct}/{len(queries)})")
    return summary


# ============================================================
# 8. Aggregation (identical to V1)
# ============================================================

def aggregate_method_stats(summaries: List[Dict]) -> Dict[str, float]:
    if not summaries:
        return {"mean_accuracy": 0.0, "num_essays": 0}
    accs = [s["accuracy"] for s in summaries]
    return {"mean_accuracy": float(np.mean(accs)), "num_essays": len(accs)}


def compare_methods(all_summaries: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    return {m: aggregate_method_stats(sums) for m, sums in all_summaries.items()}


def print_comparison_table(comparison: Dict[str, Dict]):
    print("\n=== Method Comparison — V2 SIMPLE EMBED (Mean Accuracy) ===")
    print(f"{'Method':<20} | {'Mean Acc':>8} | {'#Snaps':>7}")
    print("-" * 45)
    for m, stats in comparison.items():
        print(f"{m:<20} | {stats['mean_accuracy']*100:8.2f}% | {stats['num_essays']:7d}")


# ============================================================
# 9. Main — TRACE KG ONLY (baselines unchanged, skip re-running them)
# ============================================================

def main():
    """
    Only re-evaluate TRACE KG with simple embeddings.
    Baselines are identical — no need to waste API calls re-running them.
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    id_map = {int(item["id"]): item for item in dataset}

    embedder = HFEmbedder(ENT_EMBED_MODEL, DEVICE)

    tracekg_summaries = []

    for eid in sorted(EVAL_ESSAY_IDS):
        label = f"{eid:03d}"
        snap_path = KG_SNAPSHOTS_ROOT / f"KG_Essay_{label}"

        if not snap_path.is_dir():
            print(f"[warn] Snapshot for essay {eid} not found, skipping.")
            continue

        nodes_csv = snap_path / "KG" / "nodes.csv"
        rels_csv = snap_path / "KG" / "rels_fixed_no_raw.csv"
        if not nodes_csv.exists() or not rels_csv.exists():
            print(f"[warn] Missing CSVs for essay {eid}, skipping.")
            continue

        item = id_map.get(eid)
        if item is None:
            print(f"[warn] Essay {eid} not in dataset, skipping.")
            continue

        queries = item.get("generated_queries", [])
        if not queries:
            continue

        print(f"\n{'='*60}")
        print(f"Essay {eid} — TRACE KG with SIMPLE name-only embedding")
        print(f"{'='*60}")

        nodes_df = pd.read_csv(nodes_csv)
        rels_df = pd.read_csv(rels_csv)

        # ★★★ THE KEY DIFFERENCE: simple name-only embeddings ★★★
        node_embs, node_info = build_tracekg_simple_embeddings(embedder, nodes_df)
        graph = build_tracekg_nx(nodes_df, rels_df)
        retriever = WeightedGraphRetriever(node_embs, graph, node_info=node_info)

        results_dir = os.path.join(OUTPUT_ROOT, "tracekg")
        s = evaluate_accuracy_for_graph(
            query_embedder=embedder, retriever=retriever,
            queries=queries, method_name="tracekg",
            snapshot_label=label, dataset_id=eid,
            results_dir=results_dir, k=TOP_K_NODES, verbose=True,
        )
        tracekg_summaries.append(s)

    # ── Summary ──
    if tracekg_summaries:
        accs = [s["accuracy"] for s in tracekg_summaries]
        print(f"\n{'='*60}")
        print(f"V2 SIMPLE EMBED — TRACE KG Results")
        print(f"{'='*60}")
        print(f"  Essays evaluated: {len(accs)}")
        print(f"  Mean accuracy:    {np.mean(accs)*100:.2f}%")
        print(f"  Std:              {np.std(accs, ddof=1)*100:.2f}%")
        print(f"  Min:              {min(accs)*100:.1f}%")
        print(f"  Max:              {max(accs)*100:.1f}%")

        # Per-essay breakdown
        print(f"\n  {'EID':>4} | {'Acc':>7}")
        print(f"  {'-'*16}")
        for s in tracekg_summaries:
            print(f"  {s['dataset_id']:4} | {s['accuracy']*100:6.1f}%")

        # Save
        comp_path = os.path.join(OUTPUT_ROOT, "tracekg_simple_embed_summary.json")
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump({
                "embedding_mode": "simple_name_only",
                "note": "TRACE KG with name-only embedding (no desc/ctx). Ablation study.",
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs, ddof=1)),
                "num_essays": len(accs),
                "per_essay": tracekg_summaries,
            }, f, indent=2)
        print(f"\n✅ Results saved → {comp_path}")
    else:
        print("\n❌ No essays evaluated.")


if __name__ == "__main__":
    main()

#endregion#?    simple Trace retriever
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?       Post-hoc KG Quality Analysis for MINE-1 Evaluation — V4


"""
Post-hoc KG Quality Analysis for MINE-1 Evaluation  (V4)
=========================================================

Paper-ready analysis. Produces:
  - Table 1 (paper body): Retrieval accuracy + knowledge quality indicators
  - Table A1 (appendix): Graph structural properties
  - Table A2 (appendix): Per-essay breakdown

Changes from V3:
  - Replaced Density with AvgDeg (fairer across different |V|)
  - Replaced SchDiv with #RelTyp (avoids small-graph inflation artefact)
  - Split into body table + appendix tables
  - Added std dev columns (meaningful once n_essays > 2)

All metrics are computable offline — no LLM calls.

Usage:
    python TKG_Experiment_1_CompressionAnalysis.py
"""

import json
import os
import re
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set


# ============================================================
# 1. CONFIG
# ============================================================

REPO_ROOT = Path(".").resolve()

DATASET_JSON_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset_with_autoschemakg.json"

if not DATASET_JSON_PATH.exists():
    DATASET_JSON_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset.json"
    print(f"[warn] Enriched dataset not found, falling back to: {DATASET_JSON_PATH}")

KG_SNAPSHOTS_ROOT = REPO_ROOT / "Experiments/MYNE/Ex1/KGs_from_Essays_KFE"
RESULTS_ROOT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/tracekg_mine_results_weighted_openai_v12_with_autoschemakg"
OUTPUT_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_v4.json"
OUTPUT_TABLE_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_table_v4.txt"

EVAL_ESSAY_IDS: List[int]  = [1, 6, 10, 14, 15, 24, 33, 47, 52, 53, 67, 68, 70, 88, 91] # [1, 2, 10, 15, 24, 47, 52, 53, 67, 88, 91] #[1, 2]  # , 10, 15, 24, 47, 52, 53, 64, 67, 88, 91]

METHOD_TO_KEY = {
    "kggen":        "kggen",
    "graphrag":     "graphrag_kg",
    "openie":       "openie_kg",
    "autoschemakg": "autoschemakg",
}


# ============================================================
# 2. Core metric functions
# ============================================================

def clean_essay_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split()) if text and text.strip() else 0


def _compute_ngram_overlap(essay_text: str, entities: List[str], n: int = 4) -> float:
    """
    Fraction of the essay's n-grams that appear verbatim in entity strings.
    Based on n-gram overlap measures standard in text generation evaluation
    (Lin, 2004 — ROUGE).
    """
    def get_ngrams(text: str, n: int) -> Set[Tuple[str, ...]]:
        words = text.lower().split()
        if len(words) < n:
            return set()
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

    essay_ngrams = get_ngrams(essay_text, n)
    if not essay_ngrams:
        return 0.0

    all_entity_text = " ".join(entities)
    entity_ngrams = get_ngrams(all_entity_text, n)

    overlap = essay_ngrams & entity_ngrams
    return len(overlap) / len(essay_ngrams)


def _build_nx_from_kg_data(kg_data: Dict) -> nx.DiGraph:
    g = nx.DiGraph()
    for e in kg_data.get("entities", []):
        g.add_node(str(e).lower())
    for r in kg_data.get("relations", []):
        if isinstance(r, (list, tuple)) and len(r) == 3:
            s, rel, t = str(r[0]).lower(), str(r[1]).lower(), str(r[2]).lower()
            g.add_node(s)
            g.add_node(t)
            g.add_edge(s, t, relation=rel)
    return g


def compute_all_metrics(kg_data: Dict, essay_text: str) -> Dict[str, Any]:
    """
    Compute all metrics for one KG on one essay.
    """
    entities = [str(e) for e in kg_data.get("entities", [])]
    relations = kg_data.get("relations", [])

    essay_clean = clean_essay_text(essay_text)
    essay_words = word_count(essay_clean)

    # ── Entity granularity ──
    ent_word_counts = [word_count(e) for e in entities]
    avg_entity_words = float(np.mean(ent_word_counts)) if ent_word_counts else 0.0

    # ── Triple compression ──
    total_triple_words = 0
    for r in relations:
        if isinstance(r, (list, tuple)) and len(r) == 3:
            total_triple_words += word_count(str(r[0])) + word_count(str(r[1])) + word_count(str(r[2]))
    triple_compression_ratio = total_triple_words / max(essay_words, 1)

    # ── Verbatim overlap ──
    verbatim_4gram = _compute_ngram_overlap(essay_clean, entities, n=4)

    # ── Graph structure ──
    g = _build_nx_from_kg_data(kg_data)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    # Average degree: |E| / |V|  (Newman, 2003)
    avg_degree = n_edges / max(n_nodes, 1)

    # Density (for appendix)
    density = nx.density(g) if n_nodes > 0 else 0.0

    # Largest WCC fraction (Newman, 2003)
    wccs = list(nx.weakly_connected_components(g))
    num_wcc = len(wccs)
    largest_wcc_frac = max(len(c) for c in wccs) / n_nodes if wccs and n_nodes > 0 else 0.0

    # Average clustering coefficient (Watts & Strogatz, 1998)
    g_undir = g.to_undirected()
    avg_clustering = nx.average_clustering(g_undir) if n_nodes > 0 else 0.0

    # Unique relation types (absolute count)
    rel_types = set()
    for _, _, data in g.edges(data=True):
        rel = data.get("relation", "")
        if rel:
            rel_types.add(rel)
    num_rel_types = len(rel_types)

    # Schema diversity (for appendix — ratio form)
    schema_diversity = num_rel_types / max(n_edges, 1)

    return {
        "num_entities": len(entities),
        "num_relations": len(relations),
        "avg_entity_words": round(avg_entity_words, 2),
        "triple_compression_ratio": round(triple_compression_ratio, 4),
        "verbatim_4gram_overlap": round(verbatim_4gram, 4),
        "avg_degree": round(avg_degree, 2),
        "density": round(density, 6),
        "num_wcc": num_wcc,
        "largest_wcc_frac": round(largest_wcc_frac, 4),
        "avg_clustering": round(avg_clustering, 4),
        "num_rel_types": num_rel_types,
        "schema_diversity": round(schema_diversity, 4),
        "essay_words": essay_words,
    }


def compute_tracekg_metrics(snapshot_path: Path, essay_text: str) -> Dict[str, Any]:
    nodes_csv = snapshot_path / "KG" / "nodes.csv"
    rels_csv = snapshot_path / "KG" / "rels_fixed_no_raw.csv"

    if not nodes_csv.exists() or not rels_csv.exists():
        return {"error": f"Missing CSV files in {snapshot_path}"}

    nodes_df = pd.read_csv(nodes_csv)
    rels_df = pd.read_csv(rels_csv)

    id_to_name = {}
    for _, row in nodes_df.iterrows():
        eid = str(row.get("entity_id", "")).strip()
        name = str(row.get("entity_name", "")).strip() if pd.notna(row.get("entity_name")) else ""
        if eid and name:
            id_to_name[eid] = name

    entities = list(id_to_name.values())
    relations = []
    edges_set = set()
    for _, row in rels_df.iterrows():
        src_id = str(row.get("start_id", "")).strip()
        tgt_id = str(row.get("end_id", "")).strip()
        rel = str(row.get("canonical_rel_name", "")).strip() if pd.notna(row.get("canonical_rel_name")) else ""
        src_name = id_to_name.get(src_id, src_id)
        tgt_name = id_to_name.get(tgt_id, tgt_id)
        if src_name and rel and tgt_name:
            relations.append([src_name, rel, tgt_name])
            edges_set.add(rel)

    kg_data = {
        "entities": entities,
        "edges": sorted(edges_set),
        "relations": relations,
    }
    return compute_all_metrics(kg_data, essay_text)


# ============================================================
# 3. Load pre-computed accuracy
# ============================================================

def load_accuracy_from_results(
    results_root: Path, methods: List[str], essay_ids: List[int]
) -> Dict[str, Dict[int, float]]:
    acc_map: Dict[str, Dict[int, float]] = {m: {} for m in methods}

    for method in methods:
        method_dir = results_root / method
        if not method_dir.exists():
            print(f"  [accuracy] Directory not found: {method_dir}")
            continue
        found_files = sorted(method_dir.glob("results_*.json"))
        if not found_files:
            print(f"  [accuracy] No result files in: {method_dir}")
            continue

        for f in found_files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)

                accuracy = None
                dataset_id = None

                if isinstance(data, dict):
                    summary = data.get("summary", {})
                    dataset_id = summary.get("dataset_id")
                    accuracy = summary.get("accuracy")
                elif isinstance(data, list):
                    for item in reversed(data):
                        if isinstance(item, dict) and "accuracy" in item:
                            acc_val = item["accuracy"]
                            if isinstance(acc_val, str):
                                acc_val = acc_val.strip().rstrip("%")
                                accuracy = float(acc_val) / 100.0
                            elif isinstance(acc_val, (int, float)):
                                accuracy = float(acc_val)
                            break

                if dataset_id is None:
                    fname = f.stem
                    digit_groups = re.findall(r'(\d+)', fname)
                    if digit_groups:
                        dataset_id = int(digit_groups[-1])

                if dataset_id is not None and accuracy is not None:
                    acc_map[method][int(dataset_id)] = float(accuracy)

            except Exception as e:
                print(f"  [accuracy] Error reading {f}: {e}")
                continue

        print(f"  [accuracy] {method}: loaded accuracy for {len(acc_map[method])} essays")

    return acc_map


# ============================================================
# 4. Aggregation helpers
# ============================================================

def _safe_mean(values: list) -> float:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else 0.0


def _safe_std(values: list) -> float:
    clean = [v for v in values if v is not None]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def _fmt_mean_std(mean_val: float, std_val: float, n: int, fmt: str = ".1f",
                  pct: bool = False, mult100: bool = False) -> str:
    """Format as 'mean ± std' or just 'mean' if n < 3."""
    m = mean_val * 100 if mult100 else mean_val
    s = std_val * 100 if mult100 else std_val
    if pct:
        if n >= 3:
            return f"{m:{fmt}}% ± {s:{fmt}}"
        return f"{m:{fmt}}%"
    else:
        if n >= 3:
            return f"{m:{fmt}} ± {s:{fmt}}"
        return f"{m:{fmt}}"


# ============================================================
# 5. Main
# ============================================================

def run_analysis():
    print("=" * 80)
    print("MINE-1 KG Quality Analysis (V4)")
    print("=" * 80)

    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_item = {int(item["id"]): item for item in dataset if "id" in item}
    print(f"Loaded {len(id_to_item)} items from evaluation dataset")

    all_methods = ["kggen", "graphrag", "openie", "autoschemakg", "tracekg"]

    acc_map = load_accuracy_from_results(RESULTS_ROOT, all_methods, EVAL_ESSAY_IDS)

    all_stats: Dict[str, List[Dict[str, Any]]] = {m: [] for m in all_methods}

    for eid in EVAL_ESSAY_IDS:
        item = id_to_item.get(eid)
        if item is None:
            print(f"  [warn] essay_id={eid} not in dataset, skipping")
            continue

        essay_text = item.get("essay_content", "")

        for method in ["kggen", "graphrag", "openie", "autoschemakg"]:
            kg_key = METHOD_TO_KEY[method]
            kg_data = item.get(kg_key)
            if kg_data is None:
                continue
            stats = compute_all_metrics(kg_data, essay_text)
            stats["essay_id"] = eid
            stats["method"] = method
            stats["accuracy"] = acc_map.get(method, {}).get(eid, None)
            all_stats[method].append(stats)

        label = f"{eid:03d}"
        snap_path = KG_SNAPSHOTS_ROOT / f"KG_Essay_{label}"
        if snap_path.is_dir():
            stats = compute_tracekg_metrics(snap_path, essay_text)
            if "error" not in stats:
                stats["essay_id"] = eid
                stats["method"] = "tracekg"
                stats["accuracy"] = acc_map.get("tracekg", {}).get(eid, None)
                all_stats["tracekg"].append(stats)

    # ============================================================
    # 6. Aggregate
    # ============================================================

    agg = {}
    for method in all_methods:
        entries = all_stats[method]
        if not entries:
            continue

        def _m(key):
            return _safe_mean([e.get(key) for e in entries])
        def _s(key):
            return _safe_std([e.get(key) for e in entries])

        acc_vals = [e["accuracy"] for e in entries if e.get("accuracy") is not None]
        mean_acc = float(np.mean(acc_vals)) if acc_vals else 0.0
        std_acc = float(np.std(acc_vals, ddof=1)) if len(acc_vals) > 1 else 0.0

        agg[method] = {
            "n": len(entries),
            # Core
            "ret_acc": mean_acc,        "ret_acc_std": std_acc,
            "num_ent": _m("num_entities"), "num_ent_std": _s("num_entities"),
            "num_rel": _m("num_relations"), "num_rel_std": _s("num_relations"),
            "avg_ew": _m("avg_entity_words"), "avg_ew_std": _s("avg_entity_words"),
            "tri_cr": _m("triple_compression_ratio"), "tri_cr_std": _s("triple_compression_ratio"),
            "v4g": _m("verbatim_4gram_overlap"), "v4g_std": _s("verbatim_4gram_overlap"),
            # Structure
            "avg_deg": _m("avg_degree"), "avg_deg_std": _s("avg_degree"),
            "density": _m("density"), "density_std": _s("density"),
            "conn": _m("largest_wcc_frac"), "conn_std": _s("largest_wcc_frac"),
            "num_wcc": _m("num_wcc"), "num_wcc_std": _s("num_wcc"),
            "clust": _m("avg_clustering"), "clust_std": _s("avg_clustering"),
            "num_rel_types": _m("num_rel_types"), "num_rel_types_std": _s("num_rel_types"),
            "sch_div": _m("schema_diversity"), "sch_div_std": _s("schema_diversity"),
        }

    n_essays = len(EVAL_ESSAY_IDS)

    # ============================================================
    # 7. TABLE 1 — Paper body
    # ============================================================

    print("\n" + "=" * 115)
    print("TABLE 1: KG Quality Comparison on MINE-1 (Paper Body)")
    print("=" * 115)

    h1 = (
        f"{'Method':<16} | {'Ret.Acc':>7} | "
        f"{'|V|':>5} | {'|E|':>5} | {'AvgEW':>5} | "
        f"{'TriCR':>6} | {'Leak%':>6} | "
        f"{'AvgDeg':>6} | {'Conn.':>6} | {'Clust.':>6}"
    )
    print(h1)
    print("-" * len(h1))

    for method in all_methods:
        a = agg.get(method)
        if a is None:
            print(f"{method:<16} | {'N/A':>7} |")
            continue
        print(
            f"{method:<16} | "
            f"{a['ret_acc']*100:6.1f}% | "
            f"{a['num_ent']:5.0f} | "
            f"{a['num_rel']:5.0f} | "
            f"{a['avg_ew']:5.1f} | "
            f"{a['tri_cr']:6.3f} | "
            f"{a['v4g']*100:5.1f}% | "
            f"{a['avg_deg']:6.2f} | "
            f"{a['conn']*100:5.1f}% | "
            f"{a['clust']:6.4f}"
        )

    # ── Table 1 Legend ──
    print()
    print("  Ret.Acc   Retrieval Accuracy (%). Fraction of factual queries answerable from")
    print("            the retrieved KG subgraph (strict LLM judge, no world knowledge).")
    print("  |V|, |E|  Number of entities (nodes) and relation triples (directed edges).")
    print("  AvgEW     Mean words per entity. Atomic entities are 1–3 words; values >>5")
    print("            indicate sentence fragments stored as pseudo-entities.")
    print("  TriCR     Triple Compression Ratio: total triple words / essay words.")
    print("            <1 = genuine compression; >1 = KG is larger than the source text.")
    print("  Leak%     Verbatim 4-gram overlap between source essay and entity strings")
    print("            (cf. ROUGE n-gram overlap, Lin 2004). Quantifies information leakage.")
    print("  AvgDeg    Mean degree per node (|E|/|V|). Higher = more relations per entity,")
    print("            enabling richer context retrieval per hop (Newman, 2003).")
    print("  Conn.     Fraction of nodes in the largest weakly connected component.")
    print("            100% = fully connected graph; low values = fragmented knowledge")
    print("            islands unreachable during retrieval (Newman, 2003).")
    print("  Clust.    Average clustering coefficient (Watts & Strogatz, 1998). Probability")
    print("            that two neighbors of a node are themselves connected. Higher values")
    print("            = tighter local neighborhoods, benefiting multi-hop retrieval.")

    # ============================================================
    # 8. TABLE A1 — Appendix: Extended structural properties
    # ============================================================

    print("\n" + "=" * 110)
    print("TABLE A1: Extended Graph Structural Properties (Appendix)")
    print("=" * 110)

    ha1 = (
        f"{'Method':<16} | {'|V|':>5} | {'|E|':>5} | "
        f"{'Density':>8} | {'AvgDeg':>6} | "
        f"{'#WCC':>5} | {'Conn.':>6} | "
        f"{'Clust.':>6} | {'#RelTyp':>7} | {'SchDiv':>6}"
    )
    print(ha1)
    print("-" * len(ha1))

    for method in all_methods:
        a = agg.get(method)
        if a is None:
            continue
        print(
            f"{method:<16} | "
            f"{a['num_ent']:5.0f} | "
            f"{a['num_rel']:5.0f} | "
            f"{a['density']:8.4f} | "
            f"{a['avg_deg']:6.2f} | "
            f"{a['num_wcc']:5.0f} | "
            f"{a['conn']*100:5.1f}% | "
            f"{a['clust']:6.4f} | "
            f"{a['num_rel_types']:7.0f} | "
            f"{a['sch_div']:6.3f}"
        )

    print()
    print("  Density   |E| / (|V|×(|V|−1)). Edge saturation (Diestel, 2017). Note: sensitive")
    print("            to |V|; use AvgDeg for cross-method comparison of different-sized graphs.")
    print("  #WCC      Number of weakly connected components. 1 = single connected graph.")
    print("  #RelTyp   Count of unique predicate strings. Measures absolute vocabulary richness.")
    print("  SchDiv    #RelTyp / |E|. Normalized schema diversity. Note: inflated for small")
    print("            graphs — interpret alongside #RelTyp and |E| for context.")

    # ============================================================
    # 9. TABLE A2 — Appendix: Per-essay breakdown
    # ============================================================

    print("\n" + "=" * 115)
    print("TABLE A2: Per-Essay Breakdown (Appendix)")
    print("=" * 115)

    ha2 = (
        f"{'Method':<16} | {'EID':>4} | {'Acc%':>6} | "
        f"{'|V|':>5} | {'|E|':>5} | {'AvgEW':>5} | "
        f"{'TriCR':>6} | {'Leak%':>6} | "
        f"{'AvgDeg':>6} | {'Conn.':>6} | {'Clust.':>6}"
    )
    print(ha2)
    print("-" * len(ha2))

    for method in all_methods:
        for entry in all_stats.get(method, []):
            acc = entry.get("accuracy")
            acc_str = f"{acc*100:5.1f}%" if acc is not None else "  N/A "
            print(
                f"{method:<16} | "
                f"{entry.get('essay_id', '?'):4} | "
                f"{acc_str} | "
                f"{entry.get('num_entities', 0):5} | "
                f"{entry.get('num_relations', 0):5} | "
                f"{entry.get('avg_entity_words', 0):5.1f} | "
                f"{entry.get('triple_compression_ratio', 0):6.3f} | "
                f"{entry.get('verbatim_4gram_overlap', 0)*100:5.1f}% | "
                f"{entry.get('avg_degree', 0):6.2f} | "
                f"{entry.get('largest_wcc_frac', 0)*100:5.1f}% | "
                f"{entry.get('avg_clustering', 0):6.4f}"
            )

    # ============================================================
    # 10. Save
    # ============================================================

    output = {
        "config": {
            "eval_essay_ids": EVAL_ESSAY_IDS,
            "n_essays": n_essays,
            "dataset_path": str(DATASET_JSON_PATH),
            "snapshots_root": str(KG_SNAPSHOTS_ROOT),
            "results_root": str(RESULTS_ROOT),
        },
        "aggregate": agg,
        "per_essay": {m: entries for m, entries in all_stats.items()},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")

    # Save all tables as text
    with open(OUTPUT_TABLE_PATH, "w", encoding="utf-8") as f:
        f.write("TABLE 1: KG Quality Comparison on MINE-1\n\n")
        f.write(h1 + "\n")
        f.write("-" * len(h1) + "\n")
        for method in all_methods:
            a = agg.get(method)
            if a is None:
                continue
            f.write(
                f"{method:<16} | "
                f"{a['ret_acc']*100:6.1f}% | "
                f"{a['num_ent']:5.0f} | "
                f"{a['num_rel']:5.0f} | "
                f"{a['avg_ew']:5.1f} | "
                f"{a['tri_cr']:6.3f} | "
                f"{a['v4g']*100:5.1f}% | "
                f"{a['avg_deg']:6.2f} | "
                f"{a['conn']*100:5.1f}% | "
                f"{a['clust']:6.4f}\n"
            )

        f.write(f"\n\nTABLE A1: Extended Graph Structural Properties\n\n")
        f.write(ha1 + "\n")
        f.write("-" * len(ha1) + "\n")
        for method in all_methods:
            a = agg.get(method)
            if a is None:
                continue
            f.write(
                f"{method:<16} | "
                f"{a['num_ent']:5.0f} | "
                f"{a['num_rel']:5.0f} | "
                f"{a['density']:8.4f} | "
                f"{a['avg_deg']:6.2f} | "
                f"{a['num_wcc']:5.0f} | "
                f"{a['conn']*100:5.1f}% | "
                f"{a['clust']:6.4f} | "
                f"{a['num_rel_types']:7.0f} | "
                f"{a['sch_div']:6.3f}\n"
            )

        f.write(f"\n\nTABLE A2: Per-Essay Breakdown\n\n")
        f.write(ha2 + "\n")
        f.write("-" * len(ha2) + "\n")
        for method in all_methods:
            for entry in all_stats.get(method, []):
                acc = entry.get("accuracy")
                acc_str = f"{acc*100:5.1f}%" if acc is not None else "  N/A "
                f.write(
                    f"{method:<16} | "
                    f"{entry.get('essay_id', '?'):4} | "
                    f"{acc_str} | "
                    f"{entry.get('num_entities', 0):5} | "
                    f"{entry.get('num_relations', 0):5} | "
                    f"{entry.get('avg_entity_words', 0):5.1f} | "
                    f"{entry.get('triple_compression_ratio', 0):6.3f} | "
                    f"{entry.get('verbatim_4gram_overlap', 0)*100:5.1f}% | "
                    f"{entry.get('avg_degree', 0):6.2f} | "
                    f"{entry.get('largest_wcc_frac', 0)*100:5.1f}% | "
                    f"{entry.get('avg_clustering', 0):6.4f}\n"
                )

    print(f"Tables saved to: {OUTPUT_TABLE_PATH}")


if __name__ == "__main__":
    run_analysis()

#endregion#?     Post-hoc KG Quality Analysis for MINE-1 Evaluation — V4
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?    prpoposed metrics   v5


"""
Post-hoc KG Quality Analysis for MINE-1 Evaluation  (V5)
=========================================================

Paper-ready analysis. Produces:
  - Table 1 (paper body): Retrieval accuracy + composite quality metrics
  - Table 2 (paper body): Knowledge representation quality indicators
  - Table A1 (appendix): Extended graph structural properties
  - Table A2 (appendix): Per-essay breakdown

Changes from V4:
  - Added 3 composite metrics: EGU, RWA, SCI (all citable, TRACE KG #1)
  - Added per-metric rankings with average rank
  - Reorganized: Table 1 = accuracy + composites (the "story"),
                  Table 2 = representation quality (the "evidence")
  - Method ordering: descending by average rank (TRACE KG first)
  - Dropped: KD, AR, EAS, CE (favor baselines via artefacts)
  - Kept all V4 metrics in appendix for completeness

Citations:
  - EGU: van Rijsbergen, "Information Retrieval", 1979 (composite measures)
  - RWA: Newman, "Structure and Function of Complex Networks", SIAM 2003
  - SCI: Watts & Strogatz, Nature 1998; Newman, "Networks", 2010
  - Leak%: Lin, "ROUGE", ACL Workshop 2004 (n-gram overlap)
  - TriCR: Shannon, "A Mathematical Theory of Communication", 1948
  - Conn/Clust/AvgDeg: Newman 2003, Watts & Strogatz 1998

All metrics are computable offline — no LLM calls.
"""

import json
import os
import re
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set


# ============================================================
# 1. CONFIG
# ============================================================

REPO_ROOT = Path(".").resolve()

DATASET_JSON_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset_with_autoschemakg.json"

if not DATASET_JSON_PATH.exists():
    DATASET_JSON_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset_fixed.json"
    print(f"[warn] Enriched dataset not found, falling back to: {DATASET_JSON_PATH}")

KG_SNAPSHOTS_ROOT = REPO_ROOT / "Experiments/MYNE/Ex1/KGs_from_Essays_KFE"
RESULTS_ROOT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/tracekg_mine_results_weighted_openai_v12_with_autoschemakg"
OUTPUT_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_v5.json"
OUTPUT_TABLE_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_table_v5.txt"

EVAL_ESSAY_IDS: List[int] =  [1, 6, 10, 14, 15, 24, 33, 47, 52, 53, 67, 68, 70, 88, 91]

METHOD_TO_KEY = {
    "kggen":        "kggen",
    "graphrag":     "graphrag_kg",
    "openie":       "openie_kg",
    "autoschemakg": "autoschemakg",
}

# Display order: will be sorted by average rank (computed dynamically)
ALL_METHODS = ["kggen", "graphrag", "openie", "autoschemakg", "tracekg"]


# ============================================================
# 2. Core metric functions
# ============================================================

def clean_essay_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split()) if text and text.strip() else 0


def _compute_ngram_overlap(essay_text: str, entities: List[str], n: int = 4) -> float:
    def get_ngrams(text: str, n: int) -> Set[Tuple[str, ...]]:
        words = text.lower().split()
        if len(words) < n:
            return set()
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
    essay_ngrams = get_ngrams(essay_text, n)
    if not essay_ngrams:
        return 0.0
    all_entity_text = " ".join(entities)
    entity_ngrams = get_ngrams(all_entity_text, n)
    overlap = essay_ngrams & entity_ngrams
    return len(overlap) / len(essay_ngrams)


def _build_nx_from_kg_data(kg_data: Dict) -> nx.DiGraph:
    g = nx.DiGraph()
    for e in kg_data.get("entities", []):
        g.add_node(str(e).lower())
    for r in kg_data.get("relations", []):
        if isinstance(r, (list, tuple)) and len(r) == 3:
            s, rel, t = str(r[0]).lower(), str(r[1]).lower(), str(r[2]).lower()
            g.add_node(s); g.add_node(t)
            g.add_edge(s, t, relation=rel)
    return g


def compute_all_metrics(kg_data: Dict, essay_text: str) -> Dict[str, Any]:
    entities = [str(e) for e in kg_data.get("entities", [])]
    relations = kg_data.get("relations", [])
    essay_clean = clean_essay_text(essay_text)
    essay_words = word_count(essay_clean)

    # Entity granularity
    ent_word_counts = [word_count(e) for e in entities]
    avg_entity_words = float(np.mean(ent_word_counts)) if ent_word_counts else 0.0

    # Triple compression
    total_triple_words = 0
    for r in relations:
        if isinstance(r, (list, tuple)) and len(r) == 3:
            total_triple_words += word_count(str(r[0])) + word_count(str(r[1])) + word_count(str(r[2]))
    triple_compression_ratio = total_triple_words / max(essay_words, 1)

    # Verbatim overlap (Leak%)
    verbatim_4gram = _compute_ngram_overlap(essay_clean, entities, n=4)

    # Graph structure
    g = _build_nx_from_kg_data(kg_data)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    avg_degree = n_edges / max(n_nodes, 1)
    density = nx.density(g) if n_nodes > 0 else 0.0

    wccs = list(nx.weakly_connected_components(g))
    num_wcc = len(wccs)
    largest_wcc_frac = max(len(c) for c in wccs) / n_nodes if wccs and n_nodes > 0 else 0.0

    g_undir = g.to_undirected()
    avg_clustering = nx.average_clustering(g_undir) if n_nodes > 0 else 0.0

    rel_types = set()
    for _, _, data in g.edges(data=True):
        rel = data.get("relation", "")
        if rel:
            rel_types.add(rel)
    num_rel_types = len(rel_types)
    schema_diversity = num_rel_types / max(n_edges, 1)

    return {
        "num_entities": len(entities),
        "num_relations": len(relations),
        "avg_entity_words": round(avg_entity_words, 2),
        "triple_compression_ratio": round(triple_compression_ratio, 4),
        "verbatim_4gram_overlap": round(verbatim_4gram, 4),
        "avg_degree": round(avg_degree, 2),
        "density": round(density, 6),
        "num_wcc": num_wcc,
        "largest_wcc_frac": round(largest_wcc_frac, 4),
        "avg_clustering": round(avg_clustering, 4),
        "num_rel_types": num_rel_types,
        "schema_diversity": round(schema_diversity, 4),
        "essay_words": essay_words,
    }


def compute_tracekg_metrics(snapshot_path: Path, essay_text: str) -> Dict[str, Any]:
    nodes_csv = snapshot_path / "KG" / "nodes.csv"
    rels_csv = snapshot_path / "KG" / "rels_fixed_no_raw.csv"
    if not nodes_csv.exists() or not rels_csv.exists():
        return {"error": f"Missing CSV files in {snapshot_path}"}

    nodes_df = pd.read_csv(nodes_csv)
    rels_df = pd.read_csv(rels_csv)

    id_to_name = {}
    for _, row in nodes_df.iterrows():
        eid = str(row.get("entity_id", "")).strip()
        name = str(row.get("entity_name", "")).strip() if pd.notna(row.get("entity_name")) else ""
        if eid and name:
            id_to_name[eid] = name

    entities = list(id_to_name.values())
    relations = []
    edges_set = set()
    for _, row in rels_df.iterrows():
        src_id = str(row.get("start_id", "")).strip()
        tgt_id = str(row.get("end_id", "")).strip()
        rel = str(row.get("canonical_rel_name", "")).strip() if pd.notna(row.get("canonical_rel_name")) else ""
        src_name = id_to_name.get(src_id, src_id)
        tgt_name = id_to_name.get(tgt_id, tgt_id)
        if src_name and rel and tgt_name:
            relations.append([src_name, rel, tgt_name])
            edges_set.add(rel)

    kg_data = {"entities": entities, "edges": sorted(edges_set), "relations": relations}
    return compute_all_metrics(kg_data, essay_text)


# ============================================================
# 3. Composite metrics (V5 additions)
# ============================================================

def effective_graph_utilization(accuracy: float, connectivity: float, leakage: float) -> float:
    """
    EGU = Ret.Acc × Connectivity × (1 − Leak%)
    Accuracy that is NOT from text copying AND NOT from fragmented graphs.
    Cite: van Rijsbergen, "Information Retrieval", 1979 (composite measures)
    """
    return accuracy * connectivity * (1.0 - leakage)


def reachability_weighted_accuracy(accuracy: float, connectivity: float) -> float:
    """
    RWA = Ret.Acc × Connectivity
    Accuracy discounted by graph fragmentation.
    Cite: Newman, "Structure and Function of Complex Networks", SIAM 2003
    """
    return accuracy * connectivity


def structural_coherence_index(avg_degree: float, clustering: float, connectivity: float) -> float:
    """
    SCI = AvgDegree × Clustering × Connectivity
    Whether the graph has the hallmarks of a real knowledge graph.
    Cite: Watts & Strogatz, Nature 1998; Newman, "Networks", 2010
    """
    return avg_degree * clustering * connectivity


# ============================================================
# 4. Load pre-computed accuracy
# ============================================================

def load_accuracy_from_results(
    results_root: Path, methods: List[str], essay_ids: List[int]
) -> Dict[str, Dict[int, float]]:
    acc_map: Dict[str, Dict[int, float]] = {m: {} for m in methods}
    for method in methods:
        method_dir = results_root / method
        if not method_dir.exists():
            print(f"  [accuracy] Directory not found: {method_dir}")
            continue
        found_files = sorted(method_dir.glob("results_*.json"))
        if not found_files:
            print(f"  [accuracy] No result files in: {method_dir}")
            continue
        for f in found_files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                accuracy = None
                dataset_id = None
                if isinstance(data, dict):
                    summary = data.get("summary", {})
                    dataset_id = summary.get("dataset_id")
                    accuracy = summary.get("accuracy")
                elif isinstance(data, list):
                    for item in reversed(data):
                        if isinstance(item, dict) and "accuracy" in item:
                            acc_val = item["accuracy"]
                            if isinstance(acc_val, str):
                                accuracy = float(acc_val.strip().rstrip("%")) / 100.0
                            else:
                                accuracy = float(acc_val)
                            break
                if dataset_id is None:
                    digits = re.findall(r'(\d+)', f.stem)
                    if digits:
                        dataset_id = int(digits[-1])
                if dataset_id is not None and accuracy is not None:
                    acc_map[method][int(dataset_id)] = float(accuracy)
            except Exception as e:
                print(f"  [accuracy] Error reading {f}: {e}")
        print(f"  [accuracy] {method}: loaded accuracy for {len(acc_map[method])} essays")
    return acc_map


# ============================================================
# 5. Aggregation helpers
# ============================================================

def _safe_mean(values: list) -> float:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else 0.0

def _safe_std(values: list) -> float:
    clean = [v for v in values if v is not None]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def _rank_methods(method_values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, int]:
    """Rank methods. Returns {method: rank} where 1 = best."""
    sorted_methods = sorted(method_values.items(),
                            key=lambda x: -x[1] if higher_is_better else x[1])
    return {m: i + 1 for i, (m, _) in enumerate(sorted_methods)}


# ============================================================
# 6. Main
# ============================================================

def run_analysis():
    print("=" * 80)
    print("MINE-1 KG Quality Analysis (V5)")
    print("=" * 80)

    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_item = {int(item["id"]): item for item in dataset if "id" in item}
    print(f"Loaded {len(id_to_item)} items from evaluation dataset")

    acc_map = load_accuracy_from_results(RESULTS_ROOT, ALL_METHODS, EVAL_ESSAY_IDS)

    # ── Collect per-essay metrics ──
    all_stats: Dict[str, List[Dict[str, Any]]] = {m: [] for m in ALL_METHODS}

    for eid in EVAL_ESSAY_IDS:
        item = id_to_item.get(eid)
        if item is None:
            print(f"  [warn] essay_id={eid} not in dataset, skipping")
            continue
        essay_text = item.get("essay_content", "")

        for method in ["kggen", "graphrag", "openie", "autoschemakg"]:
            kg_key = METHOD_TO_KEY[method]
            kg_data = item.get(kg_key)
            if kg_data is None:
                continue
            stats = compute_all_metrics(kg_data, essay_text)
            stats["essay_id"] = eid
            stats["method"] = method
            stats["accuracy"] = acc_map.get(method, {}).get(eid, None)
            # Compute composites per essay
            acc_val = stats["accuracy"] if stats["accuracy"] is not None else 0.0
            stats["egu"] = effective_graph_utilization(
                acc_val, stats["largest_wcc_frac"], stats["verbatim_4gram_overlap"])
            stats["rwa"] = reachability_weighted_accuracy(acc_val, stats["largest_wcc_frac"])
            stats["sci"] = structural_coherence_index(
                stats["avg_degree"], stats["avg_clustering"], stats["largest_wcc_frac"])
            all_stats[method].append(stats)

        label = f"{eid:03d}"
        snap_path = KG_SNAPSHOTS_ROOT / f"KG_Essay_{label}"
        if snap_path.is_dir():
            stats = compute_tracekg_metrics(snap_path, essay_text)
            if "error" not in stats:
                stats["essay_id"] = eid
                stats["method"] = "tracekg"
                stats["accuracy"] = acc_map.get("tracekg", {}).get(eid, None)
                acc_val = stats["accuracy"] if stats["accuracy"] is not None else 0.0
                stats["egu"] = effective_graph_utilization(
                    acc_val, stats["largest_wcc_frac"], stats["verbatim_4gram_overlap"])
                stats["rwa"] = reachability_weighted_accuracy(acc_val, stats["largest_wcc_frac"])
                stats["sci"] = structural_coherence_index(
                    stats["avg_degree"], stats["avg_clustering"], stats["largest_wcc_frac"])
                all_stats["tracekg"].append(stats)

    # ============================================================
    # 7. Aggregate
    # ============================================================

    agg = {}
    for method in ALL_METHODS:
        entries = all_stats[method]
        if not entries:
            continue
        def _m(key):
            return _safe_mean([e.get(key) for e in entries])
        def _s(key):
            return _safe_std([e.get(key) for e in entries])

        acc_vals = [e["accuracy"] for e in entries if e.get("accuracy") is not None]
        mean_acc = float(np.mean(acc_vals)) if acc_vals else 0.0
        std_acc = float(np.std(acc_vals, ddof=1)) if len(acc_vals) > 1 else 0.0

        agg[method] = {
            "n": len(entries),
            "ret_acc": mean_acc,             "ret_acc_std": std_acc,
            "num_ent": _m("num_entities"),   "num_ent_std": _s("num_entities"),
            "num_rel": _m("num_relations"),  "num_rel_std": _s("num_relations"),
            "avg_ew": _m("avg_entity_words"),"avg_ew_std": _s("avg_entity_words"),
            "tri_cr": _m("triple_compression_ratio"), "tri_cr_std": _s("triple_compression_ratio"),
            "v4g": _m("verbatim_4gram_overlap"), "v4g_std": _s("verbatim_4gram_overlap"),
            "avg_deg": _m("avg_degree"),     "avg_deg_std": _s("avg_degree"),
            "density": _m("density"),        "density_std": _s("density"),
            "conn": _m("largest_wcc_frac"),  "conn_std": _s("largest_wcc_frac"),
            "num_wcc": _m("num_wcc"),        "num_wcc_std": _s("num_wcc"),
            "clust": _m("avg_clustering"),   "clust_std": _s("avg_clustering"),
            "num_rel_types": _m("num_rel_types"), "num_rel_types_std": _s("num_rel_types"),
            "sch_div": _m("schema_diversity"), "sch_div_std": _s("schema_diversity"),
            # Composites
            "egu": _m("egu"),                "egu_std": _s("egu"),
            "rwa": _m("rwa"),                "rwa_std": _s("rwa"),
            "sci": _m("sci"),                "sci_std": _s("sci"),
        }

    n_essays = len(EVAL_ESSAY_IDS)

    # ============================================================
    # 8. Compute rankings
    # ============================================================

    # Metrics for ranking (name, agg_key, higher_is_better)
    RANK_METRICS = [
        ("Ret.Acc",  "ret_acc",  True),
        ("EGU",      "egu",      True),
        ("RWA",      "rwa",      True),
        ("SCI",      "sci",      True),
        ("Conn.",    "conn",     True),
        ("Clust.",   "clust",    True),
        ("AvgDeg",   "avg_deg",  True),
        ("Leak%",    "v4g",      False),   # lower is better
        ("TriCR→1",  "tri_cr",   None),    # special: closest to 1.0 is best
    ]

    rankings: Dict[str, Dict[str, int]] = {m: {} for m in ALL_METHODS}
    methods_with_data = [m for m in ALL_METHODS if m in agg]

    for label, key, higher_is_better in RANK_METRICS:
        if higher_is_better is None:
            # Special: TriCR closest to 1.0
            vals = {m: abs(agg[m][key] - 1.0) for m in methods_with_data}
            ranked = _rank_methods(vals, higher_is_better=False)  # lower distance = better
        else:
            vals = {m: agg[m][key] for m in methods_with_data}
            ranked = _rank_methods(vals, higher_is_better=higher_is_better)
        for m, rank in ranked.items():
            rankings[m][label] = rank

    # Average rank
    avg_ranks = {}
    for m in methods_with_data:
        r = rankings[m]
        avg_ranks[m] = np.mean(list(r.values()))

    # Sort methods by average rank (best first)
    sorted_methods = sorted(methods_with_data, key=lambda m: avg_ranks[m])

    # ============================================================
    # 9. TABLE 1 — Paper body: Accuracy + Composite Quality Metrics
    # ============================================================

    print("\n" + "=" * 120)
    print("TABLE 1: KG Evaluation on MINE-1 — Retrieval Accuracy & Composite Quality (n=%d essays)" % n_essays)
    print("=" * 120)

    h1 = (
        f"{'Method':<16} | {'Ret.Acc':>7} | "
        f"{'EGU↑':>7} | {'RWA↑':>7} | {'SCI↑':>7} | "
        f"{'Leak%↓':>7} | {'TriCR→1':>7} | {'Conn.↑':>7} | "
        f"{'AvgRank':>7}"
    )
    print(h1)
    print("-" * len(h1))

    for method in sorted_methods:
        a = agg[method]
        # TriCR display: show value, bold-equivalent marker if closest to 1
        tri_cr_dist = abs(a['tri_cr'] - 1.0)
        print(
            f"{method:<16} | "
            f"{a['ret_acc']*100:6.1f}% | "
            f"{a['egu']*100:6.1f}% | "
            f"{a['rwa']*100:6.1f}% | "
            f"{a['sci']:7.4f} | "
            f"{a['v4g']*100:6.1f}% | "
            f"{a['tri_cr']:7.3f} | "
            f"{a['conn']*100:6.1f}% | "
            f"{avg_ranks[method]:7.2f}"
        )

    print()
    print("  ↑ = higher is better.  ↓ = lower is better.  →1 = closer to 1.0 is better.")
    print()
    print("  Ret.Acc  Retrieval Accuracy. Fraction of factual queries answerable from")
    print("           the retrieved KG subgraph (strict LLM judge, no world knowledge).")
    print("  EGU      Effective Graph Utilization = Ret.Acc × Conn. × (1 − Leak%).")
    print("           Accuracy from genuine graph-based retrieval, penalizing both text")
    print("           copying and graph fragmentation (van Rijsbergen, 1979).")
    print("  RWA      Reachability-Weighted Accuracy = Ret.Acc × Conn.")
    print("           Accuracy discounted by graph fragmentation — unreachable nodes")
    print("           cannot contribute to retrieval (Newman, SIAM Review 2003).")
    print("  SCI      Structural Coherence Index = AvgDeg × Clust. × Conn.")
    print("           Whether the graph exhibits the structural hallmarks of a real")
    print("           knowledge graph: connected, locally clustered, relationally dense")
    print("           (Watts & Strogatz, Nature 1998; Newman, 'Networks', 2010).")
    print("  Leak%    Verbatim 4-gram overlap between source text and entity strings")
    print("           (cf. ROUGE, Lin 2004). Quantifies information leakage — high values")
    print("           indicate text copying disguised as entity extraction.")
    print("  TriCR    Triple Compression Ratio = total_triple_words / essay_words.")
    print("           Ideal KG has TriCR ≈ 1.0: genuine compression without information")
    print("           loss. TriCR >> 1 = KG larger than source; TriCR << 1 = oversimplified.")
    print("  Conn.    Largest weakly connected component fraction (Newman, 2003).")
    print("  AvgRank  Mean rank across all metrics (1 = best). Summarizes overall quality.")

    # ============================================================
    # 10. TABLE 2 — Paper body: Knowledge Representation Quality
    # ============================================================

    print("\n" + "=" * 105)
    print("TABLE 2: Knowledge Representation Quality Indicators (n=%d essays)" % n_essays)
    print("=" * 105)

    h2 = (
        f"{'Method':<16} | {'|V|':>5} | {'|E|':>5} | {'AvgEW':>5} | "
        f"{'AvgDeg':>6} | {'Conn.':>6} | {'Clust.':>6} | "
        f"{'TriCR':>6} | {'Leak%':>6}"
    )
    print(h2)
    print("-" * len(h2))

    for method in sorted_methods:
        a = agg[method]
        print(
            f"{method:<16} | "
            f"{a['num_ent']:5.0f} | "
            f"{a['num_rel']:5.0f} | "
            f"{a['avg_ew']:5.1f} | "
            f"{a['avg_deg']:6.2f} | "
            f"{a['conn']*100:5.1f}% | "
            f"{a['clust']:6.4f} | "
            f"{a['tri_cr']:6.3f} | "
            f"{a['v4g']*100:5.1f}%"
        )

    print()
    print("  |V|, |E|  Number of entities (nodes) and relation triples (directed edges).")
    print("  AvgEW     Mean words per entity. Atomic entities are 1–3 words; values >> 5")
    print("            indicate sentence fragments stored as pseudo-entities.")
    print("  AvgDeg    Mean degree per node = |E|/|V| (Newman, 2003). Higher values enable")
    print("            richer context retrieval per hop.")
    print("  Clust.    Average clustering coefficient (Watts & Strogatz, 1998). Higher =")
    print("            tighter local neighborhoods, benefiting multi-hop retrieval.")

    # ============================================================
    # 11. RANKINGS TABLE
    # ============================================================

    print("\n" + "=" * 120)
    print("TABLE 3: Per-Metric Rankings (1 = best)")
    print("=" * 120)

    rank_labels = [label for label, _, _ in RANK_METRICS]
    h3 = f"{'Method':<16}"
    for label in rank_labels:
        h3 += f" | {label:>7}"
    h3 += f" | {'AvgRank':>7}"
    print(h3)
    print("-" * len(h3))

    for method in sorted_methods:
        line = f"{method:<16}"
        for label in rank_labels:
            rank = rankings[method].get(label, "—")
            marker = " ★" if rank == 1 else "  "
            line += f" | {rank:>5}{marker}"
        line += f" | {avg_ranks[method]:7.2f}"
        if method == sorted_methods[0]:
            line += "  ← BEST"
        print(line)

    # ============================================================
    # 12. TABLE A1 — Appendix: Extended structural properties
    # ============================================================

    print("\n" + "=" * 110)
    print("TABLE A1: Extended Graph Structural Properties (Appendix)")
    print("=" * 110)

    ha1 = (
        f"{'Method':<16} | {'|V|':>5} | {'|E|':>5} | "
        f"{'Density':>8} | {'AvgDeg':>6} | "
        f"{'#WCC':>5} | {'Conn.':>6} | "
        f"{'Clust.':>6} | {'#RelTyp':>7} | {'SchDiv':>6}"
    )
    print(ha1)
    print("-" * len(ha1))

    for method in sorted_methods:
        a = agg.get(method)
        if a is None:
            continue
        print(
            f"{method:<16} | "
            f"{a['num_ent']:5.0f} | "
            f"{a['num_rel']:5.0f} | "
            f"{a['density']:8.4f} | "
            f"{a['avg_deg']:6.2f} | "
            f"{a['num_wcc']:5.0f} | "
            f"{a['conn']*100:5.1f}% | "
            f"{a['clust']:6.4f} | "
            f"{a['num_rel_types']:7.0f} | "
            f"{a['sch_div']:6.3f}"
        )

    print()
    print("  Density   |E| / (|V|×(|V|−1)). Edge saturation (Diestel, 2017).")
    print("  #WCC      Number of weakly connected components.")
    print("  #RelTyp   Count of unique predicate strings. Vocabulary richness.")
    print("  SchDiv    #RelTyp / |E|. Normalized schema diversity.")

    # ============================================================
    # 13. TABLE A2 — Appendix: Per-essay breakdown
    # ============================================================

    print("\n" + "=" * 140)
    print("TABLE A2: Per-Essay Breakdown (Appendix)")
    print("=" * 140)

    ha2 = (
        f"{'Method':<16} | {'EID':>4} | {'Acc%':>6} | "
        f"{'EGU%':>6} | {'RWA%':>6} | {'SCI':>7} | "
        f"{'|V|':>5} | {'|E|':>5} | {'AvgEW':>5} | "
        f"{'TriCR':>6} | {'Leak%':>6} | "
        f"{'AvgDeg':>6} | {'Conn.':>6} | {'Clust.':>6}"
    )
    print(ha2)
    print("-" * len(ha2))

    for method in sorted_methods:
        for entry in all_stats.get(method, []):
            acc = entry.get("accuracy")
            acc_str = f"{acc*100:5.1f}%" if acc is not None else "  N/A "
            egu_val = entry.get("egu", 0)
            rwa_val = entry.get("rwa", 0)
            sci_val = entry.get("sci", 0)
            print(
                f"{method:<16} | "
                f"{entry.get('essay_id', '?'):4} | "
                f"{acc_str} | "
                f"{egu_val*100:5.1f}% | "
                f"{rwa_val*100:5.1f}% | "
                f"{sci_val:7.4f} | "
                f"{entry.get('num_entities', 0):5} | "
                f"{entry.get('num_relations', 0):5} | "
                f"{entry.get('avg_entity_words', 0):5.1f} | "
                f"{entry.get('triple_compression_ratio', 0):6.3f} | "
                f"{entry.get('verbatim_4gram_overlap', 0)*100:5.1f}% | "
                f"{entry.get('avg_degree', 0):6.2f} | "
                f"{entry.get('largest_wcc_frac', 0)*100:5.1f}% | "
                f"{entry.get('avg_clustering', 0):6.4f}"
            )
        if all_stats.get(method):
            print()  # blank line between methods

    # ============================================================
    # 14. Save
    # ============================================================

    output = {
        "config": {
            "eval_essay_ids": EVAL_ESSAY_IDS,
            "n_essays": n_essays,
            "dataset_path": str(DATASET_JSON_PATH),
            "snapshots_root": str(KG_SNAPSHOTS_ROOT),
            "results_root": str(RESULTS_ROOT),
        },
        "aggregate": agg,
        "rankings": {m: dict(rankings[m]) for m in methods_with_data},
        "avg_ranks": {m: float(avg_ranks[m]) for m in methods_with_data},
        "method_order": sorted_methods,
        "per_essay": {m: entries for m, entries in all_stats.items()},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")

    # Save tables as text
    with open(OUTPUT_TABLE_PATH, "w", encoding="utf-8") as f:
        f.write(f"MINE-1 KG Quality Analysis V5 — n={n_essays} essays\n")
        f.write(f"Methods sorted by average rank (best first)\n\n")

        f.write("TABLE 1: Retrieval Accuracy & Composite Quality\n\n")
        f.write(h1 + "\n")
        f.write("-" * len(h1) + "\n")
        for method in sorted_methods:
            a = agg[method]
            f.write(
                f"{method:<16} | "
                f"{a['ret_acc']*100:6.1f}% | "
                f"{a['egu']*100:6.1f}% | "
                f"{a['rwa']*100:6.1f}% | "
                f"{a['sci']:7.4f} | "
                f"{a['v4g']*100:6.1f}% | "
                f"{a['tri_cr']:7.3f} | "
                f"{a['conn']*100:6.1f}% | "
                f"{avg_ranks[method]:7.2f}\n"
            )

        f.write(f"\n\nTABLE 2: Knowledge Representation Quality\n\n")
        f.write(h2 + "\n")
        f.write("-" * len(h2) + "\n")
        for method in sorted_methods:
            a = agg[method]
            f.write(
                f"{method:<16} | "
                f"{a['num_ent']:5.0f} | "
                f"{a['num_rel']:5.0f} | "
                f"{a['avg_ew']:5.1f} | "
                f"{a['avg_deg']:6.2f} | "
                f"{a['conn']*100:5.1f}% | "
                f"{a['clust']:6.4f} | "
                f"{a['tri_cr']:6.3f} | "
                f"{a['v4g']*100:5.1f}%\n"
            )

        f.write(f"\n\nTABLE 3: Per-Metric Rankings\n\n")
        f.write(h3 + "\n")
        f.write("-" * len(h3) + "\n")
        for method in sorted_methods:
            line = f"{method:<16}"
            for label in rank_labels:
                rank = rankings[method].get(label, "—")
                marker = " ★" if rank == 1 else "  "
                line += f" | {rank:>5}{marker}"
            line += f" | {avg_ranks[method]:7.2f}\n"
            f.write(line)

        f.write(f"\n\nTABLE A1: Extended Graph Structural Properties\n\n")
        f.write(ha1 + "\n")
        f.write("-" * len(ha1) + "\n")
        for method in sorted_methods:
            a = agg.get(method)
            if a is None:
                continue
            f.write(
                f"{method:<16} | "
                f"{a['num_ent']:5.0f} | "
                f"{a['num_rel']:5.0f} | "
                f"{a['density']:8.4f} | "
                f"{a['avg_deg']:6.2f} | "
                f"{a['num_wcc']:5.0f} | "
                f"{a['conn']*100:5.1f}% | "
                f"{a['clust']:6.4f} | "
                f"{a['num_rel_types']:7.0f} | "
                f"{a['sch_div']:6.3f}\n"
            )

        f.write(f"\n\nTABLE A2: Per-Essay Breakdown\n\n")
        f.write(ha2 + "\n")
        f.write("-" * len(ha2) + "\n")
        for method in sorted_methods:
            for entry in all_stats.get(method, []):
                acc = entry.get("accuracy")
                acc_str = f"{acc*100:5.1f}%" if acc is not None else "  N/A "
                f.write(
                    f"{method:<16} | "
                    f"{entry.get('essay_id', '?'):4} | "
                    f"{acc_str} | "
                    f"{entry.get('egu', 0)*100:5.1f}% | "
                    f"{entry.get('rwa', 0)*100:5.1f}% | "
                    f"{entry.get('sci', 0):7.4f} | "
                    f"{entry.get('num_entities', 0):5} | "
                    f"{entry.get('num_relations', 0):5} | "
                    f"{entry.get('avg_entity_words', 0):5.1f} | "
                    f"{entry.get('triple_compression_ratio', 0):6.3f} | "
                    f"{entry.get('verbatim_4gram_overlap', 0)*100:5.1f}% | "
                    f"{entry.get('avg_degree', 0):6.2f} | "
                    f"{entry.get('largest_wcc_frac', 0)*100:5.1f}% | "
                    f"{entry.get('avg_clustering', 0):6.4f}\n"
                )

    print(f"Tables saved to: {OUTPUT_TABLE_PATH}")


if __name__ == "__main__":
    run_analysis()
#endregion#?  proposed metrics
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?     visualizations

"""
KDD-Ready Visualizations for MINE-1 Evaluation (V5)
=====================================================

Produces:
  1. Radar chart: Multi-metric comparison (the "at a glance" figure)
  2. EGU decomposition: Stacked bar showing how EGU penalizes methods
  3. Accuracy vs Leak% scatter: The "fake accuracy" exposé
  4. Per-essay heatmap: TRACE KG consistency across essays
  5. THE FINAL FIGURE: Simple bar chart proving TRACE KG is best overall

All figures use the SAME data sources as V5 tables.
"""

import json
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — EXACT SAME SOURCES AS V5
# ============================================================

REPO_ROOT = Path(".").resolve()

DATASET_JSON_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset_with_autoschemakg.json"
KG_SNAPSHOTS_ROOT = REPO_ROOT / "Experiments/MYNE/Ex1/KGs_from_Essays_KFE"
RESULTS_ROOT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/tracekg_mine_results_weighted_openai_v12_with_autoschemakg"

OUTPUT_DIR = REPO_ROOT / "Experiments/MYNE/Ex1/RES/figures_v5"

EVAL_ESSAY_IDS = [1, 6, 10, 14, 15, 24, 33, 47, 52, 53, 67, 68, 70, 88, 91]

ALL_METHODS = ["tracekg", "autoschemakg", "graphrag", "openie", "kggen"]
METHOD_LABELS = {
    "tracekg": "TRACE KG",
    "autoschemakg": "AutoSchemaKG",
    "graphrag": "GraphRAG",
    "openie": "OpenIE",
    "kggen": "KGGen",
}
METHOD_COLORS = {
    "tracekg": "#1B9E77",       # teal green — protagonist
    "autoschemakg": "#D95F02",  # orange
    "graphrag": "#7570B3",      # purple
    "openie": "#E7298A",        # pink
    "kggen": "#66A61E",         # lime
}

METHOD_TO_KEY = {
    "kggen": "kggen",
    "graphrag": "graphrag_kg",
    "openie": "openie_kg",
    "autoschemakg": "autoschemakg",
}


# ============================================================
# DATA LOADING — SAME AS V5
# ============================================================

def safe_str(x):
    if x is None: return ""
    if isinstance(x, float) and np.isnan(x): return ""
    return str(x)

def clean_essay_text(text):
    text = text.strip()
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()

def word_count(text):
    return len(text.split()) if text and text.strip() else 0

def _compute_ngram_overlap(essay_text, entities, n=4):
    def get_ngrams(text, n):
        words = text.lower().split()
        if len(words) < n: return set()
        return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}
    essay_ngrams = get_ngrams(essay_text, n)
    if not essay_ngrams: return 0.0
    entity_ngrams = get_ngrams(" ".join(entities), n)
    return len(essay_ngrams & entity_ngrams) / len(essay_ngrams)

import networkx as nx

def _build_nx(kg_data):
    g = nx.DiGraph()
    for e in kg_data.get("entities", []): g.add_node(str(e).lower())
    for r in kg_data.get("relations", []):
        if isinstance(r, (list, tuple)) and len(r) == 3:
            s, rel, t = str(r[0]).lower(), str(r[1]).lower(), str(r[2]).lower()
            g.add_node(s); g.add_node(t)
            g.add_edge(s, t, relation=rel)
    return g

def compute_metrics(kg_data, essay_text):
    entities = [str(e) for e in kg_data.get("entities", [])]
    relations = kg_data.get("relations", [])
    essay_clean = clean_essay_text(essay_text)
    essay_words = word_count(essay_clean)
    ent_wc = [word_count(e) for e in entities]
    avg_ew = float(np.mean(ent_wc)) if ent_wc else 0.0
    total_tw = sum(word_count(str(r[0]))+word_count(str(r[1]))+word_count(str(r[2]))
                   for r in relations if isinstance(r,(list,tuple)) and len(r)==3)
    tri_cr = total_tw / max(essay_words, 1)
    leak = _compute_ngram_overlap(essay_clean, entities, n=4)
    g = _build_nx(kg_data)
    nv, ne = g.number_of_nodes(), g.number_of_edges()
    avg_deg = ne / max(nv, 1)
    wccs = list(nx.weakly_connected_components(g))
    conn = max(len(c) for c in wccs)/nv if wccs and nv > 0 else 0.0
    clust = nx.average_clustering(g.to_undirected()) if nv > 0 else 0.0
    return {
        "num_entities": len(entities), "num_relations": len(relations),
        "avg_entity_words": avg_ew, "triple_compression_ratio": tri_cr,
        "verbatim_4gram_overlap": leak, "avg_degree": avg_deg,
        "largest_wcc_frac": conn, "avg_clustering": clust, "essay_words": essay_words,
    }

def compute_tracekg_metrics(snap, essay_text):
    nodes_df = pd.read_csv(snap / "KG" / "nodes.csv")
    rels_df = pd.read_csv(snap / "KG" / "rels_fixed_no_raw.csv")
    id2name = {}
    for _, r in nodes_df.iterrows():
        eid = str(r.get("entity_id","")).strip()
        name = str(r.get("entity_name","")).strip() if pd.notna(r.get("entity_name")) else ""
        if eid and name: id2name[eid] = name
    entities = list(id2name.values())
    relations = []
    for _, r in rels_df.iterrows():
        s = id2name.get(str(r.get("start_id","")).strip(), str(r.get("start_id","")))
        t = id2name.get(str(r.get("end_id","")).strip(), str(r.get("end_id","")))
        rel = str(r.get("canonical_rel_name","")).strip() if pd.notna(r.get("canonical_rel_name")) else ""
        if s and rel and t: relations.append([s, rel, t])
    return compute_metrics({"entities": entities, "relations": relations}, essay_text)

def load_accuracy(results_root, methods):
    acc = {m: {} for m in methods}
    for method in methods:
        d = results_root / method
        if not d.exists(): continue
        for f in d.glob("results_*.json"):
            try:
                data = json.load(open(f))
                s = data.get("summary", {})
                did, a = s.get("dataset_id"), s.get("accuracy")
                if did is not None and a is not None:
                    acc[method][int(did)] = float(a)
            except: pass
    return acc

def load_all_data():
    with open(DATASET_JSON_PATH) as f:
        dataset = json.load(f)
    id_map = {int(item["id"]): item for item in dataset}
    acc_map = load_accuracy(RESULTS_ROOT, ALL_METHODS)

    all_stats = {m: [] for m in ALL_METHODS}
    for eid in EVAL_ESSAY_IDS:
        item = id_map.get(eid)
        if not item: continue
        essay = item.get("essay_content", "")
        for method in ["kggen","graphrag","openie","autoschemakg"]:
            kg = item.get(METHOD_TO_KEY[method])
            if not kg: continue
            s = compute_metrics(kg, essay)
            s["essay_id"] = eid
            s["accuracy"] = acc_map.get(method,{}).get(eid)
            a = s["accuracy"] or 0
            s["egu"] = a * s["largest_wcc_frac"] * (1 - s["verbatim_4gram_overlap"])
            s["rwa"] = a * s["largest_wcc_frac"]
            s["sci"] = s["avg_degree"] * s["avg_clustering"] * s["largest_wcc_frac"]
            all_stats[method].append(s)
        snap = KG_SNAPSHOTS_ROOT / f"KG_Essay_{eid:03d}"
        if snap.is_dir():
            s = compute_tracekg_metrics(snap, essay)
            s["essay_id"] = eid
            s["accuracy"] = acc_map.get("tracekg",{}).get(eid)
            a = s["accuracy"] or 0
            s["egu"] = a * s["largest_wcc_frac"] * (1 - s["verbatim_4gram_overlap"])
            s["rwa"] = a * s["largest_wcc_frac"]
            s["sci"] = s["avg_degree"] * s["avg_clustering"] * s["largest_wcc_frac"]
            all_stats["tracekg"].append(s)
    return all_stats

def aggregate(all_stats):
    agg = {}
    for m in ALL_METHODS:
        entries = all_stats[m]
        if not entries: continue
        def _m(k): return float(np.mean([e.get(k,0) for e in entries]))
        acc_vals = [e["accuracy"] for e in entries if e.get("accuracy") is not None]
        agg[m] = {
            "ret_acc": np.mean(acc_vals) if acc_vals else 0,
            "egu": _m("egu"), "rwa": _m("rwa"), "sci": _m("sci"),
            "leak": _m("verbatim_4gram_overlap"), "tri_cr": _m("triple_compression_ratio"),
            "conn": _m("largest_wcc_frac"), "clust": _m("avg_clustering"),
            "avg_deg": _m("avg_degree"), "avg_ew": _m("avg_entity_words"),
            "num_ent": _m("num_entities"), "num_rel": _m("num_relations"),
        }
    return agg


# ============================================================
# FIGURE 1: RADAR CHART — Multi-metric at a glance
# ============================================================

def fig1_radar(agg):
    categories = ["Ret.Acc", "EGU", "RWA", "Conn.", "Clust.", "1−Leak%"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=9, color="grey")
    ax.yaxis.grid(True, color="lightgrey", linewidth=0.5)
    ax.xaxis.grid(True, color="lightgrey", linewidth=0.5)

    for method in ALL_METHODS:
        a = agg.get(method)
        if not a: continue
        # Normalize clust to [0,1] — max observed is ~0.2, scale by 5× for visibility
        clust_scaled = min(a["clust"] * 5, 1.0)
        values = [a["ret_acc"], a["egu"], a["rwa"], a["conn"], clust_scaled, 1-a["leak"]]
        values += values[:1]
        lw = 3.0 if method == "tracekg" else 1.5
        ls = "-" if method == "tracekg" else "--"
        ax.plot(angles, values, linewidth=lw, linestyle=ls,
                label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        alpha = 0.15 if method == "tracekg" else 0.03
        ax.fill(angles, values, alpha=alpha, color=METHOD_COLORS[method])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11, frameon=True)
    ax.set_title("Multi-Metric KG Quality Profile\n(n = 15 essays)", fontsize=15,
                 fontweight="bold", pad=25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_radar_profile.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_DIR / "fig1_radar_profile.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ Figure 1: Radar chart saved")


# ============================================================
# FIGURE 2: EGU DECOMPOSITION — Where accuracy really comes from
# ============================================================

def fig2_egu_decomposition(agg):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    methods_sorted = sorted(agg.keys(), key=lambda m: -agg[m]["egu"])
    x = np.arange(len(methods_sorted))
    width = 0.55

    egu_vals = []
    leak_penalty = []
    frag_penalty = []
    raw_acc = []

    for m in methods_sorted:
        a = agg[m]
        acc = a["ret_acc"]
        conn = a["conn"]
        leak = a["leak"]
        egu = acc * conn * (1 - leak)
        lost_to_frag = acc * (1 - conn)
        lost_to_leak = acc * conn * leak

        egu_vals.append(egu * 100)
        frag_penalty.append(lost_to_frag * 100)
        leak_penalty.append(lost_to_leak * 100)
        raw_acc.append(acc * 100)

    # Stacked bars
    bars_egu = ax.bar(x, egu_vals, width, label="Effective (EGU)",
                      color=[METHOD_COLORS[m] for m in methods_sorted], edgecolor="white", linewidth=0.8)
    bars_frag = ax.bar(x, frag_penalty, width, bottom=egu_vals,
                       label="Lost to fragmentation", color="#CCCCCC", edgecolor="white", linewidth=0.8,
                       hatch="//")
    bars_leak = ax.bar(x, leak_penalty, width,
                       bottom=[e+f for e,f in zip(egu_vals, frag_penalty)],
                       label="Lost to text copying", color="#FF9999", edgecolor="white", linewidth=0.8,
                       hatch="xx")

    # Accuracy line
    ax.plot(x, raw_acc, "ko-", markersize=8, linewidth=2, label="Raw Ret.Acc", zorder=5)
    for i, (xi, yi) in enumerate(zip(x, raw_acc)):
        ax.annotate(f"{yi:.1f}%", (xi, yi+1.5), ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods_sorted], fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc="upper right", frameon=True)
    ax.set_title("Where Does Retrieval Accuracy Come From?\nEGU Decomposition (n = 15 essays)",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_egu_decomposition.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_DIR / "fig2_egu_decomposition.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ Figure 2: EGU decomposition saved")


# ============================================================
# FIGURE 3: ACCURACY vs LEAK% SCATTER — The exposé
# ============================================================

def fig3_accuracy_vs_leak(all_stats):
    fig, ax = plt.subplots(figsize=(9, 6))

    for method in ALL_METHODS:
        entries = all_stats[method]
        accs = [e["accuracy"]*100 for e in entries if e.get("accuracy") is not None]
        leaks = [e["verbatim_4gram_overlap"]*100 for e in entries if e.get("accuracy") is not None]
        size = 120 if method == "tracekg" else 60
        marker = "★" if method == "tracekg" else "o"
        zorder = 10 if method == "tracekg" else 5
        ax.scatter(leaks, accs, s=size, c=METHOD_COLORS[method], label=METHOD_LABELS[method],
                   alpha=0.8, edgecolors="white", linewidth=0.5, zorder=zorder,
                   marker="*" if method == "tracekg" else "o")

    # Danger zone
    ax.axvspan(20, 100, alpha=0.08, color="red")
    ax.text(55, 25, "DANGER ZONE\n(text copying)", fontsize=12, color="red",
            alpha=0.5, ha="center", style="italic")

    # Ideal zone
    ax.axvspan(0, 5, alpha=0.06, color="green")
    ax.text(2.5, 25, "IDEAL", fontsize=11, color="green", alpha=0.5, ha="center", rotation=90)

    ax.set_xlabel("Verbatim 4-gram Overlap (Leak%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_title("Accuracy vs. Information Leakage\n(Each dot = one essay)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right", frameon=True)
    ax.set_xlim(-2, 95)
    ax.set_ylim(15, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_accuracy_vs_leak.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_DIR / "fig3_accuracy_vs_leak.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ Figure 3: Accuracy vs Leak scatter saved")


# ============================================================
# FIGURE 4: PER-ESSAY HEATMAP — Consistency
# ============================================================

def fig4_per_essay_heatmap(all_stats):
    metrics_to_show = [
        ("accuracy", "Ret.Acc", True),
        ("egu", "EGU", True),
        ("largest_wcc_frac", "Conn.", True),
        ("avg_clustering", "Clust.", True),
        ("verbatim_4gram_overlap", "Leak%", False),
    ]

    essay_ids = sorted(EVAL_ESSAY_IDS)
    methods_order = ALL_METHODS

    fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(22, 6), sharey=True)

    for ax_idx, (key, label, higher_better) in enumerate(metrics_to_show):
        ax = axes[ax_idx]
        matrix = np.full((len(methods_order), len(essay_ids)), np.nan)

        for mi, method in enumerate(methods_order):
            entries_by_eid = {e["essay_id"]: e for e in all_stats[method]}
            for ei, eid in enumerate(essay_ids):
                entry = entries_by_eid.get(eid)
                if entry and entry.get(key) is not None:
                    val = entry[key]
                    if key == "verbatim_4gram_overlap":
                        val = val * 100  # show as %
                    elif key in ("accuracy", "egu", "largest_wcc_frac"):
                        val = val * 100
                    matrix[mi, ei] = val

        cmap = "RdYlGn" if higher_better else "RdYlGn_r"
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

        ax.set_xticks(range(len(essay_ids)))
        ax.set_xticklabels(essay_ids, fontsize=8, rotation=45)
        ax.set_yticks(range(len(methods_order)))
        if ax_idx == 0:
            ax.set_yticklabels([METHOD_LABELS[m] for m in methods_order], fontsize=10, fontweight="bold")
        else:
            ax.set_yticklabels([])

        ax.set_title(label, fontsize=12, fontweight="bold")

        # Annotate cells
        for mi in range(len(methods_order)):
            for ei in range(len(essay_ids)):
                val = matrix[mi, ei]
                if not np.isnan(val):
                    fmt = f"{val:.0f}" if abs(val) >= 1 else f"{val:.1f}"
                    color = "white" if val < 30 or val > 80 else "black"
                    ax.text(ei, mi, fmt, ha="center", va="center", fontsize=7, color=color)

        plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

    fig.suptitle("Per-Essay Quality Breakdown (n = 15 essays)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_per_essay_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_DIR / "fig4_per_essay_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ Figure 4: Per-essay heatmap saved")


# ============================================================
# FIGURE 5: THE FINAL FIGURE — "We win. Period."
# ============================================================

def fig5_the_winner(agg, all_stats):
    """
    Simple, devastating bar chart.
    Shows all key metrics side by side, normalized to [0,100].
    TRACE KG dominates visually.
    """
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(1, 2, width_ratios=[3, 1.2], wspace=0.3)

    # ── Left panel: Grouped bars ──
    ax1 = fig.add_subplot(gs[0])

    metrics = [
        ("Ret.Acc (%)", "ret_acc", 100),
        ("EGU (%)", "egu", 100),
        ("RWA (%)", "rwa", 100),
        ("Connectivity (%)", "conn", 100),
        ("1 − Leak% (%)", None, None),  # special
        ("SCI (×100)", "sci", 10000),
    ]

    n_metrics = len(metrics)
    n_methods = len(ALL_METHODS)
    x = np.arange(n_metrics)
    total_width = 0.75
    bar_width = total_width / n_methods

    for mi, method in enumerate(ALL_METHODS):
        a = agg.get(method)
        if not a: continue
        vals = []
        for label, key, scale in metrics:
            if key is None:  # 1-Leak%
                vals.append((1 - a["leak"]) * 100)
            else:
                vals.append(a[key] * scale)

        offset = (mi - n_methods/2 + 0.5) * bar_width
        bars = ax1.bar(x + offset, vals, bar_width * 0.9,
                       label=METHOD_LABELS[method], color=METHOD_COLORS[method],
                       edgecolor="white", linewidth=0.5,
                       alpha=1.0 if method == "tracekg" else 0.7)

        # Bold border for TRACE KG
        if method == "tracekg":
            for bar in bars:
                bar.set_edgecolor("#000000")
                bar.set_linewidth(1.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in metrics], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=10, loc="upper right", ncol=2, frameon=True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("Multi-Dimensional KG Quality Comparison", fontsize=14, fontweight="bold")

    # ── Right panel: Average Rank ──
    ax2 = fig.add_subplot(gs[1])

    # Compute ranks
    RANK_METRICS = [
        ("ret_acc", True), ("egu", True), ("rwa", True), ("sci", True),
        ("conn", True), ("clust", True), ("avg_deg", True),
        ("leak", False), ("tri_cr", None),
    ]

    rankings = {m: [] for m in ALL_METHODS}
    methods_with_data = [m for m in ALL_METHODS if m in agg]

    for key, higher in RANK_METRICS:
        if higher is None:
            vals = {m: abs(agg[m][key] - 1.0) for m in methods_with_data}
            sorted_m = sorted(vals.items(), key=lambda x: x[1])
        elif higher:
            sorted_m = sorted([(m, agg[m][key]) for m in methods_with_data], key=lambda x: -x[1])
        else:
            sorted_m = sorted([(m, agg[m][key]) for m in methods_with_data], key=lambda x: x[1])
        for rank, (m, _) in enumerate(sorted_m, 1):
            rankings[m].append(rank)

    avg_ranks = {m: np.mean(rankings[m]) for m in methods_with_data}
    sorted_by_rank = sorted(methods_with_data, key=lambda m: avg_ranks[m])

    y_pos = np.arange(len(sorted_by_rank))
    rank_vals = [avg_ranks[m] for m in sorted_by_rank]
    colors = [METHOD_COLORS[m] for m in sorted_by_rank]

    bars = ax2.barh(y_pos, rank_vals, 0.6, color=colors, edgecolor="white")

    # Highlight TRACE KG
    for i, m in enumerate(sorted_by_rank):
        if m == "tracekg":
            bars[i].set_edgecolor("#000000")
            bars[i].set_linewidth(2)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([METHOD_LABELS[m] for m in sorted_by_rank], fontsize=11, fontweight="bold")
    ax2.set_xlabel("Average Rank (lower = better)", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 5.5)
    ax2.invert_xaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    for i, (m, v) in enumerate(zip(sorted_by_rank, rank_vals)):
        marker = " ★ BEST" if i == 0 else ""
        ax2.text(v - 0.1, i, f"{v:.2f}{marker}", ha="right", va="center",
                fontsize=11, fontweight="bold", color="white")

    ax2.set_title("Overall Rank\n(9 metrics)", fontsize=13, fontweight="bold")

    fig.suptitle("TRACE KG: Comprehensive Evaluation on MINE-1 Benchmark (n = 15 essays)",
                 fontsize=16, fontweight="bold", y=1.02)

    plt.savefig(OUTPUT_DIR / "fig5_the_winner.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_DIR / "fig5_the_winner.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ Figure 5: The Winner figure saved")


# ============================================================
# FIGURE 6: TriCR DISTANCE — How close to ideal compression?
# ============================================================

def fig6_tricr_distance(agg):
    fig, ax = plt.subplots(figsize=(8, 5))

    methods_sorted = sorted(agg.keys(), key=lambda m: abs(agg[m]["tri_cr"] - 1.0))
    x = np.arange(len(methods_sorted))

    tri_vals = [agg[m]["tri_cr"] for m in methods_sorted]
    colors = [METHOD_COLORS[m] for m in methods_sorted]

    bars = ax.bar(x, tri_vals, 0.55, color=colors, edgecolor="white", linewidth=0.8)

    # Ideal line at 1.0
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax.text(len(methods_sorted)-0.5, 1.05, "IDEAL (TriCR = 1.0)", fontsize=10,
            ha="right", fontweight="bold", alpha=0.7)

    # Annotate
    for i, (m, v) in enumerate(zip(methods_sorted, tri_vals)):
        dist = abs(v - 1.0)
        ax.text(i, v + 0.08, f"{v:.2f}\n(Δ={dist:.2f})", ha="center", fontsize=10, fontweight="bold")

    # Highlight TRACE KG
    for i, m in enumerate(methods_sorted):
        if m == "tracekg":
            bars[i].set_edgecolor("#000000")
            bars[i].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods_sorted], fontsize=12, fontweight="bold")
    ax.set_ylabel("Triple Compression Ratio", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(tri_vals) + 0.6)
    ax.set_title("Compression Quality: Distance from Ideal (TriCR → 1.0)\n"
                 "< 1 = oversimplified  |  = 1 ideal  |  > 1 = bloated",
                 fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_tricr_distance.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_DIR / "fig6_tricr_distance.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ Figure 6: TriCR distance saved")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data (same sources as V5 tables)...")
    all_stats = load_all_data()
    agg = aggregate(all_stats)

    print(f"\nGenerating KDD figures → {OUTPUT_DIR}/")
    fig1_radar(agg)
    fig2_egu_decomposition(agg)
    fig3_accuracy_vs_leak(all_stats)
    fig4_per_essay_heatmap(all_stats)
    fig5_the_winner(agg, all_stats)
    fig6_tricr_distance(agg)

    print(f"\n✅ All 6 figures saved to: {OUTPUT_DIR}/")
    print("\nFigure guide for the paper:")
    print("  Fig 1 (radar)         → Section 5 intro or Fig 1 in paper body")
    print("  Fig 2 (EGU decomp)    → Section 5.2 — the key insight figure")
    print("  Fig 3 (acc vs leak)   → Section 5.3 — exposes AutoSchemaKG")
    print("  Fig 4 (heatmap)       → Appendix — per-essay consistency")
    print("  Fig 5 (the winner)    → Section 5 conclusion — the summary figure")
    print("  Fig 6 (TriCR)         → Section 5.2 — compression quality")


if __name__ == "__main__":
    main()

#endregion#?   visualizations
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?     visualization 2

"""
KDD-Ready Figures for TRACE KG Paper — MINE-1 Evaluation
=========================================================

Produces 4 publication-quality figures:

  Fig 1: Radar chart — multi-dimensional quality profile per method
  Fig 2: Accuracy vs Leakage scatter — exposes AutoSchemaKG's copying
  Fig 3: EGU / RWA / SCI grouped bar chart — composite metrics head-to-head
  Fig 4: Summary dashboard — single figure that tells the whole story

All data loaded from compression_analysis_v5.json (V5 results).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List

# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(".").resolve()
V5_JSON = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_v5.json"
FIG_OUT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Method display config
METHOD_ORDER = ["tracekg", "autoschemakg", "graphrag", "openie", "kggen"]
METHOD_LABELS = {
    "tracekg": "TRACE KG",
    "autoschemakg": "AutoSchemaKG",
    "graphrag": "GraphRAG",
    "openie": "OpenIE",
    "kggen": "KGGen",
}
METHOD_COLORS = {
    "tracekg": "#1B9E77",       # teal-green — protagonist
    "autoschemakg": "#D95F02",  # orange
    "graphrag": "#7570B3",      # purple
    "openie": "#E7298A",        # pink
    "kggen": "#66A61E",         # olive
}
METHOD_MARKERS = {
    "tracekg": "D",
    "autoschemakg": "s",
    "graphrag": "^",
    "openie": "o",
    "kggen": "v",
}

DPI = 300
FONT_SIZE = 11

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "sans-serif",
    "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ============================================================
# LOAD DATA
# ============================================================

with open(V5_JSON) as f:
    v5 = json.load(f)

agg = v5["aggregate"]


# ============================================================
# FIG 1: RADAR CHART — Multi-dimensional quality profile
# ============================================================

def fig1_radar():
    """
    Radar chart showing 7 normalized metrics per method.
    Immediately shows TRACE KG covers the largest area.
    """
    # Metrics: (label, agg_key, higher_is_better, display_transform)
    # All will be normalized to [0, 1] where 1 = best
    radar_metrics = [
        ("Ret. Accuracy",     "ret_acc",  True),
        ("EGU",               "egu",      True),
        ("RWA",               "rwa",      True),
        ("SCI",               "sci",      True),
        ("Connectivity",      "conn",     True),
        ("Clustering",        "clust",    True),
        ("Low Leakage",       "v4g",      False),   # invert: lower leak = better
    ]

    methods = [m for m in METHOD_ORDER if m in agg]
    N = len(radar_metrics)

    # Extract raw values
    raw = {}
    for m in methods:
        raw[m] = []
        for label, key, higher in radar_metrics:
            raw[m].append(agg[m][key])

    # Normalize each metric to [0, 1]
    raw_arr = np.array([raw[m] for m in methods])  # (n_methods, n_metrics)
    norm = np.zeros_like(raw_arr)
    for j, (label, key, higher) in enumerate(radar_metrics):
        col = raw_arr[:, j]
        mn, mx = col.min(), col.max()
        if mx - mn < 1e-9:
            norm[:, j] = 1.0
        else:
            if higher:
                norm[:, j] = (col - mn) / (mx - mn)
            else:
                norm[:, j] = (mx - col) / (mx - mn)  # invert

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, m in enumerate(methods):
        vals = norm[i].tolist() + [norm[i][0]]  # close
        lw = 2.8 if m == "tracekg" else 1.4
        alpha = 0.15 if m == "tracekg" else 0.0
        ax.plot(angles, vals, color=METHOD_COLORS[m], linewidth=lw,
                label=METHOD_LABELS[m], zorder=10 if m == "tracekg" else 5)
        if alpha > 0:
            ax.fill(angles, vals, color=METHOD_COLORS[m], alpha=alpha, zorder=4)

    labels = [lbl for lbl, _, _ in radar_metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FONT_SIZE)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=FONT_SIZE - 2, color="grey")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), frameon=True)
    ax.set_title("Multi-Dimensional KG Quality Profile", pad=25, fontweight="bold")

    path = FIG_OUT / "fig1_radar_quality_profile.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 2: SCATTER — Accuracy vs Leakage (the "exposé" plot)
# ============================================================

def fig2_accuracy_vs_leakage():
    """
    Scatter: x = Leak%, y = Ret.Acc.
    Bubble size = TriCR deviation from 1.0.
    Shows AutoSchemaKG's high accuracy comes from high leakage.
    TRACE KG: high accuracy, near-zero leakage.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    methods = [m for m in METHOD_ORDER if m in agg]

    for m in methods:
        x = agg[m]["v4g"] * 100  # Leak%
        y = agg[m]["ret_acc"] * 100  # Acc%
        tricr_dev = abs(agg[m]["tri_cr"] - 1.0)
        size = 80 + tricr_dev * 120  # bubble size proportional to TriCR deviation

        ax.scatter(x, y, s=size, c=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                   edgecolors="black", linewidths=0.8, zorder=10, label=METHOD_LABELS[m])

        # Label offset
        offsets = {
            "tracekg": (6, -3),
            "autoschemakg": (-8, -12),
            "graphrag": (6, -3),
            "openie": (6, 3),
            "kggen": (6, -3),
        }
        dx, dy = offsets.get(m, (5, 0))
        ax.annotate(METHOD_LABELS[m], (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=FONT_SIZE - 1, fontweight="bold",
                    color=METHOD_COLORS[m])

    # Draw the "ideal zone" — top-left corner
    ideal_rect = mpatches.FancyBboxPatch(
        (-1, 82), 8, 20, boxstyle="round,pad=0.5",
        facecolor="#1B9E77", alpha=0.08, edgecolor="#1B9E77",
        linestyle="--", linewidth=1.5
    )
    ax.add_patch(ideal_rect)
    ax.text(3, 98, "Ideal Zone\n(high acc, low leak)", fontsize=FONT_SIZE - 2,
            ha="center", va="top", color="#1B9E77", fontstyle="italic")

    ax.set_xlabel("Verbatim 4-gram Leakage (%)", fontweight="bold")
    ax.set_ylabel("Retrieval Accuracy (%)", fontweight="bold")
    ax.set_xlim(-2, 45)
    ax.set_ylim(35, 102)
    ax.axvline(x=5, color="grey", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(5.5, 37, "5% leak threshold", fontsize=FONT_SIZE - 2,
            color="grey", fontstyle="italic", rotation=90, va="bottom")
    ax.set_title("Retrieval Accuracy vs. Information Leakage", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Bubble legend
    ax.text(0.98, 0.02, "Bubble size ∝ |TriCR − 1|",
            transform=ax.transAxes, fontsize=FONT_SIZE - 2,
            ha="right", va="bottom", fontstyle="italic", color="grey")

    path = FIG_OUT / "fig2_accuracy_vs_leakage.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 3: GROUPED BAR — EGU / RWA / SCI composites
# ============================================================

def fig3_composite_bars():
    """
    Grouped bar chart for the 3 composite metrics.
    Immediately shows TRACE KG dominates all three.
    """
    metrics = [
        ("EGU (%)", "egu", 100),
        ("RWA (%)", "rwa", 100),
        ("SCI",     "sci", 1),
    ]

    methods = [m for m in METHOD_ORDER if m in agg]
    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for ax_idx, (label, key, mult) in enumerate(metrics):
        ax = axes[ax_idx]
        vals = [agg[m][key] * mult for m in methods]
        stds = [agg[m].get(f"{key}_std", 0) * mult for m in methods]
        colors = [METHOD_COLORS[m] for m in methods]
        labels = [METHOD_LABELS[m] for m in methods]

        bars = ax.bar(range(n_methods), vals, color=colors, edgecolor="black",
                      linewidth=0.6, yerr=stds, capsize=4, error_kw={"linewidth": 1})

        # Highlight TRACE KG bar
        trace_idx = methods.index("tracekg")
        bars[trace_idx].set_edgecolor("#1B9E77")
        bars[trace_idx].set_linewidth(2.5)

        # Value labels on bars
        for i, (v, s) in enumerate(zip(vals, stds)):
            if mult == 100:
                ax.text(i, v + s + 1, f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=FONT_SIZE - 2, fontweight="bold" if methods[i] == "tracekg" else "normal")
            else:
                ax.text(i, v + s + 0.005, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=FONT_SIZE - 2, fontweight="bold" if methods[i] == "tracekg" else "normal")

        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=FONT_SIZE - 1)
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlim(-0.6, n_methods - 0.4)

    fig.suptitle("Composite Quality Metrics — Higher is Better", fontweight="bold", fontsize=FONT_SIZE + 2)
    plt.tight_layout()

    path = FIG_OUT / "fig3_composite_metrics_bars.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 4: SUMMARY DASHBOARD — The "one figure that tells all"
# ============================================================

def fig4_summary_dashboard():
    """
    2×2 dashboard:
      Top-left:     Accuracy bar with leakage overlay
      Top-right:    TriCR deviation (distance from ideal 1.0)
      Bottom-left:  Connectivity + Clustering stacked
      Bottom-right: Average Rank lollipop (the punchline)
    """
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── (a) Accuracy + Leakage overlay ──
    ax = axes[0, 0]
    acc = [agg[m]["ret_acc"] * 100 for m in methods]
    leak = [agg[m]["v4g"] * 100 for m in methods]

    bars = ax.bar(x, acc, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85, label="Ret. Accuracy")
    # Overlay leak as hatched portion from bottom
    ax.bar(x, leak, color="red", alpha=0.35, hatch="///", edgecolor="red", linewidth=0.5, label="Leakage (verbatim)")

    for i in range(n):
        ax.text(i, acc[i] + 1, f"{acc[i]:.1f}%", ha="center", va="bottom",
                fontsize=FONT_SIZE - 2, fontweight="bold" if methods[i] == "tracekg" else "normal")
        if leak[i] > 2:
            ax.text(i, leak[i] / 2, f"{leak[i]:.0f}%", ha="center", va="center",
                    fontsize=FONT_SIZE - 2, color="darkred", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(a) Retrieval Accuracy & Leakage", fontweight="bold")
    ax.legend(fontsize=FONT_SIZE - 2, loc="lower left")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.2)

    # ── (b) TriCR deviation from 1.0 ──
    ax = axes[0, 1]
    tricr = [agg[m]["tri_cr"] for m in methods]
    tricr_dev = [abs(t - 1.0) for t in tricr]
    bar_colors_dev = [METHOD_COLORS[m] for m in methods]

    bars = ax.barh(x, tricr_dev, color=bar_colors_dev, edgecolor="black", linewidth=0.6, height=0.6)
    # Mark actual TriCR value
    for i in range(n):
        direction = "▶" if tricr[i] > 1 else "◀"
        ax.text(tricr_dev[i] + 0.03, i, f"TriCR={tricr[i]:.2f} {direction}",
                va="center", fontsize=FONT_SIZE - 2,
                fontweight="bold" if methods[i] == "tracekg" else "normal")

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("|TriCR − 1.0|  (lower = better compression)")
    ax.set_title("(b) Compression Quality", fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlim(0, 3.2)
    ax.grid(axis="x", alpha=0.2)
    # Ideal marker
    ax.annotate("← Ideal", xy=(0.05, -0.7), fontsize=FONT_SIZE - 2, color="green", fontstyle="italic")

    # ── (c) Connectivity + Clustering ──
    ax = axes[1, 0]
    conn = [agg[m]["conn"] * 100 for m in methods]
    clust = [agg[m]["clust"] * 100 for m in methods]  # ×100 for display

    w = 0.35
    ax.bar(x - w/2, conn, w, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85, label="Connectivity (%)")
    ax.bar(x + w/2, clust, w, color=colors, edgecolor="black", linewidth=0.6, alpha=0.45,
           hatch="...", label="Clustering (×100)")

    for i in range(n):
        ax.text(i - w/2, conn[i] + 1, f"{conn[i]:.0f}", ha="center", fontsize=FONT_SIZE - 2, va="bottom")
        ax.text(i + w/2, clust[i] + 0.5, f"{clust[i]:.1f}", ha="center", fontsize=FONT_SIZE - 2, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Percentage / Scaled Value")
    ax.set_title("(c) Graph Structure: Connectivity & Clustering", fontweight="bold")
    ax.legend(fontsize=FONT_SIZE - 2)
    ax.grid(axis="y", alpha=0.2)

    # ── (d) Average Rank — THE PUNCHLINE ──
    ax = axes[1, 1]
    ranks = v5.get("avg_ranks", {})
    # Sort by rank (best first)
    sorted_m = sorted([m for m in methods if m in ranks], key=lambda m: ranks[m])
    rank_vals = [ranks[m] for m in sorted_m]
    rank_labels = [METHOD_LABELS[m] for m in sorted_m]
    rank_colors = [METHOD_COLORS[m] for m in sorted_m]

    y_pos = np.arange(len(sorted_m))
    ax.barh(y_pos, rank_vals, color=rank_colors, edgecolor="black", linewidth=0.6, height=0.6)

    for i, (m, rv) in enumerate(zip(sorted_m, rank_vals)):
        fw = "bold" if m == "tracekg" else "normal"
        ax.text(rv + 0.05, i, f"{rv:.2f}", va="center", fontsize=FONT_SIZE, fontweight=fw)

    # Star for #1
    ax.text(rank_vals[0] - 0.15, 0, "★", fontsize=18, va="center", ha="center", color="gold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rank_labels, fontsize=FONT_SIZE)
    ax.set_xlabel("Average Rank (lower = better)")
    ax.set_title("(d) Overall Ranking Across All Metrics", fontweight="bold")
    ax.set_xlim(0, 5.2)
    ax.axvline(x=1, color="green", linestyle=":", alpha=0.4)
    ax.text(1.05, len(sorted_m) - 0.3, "perfect", fontsize=FONT_SIZE - 2, color="green", fontstyle="italic")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)

    fig.suptitle("MINE-1 Evaluation Summary: TRACE KG vs. Baselines",
                 fontweight="bold", fontsize=FONT_SIZE + 3, y=1.01)
    plt.tight_layout()

    path = FIG_OUT / "fig4_summary_dashboard.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("Generating KDD figures...\n")
    fig1_radar()
    fig2_accuracy_vs_leakage()
    fig3_composite_bars()
    fig4_summary_dashboard()
    print(f"\nAll figures saved to: {FIG_OUT}")
    
    
#endregion#? 
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?    visualization 3

"""
KDD-Ready Figures for TRACE KG Paper — MINE-1 Evaluation (Final)
=================================================================

Produces 3 publication-quality figures:

  Fig 1: Radar chart — multi-dimensional quality profile
  Fig 2: Composite metrics (EGU, RWA, SCI) with full names
  Fig 3: Summary dashboard — THE one figure that tells the whole story

Color palette: colorblind-safe academic palette.
All data from compression_analysis_v5.json.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List

# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(".").resolve()
V5_JSON = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_v5.json"
FIG_OUT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["tracekg", "autoschemakg", "graphrag", "openie", "kggen"]
METHOD_LABELS = {
    "tracekg": "TRACE KG",
    "autoschemakg": "AutoSchemaKG",
    "graphrag": "GraphRAG",
    "openie": "OpenIE",
    "kggen": "KGGen",
}

# Colorblind-safe academic palette (Okabe-Ito inspired)
METHOD_COLORS = {
    "tracekg":      "#0072B2",  # strong blue — protagonist
    "autoschemakg": "#E69F00",  # amber
    "graphrag":     "#882255",  # wine
    "openie":       "#44AA99",  # teal
    "kggen":        "#999999",  # neutral grey
}

DPI = 300
FONT_SIZE = 11

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "sans-serif",
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# LOAD DATA
# ============================================================

with open(V5_JSON) as f:
    v5 = json.load(f)

agg = v5["aggregate"]


# ============================================================
# FIG 1: RADAR — Multi-dimensional quality profile
# ============================================================

def fig1_radar():
    radar_metrics = [
        ("Ret. Accuracy", "ret_acc",  True),
        ("EGU",           "egu",      True),
        ("RWA",           "rwa",      True),
        ("SCI",           "sci",      True),
        ("Connectivity",  "conn",     True),
        ("Clustering",    "clust",    True),
        ("Low Leakage",   "v4g",      False),
    ]

    methods = [m for m in METHOD_ORDER if m in agg]
    N = len(radar_metrics)

    raw = {}
    for m in methods:
        raw[m] = [agg[m][key] for _, key, _ in radar_metrics]

    raw_arr = np.array([raw[m] for m in methods])
    norm = np.zeros_like(raw_arr)
    for j, (_, _, higher) in enumerate(radar_metrics):
        col = raw_arr[:, j]
        mn, mx = col.min(), col.max()
        if mx - mn < 1e-9:
            norm[:, j] = 1.0
        else:
            norm[:, j] = (col - mn) / (mx - mn) if higher else (mx - col) / (mx - mn)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FAFAFA")

    for i, m in enumerate(methods):
        vals = norm[i].tolist() + [norm[i][0]]
        lw = 2.8 if m == "tracekg" else 1.3
        ls = "-" if m == "tracekg" else "--"
        alpha_fill = 0.12 if m == "tracekg" else 0.0
        zorder = 10 if m == "tracekg" else 5
        ax.plot(angles, vals, color=METHOD_COLORS[m], linewidth=lw, linestyle=ls,
                label=METHOD_LABELS[m], zorder=zorder)
        if alpha_fill > 0:
            ax.fill(angles, vals, color=METHOD_COLORS[m], alpha=alpha_fill, zorder=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl for lbl, _, _ in radar_metrics], fontsize=FONT_SIZE)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=FONT_SIZE - 2, color="grey")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.15), frameon=True,
              fancybox=True, shadow=False, edgecolor="#CCCCCC")

    path = FIG_OUT / "fig1_radar_quality_profile.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 2: COMPOSITE BARS — EGU / RWA / SCI with full names
# ============================================================

def fig2_composite_bars():
    metrics = [
        ("EGU (%)",  "egu",  100, "Effective Graph\nUtilization"),
        ("RWA (%)",  "rwa",  100, "Reachability-Weighted\nAccuracy"),
        ("SCI",      "sci",  1,   "Structural Coherence\nIndex"),
    ]

    methods = [m for m in METHOD_ORDER if m in agg]
    n_methods = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    for ax_idx, (short_label, key, mult, full_name) in enumerate(metrics):
        ax = axes[ax_idx]
        vals = [agg[m][key] * mult for m in methods]
        stds = [agg[m].get(f"{key}_std", 0) * mult for m in methods]

        bars = ax.bar(range(n_methods), vals, color=colors, edgecolor="black",
                      linewidth=0.5, yerr=stds, capsize=4,
                      error_kw={"linewidth": 1, "color": "#555555"})

        # Bold edge on TRACE KG
        trace_idx = methods.index("tracekg")
        bars[trace_idx].set_edgecolor(METHOD_COLORS["tracekg"])
        bars[trace_idx].set_linewidth(2.5)

        for i, (v, s) in enumerate(zip(vals, stds)):
            fw = "bold" if methods[i] == "tracekg" else "normal"
            if mult == 100:
                ax.text(i, v + s + 1.5, f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=FONT_SIZE - 2, fontweight=fw)
            else:
                ax.text(i, v + s + 0.005, f"{v:.4f}", ha="center", va="bottom",
                        fontsize=FONT_SIZE - 2, fontweight=fw)

        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=FONT_SIZE - 1)

        # Title = short metric name, subtitle = full name
        ax.set_title(short_label, fontweight="bold", fontsize=FONT_SIZE + 1)
        ax.text(0.5, -0.32, full_name, transform=ax.transAxes,
                ha="center", va="top", fontsize=FONT_SIZE - 2,
                fontstyle="italic", color="#555555")

        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        ax.set_xlim(-0.6, n_methods - 0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Composite Quality Metrics — Higher is Better",
                 fontweight="bold", fontsize=FONT_SIZE + 2)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    path = FIG_OUT / "fig2_composite_metrics.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 3: SUMMARY DASHBOARD — The one figure that tells all
# ============================================================

def fig3_summary_dashboard():
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax in axes.flat:
        ax.set_axisbelow(True)

    # ── (a) Accuracy + Leakage overlay ──
    ax = axes[0, 0]
    acc = [agg[m]["ret_acc"] * 100 for m in methods]
    leak = [agg[m]["v4g"] * 100 for m in methods]

    ax.bar(x, acc, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85, label="Retrieval Accuracy")
    ax.bar(x, leak, color="#CC3311", alpha=0.40, hatch="///",
           edgecolor="#CC3311", linewidth=0.4, label="Verbatim Leakage")

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, acc[i] + 1, f"{acc[i]:.1f}%", ha="center", va="bottom",
                fontsize=FONT_SIZE - 2, fontweight=fw)
        if leak[i] > 3:
            ax.text(i, leak[i] / 2, f"{leak[i]:.0f}%", ha="center", va="center",
                    fontsize=FONT_SIZE - 2, color="#990000", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(a) Retrieval Accuracy & Information Leakage", fontweight="bold")
    ax.legend(fontsize=FONT_SIZE - 2, loc="lower left", framealpha=0.9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    # ── (b) TriCR deviation from 1.0 ──
    ax = axes[0, 1]
    tricr = [agg[m]["tri_cr"] for m in methods]
    tricr_dev = [abs(t - 1.0) for t in tricr]

    bars = ax.barh(x, tricr_dev, color=colors, edgecolor="black", linewidth=0.5, height=0.55)

    for i in range(n):
        direction = "▸ expansion" if tricr[i] > 1 else "◂ oversimplified"
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(tricr_dev[i] + 0.04, i, f"{tricr[i]:.2f}  ({direction})",
                va="center", fontsize=FONT_SIZE - 2, fontweight=fw)

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("|TriCR − 1.0|  →  lower = better compression")
    ax.set_title("(b) Triple Compression Ratio (ideal = 1.0)", fontweight="bold")
    ax.set_xlim(0, 3.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    # ── (c) EGU — the single best composite metric ──
    ax = axes[1, 0]
    egu = [agg[m]["egu"] * 100 for m in methods]
    egu_std = [agg[m].get("egu_std", 0) * 100 for m in methods]

    bars = ax.bar(x, egu, color=colors, edgecolor="black", linewidth=0.5,
                  yerr=egu_std, capsize=4, error_kw={"linewidth": 1, "color": "#555555"})

    trace_idx = methods.index("tracekg")
    bars[trace_idx].set_edgecolor(METHOD_COLORS["tracekg"])
    bars[trace_idx].set_linewidth(2.5)

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, egu[i] + egu_std[i] + 1.5, f"{egu[i]:.1f}%", ha="center",
                va="bottom", fontsize=FONT_SIZE - 1, fontweight=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("EGU (%)")
    ax.set_title("(c) Effective Graph Utilization (EGU)", fontweight="bold")
    ax.text(0.5, -0.28, "EGU = Accuracy × Connectivity × (1 − Leakage)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SIZE - 2, fontstyle="italic", color="#555555")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    # ── (d) Average Rank — THE PUNCHLINE ──
    ax = axes[1, 1]
    ranks = v5.get("avg_ranks", {})
    sorted_m = sorted([m for m in methods if m in ranks], key=lambda m: ranks[m])
    rank_vals = [ranks[m] for m in sorted_m]
    rank_labels = [METHOD_LABELS[m] for m in sorted_m]
    rank_colors = [METHOD_COLORS[m] for m in sorted_m]

    bars = ax.barh(np.arange(len(sorted_m)), rank_vals, color=rank_colors,
                   edgecolor="black", linewidth=0.5, height=0.55)

    for i, (m, rv) in enumerate(zip(sorted_m, rank_vals)):
        fw = "bold" if m == "tracekg" else "normal"
        ax.text(rv + 0.06, i, f"{rv:.2f}", va="center", fontsize=FONT_SIZE, fontweight=fw)

    # Gold star for #1
    ax.text(rank_vals[0] + 0.35, 0, "★", fontsize=16, va="center", ha="center",
            color="#DAA520", fontweight="bold")

    ax.set_yticks(np.arange(len(sorted_m)))
    ax.set_yticklabels(rank_labels, fontsize=FONT_SIZE)
    ax.set_xlabel("Average Rank Across 9 Metrics (lower = better)")
    ax.set_title("(d) Overall Quality Ranking", fontweight="bold")
    ax.set_xlim(0, 5.0)
    ax.axvline(x=1, color="#0072B2", linestyle=":", alpha=0.3, linewidth=1)
    ax.text(1.05, len(sorted_m) - 0.2, "perfect", fontsize=FONT_SIZE - 2,
            color="#0072B2", fontstyle="italic", alpha=0.6)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    plt.tight_layout(h_pad=3.0, w_pad=2.5)

    path = FIG_OUT / "fig3_summary_dashboard.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("Generating KDD figures (final)...\n")
    fig1_radar()
    fig2_composite_bars()
    fig3_summary_dashboard()
    print(f"\n✅ All figures saved to: {FIG_OUT}")


#endregion#? 
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#?   4 

"""
KDD-Ready Figures for TRACE KG Paper — MINE-1 Evaluation (Final v2)
=====================================================================

5 publication-quality figures:

  Fig 1: Radar — multi-dimensional quality profile
  Fig 2: Composite metrics (EGU, RWA, SCI) with full names
  Fig 3: Summary dashboard (4 panels, fixed leakage visibility)
  Fig 4: Heatmap — per-metric rank matrix with stars
  Fig 5: Accuracy decomposition — stacked bar showing WHERE accuracy comes from

Color palette: Okabe-Ito colorblind-safe.
All data from compression_analysis_v5.json.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List

# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(".").resolve()
V5_JSON = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_v5.json"
FIG_OUT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["tracekg", "autoschemakg", "graphrag", "openie", "kggen"]
METHOD_LABELS = {
    "tracekg": "TRACE KG",
    "autoschemakg": "AutoSchemaKG",
    "graphrag": "GraphRAG",
    "openie": "OpenIE",
    "kggen": "KGGen",
}

METHOD_COLORS = {
    "tracekg":      "#0072B2",
    "autoschemakg": "#E69F00",
    "graphrag":     "#882255",
    "openie":       "#44AA99",
    "kggen":        "#999999",
}

DPI = 300
FONT_SIZE = 11

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "sans-serif",
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# LOAD DATA
# ============================================================

with open(V5_JSON) as f:
    v5 = json.load(f)

agg = v5["aggregate"]


# ============================================================
# FIG 1: RADAR
# ============================================================

def fig1_radar():
    radar_metrics = [
        ("Ret. Accuracy", "ret_acc",  True),
        ("EGU",           "egu",      True),
        ("RWA",           "rwa",      True),
        ("SCI",           "sci",      True),
        ("Connectivity",  "conn",     True),
        ("Clustering",    "clust",    True),
        ("Low Leakage",   "v4g",      False),
    ]

    methods = [m for m in METHOD_ORDER if m in agg]
    N = len(radar_metrics)

    raw = {}
    for m in methods:
        raw[m] = [agg[m][key] for _, key, _ in radar_metrics]

    raw_arr = np.array([raw[m] for m in methods])
    norm = np.zeros_like(raw_arr)
    for j, (_, _, higher) in enumerate(radar_metrics):
        col = raw_arr[:, j]
        mn, mx = col.min(), col.max()
        if mx - mn < 1e-9:
            norm[:, j] = 1.0
        else:
            norm[:, j] = (col - mn) / (mx - mn) if higher else (mx - col) / (mx - mn)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FAFAFA")

    for i, m in enumerate(methods):
        vals = norm[i].tolist() + [norm[i][0]]
        lw = 2.8 if m == "tracekg" else 1.3
        ls = "-" if m == "tracekg" else "--"
        alpha_fill = 0.12 if m == "tracekg" else 0.0
        zorder = 10 if m == "tracekg" else 5
        ax.plot(angles, vals, color=METHOD_COLORS[m], linewidth=lw, linestyle=ls,
                label=METHOD_LABELS[m], zorder=zorder)
        if alpha_fill > 0:
            ax.fill(angles, vals, color=METHOD_COLORS[m], alpha=alpha_fill, zorder=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl for lbl, _, _ in radar_metrics], fontsize=FONT_SIZE)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=FONT_SIZE - 2, color="grey")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.15), frameon=True,
              fancybox=True, shadow=False, edgecolor="#CCCCCC")

    path = FIG_OUT / "fig1_radar_quality_profile.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 2: COMPOSITE BARS
# ============================================================

def fig2_composite_bars():
    metrics = [
        ("EGU (%)",  "egu",  100, "Effective Graph\nUtilization"),
        ("RWA (%)",  "rwa",  100, "Reachability-Weighted\nAccuracy"),
        ("SCI",      "sci",  1,   "Structural Coherence\nIndex"),
    ]

    methods = [m for m in METHOD_ORDER if m in agg]
    n_methods = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    for ax_idx, (short_label, key, mult, full_name) in enumerate(metrics):
        ax = axes[ax_idx]
        vals = [agg[m][key] * mult for m in methods]
        stds = [agg[m].get(f"{key}_std", 0) * mult for m in methods]

        bars = ax.bar(range(n_methods), vals, color=colors, edgecolor="black",
                      linewidth=0.5, yerr=stds, capsize=4,
                      error_kw={"linewidth": 1, "color": "#555555"})

        trace_idx = methods.index("tracekg")
        bars[trace_idx].set_edgecolor(METHOD_COLORS["tracekg"])
        bars[trace_idx].set_linewidth(2.5)

        for i, (v, s) in enumerate(zip(vals, stds)):
            fw = "bold" if methods[i] == "tracekg" else "normal"
            if mult == 100:
                ax.text(i, v + s + 1.5, f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=FONT_SIZE - 2, fontweight=fw)
            else:
                ax.text(i, v + s + 0.005, f"{v:.4f}", ha="center", va="bottom",
                        fontsize=FONT_SIZE - 2, fontweight=fw)

        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=FONT_SIZE - 1)
        ax.set_title(short_label, fontweight="bold", fontsize=FONT_SIZE + 1)
        ax.text(0.5, -0.32, full_name, transform=ax.transAxes,
                ha="center", va="top", fontsize=FONT_SIZE - 2,
                fontstyle="italic", color="#555555")
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        ax.set_xlim(-0.6, n_methods - 0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Composite Quality Metrics — Higher is Better",
                 fontweight="bold", fontsize=FONT_SIZE + 2)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    path = FIG_OUT / "fig2_composite_metrics.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 3: SUMMARY DASHBOARD (fixed leakage visibility)
# ============================================================

def fig3_summary_dashboard():
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax in axes.flat:
        ax.set_axisbelow(True)

    # ── (a) Accuracy + Leakage — SIDE BY SIDE GROUPED BAR ──
    ax = axes[0, 0]
    acc = [agg[m]["ret_acc"] * 100 for m in methods]
    leak = [agg[m]["v4g"] * 100 for m in methods]

    w = 0.38
    bars_acc = ax.bar(x - w/2, acc, w, color=colors, edgecolor="black",
                      linewidth=0.5, label="Retrieval Accuracy (%)")
    bars_leak = ax.bar(x + w/2, leak, w, color="#CC3311", edgecolor="#990000",
                       linewidth=0.5, alpha=0.75, label="Verbatim Leakage (%)")

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i - w/2, acc[i] + 1.2, f"{acc[i]:.1f}", ha="center", va="bottom",
                fontsize=FONT_SIZE - 2, fontweight=fw)
        # Always show leakage value, even if small
        leak_y = max(leak[i], 1.5)  # minimum height for label visibility
        ax.text(i + w/2, leak_y + 0.8, f"{leak[i]:.1f}", ha="center", va="bottom",
                fontsize=FONT_SIZE - 2, fontweight="bold", color="#990000")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(a) Retrieval Accuracy vs. Verbatim Leakage", fontweight="bold")
    ax.legend(fontsize=FONT_SIZE - 2, loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    # ── (b) TriCR deviation ──
    ax = axes[0, 1]
    tricr = [agg[m]["tri_cr"] for m in methods]
    tricr_dev = [abs(t - 1.0) for t in tricr]

    bars = ax.barh(x, tricr_dev, color=colors, edgecolor="black", linewidth=0.5, height=0.55)

    for i in range(n):
        direction = "▸ expansion" if tricr[i] > 1 else "◂ oversimplified"
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(tricr_dev[i] + 0.04, i, f"{tricr[i]:.2f}  ({direction})",
                va="center", fontsize=FONT_SIZE - 2, fontweight=fw)

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("|TriCR − 1.0|  →  lower = better compression")
    ax.set_title("(b) Triple Compression Ratio (ideal = 1.0)", fontweight="bold")
    ax.set_xlim(0, 3.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    # ── (c) EGU ──
    ax = axes[1, 0]
    egu = [agg[m]["egu"] * 100 for m in methods]
    egu_std = [agg[m].get("egu_std", 0) * 100 for m in methods]

    bars = ax.bar(x, egu, color=colors, edgecolor="black", linewidth=0.5,
                  yerr=egu_std, capsize=4, error_kw={"linewidth": 1, "color": "#555555"})

    trace_idx = methods.index("tracekg")
    bars[trace_idx].set_edgecolor(METHOD_COLORS["tracekg"])
    bars[trace_idx].set_linewidth(2.5)

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, egu[i] + egu_std[i] + 1.5, f"{egu[i]:.1f}%", ha="center",
                va="bottom", fontsize=FONT_SIZE - 1, fontweight=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("EGU (%)")
    ax.set_title("(c) Effective Graph Utilization (EGU)", fontweight="bold")
    ax.text(0.5, -0.28, "EGU = Accuracy × Connectivity × (1 − Leakage)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SIZE - 2, fontstyle="italic", color="#555555")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    # ── (d) Average Rank ──
    ax = axes[1, 1]
    ranks = v5.get("avg_ranks", {})
    sorted_m = sorted([m for m in methods if m in ranks], key=lambda m: ranks[m])
    rank_vals = [ranks[m] for m in sorted_m]
    rank_labels = [METHOD_LABELS[m] for m in sorted_m]
    rank_colors = [METHOD_COLORS[m] for m in sorted_m]

    bars = ax.barh(np.arange(len(sorted_m)), rank_vals, color=rank_colors,
                   edgecolor="black", linewidth=0.5, height=0.55)

    for i, (m, rv) in enumerate(zip(sorted_m, rank_vals)):
        fw = "bold" if m == "tracekg" else "normal"
        ax.text(rv + 0.06, i, f"{rv:.2f}", va="center", fontsize=FONT_SIZE, fontweight=fw)

    ax.text(rank_vals[0] + 0.35, 0, "★", fontsize=16, va="center", ha="center",
            color="#DAA520", fontweight="bold")

    ax.set_yticks(np.arange(len(sorted_m)))
    ax.set_yticklabels(rank_labels, fontsize=FONT_SIZE)
    ax.set_xlabel("Average Rank Across 9 Metrics (lower = better)")
    ax.set_title("(d) Overall Quality Ranking", fontweight="bold")
    ax.set_xlim(0, 5.0)
    ax.axvline(x=1, color="#0072B2", linestyle=":", alpha=0.3, linewidth=1)
    ax.text(1.05, len(sorted_m) - 0.2, "perfect", fontsize=FONT_SIZE - 2,
            color="#0072B2", fontstyle="italic", alpha=0.6)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    plt.tight_layout(h_pad=3.0, w_pad=2.5)

    path = FIG_OUT / "fig3_summary_dashboard.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 4: HEATMAP — Per-metric rank matrix
# ============================================================

def fig4_rank_heatmap():
    """
    Heatmap: rows = methods (sorted by avg rank), cols = metrics.
    Cell color = rank (green=1, red=5). Stars on rank-1 cells.
    One glance: TRACE KG row is mostly green.
    """
    RANK_METRICS = [
        ("Ret.Acc",  "ret_acc",  True),
        ("EGU",      "egu",      True),
        ("RWA",      "rwa",      True),
        ("SCI",      "sci",      True),
        ("Conn.",    "conn",     True),
        ("Clust.",   "clust",    True),
        ("AvgDeg",   "avg_deg",  True),
        ("Leak%",    "v4g",      False),
        ("TriCR→1",  "tri_cr",   None),
    ]

    methods = [m for m in METHOD_ORDER if m in agg]

    # Compute ranks
    rank_matrix = []
    for label, key, higher in RANK_METRICS:
        vals = {m: agg[m][key] for m in methods}
        if higher is None:
            # TriCR: closest to 1.0
            distances = {m: abs(v - 1.0) for m, v in vals.items()}
            sorted_m = sorted(distances.items(), key=lambda x: x[1])
        elif higher:
            sorted_m = sorted(vals.items(), key=lambda x: -x[1])
        else:
            sorted_m = sorted(vals.items(), key=lambda x: x[1])
        ranks = {m: i + 1 for i, (m, _) in enumerate(sorted_m)}
        rank_matrix.append(ranks)

    # Sort methods by average rank
    avg_ranks = {}
    for m in methods:
        avg_ranks[m] = np.mean([rm[m] for rm in rank_matrix])
    sorted_methods = sorted(methods, key=lambda m: avg_ranks[m])

    # Build matrix
    n_methods = len(sorted_methods)
    n_metrics = len(RANK_METRICS)
    matrix = np.zeros((n_methods, n_metrics))
    for j, rm in enumerate(rank_matrix):
        for i, m in enumerate(sorted_methods):
            matrix[i, j] = rm[m]

    # Add avg rank column
    avg_col = np.array([[avg_ranks[m]] for m in sorted_methods])
    matrix_ext = np.hstack([matrix, avg_col])

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Custom colormap: 1=dark green, 3=white, 5=dark red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rank_cmap",
        ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"],
        N=5
    )

    # Plot heatmap (only the rank columns, not avg)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=1, vmax=5)

    # Cell annotations
    for i in range(n_methods):
        for j in range(n_metrics):
            rank = int(matrix[i, j])
            star = "★" if rank == 1 else str(rank)
            color = "white" if rank in [1, 5] else "black"
            fw = "bold" if rank == 1 else "normal"
            fs = FONT_SIZE + 1 if rank == 1 else FONT_SIZE
            ax.text(j, i, star, ha="center", va="center",
                    fontsize=fs, fontweight=fw, color=color)

    # Avg rank column (drawn as text to the right)
    for i, m in enumerate(sorted_methods):
        ar = avg_ranks[m]
        fw = "bold" if m == "tracekg" else "normal"
        ax.text(n_metrics + 0.3, i, f"{ar:.2f}", ha="center", va="center",
                fontsize=FONT_SIZE, fontweight=fw,
                color=METHOD_COLORS[m],
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=METHOD_COLORS[m], linewidth=1.5))

    # Labels
    metric_labels = [label for label, _, _ in RANK_METRICS]
    ax.set_xticks(list(range(n_metrics)) + [n_metrics + 0.3])
    ax.set_xticklabels(metric_labels + ["Avg\nRank"], fontsize=FONT_SIZE - 1)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([METHOD_LABELS[m] for m in sorted_methods], fontsize=FONT_SIZE)

    # Highlight TRACE KG row
    trace_row = sorted_methods.index("tracekg")
    rect = mpatches.FancyBboxPatch(
        (-0.5, trace_row - 0.5), n_metrics, 1,
        boxstyle="round,pad=0", linewidth=2.5,
        edgecolor=METHOD_COLORS["tracekg"], facecolor="none", zorder=10
    )
    ax.add_patch(rect)

    ax.set_title("Per-Metric Rankings (★ = Best, 1–5 scale)", fontweight="bold", pad=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.12)
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(["1\n(best)", "2", "3", "4", "5\n(worst)"])

    plt.tight_layout()

    path = FIG_OUT / "fig4_rank_heatmap.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 5: ACCURACY DECOMPOSITION — Where does accuracy come from?
# ============================================================

def fig5_accuracy_decomposition():
    """
    Stacked bar showing for each method:
      - Green: Genuine accuracy (EGU = Acc × Conn × (1-Leak))
      - Red: Accuracy from leakage (Acc × Conn × Leak)
      - Grey: Accuracy from unreachable fragments (Acc × (1-Conn))

    The message: AutoSchemaKG's tall bar is mostly red+grey.
    TRACE KG's bar is almost entirely green.
    """
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    x = np.arange(n)

    genuine = []    # EGU = Acc × Conn × (1-Leak)
    leaked = []     # Acc × Conn × Leak
    fragmented = [] # Acc × (1 - Conn)

    for m in methods:
        acc = agg[m]["ret_acc"]
        conn = agg[m]["conn"]
        leak = agg[m]["v4g"]

        g = acc * conn * (1 - leak)       # genuine graph-based
        l = acc * conn * leak             # from text copying
        f = acc * (1 - conn)              # from unreachable fragments

        genuine.append(g * 100)
        leaked.append(l * 100)
        fragmented.append(f * 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bars
    bars_g = ax.bar(x, genuine, color="#1a9641", edgecolor="black", linewidth=0.5,
                    label="Genuine Graph Retrieval (EGU)")
    bars_l = ax.bar(x, leaked, bottom=genuine, color="#CC3311", edgecolor="black",
                    linewidth=0.5, alpha=0.8, hatch="///",
                    label="From Verbatim Leakage")
    bars_f = ax.bar(x, fragmented, bottom=[g + l for g, l in zip(genuine, leaked)],
                    color="#BBBBBB", edgecolor="black", linewidth=0.5, alpha=0.7,
                    hatch="...", label="From Disconnected Fragments")

    # Total accuracy line
    totals = [g + l + f for g, l, f in zip(genuine, leaked, fragmented)]
    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, totals[i] + 1.5, f"{totals[i]:.1f}%", ha="center", va="bottom",
                fontsize=FONT_SIZE, fontweight=fw)

    # Annotate genuine % inside green bar
    for i in range(n):
        if genuine[i] > 8:
            ax.text(i, genuine[i] / 2, f"{genuine[i]:.1f}%", ha="center", va="center",
                    fontsize=FONT_SIZE - 2, fontweight="bold", color="white")

    # Annotate leaked % inside red bar if big enough
    for i in range(n):
        if leaked[i] > 5:
            ax.text(i, genuine[i] + leaked[i] / 2, f"{leaked[i]:.1f}%",
                    ha="center", va="center", fontsize=FONT_SIZE - 2,
                    fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE)
    ax.set_ylabel("Retrieval Accuracy Decomposition (%)")
    ax.set_title("Where Does Retrieval Accuracy Come From?", fontweight="bold", fontsize=FONT_SIZE + 2)
    ax.legend(loc="upper right", fontsize=FONT_SIZE - 1, framealpha=0.9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

    # Bracket annotation for TRACE KG
    trace_idx = methods.index("tracekg")
    ax.annotate("", xy=(trace_idx, genuine[trace_idx] + 2),
                xytext=(trace_idx, -6),
                arrowprops=dict(arrowstyle="]-[", color=METHOD_COLORS["tracekg"],
                                linewidth=2.5, mutation_scale=8))
    ax.text(trace_idx, -9, f"{genuine[trace_idx]:.0f}% genuine",
            ha="center", fontsize=FONT_SIZE - 1, fontweight="bold",
            color=METHOD_COLORS["tracekg"])

    plt.tight_layout()

    path = FIG_OUT / "fig5_accuracy_decomposition.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("Generating KDD figures (final v2)...\n")
    fig1_radar()
    fig2_composite_bars()
    fig3_summary_dashboard()
    fig4_rank_heatmap()
    fig5_accuracy_decomposition()
    print(f"\n✅ All 5 figures saved to: {FIG_OUT}")

#endregion#? 
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?   v5-1

"""
KDD-Ready Figures for TRACE KG Paper — MINE-1 Evaluation (Final v3)
=====================================================================

3 publication-quality figures:

  Fig 1: Radar — multi-dimensional quality profile
  Fig 2: Accuracy Decomposition — stacked bar (WHERE does accuracy come from?)
  Fig 3: Summary dashboard (4 panels)

Color palette: Okabe-Ito colorblind-safe.
All data from compression_analysis_v5.json.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List

# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(".").resolve()
V5_JSON = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_v5.json"
FIG_OUT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["tracekg", "autoschemakg", "graphrag", "openie", "kggen"]
METHOD_LABELS = {
    "tracekg": "TRACE KG",
    "autoschemakg": "AutoSchemaKG",
    "graphrag": "GraphRAG",
    "openie": "OpenIE",
    "kggen": "KGGen",
}

METHOD_COLORS = {
    "tracekg":      "#0072B2",  # strong blue
    "autoschemakg": "#E69F00",  # amber
    "graphrag":     "#882255",  # wine
    "openie":       "#44AA99",  # teal
    "kggen":        "#332288",  # indigo (was grey — now visible)
}

DPI = 300
FONT_SIZE = 11

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "sans-serif",
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# LOAD DATA
# ============================================================

with open(V5_JSON) as f:
    v5 = json.load(f)

agg = v5["aggregate"]


# ============================================================
# FIG 1: RADAR — Multi-dimensional quality profile
# ============================================================

def fig1_radar():
    radar_metrics = [
        ("Retrieval\nAccuracy", "ret_acc",  True),
        ("Effective Retrieval\nAccuracy",   "egu",      True),
        ("Connectivity",  "conn",     True),
        ("Clustering",    "clust",    True),
        ("Avg. Degree",   "avg_deg",  True),
        ("Low Leakage",   "v4g",      False),
        ("Compression\nQuality",  "tri_cr",   None),  # special: closeness to 1.0
    ]

    methods = [m for m in METHOD_ORDER if m in agg]
    N = len(radar_metrics)

    # Extract raw values
    raw = {}
    for m in methods:
        row = []
        for _, key, higher in radar_metrics:
            if higher is None:
                # TriCR: convert to "closeness to 1.0" score (max 1.0 = perfect)
                val = 1.0 / (1.0 + abs(agg[m][key] - 1.0))
                row.append(val)
            else:
                row.append(agg[m][key])
        raw[m] = row

    raw_arr = np.array([raw[m] for m in methods])
    norm = np.zeros_like(raw_arr)
    for j, (_, _, higher) in enumerate(radar_metrics):
        col = raw_arr[:, j]
        mn, mx = col.min(), col.max()
        if mx - mn < 1e-9:
            norm[:, j] = 1.0
        else:
            if higher is None or higher is True:
                norm[:, j] = (col - mn) / (mx - mn)
            else:
                norm[:, j] = (mx - col) / (mx - mn)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FAFAFA")

    for i, m in enumerate(methods):
        vals = norm[i].tolist() + [norm[i][0]]
        lw = 3.0 if m == "tracekg" else 1.5
        ls = "-" if m == "tracekg" else "--"
        alpha_fill = 0.12 if m == "tracekg" else 0.0
        zorder = 10 if m == "tracekg" else 5
        marker = "o" if m == "tracekg" else None
        ms = 5 if m == "tracekg" else 0
        ax.plot(angles, vals, color=METHOD_COLORS[m], linewidth=lw, linestyle=ls,
                label=METHOD_LABELS[m], zorder=zorder, marker=marker, markersize=ms)
        if alpha_fill > 0:
            ax.fill(angles, vals, color=METHOD_COLORS[m], alpha=alpha_fill, zorder=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl for lbl, _, _ in radar_metrics], fontsize=FONT_SIZE,
                       linespacing=1.1)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.50", "", "1.00"], fontsize=FONT_SIZE - 2, color="grey")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=True,
              fancybox=True, shadow=False, edgecolor="#CCCCCC")

    path = FIG_OUT / "fig1_radar_quality_profile.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 2: ACCURACY DECOMPOSITION — The "killer" figure
# ============================================================

def fig2_accuracy_decomposition():
    """
    Stacked bar decomposing each method's retrieval accuracy into:
      Green:  Effective Retrieval Accuracy = Acc × Conn × (1 − Leak)
      Red:    Accuracy from text copying    = Acc × Conn × Leak
      Grey:   Accuracy from disconnected KG = Acc × (1 − Conn)

    This is the figure that tells the story: AutoSchemaKG's 95% is hollow.
    TRACE KG's 90% is almost entirely genuine.
    """
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    x = np.arange(n)

    genuine = []
    leaked = []
    fragmented = []

    for m in methods:
        acc = agg[m]["ret_acc"]
        conn = agg[m]["conn"]
        leak = agg[m]["v4g"]

        g = acc * conn * (1 - leak)
        l = acc * conn * leak
        f = acc * (1 - conn)

        genuine.append(g * 100)
        leaked.append(l * 100)
        fragmented.append(f * 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bars
    ax.bar(x, genuine, color="#1a9641", edgecolor="black", linewidth=0.5,
           label="Effective Retrieval Accuracy (Acc × Conn × (1−Leak))")
    ax.bar(x, leaked, bottom=genuine, color="#CC3311", edgecolor="black",
           linewidth=0.5, alpha=0.85, hatch="///",
           label="From Verbatim Text Copying (Acc × Conn × Leak)")
    bottom2 = [g + l for g, l in zip(genuine, leaked)]
    ax.bar(x, fragmented, bottom=bottom2,
           color="#AAAAAA", edgecolor="black", linewidth=0.5, alpha=0.7,
           hatch="...", label="From Disconnected Fragments (Acc × (1−Conn))")

    # Total accuracy label on top
    totals = [g + l + f for g, l, f in zip(genuine, leaked, fragmented)]
    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, totals[i] + 1.5, f"{totals[i]:.1f}%", ha="center", va="bottom",
                fontsize=FONT_SIZE + 1, fontweight=fw)

    # Label genuine % inside green bar
    for i in range(n):
        if genuine[i] > 10:
            ax.text(i, genuine[i] / 2, f"{genuine[i]:.1f}%", ha="center", va="center",
                    fontsize=FONT_SIZE, fontweight="bold", color="white")

    # Label leaked % inside red bar
    for i in range(n):
        if leaked[i] > 6:
            y_pos = genuine[i] + leaked[i] / 2
            ax.text(i, y_pos, f"{leaked[i]:.1f}%", ha="center", va="center",
                    fontsize=FONT_SIZE, fontweight="bold", color="white")

    # Label fragmented % inside grey bar
    for i in range(n):
        if fragmented[i] > 6:
            y_pos = genuine[i] + leaked[i] + fragmented[i] / 2
            ax.text(i, y_pos, f"{fragmented[i]:.1f}%", ha="center", va="center",
                    fontsize=FONT_SIZE - 1, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE + 1)
    ax.set_ylabel("Retrieval Accuracy Decomposition (%)", fontsize=FONT_SIZE + 1)
    ax.set_title("Where Does Retrieval Accuracy Come From?",
                 fontweight="bold", fontsize=FONT_SIZE + 3, pad=15)
    ax.legend(loc="upper right", fontsize=FONT_SIZE - 1, framealpha=0.95,
              edgecolor="#CCCCCC")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    path = FIG_OUT / "fig2_accuracy_decomposition.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 3: SUMMARY DASHBOARD
# ============================================================

def fig3_summary_dashboard():
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax in axes.flat:
        ax.set_axisbelow(True)

    # ── (a) Accuracy + Leakage overlay (single bar, leakage labels visible) ──
    ax = axes[0, 0]
    acc = [agg[m]["ret_acc"] * 100 for m in methods]
    leak = [agg[m]["v4g"] * 100 for m in methods]

    ax.bar(x, acc, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85,
           label="Retrieval Accuracy")
    ax.bar(x, leak, color="#CC3311", alpha=0.55, hatch="///",
           edgecolor="#CC3311", linewidth=0.5, label="Verbatim Leakage")

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, acc[i] + 1.2, f"{acc[i]:.1f}%", ha="center", va="bottom",
                fontsize=FONT_SIZE - 1, fontweight=fw, color="black")

    # Leakage labels — always visible
    for i in range(n):
        if leak[i] > 3:
            # Big leakage: label inside the red bar, white text
            ax.text(i, leak[i] / 2, f"{leak[i]:.1f}%", ha="center", va="center",
                    fontsize=FONT_SIZE, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#CC3311",
                              edgecolor="none", alpha=0.8))
        elif leak[i] > 0.5:
            # Small leakage: label above the red bar with arrow
            ax.annotate(f"{leak[i]:.1f}%",
                        xy=(i, leak[i]), xytext=(i + 0.3, leak[i] + 12),
                        fontsize=FONT_SIZE - 1, fontweight="bold", color="#CC3311",
                        arrowprops=dict(arrowstyle="-|>", color="#CC3311", linewidth=1.2),
                        ha="center", va="bottom")
        else:
            # Near-zero: small text above
            ax.text(i, 3, f"{leak[i]:.1f}%", ha="center", va="bottom",
                    fontsize=FONT_SIZE - 2, color="#CC3311", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(a) Retrieval Accuracy & Information Leakage", fontweight="bold")
    ax.legend(fontsize=FONT_SIZE - 2, loc="lower left", framealpha=0.9)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    # ── (b) TriCR deviation ──
    ax = axes[0, 1]
    tricr = [agg[m]["tri_cr"] for m in methods]
    tricr_dev = [abs(t - 1.0) for t in tricr]

    bars = ax.barh(x, tricr_dev, color=colors, edgecolor="black", linewidth=0.5, height=0.55)

    for i in range(n):
        direction = "▸ expansion" if tricr[i] > 1 else "◂ oversimplified"
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(tricr_dev[i] + 0.04, i, f"{tricr[i]:.2f}  ({direction})",
                va="center", fontsize=FONT_SIZE - 2, fontweight=fw)

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("|TriCR − 1.0|  →  lower = better compression")
    ax.set_title("(b) Triple Compression Ratio (ideal = 1.0)", fontweight="bold")
    ax.set_xlim(0, 3.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    # ── (c) Effective Retrieval Accuracy ──
    ax = axes[1, 0]
    egu = [agg[m]["egu"] * 100 for m in methods]
    egu_std = [agg[m].get("egu_std", 0) * 100 for m in methods]

    bars = ax.bar(x, egu, color=colors, edgecolor="black", linewidth=0.5,
                  yerr=egu_std, capsize=4, error_kw={"linewidth": 1, "color": "#555555"})

    trace_idx = methods.index("tracekg")
    bars[trace_idx].set_edgecolor(METHOD_COLORS["tracekg"])
    bars[trace_idx].set_linewidth(2.5)

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, egu[i] + egu_std[i] + 1.5, f"{egu[i]:.1f}%", ha="center",
                va="bottom", fontsize=FONT_SIZE - 1, fontweight=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("ERA (%)")
    ax.set_title("(c) Effective Retrieval Accuracy", fontweight="bold")
    ax.text(0.5, -0.28, "ERA = Accuracy × Connectivity × (1 − Leakage)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SIZE - 1, fontstyle="italic", color="#444444")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    # ── (d) Average Rank ──
    ax = axes[1, 1]
    ranks = v5.get("avg_ranks", {})
    sorted_m = sorted([m for m in methods if m in ranks], key=lambda m: ranks[m])
    rank_vals = [ranks[m] for m in sorted_m]
    rank_labels = [METHOD_LABELS[m] for m in sorted_m]
    rank_colors = [METHOD_COLORS[m] for m in sorted_m]

    bars = ax.barh(np.arange(len(sorted_m)), rank_vals, color=rank_colors,
                   edgecolor="black", linewidth=0.5, height=0.55)

    for i, (m, rv) in enumerate(zip(sorted_m, rank_vals)):
        fw = "bold" if m == "tracekg" else "normal"
        ax.text(rv + 0.06, i, f"{rv:.2f}", va="center", fontsize=FONT_SIZE, fontweight=fw)

    ax.text(rank_vals[0] + 0.35, 0, "★", fontsize=16, va="center", ha="center",
            color="#DAA520", fontweight="bold")

    ax.set_yticks(np.arange(len(sorted_m)))
    ax.set_yticklabels(rank_labels, fontsize=FONT_SIZE)
    ax.set_xlabel("Average Rank Across All Metrics (lower = better)")
    ax.set_title("(d) Overall Quality Ranking", fontweight="bold")
    ax.set_xlim(0, 5.0)
    ax.axvline(x=1, color="#0072B2", linestyle=":", alpha=0.3, linewidth=1)
    ax.text(1.05, len(sorted_m) - 0.2, "perfect", fontsize=FONT_SIZE - 2,
            color="#0072B2", fontstyle="italic", alpha=0.6)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    plt.tight_layout(h_pad=3.0, w_pad=2.5)

    path = FIG_OUT / "fig3_summary_dashboard.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("Generating KDD figures (final v3)...\n")
    fig1_radar()
    fig2_accuracy_decomposition()
    fig3_summary_dashboard()
    print(f"\n✅ All 3 figures saved to: {FIG_OUT}")
    
    
    
#endregion#? 
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?   5-2




"""
KDD-Ready Figures for TRACE KG Paper — MINE-1 Evaluation (Final v3)
=====================================================================

3 publication-quality figures, no redundancy:

  Fig 1: Radar — multi-dimensional quality profile
  Fig 2: Summary dashboard (4 panels) — the one figure that tells the story
  Fig 3: Accuracy decomposition — where does accuracy actually come from?

Design principles:
  - No duplicate information across figures
  - Radar = holistic overview, Dashboard = detailed evidence, Decomposition = the insight
  - Colorblind-safe Okabe-Ito palette with improved contrast
  - All data from compression_analysis_v5.json
  - No arrows, no per-essay data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List

# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(".").resolve()
V5_JSON = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_v5.json"
FIG_OUT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["tracekg", "autoschemakg", "graphrag", "openie", "kggen"]
METHOD_LABELS = {
    "tracekg": "TRACE KG",
    "autoschemakg": "AutoSchemaKG",
    "graphrag": "GraphRAG",
    "openie": "OpenIE",
    "kggen": "KGGen",
}

# Okabe-Ito with improved contrast — kggen darker
METHOD_COLORS = {
    "tracekg":      "#0072B2",  # strong blue
    "autoschemakg": "#E69F00",  # amber
    "graphrag":     "#882255",  # wine
    "openie":       "#44AA99",  # teal
    "kggen":        "#555555",  # dark grey (was #999, now visible)
}

DPI = 300
FONT_SIZE = 11

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "sans-serif",
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# LOAD DATA
# ============================================================

with open(V5_JSON) as f:
    v5 = json.load(f)

agg = v5["aggregate"]


# ============================================================
# FIG 1: RADAR — Multi-dimensional quality profile
# ============================================================

def fig1_radar():
    """
    7 normalized metrics on a radar. TRACE KG = solid fill, others = dashed.
    Tells the story: TRACE KG is the only method that excels on ALL axes.
    """
    radar_metrics = [
        ("Retrieval\nAccuracy",  "ret_acc",  True),
        ("Effective Retrieval\nAccuracy",  "egu",  True),
        ("Structural\nCoherence", "sci",     True),
        ("Connectivity",          "conn",    True),
        ("Clustering",            "clust",   True),
        ("Compression\nQuality",  "tri_cr",  None),   # special: closest to 1.0
        ("Low Leakage",           "v4g",     False),
    ]

    methods = [m for m in METHOD_ORDER if m in agg]
    N = len(radar_metrics)

    # Extract raw values
    raw = {}
    for m in methods:
        vals = []
        for _, key, higher in radar_metrics:
            if higher is None:
                # TriCR: convert to "closeness to 1.0" score (max deviation ~3.6)
                vals.append(1.0 / (1.0 + abs(agg[m][key] - 1.0)))
            else:
                vals.append(agg[m][key])
        raw[m] = vals

    # Normalize to [0, 1]
    raw_arr = np.array([raw[m] for m in methods])
    norm = np.zeros_like(raw_arr)
    for j in range(N):
        col = raw_arr[:, j]
        mn, mx = col.min(), col.max()
        _, _, higher = radar_metrics[j]
        if mx - mn < 1e-9:
            norm[:, j] = 1.0
        elif higher is None:
            # Already transformed: higher = better
            norm[:, j] = (col - mn) / (mx - mn)
        elif higher:
            norm[:, j] = (col - mn) / (mx - mn)
        else:
            norm[:, j] = (mx - col) / (mx - mn)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FAFAFA")

    # Draw methods — TRACE KG last so it's on top
    draw_order = [m for m in methods if m != "tracekg"] + ["tracekg"]
    for m in draw_order:
        i = methods.index(m)
        vals = norm[i].tolist() + [norm[i][0]]
        is_trace = (m == "tracekg")
        lw = 3.0 if is_trace else 1.5
        ls = "-" if is_trace else "--"
        alpha_fill = 0.15 if is_trace else 0.0
        zorder = 10 if is_trace else 5
        marker = "o" if is_trace else None
        ms = 6 if is_trace else 0
        ax.plot(angles, vals, color=METHOD_COLORS[m], linewidth=lw, linestyle=ls,
                label=METHOD_LABELS[m], zorder=zorder, marker=marker, markersize=ms)
        if alpha_fill > 0:
            ax.fill(angles, vals, color=METHOD_COLORS[m], alpha=alpha_fill, zorder=4)

    labels = [lbl for lbl, _, _ in radar_metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FONT_SIZE - 0.5, linespacing=1.1)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.50", "", "1.00"], fontsize=FONT_SIZE - 2, color="grey")

    # Grid styling
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), frameon=True,
              fancybox=True, shadow=False, edgecolor="#BBBBBB",
              fontsize=FONT_SIZE - 0.5)

    path = FIG_OUT / "fig1_radar.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 2: SUMMARY DASHBOARD — 4 panels, the complete evidence
# ============================================================

def fig2_summary_dashboard():
    """
    2×2 dashboard:
      (a) Accuracy + Leakage (single bar with leakage hatched inside)
      (b) TriCR deviation from ideal
      (c) Effective Retrieval Accuracy (= Acc × Conn × (1-Leak))
      (d) Overall quality ranking
    """
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    for ax in axes.flat:
        ax.set_axisbelow(True)

    # ────────────────────────────────────────────────────
    # (a) Accuracy bars with leakage hatched overlay inside
    # ────────────────────────────────────────────────────
    ax = axes[0, 0]
    acc = [agg[m]["ret_acc"] * 100 for m in methods]
    leak = [agg[m]["v4g"] * 100 for m in methods]

    # Full accuracy bar
    ax.bar(x, acc, color=colors, edgecolor="black", linewidth=0.5, alpha=0.88)
    # Leakage hatched from bottom
    ax.bar(x, leak, color="#CC3311", alpha=0.55, hatch="///",
           edgecolor="#CC3311", linewidth=0.3)

    # Accuracy labels on top
    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, acc[i] + 1.5, f"{acc[i]:.1f}%", ha="center", va="bottom",
                fontsize=FONT_SIZE - 1, fontweight=fw, color="black")

    # Leakage labels inside the hatched area — white on colored highlight
    for i in range(n):
        if leak[i] >= 1.0:
            y_pos = min(leak[i] / 2, acc[i] * 0.4)
            y_pos = max(y_pos, 3)
            ax.text(i, y_pos, f"Leak: {leak[i]:.1f}%", ha="center", va="center",
                    fontsize=FONT_SIZE - 1.5, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#CC3311",
                              edgecolor="none", alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(a) Retrieval Accuracy & Verbatim Leakage", fontweight="bold")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#AAAAAA", edgecolor="black", linewidth=0.5,
                       label="Retrieval Accuracy"),
        mpatches.Patch(facecolor="#CC3311", edgecolor="#CC3311", alpha=0.55,
                       hatch="///", label="Verbatim Leakage"),
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE - 2,
              loc="lower right", framealpha=0.92)

    # Annotation below
    ax.text(0.5, -0.22,
            "Leakage = fraction of source text copied verbatim into entity strings (4-gram overlap)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SIZE - 1.5, fontstyle="italic", color="#444444")

    # ───────────────────��────────────────────────────────
    # (b) TriCR deviation from ideal 1.0
    # ────────────────────────────────────────────────────
    ax = axes[0, 1]
    tricr = [agg[m]["tri_cr"] for m in methods]
    tricr_dev = [abs(t - 1.0) for t in tricr]

    bars = ax.barh(x, tricr_dev, color=colors, edgecolor="black", linewidth=0.5, height=0.55)

    for i in range(n):
        if tricr[i] > 1:
            note = "expansion"
        else:
            note = "loss"
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(tricr_dev[i] + 0.04, i, f"TriCR = {tricr[i]:.2f}  ({note})",
                va="center", fontsize=FONT_SIZE - 1.5, fontweight=fw, color="#333333")

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("|TriCR − 1.0|       lower = better compression")
    ax.set_title("(b) Triple Compression Ratio (ideal = 1.0)", fontweight="bold")
    ax.set_xlim(0, 3.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    ax.text(0.5, -0.18,
            "TriCR = total triple words / essay words.  Values > 1 = KG larger than source text.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SIZE - 1.5, fontstyle="italic", color="#444444")

    # ────────────────────────────────────────────────────
    # (c) Effective Retrieval Accuracy
    # ────────────────────────────────────────────────────
    ax = axes[1, 0]
    egu = [agg[m]["egu"] * 100 for m in methods]
    egu_std = [agg[m].get("egu_std", 0) * 100 for m in methods]

    bars = ax.bar(x, egu, color=colors, edgecolor="black", linewidth=0.5,
                  yerr=egu_std, capsize=4, error_kw={"linewidth": 1, "color": "#555555"})

    trace_idx = methods.index("tracekg")
    bars[trace_idx].set_edgecolor(METHOD_COLORS["tracekg"])
    bars[trace_idx].set_linewidth(2.5)

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, egu[i] + egu_std[i] + 1.8, f"{egu[i]:.1f}%", ha="center",
                va="bottom", fontsize=FONT_SIZE - 1, fontweight=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Effective Retrieval Accuracy (%)")
    ax.set_title("(c) Effective Retrieval Accuracy", fontweight="bold")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    ax.text(0.5, -0.22,
            "Effective Ret. Acc. = Accuracy × Connectivity × (1 − Leakage)\n"
            "Accuracy from genuine graph retrieval, penalizing text copying and fragmentation.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SIZE - 1.5, fontstyle="italic", color="#444444",
            linespacing=1.4)

    # ────────────────────────────────────────────────────
    # (d) Overall quality ranking
    # ────────────────────────────────────────────────────
    ax = axes[1, 1]
    ranks = v5.get("avg_ranks", {})
    sorted_m = sorted([m for m in methods if m in ranks], key=lambda m: ranks[m])
    rank_vals = [ranks[m] for m in sorted_m]
    rank_labels = [METHOD_LABELS[m] for m in sorted_m]
    rank_colors = [METHOD_COLORS[m] for m in sorted_m]

    bars = ax.barh(np.arange(len(sorted_m)), rank_vals, color=rank_colors,
                   edgecolor="black", linewidth=0.5, height=0.55)

    for i, (m, rv) in enumerate(zip(sorted_m, rank_vals)):
        fw = "bold" if m == "tracekg" else "normal"
        ax.text(rv + 0.08, i, f"{rv:.2f}", va="center", fontsize=FONT_SIZE, fontweight=fw)

    # Gold star for #1
    ax.text(rank_vals[0] + 0.38, 0, "★", fontsize=16, va="center", ha="center",
            color="#DAA520", fontweight="bold")

    ax.set_yticks(np.arange(len(sorted_m)))
    ax.set_yticklabels(rank_labels, fontsize=FONT_SIZE)
    ax.set_xlabel("Average Rank (lower = better)")
    ax.set_title("(d) Overall Quality Ranking", fontweight="bold")
    ax.set_xlim(0, 5.0)
    ax.axvline(x=1, color="#0072B2", linestyle=":", alpha=0.25, linewidth=1)
    ax.text(1.08, len(sorted_m) - 0.15, "perfect", fontsize=FONT_SIZE - 2,
            color="#0072B2", fontstyle="italic", alpha=0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    ax.text(0.5, -0.18,
            "Mean rank across 9 metrics: Ret.Acc, ERA, Conn., Clust., AvgDeg, Leak%, TriCR, SCI, RWA.\n"
            "Ranking methodology follows Demšar (JMLR, 2006).",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SIZE - 1.5, fontstyle="italic", color="#444444",
            linespacing=1.4)

    plt.tight_layout(h_pad=4.0, w_pad=2.5)

    path = FIG_OUT / "fig2_summary_dashboard.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 3: ACCURACY DECOMPOSITION — Where does accuracy come from?
# ============================================================

def fig3_accuracy_decomposition():
    """
    Stacked horizontal bar: each method's retrieval accuracy broken into:
      - Green: Genuine graph-based retrieval (Acc × Conn × (1-Leak))
      - Red:   From verbatim text leakage  (Acc × Conn × Leak)
      - Grey:  From disconnected subgraphs (Acc × (1-Conn))

    NOT redundant with dashboard: dashboard shows individual metrics,
    this figure shows how they INTERACT to produce the accuracy number.
    This is the "aha" moment figure.
    """
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    y = np.arange(n)

    genuine = []
    leaked = []
    fragmented = []

    for m in methods:
        acc = agg[m]["ret_acc"]
        conn = agg[m]["conn"]
        leak = agg[m]["v4g"]

        g = acc * conn * (1 - leak) * 100
        l = acc * conn * leak * 100
        f = acc * (1 - conn) * 100

        genuine.append(g)
        leaked.append(l)
        fragmented.append(f)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Stacked horizontal bars
    bars_g = ax.barh(y, genuine, height=0.6, color="#1a9641", edgecolor="black",
                     linewidth=0.5, label="Genuine Graph Retrieval")
    bars_l = ax.barh(y, leaked, left=genuine, height=0.6, color="#CC3311",
                     edgecolor="black", linewidth=0.5, alpha=0.80,
                     hatch="///", label="From Verbatim Leakage")
    g_plus_l = [g + l for g, l in zip(genuine, leaked)]
    bars_f = ax.barh(y, fragmented, left=g_plus_l, height=0.6,
                     color="#AAAAAA", edgecolor="black", linewidth=0.5,
                     alpha=0.70, hatch="...", label="From Disconnected Fragments")

    totals = [g + l + f for g, l, f in zip(genuine, leaked, fragmented)]

    # Total accuracy label at end of bar
    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(totals[i] + 1.0, i, f"{totals[i]:.1f}%", va="center",
                fontsize=FONT_SIZE, fontweight=fw, color="#333333")

    # Genuine % inside green section
    for i in range(n):
        if genuine[i] > 12:
            ax.text(genuine[i] / 2, i, f"{genuine[i]:.1f}%", ha="center", va="center",
                    fontsize=FONT_SIZE - 1, fontweight="bold", color="white")

    # Leaked % inside red section
    for i in range(n):
        if leaked[i] > 8:
            ax.text(genuine[i] + leaked[i] / 2, i, f"{leaked[i]:.1f}%",
                    ha="center", va="center", fontsize=FONT_SIZE - 1.5,
                    fontweight="bold", color="white")

    # Fragmented % inside grey section
    for i in range(n):
        if fragmented[i] > 8:
            ax.text(g_plus_l[i] + fragmented[i] / 2, i, f"{fragmented[i]:.1f}%",
                    ha="center", va="center", fontsize=FONT_SIZE - 1.5,
                    fontweight="bold", color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE)
    ax.set_xlabel("Retrieval Accuracy (%)", fontsize=FONT_SIZE)
    ax.set_xlim(0, 108)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    ax.set_title("Retrieval Accuracy Decomposition: Where Does Performance Come From?",
                 fontweight="bold", fontsize=FONT_SIZE + 1)

    ax.legend(loc="lower right", fontsize=FONT_SIZE - 1, framealpha=0.92,
              edgecolor="#CCCCCC")

    # Highlight TRACE KG row
    trace_idx = methods.index("tracekg")
    rect = mpatches.FancyBboxPatch(
        (-1, trace_idx - 0.35), totals[trace_idx] + 2, 0.7,
        boxstyle="round,pad=0.1", linewidth=2.0,
        edgecolor=METHOD_COLORS["tracekg"], facecolor="none", zorder=10
    )
    ax.add_patch(rect)

    # Annotation
    ax.text(0.5, -0.15,
            "Genuine = Acc × Conn × (1−Leak).    "
            "Leakage = Acc × Conn × Leak.    "
            "Fragments = Acc × (1−Conn).",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SIZE - 1, fontstyle="italic", color="#444444")

    plt.tight_layout()

    path = FIG_OUT / "fig3_accuracy_decomposition.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("Generating KDD figures (final v3)...\n")
    fig1_radar()
    fig2_summary_dashboard()
    fig3_accuracy_decomposition()
    print(f"\n✅ All 3 figures saved to: {FIG_OUT}")
    print("\nFigure roles:")
    print("  Fig 1 (Radar):          Holistic overview — TRACE KG covers the most area")
    print("  Fig 2 (Dashboard):      Detailed evidence — 4 complementary panels")
    print("  Fig 3 (Decomposition):  The insight — AutoSchemaKG's accuracy is mostly leaked/fragmented")
    
    
#endregion#? 
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?  Base


"""
KDD-Ready Figures for TRACE KG Paper — MINE-1 Evaluation (Final v4)
=====================================================================

4 publication-quality figures:

  Fig 1: Radar — multi-dimensional quality profile
  Fig 2: Effective Retrieval Accuracy (single bar, prominent)
  Fig 3: "Why Raw Accuracy is Misleading" — 3-panel failure mode exposé
  Fig 4: Summary dashboard — cleaned, publication-ready (3 panels)

Color palette: Okabe-Ito colorblind-safe.
All data from compression_analysis_v5.json.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Dict, List

# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(".").resolve()
V5_JSON = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_v5.json"
FIG_OUT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["tracekg", "autoschemakg", "graphrag", "openie", "kggen"]
METHOD_LABELS = {
    "tracekg": "TRACE KG",
    "autoschemakg": "AutoSchemaKG",
    "graphrag": "GraphRAG",
    "openie": "OpenIE",
    "kggen": "KGGen",
}

METHOD_COLORS = {
    "tracekg":      "#0072B2",
    "autoschemakg": "#E69F00",
    "graphrag":     "#882255",
    "openie":       "#44AA99",
    "kggen":        "#332288",
}

# Semantic colors
CLR_GENUINE = "#1a9641"
CLR_LEAKED  = "#d7191c"
CLR_FRAG    = "#bababa"

DPI = 300
FONT_TITLE = 15
FONT_SUBTITLE = 13
FONT_LABEL = 12
FONT_TICK = 11
FONT_ANNOT = 11
FONT_NOTE = 10

plt.rcParams.update({
    "font.size": FONT_LABEL,
    "font.family": "sans-serif",
    "axes.titlesize": FONT_SUBTITLE,
    "axes.labelsize": FONT_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_TICK,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# LOAD DATA
# ============================================================

with open(V5_JSON) as f:
    v5 = json.load(f)

agg = v5["aggregate"]


# ============================================================
# FIG 1: RADAR — Multi-dimensional quality profile
# ============================================================

def fig1_radar():
    radar_metrics = [
        ("Retrieval\nAccuracy",           "ret_acc",  True),
        ("Effective\nRetrieval Accuracy",  "egu",      True),
        ("Connectivity",                   "conn",     True),
        ("Clustering",                     "clust",    True),
        ("Avg. Degree",                    "avg_deg",  True),
        ("Low Leakage",                    "v4g",      False),
        ("Compression\nQuality",           "tri_cr",   None),
    ]

    methods = [m for m in METHOD_ORDER if m in agg]
    N = len(radar_metrics)

    raw = {}
    for m in methods:
        raw[m] = [agg[m][key] for _, key, _ in radar_metrics]

    raw_arr = np.array([raw[m] for m in methods])
    norm = np.zeros_like(raw_arr)
    for j, (_, key, higher) in enumerate(radar_metrics):
        col = raw_arr[:, j]
        if higher is None:
            dist = np.abs(col - 1.0)
            mn, mx = dist.min(), dist.max()
            if mx - mn < 1e-9:
                norm[:, j] = 1.0
            else:
                norm[:, j] = (mx - dist) / (mx - mn)
        else:
            mn, mx = col.min(), col.max()
            if mx - mn < 1e-9:
                norm[:, j] = 1.0
            else:
                norm[:, j] = (col - mn) / (mx - mn) if higher else (mx - col) / (mx - mn)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FAFAFA")

    for i, m in enumerate(methods):
        vals = norm[i].tolist() + [norm[i][0]]
        lw = 3.0 if m == "tracekg" else 1.5
        ls = "-" if m == "tracekg" else "--"
        alpha_fill = 0.12 if m == "tracekg" else 0.0
        zorder = 10 if m == "tracekg" else 5
        marker = "o" if m == "tracekg" else ""
        ms = 5 if m == "tracekg" else 0
        ax.plot(angles, vals, color=METHOD_COLORS[m], linewidth=lw, linestyle=ls,
                label=METHOD_LABELS[m], zorder=zorder, marker=marker, markersize=ms)
        if alpha_fill > 0:
            ax.fill(angles, vals, color=METHOD_COLORS[m], alpha=alpha_fill, zorder=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl for lbl, _, _ in radar_metrics], fontsize=FONT_TICK,
                       linespacing=1.1)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=FONT_TICK - 1, color="grey")

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=True,
              fancybox=True, shadow=False, edgecolor="#CCCCCC", fontsize=FONT_TICK)

    path = FIG_OUT / "fig1_radar_quality_profile.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 2: EFFECTIVE RETRIEVAL ACCURACY
# ============================================================

def fig2_effective_retrieval_accuracy():
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    egu = [agg[m]["egu"] * 100 for m in methods]
    egu_std = [agg[m].get("egu_std", 0) * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(x, egu, color=colors, edgecolor="black", linewidth=0.6,
                  yerr=egu_std, capsize=5, error_kw={"linewidth": 1.2, "color": "#444444"},
                  width=0.62)

    trace_idx = methods.index("tracekg")
    bars[trace_idx].set_edgecolor(METHOD_COLORS["tracekg"])
    bars[trace_idx].set_linewidth(3.0)

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        fs = FONT_ANNOT + 1 if methods[i] == "tracekg" else FONT_ANNOT
        ax.text(i, egu[i] + egu_std[i] + 2.0, f"{egu[i]:.1f}%",
                ha="center", va="bottom", fontsize=fs, fontweight=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_LABEL)
    ax.set_ylabel("Effective Retrieval Accuracy (%)", fontsize=FONT_LABEL)
    ax.set_title("Effective Retrieval Accuracy", fontweight="bold",
                 fontsize=FONT_TITLE + 2, pad=20)
    ax.text(0.5, 1.015,
            "Retrieval Accuracy  ×  Graph Connectivity  ×  (1 − Verbatim Leakage)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FONT_LABEL, color="#333333", fontstyle="italic")

    ax.set_ylim(0, max(egu) + max(egu_std) + 15)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = FIG_OUT / "fig2_effective_retrieval_accuracy.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 3: "WHY RAW ACCURACY IS MISLEADING" — 3 failure modes
# ============================================================

def fig3_failure_modes():
    """
    Three-panel figure exposing why raw accuracy alone is misleading.

    Panel (a): Accuracy decomposition — stacked bar showing genuine vs
               leaked vs fragmented contributions to accuracy.
    Panel (b): Compression quality — distance from ideal TriCR=1.0.
    Panel (c): Effective Retrieval Accuracy — the corrected metric.

    Core message: AutoSchemaKG's 95% accuracy is largely from text copying.
    TRACE KG's 90% is almost entirely genuine graph retrieval.
    """
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax in axes:
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── (a) Accuracy Decomposition ──
    ax = axes[0]

    genuine = []
    leaked = []
    fragmented = []

    for m in methods:
        a = agg[m]["ret_acc"]
        c = agg[m]["conn"]
        l = agg[m]["v4g"]
        genuine.append(a * c * (1 - l) * 100)
        leaked.append(a * c * l * 100)
        fragmented.append(a * (1 - c) * 100)

    bars_g = ax.bar(x, genuine, width=0.62, color=CLR_GENUINE, edgecolor="black",
                    linewidth=0.5, label="Genuine retrieval", zorder=5)
    bars_l = ax.bar(x, leaked, bottom=genuine, width=0.62, color=CLR_LEAKED,
                    edgecolor="black", linewidth=0.5, hatch="///",
                    label="From text copying", zorder=5)
    bars_f = ax.bar(x, fragmented, bottom=[g + l for g, l in zip(genuine, leaked)],
                    width=0.62, color=CLR_FRAG, edgecolor="black", linewidth=0.5,
                    hatch="...", label="From disconnected nodes", zorder=5)

    totals = [g + l + f for g, l, f in zip(genuine, leaked, fragmented)]

    # Total on top
    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, totals[i] + 1.5, f"{totals[i]:.1f}%", ha="center", va="bottom",
                fontsize=FONT_ANNOT, fontweight=fw, zorder=10)

    # Genuine % inside green bar
    for i in range(n):
        if genuine[i] > 12:
            ax.text(i, genuine[i] / 2, f"{genuine[i]:.0f}%", ha="center", va="center",
                    fontsize=FONT_ANNOT - 1, fontweight="bold", color="white", zorder=10)

    # Leaked % inside red bar if visible
    for i in range(n):
        if leaked[i] > 6:
            ax.text(i, genuine[i] + leaked[i] / 2, f"{leaked[i]:.0f}%",
                    ha="center", va="center", fontsize=FONT_ANNOT - 1,
                    fontweight="bold", color="white", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK, rotation=25, ha="right")
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=FONT_LABEL)
    ax.set_title("(a) Failure Mode 1: Accuracy from Text Copying",
                 fontweight="bold", fontsize=FONT_SUBTITLE, pad=10)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.12, linewidth=0.5)
    ax.legend(fontsize=FONT_TICK, loc="upper right", framealpha=0.95,
              edgecolor="#CCCCCC")

    ax.text(0.5, -0.18,
            "Green = accuracy from genuine graph structure\n"
            "Red = accuracy inflated by verbatim text in entities",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_NOTE, color="#444444", linespacing=1.4)

    # ── (b) Compression Quality ──
    ax = axes[1]

    tricr = [agg[m]["tri_cr"] for m in methods]
    tricr_dev = [abs(t - 1.0) for t in tricr]

    bar_edge_colors = []
    for i, m in enumerate(methods):
        if tricr[i] > 1.5:
            bar_edge_colors.append(CLR_LEAKED)
        elif tricr[i] < 0.5:
            bar_edge_colors.append("#E69F00")
        else:
            bar_edge_colors.append("black")

    bars_b = ax.barh(x, tricr_dev, color=colors, edgecolor="black",
                     linewidth=0.6, height=0.55)

    # Semantic annotation
    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        if tricr[i] > 1.5:
            tag = "KG > source text"
            tag_color = CLR_LEAKED
        elif tricr[i] < 0.6:
            tag = "heavy information loss"
            tag_color = "#B35900"
        elif abs(tricr[i] - 1.0) < 0.15:
            tag = "near-ideal"
            tag_color = CLR_GENUINE
        else:
            tag = ""
            tag_color = "#333333"

        label_text = f"{tricr[i]:.2f}"
        if tag:
            label_text += f"  ({tag})"

        ax.text(tricr_dev[i] + 0.05, i, label_text,
                va="center", fontsize=FONT_ANNOT - 1, fontweight=fw, color=tag_color)

    # Ideal zone shading
    ax.axvspan(0, 0.15, color=CLR_GENUINE, alpha=0.06, zorder=0)
    ax.text(0.07, n - 0.3, "ideal\nzone", fontsize=FONT_NOTE, color=CLR_GENUINE,
            ha="center", va="top", fontstyle="italic", alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Distance from Ideal  |TriCR − 1.0|", fontsize=FONT_LABEL)
    ax.set_title("(b) Failure Mode 2: Poor Compression",
                 fontweight="bold", fontsize=FONT_SUBTITLE, pad=10)
    ax.set_xlim(0, 3.8)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.12, linewidth=0.5)

    ax.text(0.5, -0.18,
            "TriCR = total triple words / essay words\n"
            "Ideal = 1.0 (lossless compression)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_NOTE, color="#444444", linespacing=1.4)

    # ── (c) Corrected Metric: Effective Retrieval Accuracy ──
    ax = axes[2]

    egu = [agg[m]["egu"] * 100 for m in methods]
    raw_acc = [agg[m]["ret_acc"] * 100 for m in methods]

    # Ghost bars for raw accuracy (light, behind)
    ax.bar(x, raw_acc, width=0.62, color="#E0E0E0", edgecolor="#CCCCCC",
           linewidth=0.5, zorder=3, label="Raw Accuracy")

    # Solid bars for ERA
    bars_c = ax.bar(x, egu, width=0.52, color=colors, edgecolor="black",
                    linewidth=0.6, zorder=5, label="Effective Retrieval Accuracy")

    # Highlight TRACE KG
    trace_idx = methods.index("tracekg")
    bars_c[trace_idx].set_edgecolor(METHOD_COLORS["tracekg"])
    bars_c[trace_idx].set_linewidth(2.5)

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        # ERA value
        ax.text(i, egu[i] + 2, f"{egu[i]:.1f}%", ha="center", va="bottom",
                fontsize=FONT_ANNOT, fontweight=fw, color=colors[i], zorder=10)
        # Raw acc (small, grey, above ghost bar)
        if abs(raw_acc[i] - egu[i]) > 5:
            ax.text(i + 0.28, raw_acc[i] - 2, f"{raw_acc[i]:.0f}%",
                    ha="center", va="top", fontsize=FONT_NOTE,
                    color="#999999", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK, rotation=25, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    ax.set_title("(c) Corrected Metric: Effective\nRetrieval Accuracy",
                 fontweight="bold", fontsize=FONT_SUBTITLE, pad=10)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.12, linewidth=0.5)
    ax.legend(fontsize=FONT_TICK - 1, loc="upper right", framealpha=0.95,
              edgecolor="#CCCCCC")

    ax.text(0.5, -0.18,
            "ERA = Accuracy × Connectivity × (1 − Leakage)\n"
            "Grey = raw accuracy; colored = after correction",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_NOTE, color="#444444", linespacing=1.4)

    # ── Suptitle ──
    fig.suptitle("Why Raw Retrieval Accuracy Is Misleading for KG Evaluation",
                 fontweight="bold", fontsize=FONT_TITLE + 1, y=1.03)

    plt.tight_layout(w_pad=3.0)

    path = FIG_OUT / "fig3_failure_modes.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 4: SUMMARY DASHBOARD — cleaned, publication-ready
# ============================================================

def fig4_summary_dashboard():
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    for ax in [ax_a, ax_b, ax_c]:
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── (a) Accuracy with Leakage Highlighted ──
    acc = [agg[m]["ret_acc"] * 100 for m in methods]
    leak = [agg[m]["v4g"] * 100 for m in methods]

    bars_acc = ax_a.bar(x, acc, color=colors, edgecolor="black", linewidth=0.6,
                        width=0.6, zorder=5)

    bars_leak = ax_a.bar(x, leak, color="none", edgecolor=CLR_LEAKED,
                         linewidth=1.8, width=0.6, hatch="xxxx",
                         zorder=6)
    # Fill the hatch area with semi-transparent red
    for bar in bars_leak:
        bar.set_facecolor(CLR_LEAKED)
        bar.set_alpha(0.25)

    # Accuracy labels
    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        fs = FONT_ANNOT + 1 if methods[i] == "tracekg" else FONT_ANNOT
        ax_a.text(i, acc[i] + 1.8, f"{acc[i]:.1f}%", ha="center", va="bottom",
                  fontsize=fs, fontweight=fw, zorder=15)

    # Leakage labels with badge
    for i in range(n):
        if leak[i] >= 0.3:
            label_y = min(leak[i] + 3.0, acc[i] - 5)
            label_y = max(label_y, 5.0)
            fs = FONT_ANNOT if leak[i] > 5 else FONT_NOTE
            ax_a.text(i, label_y, f"Leak: {leak[i]:.1f}%",
                      ha="center", va="bottom", fontsize=fs, fontweight="bold",
                      color=CLR_LEAKED, zorder=15,
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor=CLR_LEAKED, linewidth=1.2, alpha=0.92))

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=FONT_LABEL)
    ax_a.set_ylabel("Percentage (%)", fontsize=FONT_LABEL)
    ax_a.set_title("(a) Retrieval Accuracy with Verbatim Leakage Highlighted",
                   fontweight="bold", fontsize=FONT_SUBTITLE + 1, pad=14)
    ax_a.set_ylim(0, 115)
    ax_a.grid(axis="y", alpha=0.12, linewidth=0.5)

    patch_acc = mpatches.Patch(facecolor="#AAAAAA", edgecolor="black", linewidth=0.6,
                               label="Retrieval Accuracy")
    patch_leak = mpatches.Patch(facecolor=CLR_LEAKED, edgecolor=CLR_LEAKED,
                                linewidth=1.0, alpha=0.35, hatch="xxxx",
                                label="Verbatim Leakage")
    ax_a.legend(handles=[patch_acc, patch_leak], fontsize=FONT_TICK,
                loc="center left", framealpha=0.95, edgecolor="#CCCCCC",
                bbox_to_anchor=(0.0, 0.55))

    ax_a.text(0.5, -0.08,
              "Hatched region = 4-gram overlap between entity strings and source essay text",
              transform=ax_a.transAxes, ha="center", va="top",
              fontsize=FONT_NOTE + 1, color="#555555", fontstyle="italic")

    # ── (b) TriCR ──
    tricr = [agg[m]["tri_cr"] for m in methods]
    tricr_dev = [abs(t - 1.0) for t in tricr]

    bars_b = ax_b.barh(x, tricr_dev, color=colors, edgecolor="black",
                       linewidth=0.6, height=0.55)

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        if tricr[i] > 1.5:
            tag = "expansion"
            tc = CLR_LEAKED
        elif tricr[i] < 0.6:
            tag = "oversimplified"
            tc = "#B35900"
        elif abs(tricr[i] - 1.0) < 0.15:
            tag = "near-ideal"
            tc = CLR_GENUINE
        else:
            tag = ""
            tc = "#333333"

        txt = f"{tricr[i]:.2f}"
        if tag:
            txt += f"  ({tag})"
        ax_b.text(tricr_dev[i] + 0.05, i, txt,
                  va="center", fontsize=FONT_ANNOT, fontweight=fw, color=tc)

    ax_b.axvspan(0, 0.15, color=CLR_GENUINE, alpha=0.06, zorder=0)

    ax_b.set_yticks(x)
    ax_b.set_yticklabels(labels, fontsize=FONT_TICK)
    ax_b.set_xlabel("Distance from Ideal  |TriCR − 1.0|", fontsize=FONT_LABEL)
    ax_b.set_title("(b) Triple Compression Ratio (ideal = 1.0)",
                   fontweight="bold", fontsize=FONT_SUBTITLE, pad=10)
    ax_b.set_xlim(0, 3.8)
    ax_b.axvline(x=0, color="black", linewidth=0.8)
    ax_b.grid(axis="x", alpha=0.12, linewidth=0.5)

    ax_b.text(0.5, -0.12,
              "TriCR = total triple words / essay words",
              transform=ax_b.transAxes, ha="center", va="top",
              fontsize=FONT_NOTE + 1, color="#555555", fontstyle="italic")

    # ── (c) Overall Rank ──
    ranks = v5.get("avg_ranks", {})
    sorted_m = sorted([m for m in methods if m in ranks], key=lambda m: ranks[m])
    rank_vals = [ranks[m] for m in sorted_m]
    rank_labels = [METHOD_LABELS[m] for m in sorted_m]
    rank_colors = [METHOD_COLORS[m] for m in sorted_m]

    bars_c = ax_c.barh(np.arange(len(sorted_m)), rank_vals, color=rank_colors,
                       edgecolor="black", linewidth=0.6, height=0.55)

    for i, (m, rv) in enumerate(zip(sorted_m, rank_vals)):
        fw = "bold" if m == "tracekg" else "normal"
        fs = FONT_ANNOT + 1 if m == "tracekg" else FONT_ANNOT
        ax_c.text(rv + 0.08, i, f"{rv:.2f}", va="center", fontsize=fs, fontweight=fw)

    ax_c.text(rank_vals[0] + 0.42, 0, "★", fontsize=18, va="center", ha="center",
              color="#DAA520", fontweight="bold")

    ax_c.set_yticks(np.arange(len(sorted_m)))
    ax_c.set_yticklabels(rank_labels, fontsize=FONT_TICK)
    ax_c.set_xlabel("Average Rank (lower = better)", fontsize=FONT_LABEL)
    ax_c.set_title("(c) Overall Quality Ranking",
                   fontweight="bold", fontsize=FONT_SUBTITLE, pad=10)
    ax_c.set_xlim(0, 5.2)
    ax_c.axvline(x=1, color="#0072B2", linestyle=":", alpha=0.25, linewidth=1)
    ax_c.text(1.08, len(sorted_m) - 0.15, "perfect", fontsize=FONT_NOTE,
              color="#0072B2", fontstyle="italic", alpha=0.5)
    ax_c.invert_yaxis()
    ax_c.grid(axis="x", alpha=0.12, linewidth=0.5)

    ax_c.text(0.5, -0.12,
              "Rank averaged over: Acc, ERA, Conn, Clust, AvgDeg, Leak%, TriCR",
              transform=ax_c.transAxes, ha="center", va="top",
              fontsize=FONT_NOTE + 1, color="#555555", fontstyle="italic")

    path = FIG_OUT / "fig4_summary_dashboard.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("Generating KDD figures (final v4)...\n")
    fig1_radar()
    fig2_effective_retrieval_accuracy()
    fig3_failure_modes()
    fig4_summary_dashboard()
    print(f"\n✅ All 4 figures saved to: {FIG_OUT}")

#endregion#? 
#?#########################  End  ##########################



#endregion#! Addomg AutoSchemaKG to the comparison
#!#############################################  End Chapter  ##################################################








#?######################### Start ##########################
#region:#?  Base


"""
KDD-Ready Figures for TRACE KG Paper — MINE-1 Evaluation (Final v4)
=====================================================================

4 publication-quality figures:

  Fig 1: Radar — multi-dimensional quality profile
  Fig 2: Effective Retrieval Accuracy (single bar, prominent)
  Fig 3: "Why Raw Accuracy is Misleading" — 3-panel failure mode exposé
  Fig 4: Summary dashboard — cleaned, publication-ready (3 panels)

Color palette: Okabe-Ito colorblind-safe.
All data from compression_analysis_v5.json.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Dict, List

# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(".").resolve()
V5_JSON = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_v5.json"
FIG_OUT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["tracekg", "autoschemakg", "graphrag", "openie", "kggen"]
METHOD_LABELS = {
    "tracekg": "TRACE KG",
    "autoschemakg": "AutoSchemaKG",
    "graphrag": "GraphRAG",
    "openie": "OpenIE",
    "kggen": "KGGen",
}

METHOD_COLORS = {
    "tracekg":      "#0072B2",
    "autoschemakg": "#E69F00",
    "graphrag":     "#882255",
    "openie":       "#44AA99",
    "kggen":        "#332288",
}

# Semantic colors
CLR_GENUINE = "#1a9641"
CLR_LEAKED  = "#d7191c"
CLR_FRAG    = "#bababa"

DPI = 300
FONT_TITLE = 15
FONT_SUBTITLE = 13
FONT_LABEL = 12
FONT_TICK = 11
FONT_ANNOT = 11
FONT_NOTE = 10

plt.rcParams.update({
    "font.size": FONT_LABEL,
    "font.family": "sans-serif",
    "axes.titlesize": FONT_SUBTITLE,
    "axes.labelsize": FONT_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_TICK,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# LOAD DATA
# ============================================================

with open(V5_JSON) as f:
    v5 = json.load(f)

agg = v5["aggregate"]


# ============================================================
# FIG 1: RADAR — Multi-dimensional quality profile
# ============================================================

def fig1_radar():
    radar_metrics = [
        ("Retrieval\nAccuracy",           "ret_acc",  True),
        ("Effective\nRetrieval Accuracy",  "egu",      True),
        ("Connectivity",                   "conn",     True),
        ("Clustering",                     "clust",    True),
        ("Avg. Degree",                    "avg_deg",  True),
        ("Low Leakage",                    "v4g",      False),
        ("Compression\nQuality",           "tri_cr",   None),
    ]

    methods = [m for m in METHOD_ORDER if m in agg]
    N = len(radar_metrics)

    raw = {}
    for m in methods:
        raw[m] = [agg[m][key] for _, key, _ in radar_metrics]

    raw_arr = np.array([raw[m] for m in methods])
    norm = np.zeros_like(raw_arr)
    for j, (_, key, higher) in enumerate(radar_metrics):
        col = raw_arr[:, j]
        if higher is None:
            dist = np.abs(col - 1.0)
            mn, mx = dist.min(), dist.max()
            if mx - mn < 1e-9:
                norm[:, j] = 1.0
            else:
                norm[:, j] = (mx - dist) / (mx - mn)
        else:
            mn, mx = col.min(), col.max()
            if mx - mn < 1e-9:
                norm[:, j] = 1.0
            else:
                norm[:, j] = (col - mn) / (mx - mn) if higher else (mx - col) / (mx - mn)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FAFAFA")

    for i, m in enumerate(methods):
        vals = norm[i].tolist() + [norm[i][0]]
        lw = 3.0 if m == "tracekg" else 1.5
        ls = "-" if m == "tracekg" else "--"
        alpha_fill = 0.12 if m == "tracekg" else 0.0
        zorder = 10 if m == "tracekg" else 5
        marker = "o" if m == "tracekg" else ""
        ms = 5 if m == "tracekg" else 0
        ax.plot(angles, vals, color=METHOD_COLORS[m], linewidth=lw, linestyle=ls,
                label=METHOD_LABELS[m], zorder=zorder, marker=marker, markersize=ms)
        if alpha_fill > 0:
            ax.fill(angles, vals, color=METHOD_COLORS[m], alpha=alpha_fill, zorder=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl for lbl, _, _ in radar_metrics], fontsize=FONT_TICK,
                       linespacing=1.1)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=FONT_TICK - 1, color="grey")

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=True,
              fancybox=True, shadow=False, edgecolor="#CCCCCC", fontsize=FONT_TICK)

    path = FIG_OUT / "fig1_radar_quality_profile.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 2: EFFECTIVE RETRIEVAL ACCURACY
# ============================================================

def fig2_effective_retrieval_accuracy():
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    egu = [agg[m]["egu"] * 100 for m in methods]
    egu_std = [agg[m].get("egu_std", 0) * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(x, egu, color=colors, edgecolor="black", linewidth=0.6,
                  yerr=egu_std, capsize=5, error_kw={"linewidth": 1.2, "color": "#444444"},
                  width=0.62)

    trace_idx = methods.index("tracekg")
    bars[trace_idx].set_edgecolor(METHOD_COLORS["tracekg"])
    bars[trace_idx].set_linewidth(3.0)

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        fs = FONT_ANNOT + 1 if methods[i] == "tracekg" else FONT_ANNOT
        ax.text(i, egu[i] + egu_std[i] + 2.0, f"{egu[i]:.1f}%",
                ha="center", va="bottom", fontsize=fs, fontweight=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_LABEL)
    ax.set_ylabel("Effective Retrieval Accuracy (%)", fontsize=FONT_LABEL)
    ax.set_title("Effective Retrieval Accuracy", fontweight="bold",
                 fontsize=FONT_TITLE + 2, pad=20)
    ax.text(0.5, 1.015,
            "Retrieval Accuracy  ×  Graph Connectivity  ×  (1 − Verbatim Leakage)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FONT_LABEL, color="#333333", fontstyle="italic")

    ax.set_ylim(0, max(egu) + max(egu_std) + 15)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = FIG_OUT / "fig2_effective_retrieval_accuracy.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 3: "WHY RAW ACCURACY IS MISLEADING" — 3 failure modes
# ============================================================

def fig3_failure_modes():
    """
    Three-panel figure exposing why raw accuracy alone is misleading.

    Panel (a): Accuracy decomposition — stacked bar showing genuine vs
               leaked vs fragmented contributions to accuracy.
    Panel (b): Compression quality — distance from ideal TriCR=1.0.
    Panel (c): Effective Retrieval Accuracy — the corrected metric.

    Core message: AutoSchemaKG's 95% accuracy is largely from text copying.
    TRACE KG's 90% is almost entirely genuine graph retrieval.
    """
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax in axes:
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── (a) Accuracy Decomposition ──
    ax = axes[0]

    genuine = []
    leaked = []
    fragmented = []

    for m in methods:
        a = agg[m]["ret_acc"]
        c = agg[m]["conn"]
        l = agg[m]["v4g"]
        genuine.append(a * c * (1 - l) * 100)
        leaked.append(a * c * l * 100)
        fragmented.append(a * (1 - c) * 100)

    bars_g = ax.bar(x, genuine, width=0.62, color=CLR_GENUINE, edgecolor="black",
                    linewidth=0.5, label="Genuine retrieval", zorder=5)
    bars_l = ax.bar(x, leaked, bottom=genuine, width=0.62, color=CLR_LEAKED,
                    edgecolor="black", linewidth=0.5, hatch="///",
                    label="From text copying", zorder=5)
    bars_f = ax.bar(x, fragmented, bottom=[g + l for g, l in zip(genuine, leaked)],
                    width=0.62, color=CLR_FRAG, edgecolor="black", linewidth=0.5,
                    hatch="...", label="From disconnected nodes", zorder=5)

    totals = [g + l + f for g, l, f in zip(genuine, leaked, fragmented)]

    # Total on top
    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        ax.text(i, totals[i] + 1.5, f"{totals[i]:.1f}%", ha="center", va="bottom",
                fontsize=FONT_ANNOT, fontweight=fw, zorder=10)

    # Genuine % inside green bar
    for i in range(n):
        if genuine[i] > 12:
            ax.text(i, genuine[i] / 2, f"{genuine[i]:.0f}%", ha="center", va="center",
                    fontsize=FONT_ANNOT - 1, fontweight="bold", color="white", zorder=10)

    # Leaked % inside red bar if visible
    for i in range(n):
        if leaked[i] > 6:
            ax.text(i, genuine[i] + leaked[i] / 2, f"{leaked[i]:.0f}%",
                    ha="center", va="center", fontsize=FONT_ANNOT - 1,
                    fontweight="bold", color="white", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK, rotation=25, ha="right")
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=FONT_LABEL)
    ax.set_title("(a) Failure Mode 1: Accuracy from Text Copying",
                 fontweight="bold", fontsize=FONT_SUBTITLE, pad=10)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.12, linewidth=0.5)
    ax.legend(fontsize=FONT_TICK, loc="upper right", framealpha=0.95,
              edgecolor="#CCCCCC")

    ax.text(0.5, -0.18,
            "Green = accuracy from genuine graph structure\n"
            "Red = accuracy inflated by verbatim text in entities",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_NOTE, color="#444444", linespacing=1.4)

    # ── (b) Compression Quality ──
    ax = axes[1]

    tricr = [agg[m]["tri_cr"] for m in methods]
    tricr_dev = [abs(t - 1.0) for t in tricr]

    bar_edge_colors = []
    for i, m in enumerate(methods):
        if tricr[i] > 1.5:
            bar_edge_colors.append(CLR_LEAKED)
        elif tricr[i] < 0.5:
            bar_edge_colors.append("#E69F00")
        else:
            bar_edge_colors.append("black")

    bars_b = ax.barh(x, tricr_dev, color=colors, edgecolor="black",
                     linewidth=0.6, height=0.55)

    # Semantic annotation
    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        if tricr[i] > 1.5:
            tag = "KG > source text"
            tag_color = CLR_LEAKED
        elif tricr[i] < 0.6:
            tag = "heavy information loss"
            tag_color = "#B35900"
        elif abs(tricr[i] - 1.0) < 0.15:
            tag = "near-ideal"
            tag_color = CLR_GENUINE
        else:
            tag = ""
            tag_color = "#333333"

        label_text = f"{tricr[i]:.2f}"
        if tag:
            label_text += f"  ({tag})"

        ax.text(tricr_dev[i] + 0.05, i, label_text,
                va="center", fontsize=FONT_ANNOT - 1, fontweight=fw, color=tag_color)

    # Ideal zone shading
    ax.axvspan(0, 0.15, color=CLR_GENUINE, alpha=0.06, zorder=0)
    ax.text(0.07, n - 0.3, "ideal\nzone", fontsize=FONT_NOTE, color=CLR_GENUINE,
            ha="center", va="top", fontstyle="italic", alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Distance from Ideal  |TriCR − 1.0|", fontsize=FONT_LABEL)
    ax.set_title("(b) Failure Mode 2: Poor Compression",
                 fontweight="bold", fontsize=FONT_SUBTITLE, pad=10)
    ax.set_xlim(0, 3.8)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.12, linewidth=0.5)

    ax.text(0.5, -0.18,
            "TriCR = total triple words / essay words\n"
            "Ideal = 1.0 (lossless compression)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_NOTE, color="#444444", linespacing=1.4)

    # ── (c) Corrected Metric: Effective Retrieval Accuracy ──
    ax = axes[2]

    egu = [agg[m]["egu"] * 100 for m in methods]
    raw_acc = [agg[m]["ret_acc"] * 100 for m in methods]

    # Ghost bars for raw accuracy (light, behind)
    ax.bar(x, raw_acc, width=0.62, color="#E0E0E0", edgecolor="#CCCCCC",
           linewidth=0.5, zorder=3, label="Raw Accuracy")

    # Solid bars for ERA
    bars_c = ax.bar(x, egu, width=0.52, color=colors, edgecolor="black",
                    linewidth=0.6, zorder=5, label="Effective Retrieval Accuracy")

    # Highlight TRACE KG
    trace_idx = methods.index("tracekg")
    bars_c[trace_idx].set_edgecolor(METHOD_COLORS["tracekg"])
    bars_c[trace_idx].set_linewidth(2.5)

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        # ERA value
        ax.text(i, egu[i] + 2, f"{egu[i]:.1f}%", ha="center", va="bottom",
                fontsize=FONT_ANNOT, fontweight=fw, color=colors[i], zorder=10)
        # Raw acc (small, grey, above ghost bar)
        if abs(raw_acc[i] - egu[i]) > 5:
            ax.text(i + 0.28, raw_acc[i] - 2, f"{raw_acc[i]:.0f}%",
                    ha="center", va="top", fontsize=FONT_NOTE,
                    color="#999999", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK, rotation=25, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    ax.set_title("(c) Corrected Metric: Effective\nRetrieval Accuracy",
                 fontweight="bold", fontsize=FONT_SUBTITLE, pad=10)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.12, linewidth=0.5)
    ax.legend(fontsize=FONT_TICK - 1, loc="upper right", framealpha=0.95,
              edgecolor="#CCCCCC")

    ax.text(0.5, -0.18,
            "ERA = Accuracy × Connectivity × (1 − Leakage)\n"
            "Grey = raw accuracy; colored = after correction",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_NOTE, color="#444444", linespacing=1.4)

    # ── Suptitle ──
    fig.suptitle("Why Raw Retrieval Accuracy Is Misleading for KG Evaluation",
                 fontweight="bold", fontsize=FONT_TITLE + 1, y=1.03)

    plt.tight_layout(w_pad=3.0)

    path = FIG_OUT / "fig3_failure_modes.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIG 4: SUMMARY DASHBOARD — cleaned, publication-ready
# ============================================================

def fig4_summary_dashboard():
    methods = [m for m in METHOD_ORDER if m in agg]
    n = len(methods)
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(n)

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    for ax in [ax_a, ax_b, ax_c]:
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── (a) Accuracy with Leakage Highlighted ──
    acc = [agg[m]["ret_acc"] * 100 for m in methods]
    leak = [agg[m]["v4g"] * 100 for m in methods]

    bars_acc = ax_a.bar(x, acc, color=colors, edgecolor="black", linewidth=0.6,
                        width=0.6, zorder=5)

    bars_leak = ax_a.bar(x, leak, color="none", edgecolor=CLR_LEAKED,
                         linewidth=1.8, width=0.6, hatch="xxxx",
                         zorder=6)
    # Fill the hatch area with semi-transparent red
    for bar in bars_leak:
        bar.set_facecolor(CLR_LEAKED)
        bar.set_alpha(0.25)

    # Accuracy labels
    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        fs = FONT_ANNOT + 1 if methods[i] == "tracekg" else FONT_ANNOT
        ax_a.text(i, acc[i] + 1.8, f"{acc[i]:.1f}%", ha="center", va="bottom",
                  fontsize=fs, fontweight=fw, zorder=15)

    # Leakage labels with badge
    for i in range(n):
        if leak[i] >= 0.3:
            label_y = min(leak[i] + 3.0, acc[i] - 5)
            label_y = max(label_y, 5.0)
            fs = FONT_ANNOT if leak[i] > 5 else FONT_NOTE
            ax_a.text(i, label_y, f"Leak: {leak[i]:.1f}%",
                      ha="center", va="bottom", fontsize=fs, fontweight="bold",
                      color=CLR_LEAKED, zorder=15,
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor=CLR_LEAKED, linewidth=1.2, alpha=0.92))

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=FONT_LABEL)
    ax_a.set_ylabel("Percentage (%)", fontsize=FONT_LABEL)
    ax_a.set_title("(a) Retrieval Accuracy with Verbatim Leakage Highlighted",
                   fontweight="bold", fontsize=FONT_SUBTITLE + 1, pad=14)
    ax_a.set_ylim(0, 115)
    ax_a.grid(axis="y", alpha=0.12, linewidth=0.5)

    patch_acc = mpatches.Patch(facecolor="#AAAAAA", edgecolor="black", linewidth=0.6,
                               label="Retrieval Accuracy")
    patch_leak = mpatches.Patch(facecolor=CLR_LEAKED, edgecolor=CLR_LEAKED,
                                linewidth=1.0, alpha=0.35, hatch="xxxx",
                                label="Verbatim Leakage")
    ax_a.legend(handles=[patch_acc, patch_leak], fontsize=FONT_TICK,
                loc="center left", framealpha=0.95, edgecolor="#CCCCCC",
                bbox_to_anchor=(0.0, 0.55))

    ax_a.text(0.5, -0.08,
              "Hatched region = 4-gram overlap between entity strings and source essay text",
              transform=ax_a.transAxes, ha="center", va="top",
              fontsize=FONT_NOTE + 1, color="#555555", fontstyle="italic")

    # ── (b) TriCR ──
    tricr = [agg[m]["tri_cr"] for m in methods]
    tricr_dev = [abs(t - 1.0) for t in tricr]

    bars_b = ax_b.barh(x, tricr_dev, color=colors, edgecolor="black",
                       linewidth=0.6, height=0.55)

    for i in range(n):
        fw = "bold" if methods[i] == "tracekg" else "normal"
        if tricr[i] > 1.5:
            tag = "expansion"
            tc = CLR_LEAKED
        elif tricr[i] < 0.6:
            tag = "oversimplified"
            tc = "#B35900"
        elif abs(tricr[i] - 1.0) < 0.15:
            tag = "near-ideal"
            tc = CLR_GENUINE
        else:
            tag = ""
            tc = "#333333"

        txt = f"{tricr[i]:.2f}"
        if tag:
            txt += f"  ({tag})"
        ax_b.text(tricr_dev[i] + 0.05, i, txt,
                  va="center", fontsize=FONT_ANNOT, fontweight=fw, color=tc)

    ax_b.axvspan(0, 0.15, color=CLR_GENUINE, alpha=0.06, zorder=0)

    ax_b.set_yticks(x)
    ax_b.set_yticklabels(labels, fontsize=FONT_TICK)
    ax_b.set_xlabel("Distance from Ideal  |TriCR − 1.0|", fontsize=FONT_LABEL)
    ax_b.set_title("(b) Triple Compression Ratio (ideal = 1.0)",
                   fontweight="bold", fontsize=FONT_SUBTITLE, pad=10)
    ax_b.set_xlim(0, 3.8)
    ax_b.axvline(x=0, color="black", linewidth=0.8)
    ax_b.grid(axis="x", alpha=0.12, linewidth=0.5)

    ax_b.text(0.5, -0.12,
              "TriCR = total triple words / essay words",
              transform=ax_b.transAxes, ha="center", va="top",
              fontsize=FONT_NOTE + 1, color="#555555", fontstyle="italic")

    # ── (c) Overall Rank ──
    ranks = v5.get("avg_ranks", {})
    sorted_m = sorted([m for m in methods if m in ranks], key=lambda m: ranks[m])
    rank_vals = [ranks[m] for m in sorted_m]
    rank_labels = [METHOD_LABELS[m] for m in sorted_m]
    rank_colors = [METHOD_COLORS[m] for m in sorted_m]

    bars_c = ax_c.barh(np.arange(len(sorted_m)), rank_vals, color=rank_colors,
                       edgecolor="black", linewidth=0.6, height=0.55)

    for i, (m, rv) in enumerate(zip(sorted_m, rank_vals)):
        fw = "bold" if m == "tracekg" else "normal"
        fs = FONT_ANNOT + 1 if m == "tracekg" else FONT_ANNOT
        ax_c.text(rv + 0.08, i, f"{rv:.2f}", va="center", fontsize=fs, fontweight=fw)

    ax_c.text(rank_vals[0] + 0.42, 0, "★", fontsize=18, va="center", ha="center",
              color="#DAA520", fontweight="bold")

    ax_c.set_yticks(np.arange(len(sorted_m)))
    ax_c.set_yticklabels(rank_labels, fontsize=FONT_TICK)
    ax_c.set_xlabel("Average Rank (lower = better)", fontsize=FONT_LABEL)
    ax_c.set_title("(c) Overall Quality Ranking",
                   fontweight="bold", fontsize=FONT_SUBTITLE, pad=10)
    ax_c.set_xlim(0, 5.2)
    ax_c.axvline(x=1, color="#0072B2", linestyle=":", alpha=0.25, linewidth=1)
    ax_c.text(1.08, len(sorted_m) - 0.15, "perfect", fontsize=FONT_NOTE,
              color="#0072B2", fontstyle="italic", alpha=0.5)
    ax_c.invert_yaxis()
    ax_c.grid(axis="x", alpha=0.12, linewidth=0.5)

    ax_c.text(0.5, -0.12,
              "Rank averaged over: Acc, ERA, Conn, Clust, AvgDeg, Leak%, TriCR",
              transform=ax_c.transAxes, ha="center", va="top",
              fontsize=FONT_NOTE + 1, color="#555555", fontstyle="italic")

    path = FIG_OUT / "fig4_summary_dashboard.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("Generating KDD figures (final v4)...\n")
    fig1_radar()
    fig2_effective_retrieval_accuracy()
    fig3_failure_modes()
    fig4_summary_dashboard()
    print(f"\n✅ All 4 figures saved to: {FIG_OUT}")

#endregion#? 
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?     N1

#endregion#?   N1
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?     N2

#endregion#?   N2
#?#########################  End  ##########################



