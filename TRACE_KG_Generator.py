

import Trace_KG

#?######################### Start ##########################
#region:#?   Pipeline for producing 100 KG - V9

#todo: The ID on the folder name and ID used in TRACE_KG_per_essay_stats.json, 
#todo  should be checked because sometimes they don't match the real essay ID because they are not always incremnted by 1. 
#!For the evaluation though, it turned out that the problem is with the dataset used for evaluation (huggingface one: Topic does not match their produced KG and responses)!


#!/usr/bin/env python3
"""
Drive the full TRACE KG pipeline in Trace_KG.py for a SELECTED set of essays.

Pipeline per essay:

1) Write essay i into:
       data/pdf_to_json/Plain_Text.json

2) Entity & Class pipeline (in this order, STOP on first failure):
       sentence_chunks_token_driven(...)
       embed_and_index_chunks(...)
       run_entity_extraction_on_chunks(...)
       iterative_resolution()
       produce_clean_jsonl(...)
       classrec_iterative_main()
       main_input_for_cls_res()
       run_pipeline_iteratively()          # ClassRes multi-run

3) Relation pipeline (strict order, STOP on first failure):
       run_rel_rec(...)                    # writes data/Relations/Rel Rec/relations_raw.jsonl
       run_relres_iteratively()            # reads relations_raw.jsonl and resolves relations

4) KG export:
       export_relations_and_nodes_to_csv()

5) Snapshot data/ into:
       KGs_from_Essays/KG_Essay_{i} or ..._FAILED

6) Clear only:
       data/Chunks, data/Classes, data/Entities, data/KG, data/Relations

We ALWAYS write the global stats file at the end, even if we stop early.

Assumption: run this script from the SGCE-KG repo root.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tqdm import tqdm

from Trace_KG import (  
    sentence_chunks_token_driven,
    embed_and_index_chunks,
    run_entity_extraction_on_chunks,
    iterative_resolution,
    produce_clean_jsonl,
    classrec_iterative_main,
    main_input_for_cls_res,
    run_pipeline_iteratively,
    run_rel_rec,
    run_relres_iteratively,
    export_relations_and_nodes_to_csv,
)

import inspect
print("TRACE_KG loaded from:", inspect.getfile(sentence_chunks_token_driven))

# ------------------------------------------------------------------------------------
# CONFIG PATHS
# ------------------------------------------------------------------------------------

REPO_ROOT = Path(".").resolve()

PLAIN_TEXT_JSON = REPO_ROOT / "data/pdf_to_json/Plain_Text.json"
ESSAYS_JSON = REPO_ROOT / "data/In_Plain_Text.json"
DATA_DIR = REPO_ROOT / "data"
KG_OUT_ROOT = REPO_ROOT / "KGs_from_Essays"

DATA_SUBDIRS_TO_CLEAR = [
    DATA_DIR / "Chunks",
    DATA_DIR / "Classes",
    DATA_DIR / "Entities",
    DATA_DIR / "KG",
    DATA_DIR / "Relations",
]

# Instead of ESSAY_START / ESSAY_END, specify exactly which essays to run (1‑based indices).
# Example: run essays 2 and 5 only:
ESSAY_IDS: List[int] = [87, 123, 23, 64, 46, 52, 84, 10, 51, 15]   # <--- edit this list as needed


# ------------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clear_subdir_contents(path: Path) -> None:
    ensure_dir(path)
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except FileNotFoundError:
                pass


def clear_pipeline_state() -> None:
    for sub in DATA_SUBDIRS_TO_CLEAR:
        clear_subdir_contents(sub)


def copy_data_for_essay(essay_index: int, ok: bool) -> Path:
    ensure_dir(KG_OUT_ROOT)
    suffix = "" if ok else "_FAILED"
    dest = KG_OUT_ROOT / f"KG_Essay_{essay_index:03d}{suffix}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(DATA_DIR, dest)
    return dest


def load_essays(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("essays", "data", "items", "documents"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Cannot interpret essays structure in {path}")


def write_single_plain_text_json(essay: Dict[str, Any]) -> None:
    ensure_dir(PLAIN_TEXT_JSON.parent)

    text = essay.get("text") or essay.get("content") or ""
    title = essay.get("title") or essay.get("id") or f"essay_{essay.get('index', '')}"

    payload = [
        {
            "title": str(title),
            "text": str(text),
            "start_page": 1,
            "end_page": 1,
        }
    ]

    with PLAIN_TEXT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------------------------
# MAIN PER‑ESSAY PIPELINE WRAPPER
# ------------------------------------------------------------------------------------

def run_full_pipeline_for_current_plain_text() -> Dict[str, Any]:
    """
    Run all pipeline steps in order, STOPPING at the first failure.
    """
    stats: Dict[str, Any] = {
        "steps": {},
        "ok": True,
        "error": None,
    }
    
    def _run_step(name: str, fn, *args, **kwargs) -> bool:
        """
        Run one step, record timing and error.
        Return True if succeeded, False if failed.
        """
        import traceback

        t0 = time.time()
        step_info: Dict[str, Any] = {"ok": True, "error": None, "seconds": None}
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print(f"\n[STEP ERROR] {name} raised an exception:")
            traceback.print_exc()

            step_info["ok"] = False
            step_info["error"] = repr(e)
            if stats["ok"]:
                stats["ok"] = False
                stats["error"] = f"{name} failed: {e}"
        finally:
            step_info["seconds"] = time.time() - t0
            stats["steps"][name] = step_info
        return step_info["ok"]

    # 1) Chunking
    if not _run_step(
        "sentence_chunks_token_driven",
        sentence_chunks_token_driven,
        str(PLAIN_TEXT_JSON),
        "data/Chunks/chunks_sentence.jsonl",
        max_tokens_per_chunk=200,
        min_tokens_per_chunk=100,
        sentence_per_line=True,
        keep_ref_text=False,
        strip_leading_headings=True,
        force=True,
        debug=False,
    ):
        return stats  # stop on first failure

    # 2) Embed + index chunks
    if not _run_step(
        "embed_and_index_chunks",
        embed_and_index_chunks,
        "data/Chunks/chunks_sentence.jsonl",
        "data/Chunks/chunks_emb",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        False,
        32,
        None,
        True,
        True,
    ):
        return stats

    # 3) Entity Recognition
    if not _run_step(
        "run_entity_extraction_on_chunks",
        run_entity_extraction_on_chunks,
        chunk_ids=None,
        prev_chunks=4,
        save_debug=False,
        model="gpt-5.1",
        max_tokens=8000,
    ):
        return stats

    # 4) Iterative entity resolution
    if not _run_step("iterative_resolution", iterative_resolution):
        return stats

    # 5) Class‑rec input producer
    if not _run_step(
        "produce_clean_jsonl",
        produce_clean_jsonl,
        None,
        None,
    ):
        return stats

    # 6) Class Recognition
    if not _run_step("classrec_iterative_main", classrec_iterative_main):
        return stats

    # 7) Create input for Cls Res
    if not _run_step("main_input_for_cls_res", main_input_for_cls_res):
        return stats

    # 8) Class Res Multi Run
    if not _run_step("run_pipeline_iteratively", run_pipeline_iteratively):
        return stats

    # 9) Relation Recognition (Rel Rec) – produce relations_raw.jsonl
    entities_with_class_primary = Path(
        "data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl"
    )
    entities_with_class_fallback = Path(
        "data/Relations/Rel Res_IterativeRuns/overall_summary/entities_with_class.jsonl"
    )

    if entities_with_class_primary.exists():
        entities_with_class_path = entities_with_class_primary
    elif entities_with_class_fallback.exists():
        entities_with_class_path = entities_with_class_fallback
    else:
        stats["ok"] = False
        stats["error"] = (
            "run_rel_rec skipped: entities_with_class.jsonl not found at either "
            f"{entities_with_class_primary} or {entities_with_class_fallback}. "
            "This means neither ClassRes nor RelRes multi-runs produced their overall_summary outputs."
        )
        return stats

    if not _run_step(
        "run_rel_rec",
        run_rel_rec,
        entities_path=str(entities_with_class_path),
        chunks_path="data/Chunks/chunks_sentence.jsonl",
        output_path="data/Relations/Rel Rec/relations_raw.jsonl",
        model="gpt-5.1",
    ):
        return stats

    # 10) Relation Res Multi Run over the recognized relations (relations_raw.jsonl)
    if not _run_step("run_relres_iteratively", run_relres_iteratively):
        return stats

    # 11) Export KG to CSVs
    if not _run_step("export_relations_and_nodes_to_csv", export_relations_and_nodes_to_csv):
        return stats

    return stats


# ------------------------------------------------------------------------------------
# ORCHESTRATOR OVER SELECTED ESSAYS
# ------------------------------------------------------------------------------------

def main():
    print("CWD:", os.getcwd())
    print("Trace_KG imported from:", sentence_chunks_token_driven.__module__)
    
    ensure_dir(KG_OUT_ROOT)

    try:
        essays = load_essays(ESSAYS_JSON)
    except Exception as e:
        print(f"FATAL: cannot load essays from {ESSAYS_JSON}: {e}")
        log_path = KG_OUT_ROOT / "Trace_KG_per_essay_stats.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump({"_fatal_error": repr(e)}, f, ensure_ascii=False, indent=2)
        return

    # total = len(essays)

    # # Filter essays by explicit ESSAY_IDS (1-based indices)
    # requested = sorted(set(ESSAY_IDS))
    # indexed: List[Tuple[int, Dict[str, Any]]] = [
    #     (i + 1, essays[i]) for i in range(total)
    #     if (i + 1) in requested
    # ]
        
    total = len(essays)

    # Map from explicit essay ID -> essay dict.
    # We require that each essay has a unique integer "id".
    id_to_essay: Dict[int, Dict[str, Any]] = {}
    for e in essays:
        if "id" not in e:
            raise ValueError(
                f'Essay is missing required "id" field: {e.get("title") or e}'
            )
        try:
            eid = int(e["id"])
        except Exception:
            raise ValueError(f'Essay has non-integer "id": {e["id"]!r}')

        if eid in id_to_essay:
            raise ValueError(f"Duplicate essay id {eid} in In_Plain_Text.json")
        id_to_essay[eid] = e

    # Filter essays by explicit ESSAY_IDS (explicit IDs, not positions)
    requested = sorted(set(ESSAY_IDS))

    indexed: List[Tuple[int, Dict[str, Any]]] = [
        (eid, id_to_essay[eid])
        for eid in requested
        if eid in id_to_essay
    ]




    print(f"Found {total} essays in source JSON.")
    print(f"Requested essay IDs: {requested}")
    print(f"Total to process now: {len(indexed)}")

    global_stats: Dict[int, Dict[str, Any]] = {}

    for essay_idx, essay in tqdm(indexed, desc="Essays", unit="essay"):
        print(f"\n================ Essay {essay_idx} =================\n")
        t0_essay = time.time()

        clear_pipeline_state()

        try:
            write_single_plain_text_json(essay)
        except Exception as e:
            print(f"[Essay {essay_idx}] ERROR writing Plain_Text.json: {e}")
            stats = {
                "ok": False,
                "error": f"Failed to write Plain_Text.json: {e}",
                "steps": {},
                "seconds_total": time.time() - t0_essay,
                "snapshot_dir": None,
            }
            snap_dir = copy_data_for_essay(essay_idx, ok=False)
            stats["snapshot_dir"] = str(snap_dir)
            global_stats[essay_idx] = stats
            continue

        stats = run_full_pipeline_for_current_plain_text()
        essay_ok = stats.get("ok", False)

        snapshot_dir = copy_data_for_essay(essay_idx, ok=essay_ok)

        clear_pipeline_state()

        stats["seconds_total"] = time.time() - t0_essay
        stats["snapshot_dir"] = str(snapshot_dir)
        global_stats[essay_idx] = stats

        if essay_ok:
            print(f"[Essay {essay_idx}] ✅ Completed in {stats['seconds_total']:.1f}s; snapshot: {snapshot_dir}")
        else:
            print(f"[Essay {essay_idx}] ❌ FAILED (stopped at first failing step). Snapshot: {snapshot_dir}")
            print(f"  Error: {stats.get('error')}")

    log_path = KG_OUT_ROOT / "Trace_KG_per_essay_stats.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)

    print(f"\nAll done. Per‑essay stats written to: {log_path}")


if __name__ == "__main__":
    main()

#endregion#? Pipeline for producing 100 KG - V9
#?#########################  End  ##########################
