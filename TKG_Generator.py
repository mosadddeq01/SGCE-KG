

import TKG_Main


#?######################### Start ##########################
#region:#?   Pipeline for producing KG - V11 (signature-aware DSPy LLM Config)



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
       iterative_resolution(...)
       produce_clean_jsonl(...)
       classrec_iterative_main(...)
       main_input_for_cls_res(...)
       run_pipeline_iteratively(...)        # ClassRes multi-run

3) Relation pipeline (strict order, STOP on first failure):
       run_rel_rec(...)                    # writes data/Relations/Rel Rec/relations_raw.jsonl
       run_relres_iteratively(...)         # reads relations_raw.jsonl and resolves relations

4) KG export:
       export_relations_and_nodes_to_csv(...)

5) Snapshot data/ into:
       KGs_from_Essays/KG_Essay_{i} or ..._FAILED

6) Clear only:
       data/Chunks, data/Classes, data/Entities, data/KG, data/Relations

We ALWAYS write the global stats file at the end, even if we stop early.

Assumption: run this script from the SGCE-KG repo root.

LLM config:
- All LLM-using steps can optionally accept a TraceKGLLMConfig.
- This generator is *signature-aware*: it will pass llm_config only to steps
  whose function signatures actually accept it. Older steps keep working.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm

from TKG_Main import (  # core pipeline functions
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
    # central LLM configuration
    TraceKGLLMConfig,
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

# Instead of ESSAY_START / ESSAY_END, specify exactly which essays to run by explicit ID (from In_Plain_Text.json).
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

def run_full_pipeline_for_current_plain_text(
    llm_config: Optional[TraceKGLLMConfig] = None,
) -> Dict[str, Any]:
    """
    Run all pipeline steps in order, STOPPING at the first failure.

    llm_config:
        If provided, is passed through to TRACE KG steps that *accept* an
        llm_config parameter. Older steps that don't declare this parameter
        are automatically called without it.
    """
    stats: Dict[str, Any] = {
        "steps": {},
        "ok": True,
        "error": None,
    }
    
    def _run_step(name: str, fn, *args, **kwargs) -> bool:
        """
        Run one step, record timing and error.
        - If llm_config is in kwargs but the function signature does NOT accept
          an 'llm_config' parameter, we silently drop it.
        Return True if succeeded, False if failed.
        """
        import traceback

        t0 = time.time()
        step_info: Dict[str, Any] = {"ok": True, "error": None, "seconds": None}
        try:
            # Signature-aware: strip llm_config if fn doesn't accept it
            sig = inspect.signature(fn)
            if "llm_config" not in sig.parameters and "llm_config" in kwargs:
                # Optional debug print if you want to see this happening:
                # print(f"[debug] step {name} does not accept llm_config; ignoring it.")
                kwargs = {k: v for k, v in kwargs.items() if k != "llm_config"}
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

    # 3) Entity Recognition (supports llm_config)
    if not _run_step(
        "run_entity_extraction_on_chunks",
        run_entity_extraction_on_chunks,
        chunk_ids=None,
        prev_chunks=4,
        save_debug=False,
        model="gpt-5.1",      # kept for backward compatibility; overridden by llm_config if provided
        max_tokens=8000,
        llm_config=llm_config,
    ):
        return stats

    # 4) Iterative entity resolution (may or may not accept llm_config depending on your version)
    if not _run_step(
        "iterative_resolution",
        iterative_resolution,
        llm_config=llm_config,
    ):
        return stats

    # 5) Class‑rec input producer
    if not _run_step(
        "produce_clean_jsonl",
        produce_clean_jsonl,
        None,
        None,
    ):
        return stats

    # 6) Class Recognition (may accept llm_config in newer version)
    if not _run_step(
        "classrec_iterative_main",
        classrec_iterative_main,
        llm_config=llm_config,
    ):
        return stats

    # 7) Create input for Cls Res
    if not _run_step("main_input_for_cls_res", main_input_for_cls_res):
        return stats

    # 8) Class Res Multi Run (may accept llm_config in newer version)
    if not _run_step(
        "run_pipeline_iteratively",
        run_pipeline_iteratively,
        llm_config=llm_config,
    ):
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
        model="gpt-5.1",      # default; overridden by llm_config if step supports it
        llm_config=llm_config,
    ):
        return stats

    # 10) Relation Res Multi Run over the recognized relations (relations_raw.jsonl)
    if not _run_step(
        "run_relres_iteratively",
        run_relres_iteratively,
        llm_config=llm_config,
    ):
        return stats

    # 11) Export KG to CSVs
    if not _run_step("export_relations_and_nodes_to_csv", export_relations_and_nodes_to_csv):
        return stats

    return stats


# ------------------------------------------------------------------------------------
# ORCHESTRATOR OVER SELECTED ESSAYS
# ------------------------------------------------------------------------------------

def main(
    essay_ids: Optional[List[int]] = None,
    llm_config: Optional[TraceKGLLMConfig] = None,
) -> None:
    """
    Top-level entry point when running as a script.

    Parameters
    ----------
    essay_ids:
        If provided, overrides the global ESSAY_IDS for this run.
        Interpreted as explicit `id` values in data/In_Plain_Text.json.
    llm_config:
        Optional TraceKGLLMConfig instance to control all LLM usage.
        If None, a default config with model="gpt-5.1" (and its defaults) is used.
    """
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

    # Determine which essays to run:
    ids_to_use = essay_ids if essay_ids is not None else ESSAY_IDS
    requested = sorted(set(ids_to_use))

    indexed: List[Tuple[int, Dict[str, Any]]] = [
        (eid, id_to_essay[eid])
        for eid in requested
        if eid in id_to_essay
    ]

    print(f"Found {total} essays in source JSON.")
    print(f"Requested essay IDs: {requested}")
    print(f"Total to process now: {len(indexed)}")

    if llm_config is None:
        llm_config = TraceKGLLMConfig()  # default: gpt-5.1 everywhere

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

        stats = run_full_pipeline_for_current_plain_text(llm_config=llm_config)
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


# ------------------------------------------------------------------------------------
# High-level API: generate_trace_kgs
# ------------------------------------------------------------------------------------

def generate_trace_kgs(
    essay_ids: Optional[List[int]] = None,
    default_model: str = "gpt-5.1",
    rec_model: Optional[str] = None,
    res_model: Optional[str] = None,
    entity_rec_model: Optional[str] = None,
    entity_res_model: Optional[str] = None,
    class_rec_model: Optional[str] = None,
    class_res_model: Optional[str] = None,
    rel_rec_model: Optional[str] = None,
    rel_res_model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 16000,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    disable_cache: bool = False,
    enforce_gpt5_constraints: bool = True,
    #new
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> None:
    """
    High-level convenience function to run the full TRACE KG pipeline over selected essays.

    This lets callers configure all LLM-related knobs in one place, while keeping a
    simple call signature for typical usages.
    """
    cfg = TraceKGLLMConfig(
        default_model=default_model,
        rec_model=rec_model,
        res_model=res_model,
        entity_rec_model=entity_rec_model,
        entity_res_model=entity_res_model,
        class_rec_model=class_rec_model,
        class_res_model=class_res_model,
        rel_rec_model=rel_rec_model,
        rel_res_model=rel_res_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        api_base=api_base,
        disable_cache=disable_cache,
        enforce_gpt5_constraints=enforce_gpt5_constraints,
        # 🔹 pass through
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )

    main(essay_ids=essay_ids, llm_config=cfg)


# if __name__ == "__main__":
    # Script entry point: use defaults from ESSAY_IDS and a default config (gpt-5.1).
    # main()

#endregion#? Pipeline for producing KG - V11 (signature-aware DSPy LLM Config)
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?   Some Test Calls



# # from TRACE_KG_Generator import generate_trace_kgs

# generate_trace_kgs(
#     essay_ids=[123],
#     default_model="openai/gpt-5-nano",
# )

# from Trace_KG import TraceKGLLMConfig, main


generate_trace_kgs(
    essay_ids=[123],
    default_model="openai/gpt-4.1-nano",
)


# cfg = TraceKGLLMConfig(
#     default_model="openai/gpt-5-nano",
#     temperature=None,              # do NOT send temperature to gpt-5 family
#     max_tokens=16000,
#     reasoning_effort="low",        # forwarded for responses-style models
#     verbosity="low",               # optionally forwarded
# )
# main(essay_ids=[123], llm_config=cfg)





# cfg = TraceKGLLMConfig(
#     default_model="openai/gpt-5-nano",
#     temperature=None,              # do NOT send temperature to gpt-5 family
#     max_tokens=16000,
#     reasoning_effort="low",        # forwarded for responses-style models
#     verbosity="low",               # optionally forwarded
# )
# main(essay_ids=[123], llm_config=cfg)









# from TRACE_KG_Generator import generate_trace_kgs
# generate_trace_kgs(
#     essay_ids=[123],
#     default_model="openai/gpt-4o",
#     entity_rec_model="openai/gpt-5.1",
# )



#endregion#? Some Test Calls
#?#########################  End  ##########################

