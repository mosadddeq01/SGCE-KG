






  
  

#!############################################# Start Chapter ##################################################
#region:#!   Experiments 4 - Text2KGBench Reverse

   
  


#?######################### Start ##########################
#region:#?   Download Raw Data from Text2KGBench repository 


import os
import zipfile
import requests
import io

URL = "https://github.com/cenguix/Text2KGBench/archive/refs/heads/main.zip"
TARGET_DIR = "Experiments/MYNE/Ex4_T2KGBench/dbpedia-webnlg/Raw"
SUBDIR = "Text2KGBench-main/data/dbpedia_webnlg/"

os.makedirs(TARGET_DIR, exist_ok=True)

r = requests.get(URL)
r.raise_for_status()

with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    for member in z.namelist():
        if member.startswith(SUBDIR) and not member.endswith("/"):
            rel_path = member[len(SUBDIR):]
            out_path = os.path.join(TARGET_DIR, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with z.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

print("Downloaded dbpedia_webnlg to:", TARGET_DIR)




#endregion#? Download Raw Data from Text2KGBench repository
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   


import os
import json
from pathlib import Path
from collections import defaultdict
import re

RAW_ROOT = Path("Experiments/MYNE/Ex4_T2KGBench/dbpedia-webnlg/Raw")
TRAIN_DIR = RAW_ROOT / "train"
TEST_DIR = RAW_ROOT / "test"
OUT_DIR = RAW_ROOT.parent / "Input_to_TRACE-KG"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "In_Plain_Text.json"

# Helper to get ontology prefix and numeric suffix from mention id
# Examples:
#  - "ont_1_university_train_1" -> ("ont_1_university", 1)
#  - "ont_1_university_test_12" -> ("ont_1_university", 12)
# If no numeric suffix found, returns -1 for suffix.
def parse_source_id(src_id: str):
    # try explicit separators first
    m = re.match(r"^(.*?)(_train|_test)_(\d+)$", src_id)
    if m:
        prefix = m.group(1)
        try:
            suffix_num = int(m.group(3))
        except:
            suffix_num = -1
        return prefix, suffix_num
    # fallback: try to split on last underscore with trailing digits
    m2 = re.match(r"^(.*)_([0-9]+)$", src_id)
    if m2:
        return m2.group(1), int(m2.group(2))
    # otherwise return full id as prefix and -1 as suffix
    return src_id, -1

# Read all .jsonl files from a directory (non-recursive)
def read_jsonl_dir(dir_path: Path):
    items = []
    if not dir_path.exists():
        return items
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        # skip bad lines but print a warning
                        print(f"Warning: failed to parse line in {p}: {e}")
                        continue
                    items.append((p.name, obj))
    return items

# Collect items from train and test
all_items = []
all_items += read_jsonl_dir(TRAIN_DIR)
all_items += read_jsonl_dir(TEST_DIR)

# Group by ontology prefix
groups = defaultdict(list)  # prefix -> list of (suffix_num, src_filename, obj)
for src_fname, obj in all_items:
    src_id = obj.get("id") or ""
    sent = obj.get("sent") or obj.get("text") or ""
    # normalize obj minimally
    prefix, suffix = parse_source_id(src_id)
    groups[prefix].append((suffix, src_fname, src_id, sent, obj))

# For deterministic output: sort groups by prefix name
ordered_prefixes = sorted(groups.keys())

out_records = []
next_id = 100

for prefix in ordered_prefixes:
    entries = groups[prefix]
    # sort by suffix (numeric) ascending, then by src_filename for tie-breaker
    entries_sorted = sorted(entries, key=lambda x: (x[0] if x[0] is not None else -1, x[1]))
    for suffix_num, src_fname, src_id, sent, raw_obj in entries_sorted:
        # Build destination object according to your mapping
        dest = {
            "id": next_id,
            "title": src_id,                   # mapped from original "id"
            "start_page": suffix_num if isinstance(suffix_num, int) and suffix_num >= 0 else -1,
            "end_page": suffix_num if isinstance(suffix_num, int) and suffix_num >= 0 else -1,
            "text": f"```{sent}```",            # mapped from "sent", wrapped as you showed
            "kind": "Disjoint Sentences"
        }
        out_records.append(dest)
        next_id += 1

# Write output JSON array
with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(out_records, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(out_records)} records grouped by {len(ordered_prefixes)} ontologies to:")
print(OUT_PATH)



#endregion#? 
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?     Produce Chunks per Ontology for Ex4_T2KGBench








#!/usr/bin/env python3
"""
make_chunks_per_ontology.py

Create one chunks_sentence.jsonl per ontology prefix found in:
Experiments/MYNE/Ex4_T2KGBench/dbpedia-webnlg/Input_to_TRACE-KG/In_Plain_Text.json

Output layout (example):
  data/Chunks/ont_10_comicscharacter/chunks_sentence.jsonl
  data/Chunks/ont_11_university/chunks_sentence.jsonl
  ...

Each output file is NDJSON where each line is a chunk object consumable by TRACE-KG.
By default we create one chunk per essay (ONE_CHUNK_PER_ESSAY=True). Set it to False
to emit one chunk per sentence instead.
"""
import json
from pathlib import Path
import re
from collections import defaultdict

# ---------- CONFIG ----------
IN_PLAIN_JSON = Path("Experiments/MYNE/Ex4_T2KGBench/dbpedia-webnlg/Input_to_TRACE-KG/In_Plain_Text.json")
CHUNKS_BASEDIR = Path("data/Chunks")
ONE_CHUNK_PER_ESSAY = True   # True: one chunk per essay; False: one chunk per sentence
CHUNK_ID_PREFIX = "Ch"
# ----------------------------

CHUNKS_BASEDIR.mkdir(parents=True, exist_ok=True)

def parse_ontology_prefix(title: str):
    """
    Extract ontology prefix by removing _train_<n> or _test_<n> suffix.
    Examples:
      'ont_10_comicscharacter_train_1' -> 'ont_10_comicscharacter'
      'ont_10_comicscharacter_test_1'  -> 'ont_10_comicscharacter'
    If no match, fallback to taking text up to last underscore-number or full title.
    """
    if not title:
        return "unknown_ontology"
    m = re.match(r"^(.*?)(?:_train|_test)_[0-9]+$", title)
    if m:
        return m.group(1)
    # fallback: match trailing _<number>
    m2 = re.match(r"^(.*)_([0-9]+)$", title)
    if m2:
        return m2.group(1)
    # final fallback: use entire title but sanitize
    return re.sub(r"\s+", "_", title.strip())

def simple_sentence_split(text: str):
    """Simple sentence splitter (works well for your short sentences).
       - If newlines present we use them.
       - Else split on punctuation boundaries.
    """
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    if "\n" in text:
        parts = [s.strip() for s in text.splitlines() if s.strip()]
        if len(parts) > 0:
            return parts
    # fallback regex split: keep punctuation
    parts = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9"\'\(\[])', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]

def estimate_tokens(text: str):
    return max(1, len(text.split()))

def make_chunk_obj(global_idx:int, title:str, start_page:int, end_page:int,
                   chunk_index_in_section:int, chunk_text:str, sentences:list):
    chunk_id = f"{CHUNK_ID_PREFIX}_{global_idx:06d}"
    n_words = len(chunk_text.split())
    n_tokens_est = estimate_tokens(chunk_text)
    span = [0, max(0, len(chunk_text)-1)]
    obj = {
        "id": chunk_id,
        "ref_title": title,
        "ref_index": None,
        "ref_heading": None,
        "start_page": start_page,
        "end_page": end_page,
        "chunk_index_in_section": chunk_index_in_section,
        "text": chunk_text,
        "sentences": sentences,
        "span": span,
        "n_words": n_words,
        "n_tokens_est": n_tokens_est,
        "stripped_heading": None,
        "reason": "manual_chunk_created_per_ontology"
    }
    return obj

def main():
    if not IN_PLAIN_JSON.exists():
        raise FileNotFoundError(f"In_Plain_Text.json not found at {IN_PLAIN_JSON}")

    with IN_PLAIN_JSON.open("r", encoding="utf-8") as f:
        essays = json.load(f)
    if not isinstance(essays, list):
        raise ValueError("In_Plain_Text.json must be a JSON array (list) of essay objects.")

    # group essays by ontology prefix
    groups = defaultdict(list)
    for essay in essays:
        title = str(essay.get("title", "")).strip()
        prefix = parse_ontology_prefix(title)
        groups[prefix].append(essay)

    print(f"Found {len(groups)} ontology groups.")

    # Produce one chunks file per ontology
    for prefix, items in groups.items():
        out_dir = CHUNKS_BASEDIR / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chunks_sentence.jsonl"
        print(f"Writing {len(items)} chunk-entries into {out_path} (ontology: {prefix})")

        with out_path.open("w", encoding="utf-8") as out_f:
            global_counter = 1
            # sort items deterministically by title (you can change ordering)
            items_sorted = sorted(items, key=lambda e: str(e.get("title","")))
            for essay in items_sorted:
                title = str(essay.get("title",""))
                start_page = essay.get("start_page", -1)
                end_page = essay.get("end_page", start_page if start_page is not None else -1)
                raw_text = str(essay.get("text","")).strip()
                if raw_text.startswith("```") and raw_text.endswith("```"):
                    raw_text = raw_text[3:-3].strip()

                if ONE_CHUNK_PER_ESSAY:
                    sentences = simple_sentence_split(raw_text)
                    chunk_text = "\n".join(sentences) if len(sentences) > 1 else raw_text
                    obj = make_chunk_obj(global_counter, title, start_page, end_page, 0, chunk_text, sentences)
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    global_counter += 1
                else:
                    # emit one chunk per sentence
                    sentences = simple_sentence_split(raw_text)
                    for idx, sent in enumerate(sentences):
                        obj = make_chunk_obj(global_counter, title, start_page, end_page, idx, sent, [sent])
                        out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        global_counter += 1

    print("Done. Per-ontology chunks written under:", CHUNKS_BASEDIR)

if __name__ == "__main__":
    main()

  
  
#endregion#?   Produce Chunks per Ontology for Ex4_T2KGBench
#?#########################  End  ##########################




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
# ESSAY_IDS: List[int] = [87, 123, 23, 64, 46, 52, 84, 10, 51, 15]   # <--- edit this list as needed
ESSAY_IDS: List[int] = [ 100, 101, 4958, 4959]

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

    # # 1) Chunking
    # if not _run_step(
    #     "sentence_chunks_token_driven",
    #     sentence_chunks_token_driven,
    #     str(PLAIN_TEXT_JSON),
    #     "data/Chunks/chunks_sentence.jsonl",
    #     max_tokens_per_chunk=None,
    #     min_tokens_per_chunk=None,
    #     sentence_per_line=True,
    #     keep_ref_text=False,
    #     strip_leading_headings=True,
    #     force=True,
    #     debug=False,
    # ):
    #     return stats  # stop on first failure
    
    
        # 1) Chunking
    chunks_path = Path("data/Chunks/chunks_sentence.jsonl")
    if chunks_path.exists():
        print(f"[SKIP] Precomputed chunks found at {chunks_path} — skipping sentence_chunks_token_driven.")
    else:
        if not _run_step(
            "sentence_chunks_token_driven",
            sentence_chunks_token_driven,
            str(PLAIN_TEXT_JSON),
            str(chunks_path),
            max_tokens_per_chunk=None,
            min_tokens_per_chunk=None,
            sentence_per_line=True,
            keep_ref_text=False,
            strip_leading_headings=True,
            force=False,   # safer: do not force overwrite existing files (we just checked)
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
        prev_chunks=0,
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
    default_model: str = "gpt-5-nano",
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




# Run on all essays in In_Plain_Text.json
generate_trace_kgs(
    essay_ids=None,          # or a list of specific ids
    # default_model= "gpt-5-nano" #"gpt-5.1", # or another model name
    temperature=0.0,
    max_tokens=16000,
)











#?######################### Start ##########################
#region:#?        Create KG from Text2KGBench Reverse Chunks



#!/usr/bin/env python3
"""
TRACE-KG generator (ontology / pre-chunked single-run mode)

This revised driver **does not** look for or use In_Plain_Text.json at all.
It assumes you have already created the chunks file at:

    data/Chunks/chunks_sentence.jsonl

and will skip the chunking step entirely. The script runs the remainder of the
TRACE-KG pipeline once and produces a single KG snapshot under KGs_from_Essays.

Save this file and run it from the repo root.

Notes:
- Entity recognition is called with prev_chunks=0 (no cross-chunk context).
- If data/Chunks/chunks_sentence.jsonl is missing the script exits with an error.
- The output snapshot folder is named KG_Run_001 (you can change this).
"""
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from tqdm import tqdm

from TKG_Main import (  # core pipeline functions
    # chunking function still imported for compatibility but not invoked
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
DATA_DIR = REPO_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "Chunks" / "chunks_sentence.jsonl"
CHUNKS_EMB_DIR = DATA_DIR / "Chunks" / "chunks_emb"
KG_OUT_ROOT = REPO_ROOT / "KGs_from_Essays"
STATS_OUT = KG_OUT_ROOT / "Trace_KG_run_stats.json"

# Subdirectories that the pipeline may create and which we clear between runs
DATA_SUBDIRS_TO_CLEAR = [
    DATA_DIR / "Classes",
    DATA_DIR / "Entities",
    DATA_DIR / "KG",
    DATA_DIR / "Relations",
    # NOTE: we intentionally DO NOT clear data/Chunks here because you manage it manually
]

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
    """
    Clear pipeline-generated state *except* for data/Chunks (you provided chunks).
    """
    for sub in DATA_SUBDIRS_TO_CLEAR:
        clear_subdir_contents(sub)


def copy_data_for_snapshot(snapshot_name: str, ok: bool) -> Path:
    """
    Copy the entire data/ directory into KGs_from_Essays/<snapshot_name> or
    <snapshot_name>_FAILED depending on ok flag.
    """
    ensure_dir(KG_OUT_ROOT)
    suffix = "" if ok else "_FAILED"
    dest = KG_OUT_ROOT / f"{snapshot_name}{suffix}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(DATA_DIR, dest)
    return dest


# ------------------------------------------------------------------------------------
# MAIN PIPELINE (single-run, pre-chunked)
# ------------------------------------------------------------------------------------

def run_full_pipeline_from_precomputed_chunks(
    llm_config: Optional[TraceKGLLMConfig] = None,
) -> Dict[str, Any]:
    """
    Run the TRACE-KG pipeline starting from an existing chunks file.
    This runs the steps:
      - embed_and_index_chunks
      - run_entity_extraction_on_chunks (prev_chunks=0)
      - iterative_resolution
      - produce_clean_jsonl
      - classrec_iterative_main
      - main_input_for_cls_res
      - run_pipeline_iteratively
      - run_rel_rec
      - run_relres_iteratively
      - export_relations_and_nodes_to_csv

    Returns a stats dict similar to the original generator.
    """
    stats: Dict[str, Any] = {
        "steps": {},
        "ok": True,
        "error": None,
    }

    def _run_step(name: str, fn, *args, **kwargs) -> bool:
        import traceback
        t0 = time.time()
        step_info: Dict[str, Any] = {"ok": True, "error": None, "seconds": None}
        try:
            # Signature-aware: strip llm_config if fn doesn't accept it
            import inspect as _inspect
            sig = _inspect.signature(fn)
            if "llm_config" not in sig.parameters and "llm_config" in kwargs:
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

    # check precomputed chunks exist
    if not CHUNKS_PATH.exists():
        stats["ok"] = False
        stats["error"] = f"Precomputed chunks file not found at {CHUNKS_PATH}"
        print(stats["error"])
        return stats

    # 1) Embed + index chunks
    if not _run_step(
        "embed_and_index_chunks",
        embed_and_index_chunks,
        str(CHUNKS_PATH),
        str(CHUNKS_EMB_DIR),
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        False,   # use_small_model_for_dev
        32,      # batch_size
        None,    # device (let function decide)
        True,    # create_faiss
        True,    # normalize
    ):
        return stats

    # 2) Entity Recognition (no previous-chunk context)
    if not _run_step(
        "run_entity_extraction_on_chunks",
        run_entity_extraction_on_chunks,
        chunk_ids=None,
        prev_chunks=0,         # IMPORTANT: no cross-chunk context
        save_debug=False,
        model="gpt-5.1",
        max_tokens=8000,
        llm_config=llm_config,
    ):
        return stats

    # 3) Iterative entity resolution
    if not _run_step(
        "iterative_resolution",
        iterative_resolution,
        llm_config=llm_config,
    ):
        return stats

    # 4) Class-rec input producer
    if not _run_step(
        "produce_clean_jsonl",
        produce_clean_jsonl,
        None,
        None,
    ):
        return stats

    # 5) Class Recognition
    if not _run_step(
        "classrec_iterative_main",
        classrec_iterative_main,
        llm_config=llm_config,
    ):
        return stats

    # 6) Create input for Cls Res
    if not _run_step("main_input_for_cls_res", main_input_for_cls_res):
        return stats

    # 7) Class Res Multi Run
    if not _run_step(
        "run_pipeline_iteratively",
        run_pipeline_iteratively,
        llm_config=llm_config,
    ):
        return stats

    # 8) Relation Recognition (Rel Rec) – produce relations_raw.jsonl
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
        print(stats["error"])
        return stats

    if not _run_step(
        "run_rel_rec",
        run_rel_rec,
        entities_path=str(entities_with_class_path),
        chunks_path=str(CHUNKS_PATH),
        output_path="data/Relations/Rel Rec/relations_raw.jsonl",
        model="gpt-5.1",
        llm_config=llm_config,
    ):
        return stats

    # 9) Relation Res Multi Run
    if not _run_step(
        "run_relres_iteratively",
        run_relres_iteratively,
        llm_config=llm_config,
    ):
        return stats

    # 10) Export KG to CSVs
    if not _run_step("export_relations_and_nodes_to_csv", export_relations_and_nodes_to_csv):
        return stats

    return stats


# ------------------------------------------------------------------------------------
# SIMPLE MAIN (single-run)
# ------------------------------------------------------------------------------------

def main(
    llm_config: Optional[TraceKGLLMConfig] = None,
) -> None:
    """
    Single-run main entrypoint for pre-chunked mode.
    - Expects data/Chunks/chunks_sentence.jsonl to exist and represent one ontology.
    - Runs the pipeline once and snapshots the data/ directory to KGs_from_Essays/KG_Run_001.
    """
    print("CWD:", os.getcwd())
    print("TRACE-KG running in pre-chunked single-run mode.")
    ensure_dir(KG_OUT_ROOT)

    # basic check: chunks file must exist
    if not CHUNKS_PATH.exists():
        print(f"ERROR: required chunks file not found at {CHUNKS_PATH}")
        return

    if llm_config is None:
        llm_config = TraceKGLLMConfig()

    # clear previous pipeline state (but do NOT erase data/Chunks)
    clear_pipeline_state()

    t0 = time.time()
    stats = run_full_pipeline_from_precomputed_chunks(llm_config=llm_config)
    ok = stats.get("ok", False)

    # snapshot & copy results
    snapshot_dir = copy_data_for_snapshot("KG_Run_001", ok=ok)

    stats["seconds_total"] = time.time() - t0
    stats["snapshot_dir"] = str(snapshot_dir)

    # write a run-level stats file
    ensure_dir(KG_OUT_ROOT)
    with STATS_OUT.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if ok:
        print(f"✅ TRACE-KG completed successfully. Snapshot: {snapshot_dir}")
    else:
        print(f"❌ TRACE-KG failed. Snapshot (partial): {snapshot_dir}")
        print("Error:", stats.get("error"))


# ------------------------------------------------------------------------------------
# High-level API: generate_trace_kgs (keeps compatibility with previous code)
# ------------------------------------------------------------------------------------

def generate_trace_kgs(
    default_model: str = "gpt-5-nano",
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
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> None:
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
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )

    main(llm_config=cfg)


# ------------------------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the single, pre-chunked pipeline once using default LLM config
    generate_trace_kgs(
        default_model="gpt-5-nano",
        temperature=0.0,
        max_tokens=16000,
    )


#endregion#?      Create KG from Text2KGBench Reverse Chunks
#?#########################  End  ##########################


  

#endregion#! Experiments 4 - Text2KGBench Reverse
#!############################################# End Chapter ##################################################
  
  
  
  


#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################



