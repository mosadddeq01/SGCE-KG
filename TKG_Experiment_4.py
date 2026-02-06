






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
    # entities_with_class_fallback = Path(
    #     "data/Relations/Rel Res_IterativeRuns/overall_summary/entities_with_class.jsonl"
    # )

    if entities_with_class_primary.exists():
        entities_with_class_path = entities_with_class_primary
    # elif entities_with_class_fallback.exists():
    #     entities_with_class_path = entities_with_class_fallback
    else:
        stats["ok"] = False
        stats["error"] = (
            "run_rel_rec skipped: entities_with_class.jsonl not found at"
            f"{entities_with_class_primary}."
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
    default_model= "openai/gpt-4.1-nano", #"gpt-5-nano" #"gpt-5.1", # or another model name,
    # temperature=0.0,
    max_tokens=16000,
    # Disable DSPy LM cache so entity extraction and other LLM calls
    # always re-run instead of reusing cached outputs.
    disable_cache=True,
)




from TKG_Generator import clear_pipeline_state
clear_pipeline_state()




import TKG_Main


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
KG_OUT_ROOT = REPO_ROOT / "Experiments/MYNE/Ex4_T2KGBench/KGs_from_Essays"
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


#endregion#?      Create KG from Text2KGBench Reverse Chunks
#?#########################  End  ##########################


# ------------------------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------------------------


if __name__ == "__main__":
    # Run the single, pre-chunked pipeline once using default LLM config
    generate_trace_kgs(
        default_model=  "gpt-5.1",  #"gpt-4.1-nano", # # "gpt-4.1-nano",  #"gpt-5-mini", #"gpt-5-nano", #"openai/gpt-4.1-nano", #"gpt-5-nano",,
        max_tokens=16000,
        disable_cache=True,
    )




# if __name__ == "__main__":
#     # Run the single, pre-chunked pipeline once using default LLM config
#     generate_trace_kgs(
#         # default_model=  "gpt-5.1", # "gpt-4.1-nano",  #"gpt-5-mini", #"gpt-5-nano", #"openai/gpt-4.1-nano", #"gpt-5-nano",,
#         entity_rec_model="gpt-4.1-nano",
#         entity_res_model="gpt-5.1",
#         class_rec_model="gpt-4.1-nano",
#         class_res_model="gpt-5.1",
#         rel_rec_model="gpt-4.1-nano",
#         rel_res_model="gpt-5.1",
#         # temperature=0.0,
#         # reasoning_effort="low",
#         max_tokens=16000,
#     )

  
  #endregion#! Experiments 4 - Text2KGBench Reverse
  #!#############################################  End Chapter  ##################################################
  





    
    
#?######################### Start ##########################
#region:#?   Schema Extraction from produced KG - Detailed View (V2)

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------------------------
# Small IO helpers
# -------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def _safe_json_loads(s: Any, default: Any) -> Any:
    if s is None:
        return default
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

def _first_existing(root: Path, candidates: List[str]) -> Optional[Path]:
    for rel in candidates:
        p = (root / rel).resolve()
        if p.exists():
            return p
    return None

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# -------------------------
# Normalization helpers
# -------------------------

def _norm(s: Any, fallback: str = "") -> str:
    if s is None:
        return fallback
    s = str(s).strip()
    return s if s else fallback

def _norm_upper(s: Any, fallback: str = "") -> str:
    return _norm(s, fallback=fallback).upper()

def _first_non_empty(*vals: Any) -> Optional[Any]:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None

# TRACE-KG internal IDs often look like En_xxxxxxxx / Can_xxxxxxxx / ClsR_xxxxxxxx etc.
_INTERNAL_ID_RE = re.compile(r"^(En|Can|Cls|ClsR|ClsC|RelR)_[0-9a-fA-F]{8}$")

def _looks_like_internal_id(s: str) -> bool:
    return bool(_INTERNAL_ID_RE.match((s or "").strip()))

# -------------------------
# Records
# -------------------------

@dataclass
class EntityRec:
    entity_id: str
    entity_name: str
    entity_description: str
    entity_type_hint: str
    class_label: str
    class_group: str
    node_properties: List[Dict[str, Any]]

@dataclass
class RelRec:
    relation_id: str
    subject_entity_id: str
    object_entity_id: str
    canonical_rel_name: str
    canonical_rel_desc: str
    rel_cls: str
    rel_cls_group: str
    qualifiers: Dict[str, Any]
    raw_relation_name: str  # original/raw relation_name token (if present)

# -------------------------
# Loaders
# -------------------------

def load_entities(path: Path) -> Tuple[List[EntityRec], Dict[str, EntityRec]]:
    entities: List[EntityRec] = []

    if path.suffix.lower() == ".jsonl":
        raw = _read_jsonl(path)
        for obj in raw:
            nested_ent = obj.get("entity") or {}

            # entity_id can be top-level or nested
            eid = _norm(_first_non_empty(obj.get("entity_id"), obj.get("id"), nested_ent.get("id"), nested_ent.get("entity_id")))
            if not eid:
                continue

            # --- IMPORTANT FIX: name/desc/type often live under nested_ent in TRACE-KG ---
            name_val = _first_non_empty(
                nested_ent.get("entity_name"),
                nested_ent.get("name"),
                nested_ent.get("canonical_name"),
                obj.get("entity_name"),
                obj.get("canonical_name"),
                obj.get("name"),
                obj.get("label"),
            )
            desc_val = _first_non_empty(
                nested_ent.get("entity_description"),
                nested_ent.get("description"),
                obj.get("entity_description"),
                obj.get("description"),
            )
            type_val = _first_non_empty(
                nested_ent.get("entity_type_hint"),
                obj.get("entity_type_hint"),
                obj.get("type_hint"),
                obj.get("type"),
            )

            # Class info may be top-level or stored as _class_* inside nested entity
            cls_label_val = _first_non_empty(obj.get("class_label"), nested_ent.get("_class_label"), obj.get("_class_label"))
            cls_group_val = _first_non_empty(obj.get("class_group"), nested_ent.get("_class_group"), obj.get("_class_group"))

            # node_properties can be nested or top-level
            node_props = _first_non_empty(nested_ent.get("node_properties"), obj.get("node_properties"), [])
            if not isinstance(node_props, list):
                node_props = []

            entities.append(
                EntityRec(
                    entity_id=eid,
                    entity_name=_norm(name_val, fallback=""),
                    entity_description=_norm(desc_val, fallback=""),
                    entity_type_hint=_norm(type_val, fallback=""),
                    class_label=_norm(cls_label_val, fallback="TBD"),
                    class_group=_norm(cls_group_val, fallback="TBD"),
                    node_properties=node_props,
                )
            )

    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = _norm(_first_non_empty(row.get("entity_id"), row.get("id"), row.get("node_id")))
                if not eid:
                    continue

                # node_properties stored as JSON string sometimes
                node_props = _safe_json_loads(row.get("node_properties"), default=[])
                if not isinstance(node_props, list):
                    node_props = []

                name_val = _first_non_empty(
                    row.get("entity_name"),
                    row.get("name"),
                    row.get("canonical_name"),
                    row.get("label"),
                )
                desc_val = _first_non_empty(
                    row.get("entity_description"),
                    row.get("description"),
                )
                type_val = _first_non_empty(
                    row.get("entity_type_hint"),
                    row.get("type_hint"),
                    row.get("type"),
                )

                entities.append(
                    EntityRec(
                        entity_id=eid,
                        entity_name=_norm(name_val, fallback=""),
                        entity_description=_norm(desc_val, fallback=""),
                        entity_type_hint=_norm(type_val, fallback=""),
                        class_label=_norm(row.get("class_label"), fallback="TBD"),
                        class_group=_norm(row.get("class_group"), fallback="TBD"),
                        node_properties=node_props,
                    )
                )
    else:
        raise ValueError(f"Unsupported entities file: {path}")

    by_id = {e.entity_id: e for e in entities}
    return entities, by_id

def load_relations(path: Path) -> List[RelRec]:
    rels: List[RelRec] = []

    if path.suffix.lower() == ".jsonl":
        raw = _read_jsonl(path)
        for r in raw:
            rid = _norm(r.get("relation_id") or r.get("id") or r.get("rid") or ("RelR_" + _norm(r.get("relation_name"), fallback="")))
            sid = _norm(r.get("subject_entity_id") or r.get("start_id") or r.get("start"))
            oid = _norm(r.get("object_entity_id") or r.get("end_id") or r.get("end"))
            if not (rid and sid and oid):
                continue

            raw_name = _norm(r.get("relation_name") or r.get("raw_relation_name") or "")
            canonical = _norm(r.get("canonical_rel_name"), fallback=_norm(r.get("relation_name"), fallback="TBD"))

            rels.append(
                RelRec(
                    relation_id=rid,
                    subject_entity_id=sid,
                    object_entity_id=oid,
                    canonical_rel_name=canonical,
                    canonical_rel_desc=_norm(r.get("canonical_rel_desc")),
                    rel_cls=_norm(r.get("rel_cls"), fallback="TBD"),
                    rel_cls_group=_norm_upper(r.get("rel_cls_group"), fallback=_norm_upper(r.get("rel_hint_type"), fallback="TBD")),
                    qualifiers=(r.get("qualifiers") if isinstance(r.get("qualifiers"), dict) else {}),
                    raw_relation_name=raw_name,
                )
            )

    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = _norm(row.get("relation_id"))
                sid = _norm(row.get("start_id") or row.get("subject_entity_id"))
                oid = _norm(row.get("end_id") or row.get("object_entity_id"))
                if not (rid and sid and oid):
                    continue

                qualifiers = _safe_json_loads(row.get("qualifiers"), default={})
                if not isinstance(qualifiers, dict):
                    qualifiers = {}

                raw_name = _norm(row.get("relation_name") or row.get("raw_relation_name") or "")
                canonical = _norm(row.get("canonical_rel_name"), fallback=_norm(row.get("relation_name"), fallback="TBD"))

                rels.append(
                    RelRec(
                        relation_id=rid,
                        subject_entity_id=sid,
                        object_entity_id=oid,
                        canonical_rel_name=canonical,
                        canonical_rel_desc=_norm(row.get("canonical_rel_desc")),
                        rel_cls=_norm(row.get("rel_cls"), fallback="TBD"),
                        rel_cls_group=_norm_upper(row.get("rel_cls_group"), fallback="TBD"),
                        qualifiers=qualifiers,
                        raw_relation_name=raw_name,
                    )
                )
    else:
        raise ValueError(f"Unsupported relations file: {path}")

    return rels

# -------------------------
# Schema builders
# -------------------------

def build_entity_tree(entities: List[EntityRec]) -> Dict[str, Dict[str, List[str]]]:
    tree: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for e in entities:
        tree[e.class_group][e.class_label].append(e.entity_id)
    return tree

def build_entity_tree_extended(entities: List[EntityRec]) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """
    Extended entity tree:
      class_group -> class_label -> list of (entity_id, display_name)
    """
    tree: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for e in entities:
        # Prefer real entity_name; if it's missing or still looks like an internal id, fall back to description.
        name = (e.entity_name or "").strip()
        if not name or _looks_like_internal_id(name):
            desc = (e.entity_description or "").strip()
            if desc and not _looks_like_internal_id(desc):
                # keep it short-ish for tree readability
                name = desc[:80] + ("…" if len(desc) > 80 else "")
            else:
                name = ""  # keep empty; renderer will still show ID
        tree[e.class_group][e.class_label].append((e.entity_id, name))
    return tree

def build_relation_tree(rels: List[RelRec]) -> Dict[str, Dict[str, Dict[str, int]]]:
    tree: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rels:
        tree[r.rel_cls_group][r.rel_cls][r.canonical_rel_name] += 1
    return tree

def build_relation_tree_extended(rels: List[RelRec]) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
    tree: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )
    for r in rels:
        raw = r.raw_relation_name if r.raw_relation_name else "<raw:N/A>"
        tree[r.rel_cls_group][r.rel_cls][r.canonical_rel_name][raw] += 1
    return tree

def compute_domain_range(
    rels: List[RelRec],
    ent_by_id: Dict[str, EntityRec],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for r in rels:
        pred = r.canonical_rel_name or "TBD"
        subj = ent_by_id.get(r.subject_entity_id)
        obj = ent_by_id.get(r.object_entity_id)

        subj_cls = subj.class_label if subj else "TBD"
        subj_grp = subj.class_group if subj else "TBD"
        obj_cls = obj.class_label if obj else "TBD"
        obj_grp = obj.class_group if obj else "TBD"

        rec = out.setdefault(pred, {
            "canonical_rel_name": pred,
            "rel_cls_groups": set(),
            "rel_classes": set(),
            "domain_class_groups": set(),
            "domain_classes": set(),
            "range_class_groups": set(),
            "range_classes": set(),
            "count": 0,
            "example_pairs": set(),
        })

        rec["rel_cls_groups"].add(r.rel_cls_group or "TBD")
        rec["rel_classes"].add(r.rel_cls or "TBD")
        rec["domain_class_groups"].add(subj_grp)
        rec["domain_classes"].add(subj_cls)
        rec["range_class_groups"].add(obj_grp)
        rec["range_classes"].add(obj_cls)
        rec["count"] += 1
        rec["example_pairs"].add((subj_cls, obj_cls))

    for pred, rec in out.items():
        rec["rel_cls_groups"] = sorted(rec["rel_cls_groups"])
        rec["rel_classes"] = sorted(rec["rel_classes"])
        rec["domain_class_groups"] = sorted(rec["domain_class_groups"])
        rec["domain_classes"] = sorted(rec["domain_classes"])
        rec["range_class_groups"] = sorted(rec["range_class_groups"])
        rec["range_classes"] = sorted(rec["range_classes"])
        rec["example_pairs"] = sorted(list(rec["example_pairs"]))[:25]

    return out

def extract_data_properties(entities: List[EntityRec]) -> Dict[str, Any]:
    global_counts = Counter()
    per_class_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for e in entities:
        for p in (e.node_properties or []):
            if not isinstance(p, dict):
                continue
            name = _norm(p.get("prop_name"))
            if not name:
                continue
            global_counts[name] += 1
            per_class_counts[(e.class_group, e.class_label)][name] += 1

    return {
        "global_property_counts": dict(global_counts.most_common()),
        "per_class_property_counts": {
            f"{grp} :: {lbl}": dict(cnt.most_common())
            for (grp, lbl), cnt in per_class_counts.items()
        }
    }

def find_duplicate_edges(rels: List[RelRec]) -> List[Dict[str, Any]]:
    key_counts = Counter((r.subject_entity_id, r.object_entity_id, r.canonical_rel_name) for r in rels)
    dups = []
    for (sid, oid, pred), c in key_counts.items():
        if c > 1:
            dups.append({
                "subject_entity_id": sid,
                "object_entity_id": oid,
                "canonical_rel_name": pred,
                "count": c,
            })
    dups.sort(key=lambda x: x["count"], reverse=True)
    return dups

# -------------------------
# Pretty printing (ASCII trees)
# -------------------------

def render_entity_tree(tree: Dict[str, Dict[str, List[str]]]) -> str:
    lines: List[str] = []
    groups = sorted(tree.keys(), key=lambda g: (-sum(len(v) for v in tree[g].values()), g.lower()))
    for g in groups:
        classes = tree[g]
        total = sum(len(ids) for ids in classes.values())
        lines.append(f"{g}  ({total} entities)")
        class_items = sorted(classes.items(), key=lambda kv: (-len(kv[1]), kv[0].lower()))
        for i, (cls, ids) in enumerate(class_items):
            branch = "└─" if i == len(class_items) - 1 else "├─"
            lines.append(f"  {branch} {cls}  ({len(ids)})")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def render_entity_tree_extended(tree: Dict[str, Dict[str, List[Tuple[str, str]]]]) -> str:
    """
    Extended rendering: Group -> Class -> Entity as "ID: Name"
    """
    lines: List[str] = []
    groups = sorted(tree.keys(), key=lambda g: (-sum(len(v) for v in tree[g].values()), g.lower()))
    for g in groups:
        classes = tree[g]
        total = sum(len(members) for members in classes.values())
        lines.append(f"{g}  ({total} entities)")
        class_items = sorted(classes.items(), key=lambda kv: (-len(kv[1]), kv[0].lower()))
        for ci, (cls, members) in enumerate(class_items):
            cls_branch = "└─" if ci == len(class_items) - 1 else "├─"
            lines.append(f"  {cls_branch} {cls}  ({len(members)})")

            # sort by name if available, otherwise by id
            mem_items = sorted(members, key=lambda x: ((x[1] or "").lower(), x[0]))
            for mi, (eid, ename) in enumerate(mem_items):
                mem_branch = "    └─" if mi == len(mem_items) - 1 else "    ├─"
                show_name = ename if ename else "<missing_entity_name>"
                # REQUIRED FORMAT: id + ":" + name
                lines.append(f"{mem_branch} {eid}: {show_name}")

        lines.append("")
    return "\n".join(lines).strip() + "\n"

def render_relation_tree(tree: Dict[str, Dict[str, Dict[str, int]]]) -> str:
    lines: List[str] = []
    groups = sorted(tree.keys(), key=lambda g: g.lower())
    for g in groups:
        total = sum(cnt for cls in tree[g].values() for cnt in cls.values())
        lines.append(f"{g}  ({total} relations)")
        rel_classes = tree[g]
        cls_items = sorted(rel_classes.items(), key=lambda kv: (-sum(kv[1].values()), kv[0].lower()))
        for ci, (cls, preds) in enumerate(cls_items):
            cls_total = sum(preds.values())
            cls_branch = "└─" if ci == len(cls_items) - 1 else "├─"
            lines.append(f"  {cls_branch} {cls}  ({cls_total})")
            pred_items = sorted(preds.items(), key=lambda kv: (-kv[1], kv[0].lower()))
            for pi, (pred, cnt) in enumerate(pred_items):
                pred_branch = "└─" if pi == len(pred_items) - 1 else "├─"
                lines.append(f"      {pred_branch} {pred}  ({cnt})")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def render_relation_tree_extended(tree: Dict[str, Dict[str, Dict[str, Dict[str, int]]]]) -> str:
    lines: List[str] = []
    groups = sorted(tree.keys(), key=lambda g: g.lower())
    for g in groups:
        total = sum(cnt for cls in tree[g].values() for canon in cls.values() for cnt in canon.values())
        lines.append(f"{g}  ({total} relations)")
        rel_classes = tree[g]

        cls_items = sorted(
            rel_classes.items(),
            key=lambda kv: (-sum(sum(raws.values()) for raws in kv[1].values()), kv[0].lower())
        )

        for ci, (cls, canon_map) in enumerate(cls_items):
            cls_total = sum(sum(raws.values()) for raws in canon_map.values())
            cls_branch = "└─" if ci == len(cls_items) - 1 else "├─"
            lines.append(f"  {cls_branch} {cls}  ({cls_total})")

            canon_items = sorted(canon_map.items(), key=lambda kv: (-sum(kv[1].values()), kv[0].lower()))
            for pi, (canon, raw_map) in enumerate(canon_items):
                canon_branch = "    └─" if pi == len(canon_items) - 1 else "    ├─"
                canon_count = sum(raw_map.values())
                lines.append(f"{canon_branch} {canon}  ({canon_count})")

                raw_items = sorted(raw_map.items(), key=lambda kv: (-kv[1], kv[0].lower()))
                for ri, (raw, cnt) in enumerate(raw_items):
                    raw_branch = "      └─" if ri == len(raw_items) - 1 else "      ├─"
                    lines.append(f"{raw_branch} {raw}  ({cnt})")

        lines.append("")
    return "\n".join(lines).strip() + "\n"

# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root (where data/ exists).")
    ap.add_argument("--out", default="data/Schema", help="Output directory for schema artifacts.")
    # In Jupyter / VS Code interactive, extra args like "--f=..." are injected.
    # parse_known_args() ignores those unknown args so the script can still run.
    args, _ = ap.parse_known_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ent_path = _first_existing(root, [
        "data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl",
        "data/Relations/Rel Res_IterativeRuns/overall_summary/entities_with_class.jsonl",
        "data/KG/nodes.csv",
        "data/Classes/entities_with_class.jsonl",
    ])
    rel_path = _first_existing(root, [
        "data/Relations/Rel Res_IterativeRuns/overall_summary/relations_resolved.jsonl",
        "data/KG/rels_fixed_no_raw.csv",
        "data/KG/relations_resolved.jsonl",
        "data/Relations/relations_resolved.jsonl",
    ])

    if ent_path is None:
        raise FileNotFoundError("Could not find entities input (entities_with_class.jsonl or nodes.csv).")
    if rel_path is None:
        raise FileNotFoundError("Could not find relations input (relations_resolved.jsonl or rels_fixed_no_raw.csv).")

    print("[schema] Using entities:", ent_path)
    print("[schema] Using relations:", rel_path)

    entities, ent_by_id = load_entities(ent_path)
    rels = load_relations(rel_path)

    ent_tree = build_entity_tree(entities)
    ent_tree_ext = build_entity_tree_extended(entities)
    rel_tree = build_relation_tree(rels)
    rel_tree_ext = build_relation_tree_extended(rels)

    domain_range = compute_domain_range(rels, ent_by_id)
    data_props = extract_data_properties(entities)
    dup_edges = find_duplicate_edges(rels)

    # original trees
    _write_text(out_dir / "entity_schema_tree.txt", render_entity_tree(ent_tree))
    _write_text(out_dir / "relation_schema_tree.txt", render_relation_tree(rel_tree))

    # extended trees (now: ID: Name)
    _write_text(out_dir / "entity_schema_tree_extended.txt", render_entity_tree_extended(ent_tree_ext))
    _write_text(out_dir / "relation_schema_tree_extended.txt", render_relation_tree_extended(rel_tree_ext))

    _write_json(out_dir / "domain_range_by_canonical_rel.json", domain_range)
    _write_json(out_dir / "data_properties_schema.json", data_props)
    _write_json(out_dir / "duplicate_edges_report.json", dup_edges)

    # CSV for domain/range
    dr_csv = out_dir / "domain_range_by_canonical_rel.csv"
    with dr_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "canonical_rel_name",
            "count",
            "rel_cls_groups",
            "rel_classes",
            "domain_class_groups",
            "domain_classes",
            "range_class_groups",
            "range_classes",
            "example_pairs",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for pred, rec in sorted(domain_range.items(), key=lambda kv: (-kv[1]["count"], kv[0].lower())):
            w.writerow({
                "canonical_rel_name": pred,
                "count": rec["count"],
                "rel_cls_groups": json.dumps(rec["rel_cls_groups"], ensure_ascii=False),
                "rel_classes": json.dumps(rec["rel_classes"], ensure_ascii=False),
                "domain_class_groups": json.dumps(rec["domain_class_groups"], ensure_ascii=False),
                "domain_classes": json.dumps(rec["domain_classes"], ensure_ascii=False),
                "range_class_groups": json.dumps(rec["range_class_groups"], ensure_ascii=False),
                "range_classes": json.dumps(rec["range_classes"], ensure_ascii=False),
                "example_pairs": json.dumps(rec["example_pairs"], ensure_ascii=False),
            })

    print(f"\n[schema] Wrote outputs to: {out_dir}")

#endregion#? Schema Extraction from produced KG - Detailed View (V2)
#?#########################  End  ##########################

if __name__ == "__main__":
    main()






#!############################################# Start Chapter ##################################################
#region:#!   000






#!############################################# Start Chapter ##################################################
#region:#!   OLLM - NeurIPS 2024






#?######################### Start ##########################
#region:#?   Download datasets

# Install if needed:
# pip install huggingface-hub datasets

from huggingface_hub import snapshot_download
from pathlib import Path
import json

# Optional: set your token if the repo is private or rate-limited
# export HF_TOKEN="your_token"  (or set here)
# from huggingface_hub import login
# login(token="YOUR_TOKEN")

repo_id = "andylolu24/wiki-ol"   # dataset repo (as shown on Hugging Face)
out_dir = Path("Experiments/MYNE/Ex5_OLLM/wiki-ol_repo")  # local folder to store everything

# This downloads the whole repo (all files + history at the specified revision)
# It returns the path to the snapshot directory.
snapshot_path = snapshot_download(repo_id=repo_id, local_dir=str(out_dir), repo_type="dataset")

print("Downloaded snapshot to:", snapshot_path)

# Example: list downloaded files
for p in Path(snapshot_path).rglob("*"):
    if p.is_file():
        print(p.relative_to(snapshot_path))

# Example: open a JSON file if present (adjust filename as needed)
sample_json = Path(snapshot_path) / "out" / "data" / "wikipedia" / "v2" / "train_test_split" / "train_graph.json"
if sample_json.exists():
    print("Found", sample_json)
    with open(sample_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    print("Top-level keys:", list(obj.keys())[:20])
else:
    print("Sample file not found: adjust path to a file present in the downloaded repo.")









from huggingface_hub import snapshot_download
from pathlib import Path
import json

# Optional: set your token if the repo is private or rate-limited
# export HF_TOKEN="your_token"  (or set here)
# from huggingface_hub import login
# login(token="YOUR_TOKEN")

repo_id = "andylolu24/arxiv-ol"   # dataset repo (as shown on Hugging Face)
out_dir = Path("Experiments/MYNE/Ex5_OLLM/arxiv-ol_repo")  # local folder to store everything

# This downloads the whole repo (all files + history at the specified revision)
# It returns the path to the snapshot directory.
snapshot_path = snapshot_download(repo_id=repo_id, local_dir=str(out_dir), repo_type="dataset")

print("Downloaded snapshot to:", snapshot_path)

# Example: list downloaded files
for p in Path(snapshot_path).rglob("*"):
    if p.is_file():
        print(p.relative_to(snapshot_path))

# Example: open a JSON file if present (adjust filename as needed)
sample_json = Path(snapshot_path) / "out" / "data" / "wikipedia" / "v2" / "train_test_split" / "train_graph.json"
if sample_json.exists():
    print("Found", sample_json)
    with open(sample_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    print("Top-level keys:", list(obj.keys())[:20])
else:
    print("Sample file not found: adjust path to a file present in the downloaded repo.")



#endregion#? Download datasets
#?#########################  End  ##########################



# find and print directory tree

from pathlib import Path

def print_tree(root, prefix=""):
    root = Path(root)
    print(prefix + root.name + "/")
    children = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for i, path in enumerate(children):
        is_last = (i == len(children) - 1)
        connector = "└── " if is_last else "├── "
        if path.is_dir():
            print(prefix + connector + path.name + "/")
            print_tree(path, prefix + ("    " if is_last else "│   "))
        else:
            print(prefix + connector + path.name)

# 🔽 CHANGE THIS to the directory you downloaded from HuggingFace
DATA_ROOT = "Experiments/MYNE/Ex5_OLLM"

print_tree(DATA_ROOT)











# print a few line from Experiments/MYNE/Ex5_OLLM/arxiv-ol_repo/pages/raw/arxiv-metadata-oai-snapshot.json
sample_file = Path("Experiments/MYNE/Ex5_OLLM/arxiv-ol_repo/pages/raw/arxiv-metadata-oai-snapshot.json")
if sample_file.exists():
    print(f"Reading sample file: {sample_file}")
    with sample_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i >= 9:
                break
else:
    print(f"Sample file not found: {sample_file}")
    
    
    
# Beautify JSON lines
import json
sample_file = Path("Experiments/MYNE/Ex5_OLLM/arxiv-ol_repo/pages/raw/arxiv-metadata-oai-snapshot.json")
if sample_file.exists():
    print(f"Reading sample file: {sample_file}")
    with sample_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            pretty = json.dumps(obj, indent=2)
            print(pretty)
            if i >= 1:
                break
else:
    print(f"Sample file not found: {sample_file}")
    

# Beautify JSON lines but for id "0704.0001"
import json
sample_file = Path("Experiments/MYNE/Ex5_OLLM/arxiv-ol_repo/pages/raw/arxiv-metadata-oai-snapshot.json")
target_id = "2001.00279"
if sample_file.exists():
    print(f"Reading sample file: {sample_file}")
    with sample_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if obj.get("id") == target_id:
                pretty = json.dumps(obj, indent=2)
                print(pretty)
                break
else:
    print(f"Sample file not found: {sample_file}")
    





# now just head line from this Experiments/MYNE/Ex5_OLLM/arxiv-ol_repo/train_test_split/train_graph.json


import json
from pathlib import Path

sample_file = Path("Experiments/MYNE/Ex5_OLLM/arxiv-ol_repo/train_test_split/train_graph.json")
if sample_file.exists():
    print(f"Reading sample file: {sample_file}")
    with sample_file.open("r", encoding="utf-8") as f:
        obj = json.load(f)
        pretty = json.dumps(obj, indent=2)
        print(pretty)
else:
    print(f"Sample file not found: {sample_file}")
    
    

# now just head line from this Experiments/MYNE/Ex5_OLLM/arxiv-ol_repo/train_test_split/train_graph.json
# with id "0704.0001"

import json
from pathlib import Path
sample_file = Path("Experiments/MYNE/Ex5_OLLM/arxiv-ol_repo/train_test_split/train_graph.json")
if sample_file.exists():
    print(f"Reading sample file: {sample_file}")
    with sample_file.open("r", encoding="utf-8") as f:
        obj = json.load(f)
        target_id = "2001.00279"
        if target_id in obj:
            paper = obj[target_id]
            pretty = json.dumps(paper, indent=2)
            print(pretty)
        else:
            print(f"ID {target_id} not found in the graph.")
else:
    print(f"Sample file not found: {sample_file}")    





import json
from pathlib import Path
from pprint import pprint

sample_file = Path(
    "Experiments/MYNE/Ex5_OLLM/arxiv-ol_repo/train_test_split/train_graph.json"
)

# ---- 1. Read only the top-level JSON object (no deep traversal) ----
with open(sample_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n=== TOP-LEVEL KEYS ===")
for k, v in data.items():
    print(f"- {k}: {type(v).__name__}")

# ---- 2. Inspect ONE example element per major key ----
def show_example(name, obj, max_depth=2):
    print(f"\n=== EXAMPLE FROM `{name}` ===")
    if isinstance(obj, list) and obj:
        pprint(obj[0], depth=max_depth)
    elif isinstance(obj, dict):
        # show first key-value pair
        first_key = next(iter(obj))
        print(f"key: {first_key}")
        pprint(obj[first_key], depth=max_depth)
    else:
        print(obj)




for key in data:
    show_example(key, data[key])






#endregion#! OLLM - NeurIPS 2024
#!#############################################  End Chapter  ##################################################



#endregion#! 000
#!#############################################  End Chapter  ##################################################







#!############################################# Start Chapter ##################################################
#region:#!   Comparing our schema with Text2KG Benchmark Ontology


#?######################### Start ##########################
#region:#?   convert ont_19_film_ground_truth.jsonl → gold_triples.jsonl


from pathlib import Path
import json

src = Path("Experiments/MYNE/Ex4_T2KGBench/dbpedia-webnlg/Raw/ground_truth/ont_19_film_ground_truth.jsonl")
out = Path("Experiments/MYNE/Ex4_T2KGBench/KGs_from_Essays/KG_Run_F3/OntCompResults/gold_triples.jsonl")   # evaluator default
out_lines = []

if not src.exists():
    raise FileNotFoundError(f"{src} not found")

with src.open("r", encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        j = json.loads(ln)
        # j has fields: id, sent, triples: [ {sub,rel,obj}, ...]
        tlist = j.get("triples", [])
        for t in tlist:
            # normalize keys if necessary
            s = t.get("sub") or t.get("subject") or ""
            p = t.get("rel") or t.get("predicate") or ""
            o = t.get("obj") or t.get("object") or ""
            if not (s and p and o):
                continue
            out_lines.append({"sentence_id": j.get("id"), "subject": s, "predicate": p, "object": o})

# write out simplified JSONL for the evaluator
with out.open("w", encoding="utf-8") as f:
    for rec in out_lines:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote {len(out_lines)} gold triples to {out}")



#endregion#? convert ont_19_film_ground_truth.jsonl → gold_triples.jsonl
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?    Cluster-based Concept Mapping v5


from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import hdbscan
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False


# ---------------------------
# Paths (you can edit ROOT only)
# ---------------------------
ROOT = Path("Experiments/MYNE/Ex4_T2KGBench/KGs_from_Essays/KG_Run_F3").resolve()
TRACE_ROOT = ROOT  # keep naming explicit
REF_ONTOLOGY = Path("Experiments/MYNE/Ex4_T2KGBench/dbpedia-webnlg/Raw/ontologies/19_film_ontology.json").resolve()
GOLD_TRIPLES = ROOT / "OntCompResults" / "gold_triples.jsonl"

OUT_DIR = ROOT / "OntCompResults" / "AnchoredClusters"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# IO helpers
# ---------------------------
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def read_json(p: Path) -> Any:
    return json.loads(read_text(p))

def read_jsonl(p: Path) -> List[dict]:
    out=[]
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonl(p: Path, rows: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def first_existing(cands: List[Path]) -> Optional[Path]:
    for p in cands:
        if p.exists():
            return p
    return None


# ---------------------------
# Normalization
# ---------------------------
def clean_label(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------
# Locate TRACE canonical files
# ---------------------------
def locate_trace_class_file(root: Path) -> Path:
    cands = [
        root / "Schema" / "Classes" / "Cls_Res" / "Cls_Res_IterativeRuns" / "overall_summary" / "final_classes_resolved.json",
        root / "data" / "Classes" / "Cls_Res" / "Cls_Res_IterativeRuns" / "overall_summary" / "final_classes_resolved.json",
        root / "Classes" / "Cls_Res" / "Cls_Res_IterativeRuns" / "overall_summary" / "final_classes_resolved.json",
    ]
    p = first_existing(cands)
    if not p:
        raise FileNotFoundError("Could not find final_classes_resolved.json in expected TRACE folders.")
    return p

def locate_trace_relation_file(root: Path) -> Path:
    cands = [
        # user stated location with spaces:
        root / "Schema" / "Relations" / "Rel Res_IterativeRuns" / "overall_summary" / "relations_resolved.jsonl",
        root / "Schema" / "Relations" / "Rel Res_IterativeRuns" / "overall_summary" / "relations_resolved.json",
        # common TRACE locations (no Schema):
        root / "data" / "Relations" / "Rel Res_IterativeRuns" / "overall_summary" / "relations_resolved.jsonl",
        root / "data" / "Relations" / "Rel Res_IterativeRuns" / "overall_summary" / "relations_resolved.json",
        root / "data" / "Relations" / "relations_resolved.jsonl",
        root / "data" / "Relations" / "relations_resolved.json",
        root / "Relations" / "Rel Res_IterativeRuns" / "overall_summary" / "relations_resolved.jsonl",
        root / "Relations" / "Rel Res_IterativeRuns" / "overall_summary" / "relations_resolved.json",
    ]
    p = first_existing(cands)
    if not p:
        raise FileNotFoundError("Could not find relations_resolved.jsonl/json in expected TRACE folders.")
    return p


# ---------------------------
# Load TRACE entity classes (correct source)
# ---------------------------
def load_trace_entity_classes(final_classes_path: Path, max_member_samples: int = 25) -> List[dict]:
    data = read_json(final_classes_path)
    if not isinstance(data, list):
        raise ValueError("final_classes_resolved.json must be a JSON list.")

    items=[]
    for rec in data:
        cls_label = clean_label(rec.get("class_label"))
        if not cls_label:
            continue

        cls_group = clean_label(rec.get("class_group"))
        cls_type_hint = clean_label(rec.get("class_type_hint"))
        cls_desc = clean_label(rec.get("class_description"))
        evidence = clean_label(rec.get("evidence_excerpt"))

        members = rec.get("members") or []
        member_names=[]
        member_evidence=[]
        for m in members:
            nm = clean_label(m.get("entity_name"))
            ds = clean_label(m.get("entity_description"))
            if nm:
                member_names.append(nm)
            if nm and ds:
                member_evidence.append(f"{nm}: {ds}")

        member_names = list(dict.fromkeys(member_names))[:max_member_samples]
        member_evidence = list(dict.fromkeys(member_evidence))[:max_member_samples]

        type_hint = " :: ".join([x for x in [cls_group, cls_type_hint] if x]) or "TRACE_CLASS"
        desc = cls_desc
        # evidence bucket: evidence excerpt + a couple member descriptions
        ev = evidence
        if member_evidence:
            ev = (ev + " | " if ev else "") + " ; ".join(member_evidence[:8])

        items.append({
            "source": "trace",
            "kind": "entity_class",
            "ref_anchor_ok": False,
            "label": cls_label,
            "desc": desc,
            "type_hint": type_hint,
            "evidence": ev,
            "members": " ; ".join(member_names),
            "meta": {
                "class_group": cls_group,
                "class_type_hint": cls_type_hint,
                "confidence": rec.get("confidence"),
                "candidate_id": rec.get("candidate_id") or rec.get("candidate_ids"),
                "source_files": rec.get("source_files"),
                "member_ids": rec.get("member_ids"),
            }
        })

    return items


# ---------------------------
# Load TRACE relations (correct source)
# ---------------------------
def load_trace_relations(rel_path: Path, max_surface_samples: int = 25, max_ev_samples: int = 20) -> List[dict]:
    # relations file may be jsonl or json list
    if rel_path.suffix.lower() == ".jsonl":
        rows = read_jsonl(rel_path)
    else:
        rows = read_json(rel_path)
        if isinstance(rows, dict) and "relations" in rows:
            rows = rows["relations"]
        if not isinstance(rows, list):
            raise ValueError("relations_resolved must be a list or jsonl.")

    # aggregate by canonical_rel_name
    agg = {}
    for r in rows:
        canon = clean_label(r.get("canonical_rel_name") or r.get("relation_name"))
        if not canon:
            continue

        rec = agg.setdefault(canon, {
            "canon": canon,
            "canon_desc": clean_label(r.get("canonical_rel_desc") or ""),
            "rel_cls": clean_label(r.get("rel_cls") or ""),
            "rel_cls_group": clean_label(r.get("rel_cls_group") or r.get("rel_hint_type") or ""),
            "surfaces": [],
            "evidence_excerpts": [],
            "domain_classes": [],
            "range_classes": [],
            "domain_groups": [],
            "range_groups": [],
            "examples": [],
            "count": 0,
        })

        rec["count"] += 1

        surf = clean_label(r.get("relation_surface") or "")
        if surf:
            rec["surfaces"].append(surf)

        ev = clean_label(r.get("evidence_excerpt") or "")
        if ev:
            rec["evidence_excerpts"].append(ev)

        # domain/range classes/groups from resolved relation record
        dcls = clean_label(r.get("subject_class_label") or "")
        rcls = clean_label(r.get("object_class_label") or "")
        dgrp = clean_label(r.get("subject_class_group") or "")
        rgrp = clean_label(r.get("object_class_group") or "")
        if dcls:
            rec["domain_classes"].append(dcls)
        if rcls:
            rec["range_classes"].append(rcls)
        if dgrp:
            rec["domain_groups"].append(dgrp)
        if rgrp:
            rec["range_groups"].append(rgrp)

        sname = clean_label(r.get("subject_entity_name") or "")
        oname = clean_label(r.get("object_entity_name") or "")
        if sname and oname:
            rec["examples"].append(f"{sname} -> {oname}")

    items=[]
    for canon, rec in agg.items():
        surfaces = list(dict.fromkeys(rec["surfaces"]))[:max_surface_samples]
        excerpts = list(dict.fromkeys(rec["evidence_excerpts"]))[:max_ev_samples]
        dclasses = sorted(set(rec["domain_classes"]))
        rclasses = sorted(set(rec["range_classes"]))

        # description = canonical description + domain/range classes (top few)
        desc_parts=[]
        if rec["canon_desc"]:
            desc_parts.append(rec["canon_desc"])
        if dclasses or rclasses:
            desc_parts.append(f"domain={dclasses[:6]}; range={rclasses[:6]}")
        desc = " | ".join(desc_parts)

        type_hint = " :: ".join([x for x in [rec["rel_cls_group"], rec["rel_cls"]] if x]) or "TRACE_REL"

        ev_parts=[]
        if excerpts:
            ev_parts.append(" ; ".join(excerpts[:8]))
        ex_pairs = list(dict.fromkeys(rec["examples"]))[:10]
        if ex_pairs:
            ev_parts.append("examples=" + " ; ".join(ex_pairs[:6]))
        evidence = " | ".join(ev_parts)

        items.append({
            "source": "trace",
            "kind": "relation",
            "ref_anchor_ok": False,
            "label": canon,
            "desc": desc,
            "type_hint": type_hint,
            "evidence": evidence,
            "members": " ; ".join(surfaces),  # surface forms are best "members" for relations
            "meta": {
                "canonical_rel_desc": rec["canon_desc"],
                "rel_cls_group": rec["rel_cls_group"],
                "rel_cls": rec["rel_cls"],
                "count": rec["count"],
                "domain_classes": dclasses,
                "range_classes": rclasses,
            }
        })

    return items


# ---------------------------
# Load REF ontology (correct structure)
# ---------------------------
def load_ref_ontology(ref_path: Path) -> Tuple[List[dict], List[dict], Dict[str, Tuple[str,str]]]:
    data = read_json(ref_path)
    concepts = data.get("concepts", [])
    relations = data.get("relations", [])

    ref_classes=[]
    for c in concepts:
        lbl = clean_label(c.get("label") or c.get("qid") or "")
        if not lbl:
            continue
        ref_classes.append({
            "source":"ref",
            "kind":"entity_class",
            "ref_anchor_ok":True,
            "label": lbl,
            "desc": "",
            "type_hint":"REF_ONTOLOGY",
            "evidence":"",
            "members":"",
            "meta":{"ref_qid": c.get("qid"), "ref_label": lbl}
        })

    ref_rels=[]
    rel_dr={}
    for r in relations:
        lbl = clean_label(r.get("label") or r.get("pid") or "")
        if not lbl:
            continue
        dom = clean_label(r.get("domain") or "")
        rng = clean_label(r.get("range") or "")
        rel_dr[lbl] = (dom, rng)
        ref_rels.append({
            "source":"ref",
            "kind":"relation",
            "ref_anchor_ok":True,
            "label": lbl,
            "desc": (f"domain={dom}; range={rng}" if (dom or rng) else ""),
            "type_hint":"REF_ONTOLOGY",
            "evidence":"",
            "members":"",
            "meta":{"ref_pid": r.get("pid"), "domain": dom, "range": rng}
        })

    # dedup (safety)
    def dedup(items):
        seen=set(); out=[]
        for it in items:
            if it["label"] in seen: 
                continue
            seen.add(it["label"]); out.append(it)
        return out

    return dedup(ref_classes), dedup(ref_rels), rel_dr


# ---------------------------
# Gold triples (your format): one triple per row
# ---------------------------
def load_gold_triples(gold_path: Path) -> List[dict]:
    rows = read_jsonl(gold_path)
    out=[]
    for r in rows:
        sub = clean_label(r.get("subject") or r.get("sub") or "")
        pred = clean_label(r.get("predicate") or r.get("rel") or "")
        obj = clean_label(r.get("object") or r.get("obj") or "")
        if pred and (sub or obj):
            out.append({"sub": sub, "pred": pred, "obj": obj, "sentence_id": r.get("sentence_id")})
    return out

def build_gold_index(gold_rows: List[dict], k_per_rel: int = 25) -> Tuple[Dict[str,List[str]], Dict[str,List[str]], Dict[str,List[str]]]:
    """
    Returns:
      rel2pairs[pred] = [ "sub -> obj", ... ]
      rel2subs[pred]  = [ sub, ... ]
      rel2objs[pred]  = [ obj, ... ]
    """
    rel2pairs=defaultdict(list)
    rel2subs=defaultdict(list)
    rel2objs=defaultdict(list)
    for r in gold_rows:
        p=r["pred"]; s=r["sub"]; o=r["obj"]
        if p:
            if s and o:
                rel2pairs[p].append(f"{s} -> {o}")
            if s:
                rel2subs[p].append(s)
            if o:
                rel2objs[p].append(o)
    # dedup + cut
    for d in (rel2pairs, rel2subs, rel2objs):
        for k,v in list(d.items()):
            d[k] = list(dict.fromkeys(v))[:k_per_rel]
    return rel2pairs, rel2subs, rel2objs


def attach_ref_members_from_gold(
    ref_classes: List[dict],
    ref_rels: List[dict],
    rel_domain_range: Dict[str,Tuple[str,str]],
    gold_rel2pairs: Dict[str,List[str]],
    gold_rel2subs: Dict[str,List[str]],
    gold_rel2objs: Dict[str,List[str]],
    max_members: int = 25
):
    # relations: members are gold examples
    for it in ref_rels:
        p = it["label"]
        ex = gold_rel2pairs.get(p, [])
        if ex:
            it["members"] = " ; ".join(ex[:max_members])
            it["evidence"] = f"{len(ex)} gold examples (sample): " + " ; ".join(ex[:10])

    # concepts: members gathered via domain/range predicates
    # - if C is domain of p: use subjects from p
    # - if C is range of p: use objects from p
    dom_map=defaultdict(list)
    rng_map=defaultdict(list)
    for p,(d,r) in rel_domain_range.items():
        if d:
            dom_map[d].append(p)
        if r:
            rng_map[r].append(p)

    for it in ref_classes:
        c = it["label"]
        mem=[]
        # domain preds
        for p in dom_map.get(c, []):
            mem += gold_rel2subs.get(p, [])
        # range preds
        for p in rng_map.get(c, []):
            mem += gold_rel2objs.get(p, [])
        mem = list(dict.fromkeys(mem))[:max_members]
        if mem:
            it["members"] = " ; ".join(mem)
            it["evidence"] = f"{len(mem)} instances from gold (sample): " + " ; ".join(mem[:10])
        # desc: helpful structural context
        dom_preds = dom_map.get(c, [])
        rng_preds = rng_map.get(c, [])
        if dom_preds or rng_preds:
            it["desc"] = f"domain_of={sorted(dom_preds)[:8]}; range_of={sorted(rng_preds)[:8]}"


# ---------------------------
# Embedding (same 5 buckets idea)
# ---------------------------
def embed_items(model: SentenceTransformer, items: List[dict], weights: Dict[str,float]) -> np.ndarray:
    N=len(items)
    if N==0:
        return np.zeros((0,384), dtype=np.float32)

    def col(k):
        return [str((it.get(k) or "")).strip()[:1600] for it in items]

    buckets = {
        "label": col("label"),
        "desc": col("desc"),
        "type_hint": col("type_hint"),
        "evidence": col("evidence"),
        "members": col("members"),
    }

    embs={}
    D=None
    for k,txts in buckets.items():
        if any(t for t in txts):
            e=model.encode(txts, normalize_embeddings=True, show_progress_bar=False)
            embs[k]=np.asarray(e, dtype=np.float32)
            D=embs[k].shape[1]
        else:
            embs[k]=None

    if D is None:
        raise ValueError("All text buckets empty; cannot embed.")

    for k in buckets.keys():
        if embs[k] is None:
            embs[k]=np.zeros((N,D), dtype=np.float32)

    w={k: float(weights.get(k,0.0)) for k in buckets.keys()}
    W=sum(max(0.0,x) for x in w.values())
    if W<=0:
        raise ValueError("Weights sum to 0.")
    for k in w:
        w[k]=max(0.0,w[k])/W

    X = sum(w[k]*embs[k] for k in buckets.keys())
    X = normalize(X, axis=1)
    return X


def run_hdbscan_diag(emb: np.ndarray, min_cluster_size=3, min_samples=2, use_umap=True):
    X=emb
    N=X.shape[0]
    if use_umap and UMAP_AVAILABLE and N>=6:
        import umap
        comp=min(15, max(2, N-2))
        neigh=min(15, max(2, N-1))
        try:
            reducer=umap.UMAP(n_components=comp, n_neighbors=neigh, min_dist=0.1, metric="cosine", random_state=42)
            Xr=reducer.fit_transform(X)
            if Xr.shape[0]==N:
                X=Xr
        except Exception:
            X=emb

    clusterer=hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    return clusterer.fit_predict(X)


# ---------------------------
# Anchored assignment: exactly one REF per cluster
# ---------------------------
def anchored_assign(ref_items, trace_items, ref_emb, trace_emb, min_sim=0.20):
    if len(ref_items)==0:
        raise ValueError("No ref anchors.")
    # cosine sim because normalized
    S = trace_emb @ ref_emb.T
    best = np.argmax(S, axis=1) if len(trace_items) else np.array([], dtype=int)
    best_sim = S[np.arange(S.shape[0]), best] if len(trace_items) else np.array([], dtype=float)

    clusters={}
    for r in ref_items:
        rid=f"REF::{r['kind']}::{r['label']}"
        clusters[rid]={"anchor": r, "members": [], "stats": {"n_trace":0, "dropped":0}}

    dropped_total=0
    for i,t in enumerate(trace_items):
        j=int(best[i]); sim=float(best_sim[i])
        rid=f"REF::{ref_items[j]['kind']}::{ref_items[j]['label']}"
        if sim>=min_sim:
            t2=dict(t)
            t2["anchor_label"]=ref_items[j]["label"]
            t2["anchor_sim"]=sim
            clusters[rid]["members"].append(t2)
            clusters[rid]["stats"]["n_trace"]+=1
        else:
            clusters[rid]["stats"]["dropped"]+=1
            dropped_total+=1

    clusters["_global"]={"min_sim": min_sim, "dropped_total": dropped_total}
    return clusters


# ---------------------------
# MAIN
# ---------------------------
def build_and_cluster():
    # locate canonical TRACE files
    trace_class_path = locate_trace_class_file(TRACE_ROOT)
    trace_rel_path   = locate_trace_relation_file(TRACE_ROOT)

    print("[TRACE] class file:", trace_class_path)
    print("[TRACE] rel file  :", trace_rel_path)
    print("[REF]   ontology  :", REF_ONTOLOGY)
    print("[GOLD]  triples   :", GOLD_TRIPLES)

    # load TRACE pools (correct)
    trace_ent_items = load_trace_entity_classes(trace_class_path)
    trace_rel_items = load_trace_relations(trace_rel_path)

    # load REF pools
    ref_ent_items, ref_rel_items, ref_rel_dr = load_ref_ontology(REF_ONTOLOGY)

    # gold evidence
    gold_rows = load_gold_triples(GOLD_TRIPLES)
    gold_rel2pairs, gold_rel2subs, gold_rel2objs = build_gold_index(gold_rows)

    attach_ref_members_from_gold(
        ref_ent_items, ref_rel_items, ref_rel_dr,
        gold_rel2pairs, gold_rel2subs, gold_rel2objs
    )

    # embedding model
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(MODEL_NAME)

    # weights: make "label" dominant, but TRACE benefits from desc/type/evidence/members
    ENT_WEIGHTS = {"label":0.45, "desc":0.20, "type_hint":0.15, "evidence":0.10, "members":0.10}
    REL_WEIGHTS = {"label":0.45, "desc":0.20, "type_hint":0.15, "evidence":0.10, "members":0.10}

    # embed separately
    ref_ent_emb   = embed_items(model, ref_ent_items, ENT_WEIGHTS)
    trace_ent_emb = embed_items(model, trace_ent_items, ENT_WEIGHTS)

    ref_rel_emb   = embed_items(model, ref_rel_items, REL_WEIGHTS)
    trace_rel_emb = embed_items(model, trace_rel_items, REL_WEIGHTS)

    # diagnostics: pooled HDBSCAN labels
    ent_pool = ref_ent_items + trace_ent_items
    ent_pool_emb = np.vstack([ref_ent_emb, trace_ent_emb]) if len(ent_pool) else np.zeros((0,384), np.float32)
    ent_labels = run_hdbscan_diag(ent_pool_emb) if ent_pool_emb.shape[0] else np.array([])

    rel_pool = ref_rel_items + trace_rel_items
    rel_pool_emb = np.vstack([ref_rel_emb, trace_rel_emb]) if len(rel_pool) else np.zeros((0,384), np.float32)
    rel_labels = run_hdbscan_diag(rel_pool_emb) if rel_pool_emb.shape[0] else np.array([])

    # anchored clusters
    ent_clusters = anchored_assign(ref_ent_items, trace_ent_items, ref_ent_emb, trace_ent_emb, min_sim=0.20)
    rel_clusters = anchored_assign(ref_rel_items, trace_rel_items, ref_rel_emb, trace_rel_emb, min_sim=0.20)

    # write pools
    ent_rows=[]
    for i,it in enumerate(ent_pool):
        row=dict(it)
        row["hdbscan_label"]=int(ent_labels[i]) if ent_labels.size else -1
        ent_rows.append(row)
    write_jsonl(OUT_DIR / "entity_pool_with_hdbscan_labels.jsonl", ent_rows)

    rel_rows=[]
    for i,it in enumerate(rel_pool):
        row=dict(it)
        row["hdbscan_label"]=int(rel_labels[i]) if rel_labels.size else -1
        rel_rows.append(row)
    write_jsonl(OUT_DIR / "relation_pool_with_hdbscan_labels.jsonl", rel_rows)

    # write clusters
    write_json(OUT_DIR / "entity_anchored_clusters.json", ent_clusters)
    write_json(OUT_DIR / "relation_anchored_clusters.json", rel_clusters)

    summary = {
        "paths": {
            "trace_final_classes_resolved": str(trace_class_path),
            "trace_relations_resolved": str(trace_rel_path),
            "ref_ontology": str(REF_ONTOLOGY),
            "gold_triples": str(GOLD_TRIPLES),
            "out_dir": str(OUT_DIR),
        },
        "counts": {
            "ref_concepts": len(ref_ent_items),
            "ref_relations": len(ref_rel_items),
            "trace_classes": len(trace_ent_items),
            "trace_relations": len(trace_rel_items),
            "gold_triples_rows": len(gold_rows),
        },
        "anchored": {
            "min_sim": ent_clusters["_global"]["min_sim"],
            "entity_dropped": ent_clusters["_global"]["dropped_total"],
            "relation_dropped": rel_clusters["_global"]["dropped_total"],
        }
    }
    write_json(OUT_DIR / "summary.json", summary)

    print("\n[OK] v5 complete. Outputs:")
    print(" -", OUT_DIR / "entity_pool_with_hdbscan_labels.jsonl")
    print(" -", OUT_DIR / "relation_pool_with_hdbscan_labels.jsonl")
    print(" -", OUT_DIR / "entity_anchored_clusters.json")
    print(" -", OUT_DIR / "relation_anchored_clusters.json")
    print(" -", OUT_DIR / "summary.json")


# RUN
build_and_cluster()

#endregion#?  Cluster-based Concept Mapping v5
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?     Trace_ref_schema_eval  -  V0

# trace_ref_schema_eval.py
# KDD-oriented evaluation of TRACE schema vs Text2KGBench REF ontology
# Outputs:
#  - summary.csv (one row per ontology)
#  - by_relation.csv (per REF relation)
#  - by_concept.csv (per REF concept)
#  - llm_judgements_rel.jsonl / llm_judgements_ent.jsonl (optional)

from __future__ import annotations

import json, os, re, time
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# Optional: soft matching + coherence
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False


# -------------------------
# IO
# -------------------------
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def read_json(p: Path) -> Any:
    return json.loads(read_text(p))

def read_jsonl(p: Path) -> List[dict]:
    out = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonl(p: Path, rows: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(p: Path, rows: List[dict]):
    import csv
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------
# Parsing: anchored clusters
# -------------------------
def load_anchored_clusters(path: Path) -> Tuple[Dict[str, dict], dict]:
    data = read_json(path)
    global_cfg = data.get("_global", {})
    clusters = {k: v for k, v in data.items() if k != "_global"}
    return clusters, global_cfg

def key_to_ref_label(ref_key: str) -> str:
    # "REF::relation::director" -> "director"
    parts = ref_key.split("::")
    return parts[-1] if parts else ref_key

def build_trace_to_ref_map_from_clusters(clusters: Dict[str, dict]) -> Dict[str, Tuple[str, float]]:
    """
    Returns mapping: trace_label -> (ref_label, best_sim)
    """
    out: Dict[str, Tuple[str, float]] = {}
    for ref_key, blk in clusters.items():
        ref_label = key_to_ref_label(ref_key)
        for m in blk.get("members", []) or []:
            tl = (m.get("label") or "").strip()
            sim = float(m.get("anchor_sim") or 0.0)
            if not tl:
                continue
            if (tl not in out) or (sim > out[tl][1]):
                out[tl] = (ref_label, sim)
    return out


# -------------------------
# REF ontology + gold
# -------------------------
@dataclass
class RefRelation:
    label: str
    domain: str
    range: str

def load_ref_ontology(ref_path: Path) -> Tuple[List[str], List[RefRelation]]:
    data = read_json(ref_path)
    concepts = [c.get("label") or c.get("qid") for c in data.get("concepts", [])]
    concepts = [c for c in concepts if c]
    rels = []
    for r in data.get("relations", []):
        label = (r.get("label") or r.get("pid") or "").strip()
        if not label:
            continue
        rels.append(RefRelation(
            label=label,
            domain=(r.get("domain") or "").strip(),
            range=(r.get("range") or "").strip(),
        ))
    return concepts, rels

def load_gold_triples(gold_jsonl: Path) -> List[dict]:
    rows = read_jsonl(gold_jsonl)
    out = []
    for r in rows:
        s = (r.get("subject") or r.get("sub") or "").strip()
        p = (r.get("predicate") or r.get("pred") or r.get("rel") or "").strip()
        o = (r.get("object") or r.get("obj") or "").strip()
        if p:
            out.append({"sub": s, "pred": p, "obj": o})
    return out

def active_sets_from_gold(ref_rels: List[RefRelation], gold: List[dict]) -> Tuple[Counter, Dict[str,int], Dict[str,int]]:
    """
    Returns:
      pred_freq: Counter(pred -> count)
      active_concept_freq: concept -> weighted count (domain+range contributions)
      active_pred_freq: pred -> count (only those in ref)
    """
    pred_freq_all = Counter([t["pred"] for t in gold if t.get("pred")])
    ref_pred_set = {r.label for r in ref_rels}
    active_pred_freq = Counter({p: c for p, c in pred_freq_all.items() if p in ref_pred_set})

    # concept weights: sum of predicate frequencies for predicates that use them as domain/range
    concept_freq = Counter()
    pred2dr = {r.label: (r.domain, r.range) for r in ref_rels}
    for p, c in active_pred_freq.items():
        d, r = pred2dr.get(p, ("", ""))
        if d: concept_freq[d] += c
        if r: concept_freq[r] += c

    return pred_freq_all, dict(concept_freq), dict(active_pred_freq)


# -------------------------
# Metrics: coverage + refinement + domain/range conformance
# -------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

def soft_match_label(a: str, b: str, model: SentenceTransformer, thr: float = 0.55) -> bool:
    # cheap soft match for domain/range: embedding similarity over labels
    ea = model.encode([a], normalize_embeddings=True, show_progress_bar=False)[0]
    eb = model.encode([b], normalize_embeddings=True, show_progress_bar=False)[0]
    return float(np.dot(ea, eb)) >= thr

def evaluate_relations(
    rel_clusters: Dict[str, dict],
    ent_trace2ref: Dict[str, Tuple[str,float]],
    ref_rels: List[RefRelation],
    active_pred_freq: Dict[str,int],
    sim_thresh: float = 0.20,
    soft_match: bool = True,
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[List[dict], dict]:
    """
    Per relation anchor:
      - covered_strict
      - best_sim
      - domain_match_strict / range_match_strict
      - domain_match_soft / range_match_soft (optional)
      - refinement_count (candidates >= sim_thresh)
    Summary (gold-weighted):
      - rel_coverage_weighted
      - rel_best_sim_weighted
      - dom_rng_acc_weighted (strict/soft)
      - refinement_mean (over covered)
    """
    pred2dr = {r.label: (r.domain, r.range) for r in ref_rels}
    st_model = SentenceTransformer(st_model_name) if (soft_match and ST_AVAILABLE) else None

    rows = []
    total_w = sum(active_pred_freq.values()) or 1.0

    cov_w = 0.0
    sim_w = 0.0
    dr_strict_w = 0.0
    dr_soft_w = 0.0

    refinement_counts = []

    for ref_key, blk in rel_clusters.items():
        ref_pred = key_to_ref_label(ref_key)
        if ref_pred not in pred2dr:
            continue

        w = float(active_pred_freq.get(ref_pred, 0))
        is_active = w > 0

        anchor = blk.get("anchor", {}) or {}
        dom = (anchor.get("meta", {}) or {}).get("domain", "") or pred2dr[ref_pred][0]
        rng = (anchor.get("meta", {}) or {}).get("range", "") or pred2dr[ref_pred][1]

        members = blk.get("members", []) or []
        # filter by sim threshold
        cand = [m for m in members if float(m.get("anchor_sim") or 0.0) >= sim_thresh]
        covered = len(cand) > 0
        best_sim = max([float(m.get("anchor_sim") or 0.0) for m in members], default=0.0)

        # pick best candidate for domain/range check (highest sim among filtered, else highest overall)
        cand_sorted = sorted(members, key=lambda x: float(x.get("anchor_sim") or 0.0), reverse=True)
        best = cand_sorted[0] if cand_sorted else None

        def map_trace_classes_to_ref(trace_classes: List[str]) -> List[str]:
            mapped = []
            for tc in trace_classes:
                tc = (tc or "").strip()
                if not tc:
                    continue
                if tc in ent_trace2ref:
                    mapped.append(ent_trace2ref[tc][0])
            return list(dict.fromkeys(mapped))

        dom_match_strict = False
        rng_match_strict = False
        dom_match_soft = False
        rng_match_soft = False

        if best:
            meta = best.get("meta", {}) or {}
            dom_trace = meta.get("domain_classes") or []
            rng_trace = meta.get("range_classes") or []
            dom_ref_mapped = map_trace_classes_to_ref(dom_trace)
            rng_ref_mapped = map_trace_classes_to_ref(rng_trace)

            dom_match_strict = (dom in dom_ref_mapped) if dom else False
            rng_match_strict = (rng in rng_ref_mapped) if rng else False

            if st_model and dom and dom_ref_mapped:
                dom_match_soft = any(soft_match_label(dom, x, st_model) for x in dom_ref_mapped)
            if st_model and rng and rng_ref_mapped:
                rng_match_soft = any(soft_match_label(rng, x, st_model) for x in rng_ref_mapped)

        row = {
            "ref_predicate": ref_pred,
            "active_gold_freq": int(active_pred_freq.get(ref_pred, 0)),
            "covered_strict_sim>=thr": int(covered),
            "best_anchor_sim": round(best_sim, 4),
            "trace_candidates>=thr": len(cand),
            "best_trace_label": (best.get("label") if best else ""),
            "best_trace_sim": round(float(best.get("anchor_sim") or 0.0), 4) if best else 0.0,
            "ref_domain": dom,
            "ref_range": rng,
            "domain_match_strict": int(dom_match_strict),
            "range_match_strict": int(rng_match_strict),
            "domain_match_soft": int(dom_match_soft) if st_model else "",
            "range_match_soft": int(rng_match_soft) if st_model else "",
        }
        rows.append(row)

        if is_active:
            cov_w += w * (1.0 if covered else 0.0)
            sim_w += w * best_sim
            dr_strict_w += w * (0.5 * (float(dom_match_strict) + float(rng_match_strict)))
            if st_model:
                dr_soft_w += w * (0.5 * (float(dom_match_soft) + float(rng_match_soft)))
            if covered:
                refinement_counts.append(len(cand))

    summary = {
        "rel_coverage_weighted": cov_w / total_w,
        "rel_best_sim_weighted": sim_w / total_w,
        "rel_dom_rng_acc_strict_weighted": dr_strict_w / total_w,
        "rel_dom_rng_acc_soft_weighted": (dr_soft_w / total_w) if st_model else None,
        "rel_refinement_mean_candidates": float(np.mean(refinement_counts)) if refinement_counts else 0.0,
    }
    return rows, summary


def evaluate_concepts(
    ent_clusters: Dict[str, dict],
    active_concept_freq: Dict[str,int],
    sim_thresh: float = 0.20,
) -> Tuple[List[dict], dict]:
    """
    Concept coverage + refinement on active REF concepts (weighted).
    """
    total_w = sum(active_concept_freq.values()) or 1.0
    cov_w = 0.0
    sim_w = 0.0
    refinement_counts = []

    rows = []
    for ref_key, blk in ent_clusters.items():
        ref_concept = key_to_ref_label(ref_key)
        w = float(active_concept_freq.get(ref_concept, 0))
        is_active = w > 0

        members = blk.get("members", []) or []
        cand = [m for m in members if float(m.get("anchor_sim") or 0.0) >= sim_thresh]
        covered = len(cand) > 0
        best_sim = max([float(m.get("anchor_sim") or 0.0) for m in members], default=0.0)

        best = None
        if members:
            best = sorted(members, key=lambda x: float(x.get("anchor_sim") or 0.0), reverse=True)[0]

        row = {
            "ref_concept": ref_concept,
            "active_gold_weight": int(active_concept_freq.get(ref_concept, 0)),
            "covered_strict_sim>=thr": int(covered),
            "best_anchor_sim": round(best_sim, 4),
            "trace_candidates>=thr": len(cand),
            "best_trace_label": (best.get("label") if best else ""),
            "best_trace_sim": round(float(best.get("anchor_sim") or 0.0), 4) if best else 0.0,
        }
        rows.append(row)

        if is_active:
            cov_w += w * (1.0 if covered else 0.0)
            sim_w += w * best_sim
            if covered:
                refinement_counts.append(len(cand))

    summary = {
        "concept_coverage_weighted": cov_w / total_w,
        "concept_best_sim_weighted": sim_w / total_w,
        "concept_refinement_mean_candidates": float(np.mean(refinement_counts)) if refinement_counts else 0.0,
    }
    return rows, summary


# -------------------------
# LLM Judge (optional)
# -------------------------
LLM_JUDGEMENT_SET = ["Equivalent", "Narrower", "Broader", "Unrelated"]

def _extract_json(text: str) -> Any:
    # Robustly extract JSON object or array from LLM output
    text = text.strip()
    # find first {...} or [...]
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON found in LLM output.")
    return json.loads(m.group(1))

def build_llm_prompt_relation(anchor: dict, trace_items: List[dict]) -> str:
    a_label = anchor.get("label", "")
    meta = anchor.get("meta", {}) or {}
    dom = meta.get("domain", "")
    rng = meta.get("range", "")
    gold_ex = anchor.get("members", "") or anchor.get("evidence", "")

    lines = []
    lines.append(f"REF relation: {a_label}")
    lines.append(f"REF domain: {dom}")
    lines.append(f"REF range: {rng}")
    if gold_ex:
        lines.append(f"Gold examples (may be partial): {gold_ex}")

    lines.append("\nTRACE candidates:")
    for i, t in enumerate(trace_items, 1):
        lines.append(f"{i}. label={t.get('label','')}")
        lines.append(f"   type_hint={t.get('type_hint','')}")
        lines.append(f"   desc={t.get('desc','')[:400]}")
        lines.append(f"   evidence={t.get('evidence','')[:400]}")
        lines.append(f"   anchor_sim={t.get('anchor_sim',0)}")

    lines.append("""
Task:
For each TRACE candidate, decide its semantic relation to the REF relation:
- Equivalent: same meaning
- Narrower: a correct sub-relation/refinement of the REF relation
- Broader: more general than REF
- Unrelated: not the same relation

Return STRICT JSON:
{
  "ref": "<REF label>",
  "items": [
    {"trace_label": "...", "judgement": "Equivalent|Narrower|Broader|Unrelated", "confidence": 0.0-1.0, "note": "short"}
  ]
}
""".strip())
    return "\n".join(lines)

def build_llm_prompt_concept(anchor: dict, trace_items: List[dict]) -> str:
    a_label = anchor.get("label", "")
    gold_ex = anchor.get("members", "") or anchor.get("evidence", "")
    lines = [f"REF concept: {a_label}"]
    if gold_ex:
        lines.append(f"Gold instances (sample): {gold_ex}")

    lines.append("\nTRACE candidates:")
    for i, t in enumerate(trace_items, 1):
        lines.append(f"{i}. label={t.get('label','')}")
        lines.append(f"   type_hint={t.get('type_hint','')}")
        lines.append(f"   desc={t.get('desc','')[:400]}")
        lines.append(f"   evidence={t.get('evidence','')[:400]}")
        lines.append(f"   members={t.get('members','')[:250]}")
        lines.append(f"   anchor_sim={t.get('anchor_sim',0)}")

    lines.append("""
Task:
For each TRACE candidate, decide its semantic relation to the REF concept:
- Equivalent, Narrower (valid subtype/refinement), Broader, Unrelated

Return STRICT JSON:
{
  "ref": "<REF label>",
  "items": [
    {"trace_label": "...", "judgement": "Equivalent|Narrower|Broader|Unrelated", "confidence": 0.0-1.0, "note": "short"}
  ]
}
""".strip())
    return "\n".join(lines)

def call_openai_chat(model: str, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1200) -> str:
    # Minimal OpenAI call; uses OPENAI_API_KEY env var.
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""

def llm_judge_clusters(
    clusters: Dict[str, dict],
    kind: str,  # "relation" or "concept"
    active_weights: Dict[str,int],
    out_jsonl: Path,
    model: str = "gpt-4o-mini",
    top_k: int = 5,
    sim_thresh: float = 0.20,
    sleep_s: float = 0.0,
) -> List[dict]:
    """
    Judges only ACTIVE anchors (present in active_weights).
    Saves jsonl decisions. Returns list of rows for metric aggregation.
    """
    sys = "You are a precise ontology alignment expert. Follow the rubric and return strict JSON only."

    judged_rows = []
    outputs = []

    for ref_key, blk in clusters.items():
        ref_label = key_to_ref_label(ref_key)
        if active_weights.get(ref_label, 0) <= 0:
            continue

        anchor = blk.get("anchor", {}) or {}
        members = blk.get("members", []) or []
        # judge top-k above threshold first, else top-k overall
        above = [m for m in members if float(m.get("anchor_sim") or 0.0) >= sim_thresh]
        pool = sorted(above if above else members, key=lambda x: float(x.get("anchor_sim") or 0.0), reverse=True)[:top_k]

        if not pool:
            continue

        if kind == "relation":
            prompt = build_llm_prompt_relation(anchor, pool)
        else:
            prompt = build_llm_prompt_concept(anchor, pool)

        txt = call_openai_chat(model=model, system=sys, user=prompt, temperature=0.0)
        try:
            js = _extract_json(txt)
        except Exception as e:
            js = {"ref": ref_label, "error": str(e), "raw": txt[:1500]}

        outputs.append(js)

        # flatten for metrics
        if isinstance(js, dict) and "items" in js and isinstance(js["items"], list):
            for it in js["items"]:
                j = (it.get("judgement") or "").strip()
                if j not in LLM_JUDGEMENT_SET:
                    continue
                judged_rows.append({
                    "ref": ref_label,
                    "trace_label": it.get("trace_label",""),
                    "judgement": j,
                    "confidence": float(it.get("confidence") or 0.0),
                })

        if sleep_s > 0:
            time.sleep(sleep_s)

    write_jsonl(out_jsonl, outputs)
    return judged_rows

def aggregate_llm_metrics(judged_rows: List[dict], active_weights: Dict[str,int]) -> dict:
    """
    Produces:
      - precision_valid (Equivalent or Narrower among judged)
      - valid_coverage_weighted (active anchors with >=1 valid)
      - narrower_rate (among valid)
    """
    if not judged_rows:
        return {
            "llm_precision_valid": None,
            "llm_valid_coverage_weighted": None,
            "llm_narrower_rate_among_valid": None,
        }

    judged_rows = [r for r in judged_rows if r.get("ref") in active_weights]
    if not judged_rows:
        return {
            "llm_precision_valid": None,
            "llm_valid_coverage_weighted": None,
            "llm_narrower_rate_among_valid": None,
        }

    valid = [r for r in judged_rows if r["judgement"] in ("Equivalent", "Narrower")]
    precision = (len(valid) / len(judged_rows)) if judged_rows else 0.0

    # weighted coverage: anchor is "valid" if it has at least one valid item
    ref2has_valid = defaultdict(bool)
    ref2valid_j = defaultdict(list)
    for r in valid:
        ref2has_valid[r["ref"]] = True
        ref2valid_j[r["ref"]].append(r["judgement"])

    total_w = sum(active_weights.values()) or 1.0
    cov_w = 0.0
    for ref, w in active_weights.items():
        if w <= 0:
            continue
        cov_w += float(w) * (1.0 if ref2has_valid.get(ref, False) else 0.0)

    # narrower rate (among valid judgements)
    narrower_cnt = sum(1 for r in valid if r["judgement"] == "Narrower")
    narrower_rate = (narrower_cnt / len(valid)) if valid else 0.0

    return {
        "llm_precision_valid": precision,
        "llm_valid_coverage_weighted": cov_w / total_w,
        "llm_narrower_rate_among_valid": narrower_rate,
    }


# -------------------------
# Main orchestration
# -------------------------
def evaluate_trace_vs_ref(
    ontology_id: str,
    ref_ontology_path: Path,
    gold_triples_path: Path,
    entity_clusters_path: Path,
    relation_clusters_path: Path,
    out_dir: Path,
    sim_thresh: float = 0.20,
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
    top_k_llm: int = 5,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    ent_clusters, ent_global = load_anchored_clusters(entity_clusters_path)
    rel_clusters, rel_global = load_anchored_clusters(relation_clusters_path)

    ref_concepts, ref_rels = load_ref_ontology(ref_ontology_path)
    gold = load_gold_triples(gold_triples_path)

    _, active_concept_freq, active_pred_freq = active_sets_from_gold(ref_rels, gold)

    # build trace->ref mapping for entity labels (used for domain/range conformance)
    ent_trace2ref = build_trace_to_ref_map_from_clusters(ent_clusters)

    # deterministic metrics
    rel_rows, rel_sum = evaluate_relations(
        rel_clusters=rel_clusters,
        ent_trace2ref=ent_trace2ref,
        ref_rels=ref_rels,
        active_pred_freq=active_pred_freq,
        sim_thresh=sim_thresh,
        soft_match=True,
    )
    ent_rows, ent_sum = evaluate_concepts(
        ent_clusters=ent_clusters,
        active_concept_freq=active_concept_freq,
        sim_thresh=sim_thresh,
    )

    # LLM judge (optional)
    llm_rel_metrics = {}
    llm_ent_metrics = {}
    if use_llm:
        # Only active anchors
        judged_rel = llm_judge_clusters(
            clusters=rel_clusters,
            kind="relation",
            active_weights=active_pred_freq,
            out_jsonl=out_dir / "llm_judgements_rel.jsonl",
            model=llm_model,
            top_k=top_k_llm,
            sim_thresh=sim_thresh,
        )
        llm_rel_metrics = aggregate_llm_metrics(judged_rel, active_pred_freq)

        judged_ent = llm_judge_clusters(
            clusters=ent_clusters,
            kind="concept",
            active_weights=active_concept_freq,
            out_jsonl=out_dir / "llm_judgements_ent.jsonl",
            model=llm_model,
            top_k=top_k_llm,
            sim_thresh=sim_thresh,
        )
        llm_ent_metrics = aggregate_llm_metrics(judged_ent, active_concept_freq)

    # Build summary row (single ontology)
    summary_row = {
        "ontology_id": ontology_id,

        "n_ref_concepts_total": len(ref_concepts),
        "n_ref_relations_total": len(ref_rels),

        "n_active_ref_concepts": sum(1 for k,v in active_concept_freq.items() if v > 0),
        "n_active_ref_relations": sum(1 for k,v in active_pred_freq.items() if v > 0),

        "concept_coverage_weighted": ent_sum["concept_coverage_weighted"],
        "concept_best_sim_weighted": ent_sum["concept_best_sim_weighted"],
        "concept_refinement_mean_candidates": ent_sum["concept_refinement_mean_candidates"],

        "rel_coverage_weighted": rel_sum["rel_coverage_weighted"],
        "rel_best_sim_weighted": rel_sum["rel_best_sim_weighted"],
        "rel_dom_rng_acc_strict_weighted": rel_sum["rel_dom_rng_acc_strict_weighted"],
        "rel_dom_rng_acc_soft_weighted": rel_sum["rel_dom_rng_acc_soft_weighted"],
        "rel_refinement_mean_candidates": rel_sum["rel_refinement_mean_candidates"],
    }
    summary_row.update({f"rel_{k}": v for k,v in llm_rel_metrics.items()})
    summary_row.update({f"concept_{k}": v for k,v in llm_ent_metrics.items()})

    # Write outputs
    write_csv(out_dir / "by_relation.csv", rel_rows)
    write_csv(out_dir / "by_concept.csv", ent_rows)
    write_csv(out_dir / "summary.csv", [summary_row])
    write_json(out_dir / "summary.json", {
        "ontology_id": ontology_id,
        "paths": {
            "ref_ontology": str(ref_ontology_path),
            "gold_triples": str(gold_triples_path),
            "entity_clusters": str(entity_clusters_path),
            "relation_clusters": str(relation_clusters_path),
        },
        "cluster_globals": {"entity": ent_global, "relation": rel_global},
        "active_counts": {
            "active_predicates": len(active_pred_freq),
            "active_concepts": len(active_concept_freq),
        },
        "metrics": summary_row,
    })

    return out_dir / "summary.csv"



# # Example usage (film):
# evaluate_trace_vs_ref(
#   ontology_id="film",
#   ref_ontology_path=Path(".../19_film_ontology.json"),
#   gold_triples_path=Path(".../gold_triples.jsonl"),
#   entity_clusters_path=Path(".../AnchoredClusters/entity_anchored_clusters.json"),
#   relation_clusters_path=Path(".../AnchoredClusters/relation_anchored_clusters.json"),
#   out_dir=Path(".../OntCompResults/SchemaEval"),
#   sim_thresh=0.20,
#   use_llm=True,
#   llm_model="gpt-5.1",
# )

from pathlib import Path

# =========================
# CONFIGURE THESE TWO ONLY
# =========================
DATA_ROOT = Path("Experiments/MYNE/Ex4_T2KGBench").resolve()
OUT_ROOT  = DATA_ROOT / "KGs_from_Essays" / "KG_Run_F3" / "OntCompResults" / "SchemaEval"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# DERIVED PATHS (FILM)
# =========================
ontology_id = "film"

ref_ontology_path = (
    DATA_ROOT
    / "dbpedia-webnlg"
    / "Raw"
    / "ontologies"
    / "19_film_ontology.json"
)

gold_triples_path = (
    DATA_ROOT
    / "KGs_from_Essays"
    / "KG_Run_F3"
    / "OntCompResults"
    / "gold_triples.jsonl"
)

entity_clusters_path = (
    DATA_ROOT
    / "KGs_from_Essays"
    / "KG_Run_F3"
    / "OntCompResults"
    / "AnchoredClusters"
    / "entity_anchored_clusters.json"
)

relation_clusters_path = (
    DATA_ROOT
    / "KGs_from_Essays"
    / "KG_Run_F3"
    / "OntCompResults"
    / "AnchoredClusters"
    / "relation_anchored_clusters.json"
)

# =========================
# RUN EVALUATION
# =========================
evaluate_trace_vs_ref(
    ontology_id=ontology_id,
    ref_ontology_path=ref_ontology_path,
    gold_triples_path=gold_triples_path,
    entity_clusters_path=entity_clusters_path,
    relation_clusters_path=relation_clusters_path,
    out_dir=OUT_ROOT,
    sim_thresh=0.20,
    use_llm=True,
    llm_model="gpt-4.1-nano",
)



#endregion#?   Trace_ref_schema_eval -  V0
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#?   Trace_ref_schema_eval - v1


#!/usr/bin/env python3
"""
TRACE-KG vs Text2KGBench Ontology/Schema Evaluation (KDD-grade)

Core design:
- Gold triples are used ONLY to:
    (1) identify ACTIVE reference predicates (relations) used by the text
    (2) weight evaluation by predicate frequency (fairness)
    (3) induce ACTIVE reference concepts as union of domain/range of active predicates
- AnchoredClusters provide candidate mappings (REF anchors + TRACE members).
- LLM is the PRIMARY judge for coverage and correctness:
    Covered(REF anchor) = exists TRACE candidate judged {Equivalent|Narrower} with usable_as_schema=True.
- Similarity is NOT thresholded for coverage. Similarity is used only for ranking
  (top-K candidates) and as a continuous score (AP/PR curves, MRR@K, Hits@K).

Outputs:
- summary.csv
- by_relation.csv
- by_concept.csv
- llm_rel_judgements.jsonl
- llm_ent_judgements.jsonl
- sim_calibration_rel.csv (optional, if sklearn available)
- sim_calibration_ent.csv (optional, if sklearn available)

LLM Backend:
- Prefer DSPy + your TKG_Main.TraceKGLLMConfig/make_lm_for_step if available.
- Fallback to OpenAI python SDK if you set prefer_dspy=False.

This file is notebook-friendly: argparse uses parse_known_args().
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

# Optional numeric / PR metrics
try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ============================================================
# IO helpers
# ============================================================

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def read_json(p: Path) -> Any:
    return json.loads(read_text(p))

def read_jsonl(p: Path) -> List[dict]:
    out = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonl(p: Path, rows: List[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(p: Path, rows: List[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


# ============================================================
# Data structures
# ============================================================

@dataclass
class EvalPaths:
    ref_ontology_path: Path
    gold_triples_path: Path
    entity_clusters_path: Path
    relation_clusters_path: Path
    out_dir: Path
    ontology_id: str = "unknown"

@dataclass
class LLMConfig:
    use_llm: bool = True
    model: str = "gpt-4o-mini"
    max_tokens: int = 1400
    top_k: int = 6
    prefer_dspy: bool = True
    # Cache behavior
    reuse_cached: bool = True
    # If True, judge all anchors (active+inactive). Usually False to save cost.
    judge_inactive_anchors: bool = False

# @dataclass
# class EvalConfig:
#     llm: LLMConfig = LLMConfig()
#     # Split handling
#     compute_train_test: bool = True
#     # Similarity calibration / PR curve
#     compute_pr_curve: bool = True

from dataclasses import dataclass, field

@dataclass
class EvalConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    # Split handling
    compute_train_test: bool = True
    # Similarity calibration / PR curve
    compute_pr_curve: bool = True

# ============================================================
# Parsing: AnchoredClusters
# ============================================================

def load_anchored_clusters(path: Path) -> Tuple[Dict[str, dict], dict]:
    data = read_json(path)
    global_cfg = data.get("_global", {}) if isinstance(data, dict) else {}
    clusters = {k: v for k, v in data.items() if k != "_global"} if isinstance(data, dict) else {}
    return clusters, global_cfg

def ref_key_to_label(ref_key: str) -> str:
    # "REF::relation::releaseDate" -> "releaseDate"
    parts = (ref_key or "").split("::")
    return parts[-1] if parts else ref_key

def ref_key_to_kind(ref_key: str) -> str:
    # "REF::relation::releaseDate" -> "relation"
    parts = (ref_key or "").split("::")
    return parts[1] if len(parts) >= 3 else "unknown"


# ============================================================
# REF ontology and gold triples
# ============================================================

@dataclass
class RefRelation:
    label: str
    domain: str
    range: str

def load_ref_ontology(ref_path: Path) -> Tuple[List[str], List[RefRelation]]:
    data = read_json(ref_path)
    concepts = data.get("concepts", []) if isinstance(data, dict) else []
    relations = data.get("relations", []) if isinstance(data, dict) else []

    ref_concepts = []
    for c in concepts:
        lbl = (c.get("label") or c.get("qid") or "").strip()
        if lbl:
            ref_concepts.append(lbl)

    ref_rels = []
    for r in relations:
        lbl = (r.get("label") or r.get("pid") or "").strip()
        if not lbl:
            continue
        dom = (r.get("domain") or "").strip()
        rng = (r.get("range") or "").strip()
        ref_rels.append(RefRelation(label=lbl, domain=dom, range=rng))

    # Dedup safety
    ref_concepts = list(dict.fromkeys(ref_concepts))
    seen = set()
    dedup_rels = []
    for rr in ref_rels:
        if rr.label in seen:
            continue
        seen.add(rr.label)
        dedup_rels.append(rr)
    return ref_concepts, dedup_rels

def infer_split(sentence_id: str) -> str:
    s = (sentence_id or "").lower()
    # robust: match _train_, _test_, _valid_ or common aliases
    if re.search(r"(^|_)train(_|$)", s):
        return "train"
    if re.search(r"(^|_)valid(_|$)|(^|_)dev(_|$)", s):
        return "valid"
    if re.search(r"(^|_)test(_|$)", s):
        return "test"
    return "all"

def load_gold_triples_any_format(path: Path) -> List[dict]:
    """
    Supports either:
    - simplified one-triple-per-line: {sentence_id, subject, predicate, object}
    - original Text2KGBench ground_truth format: {id, sent, triples:[{sub,rel,obj},...]}
    Returns list of dicts: {split, sentence_id, sub, pred, obj}
    """
    rows = read_jsonl(path)
    out = []
    for r in rows:
        # original format
        if "triples" in r and isinstance(r.get("triples"), list):
            sid = r.get("id") or r.get("sentence_id") or ""
            sp = infer_split(str(sid))
            for t in r.get("triples", []):
                sub = (t.get("sub") or t.get("subject") or "").strip()
                pred = (t.get("rel") or t.get("predicate") or "").strip()
                obj = (t.get("obj") or t.get("object") or "").strip()
                if pred:
                    out.append({"split": sp, "sentence_id": sid, "sub": sub, "pred": pred, "obj": obj})
        else:
            sid = r.get("sentence_id") or r.get("id") or ""
            sp = infer_split(str(sid))
            sub = (r.get("subject") or r.get("sub") or "").strip()
            pred = (r.get("predicate") or r.get("pred") or r.get("rel") or "").strip()
            obj = (r.get("object") or r.get("obj") or "").strip()
            if pred:
                out.append({"split": sp, "sentence_id": sid, "sub": sub, "pred": pred, "obj": obj})
    return out

def compute_active_sets(
    ref_rels: List[RefRelation],
    gold: List[dict],
) -> Dict[str, dict]:
    """
    Returns per-split:
      {
        split: {
          "active_pred_freq": {pred -> count},
          "active_concept_weight": {concept -> weight},  # from domain/range of active preds
          "all_pred_freq": {pred -> count}               # includes any pred seen
        }
      }
    """
    pred2dr = {r.label: (r.domain, r.range) for r in ref_rels}
    ref_pred_set = set(pred2dr.keys())

    by_split = defaultdict(list)
    for t in gold:
        by_split[t["split"]].append(t)
        by_split["all"].append(t)  # always include

    out = {}
    for sp, triples in by_split.items():
        all_pred = Counter([x["pred"] for x in triples if x.get("pred")])
        active_pred = Counter({p: c for p, c in all_pred.items() if p in ref_pred_set})

        concept_w = Counter()
        for p, c in active_pred.items():
            d, r = pred2dr.get(p, ("", ""))
            if d:
                concept_w[d] += c
            if r:
                concept_w[r] += c

        out[sp] = {
            "all_pred_freq": dict(all_pred),
            "active_pred_freq": dict(active_pred),
            "active_concept_weight": dict(concept_w),
        }
    return out


# ============================================================
# Candidate extraction (top-K) from anchored clusters
# ============================================================

def get_topk_members(cluster_block: dict, k: int) -> List[dict]:
    members = cluster_block.get("members", []) or []
    # Sort by anchor_sim descending if present
    def sim(m): 
        try:
            return float(m.get("anchor_sim") or 0.0)
        except Exception:
            return 0.0
    members_sorted = sorted(members, key=sim, reverse=True)
    return members_sorted[:max(0, int(k))]

def build_auto_entity_map(entity_clusters: Dict[str, dict]) -> Dict[str, str]:
    """
    Deterministic map: TRACE class label -> REF concept label (anchor it was assigned to).
    No thresholds, no LLM.
    """
    m = {}
    for ref_key, blk in entity_clusters.items():
        ref_label = ref_key_to_label(ref_key)
        for it in blk.get("members", []) or []:
            tl = (it.get("label") or "").strip()
            if tl:
                m[tl] = ref_label
    return m


# ============================================================
# LLM backends
# ============================================================

class LLMBackend:
    def complete(self, system: str, user: str, max_tokens: int) -> str:
        raise NotImplementedError

# class DSPyBackend(LLMBackend):
#     """
#     Uses your TRACE KG DSPy gateway from TKG_Main.py:
#       TraceKGLLMConfig + make_lm_for_step
#     """
#     def __init__(self, model: str, max_tokens: int, step: str = "rel_res", tkg_module_name: str = "TKG_Main"):
#         import importlib
#         self.tkg = importlib.import_module(tkg_module_name)
#         self.cfg = self.tkg.TraceKGLLMConfig(default_model=model, max_tokens=max_tokens, temperature=None)
#         self.lm = self.tkg.make_lm_for_step(self.cfg, step)

#     def complete(self, system: str, user: str, max_tokens: int) -> str:
#         # We embed system into the user prompt (DSPy LM interface)
#         prompt = f"{system}\n\n{user}"
#         try:
#             out = self.lm(prompt)
#         except Exception as e:
#             return f'{{"error":"DSPy LM call failed: {str(e)}"}}'
#         if isinstance(out, list):
#             return str(out[0] if out else "")
#         return str(out or "")



class DSPyBackend(LLMBackend):
    """
    Uses your TRACE KG DSPy gateway from TKG_Main.py:
      TraceKGLLMConfig + make_lm_for_step
    """
    def __init__(self, model: str, max_tokens: int, step: str = "rel_res", tkg_module_name: str = "TKG_Main"):
        import importlib
        self.tkg = importlib.import_module(tkg_module_name)
        self.cfg = self.tkg.TraceKGLLMConfig(default_model=model, max_tokens=max_tokens, temperature=None)
        self.lm = self.tkg.make_lm_for_step(self.cfg, step)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        prompt = f"{system}\n\n{user}"
        try:
            out = self.lm(prompt)
        except Exception as e:
            return f'{{"error":"DSPy LM call failed: {str(e)}"}}'

        # ✅ CRITICAL: unwrap DSPy/Responses objects into plain text
        txt = self.tkg.coerce_llm_text(out).strip()

        # strip markdown fences if model adds them
        if txt.startswith("```"):
            txt = txt.strip("`").strip()
            if txt.lower().startswith("json"):
                txt = txt[4:].strip()

        return txt

class OpenAIBackend(LLMBackend):
    """
    Fallback backend using OpenAI python SDK (minimal).
    """
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        # Use chat.completions for broad compatibility.
        # If you prefer DSPy (recommended in your repo), set prefer_dspy=True.
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f'{{"error":"OpenAI call failed: {str(e)}"}}'

def make_backend(llm_cfg: LLMConfig, step: str) -> Optional[LLMBackend]:
    if not llm_cfg.use_llm:
        return None
    if llm_cfg.prefer_dspy:
        try:
            return DSPyBackend(model=llm_cfg.model, max_tokens=llm_cfg.max_tokens, step=step)
        except Exception:
            # fallback
            try:
                return OpenAIBackend(model=llm_cfg.model)
            except Exception:
                return None
    else:
        try:
            return OpenAIBackend(model=llm_cfg.model)
        except Exception:
            return None


# ============================================================
# LLM prompting + JSON extraction
# ============================================================

JUDGEMENTS = ["Equivalent", "Narrower", "Broader", "Unrelated"]
ACTIONS = ["Keep", "Merge", "Split", "Reject"]

# def _extract_json_obj(text: str) -> dict:
#     """
#     Robustly extract a JSON object from model output.
#     """
#     s = (text or "").strip()
#     # find first {...}
#     m = re.search(r"\{.*\}", s, flags=re.DOTALL)
#     if not m:
#         return {"error": "no_json_object_found", "raw": s[:1200]}
#     blob = m.group(0)
#     # remove trailing commas
#     blob = re.sub(r",\s*([\]}])", r"\1", blob)
#     try:
#         return json.loads(blob)
#     except Exception as e:
#         return {"error": f"json_parse_failed: {str(e)}", "raw": blob[:1200]}


def _extract_json_obj(text: str) -> dict:
    s = (text or "").strip()
    if not s:
        return {"error": "empty_response"}

    # remove markdown fences
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()

    # find first JSON object by brace matching
    i = s.find("{")
    if i == -1:
        return {"error": "no_open_brace_found", "raw": s[:1200]}

    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = s[i:j+1]
                    blob = re.sub(r",\s*([\]}])", r"\1", blob)  # trailing commas
                    try:
                        obj = json.loads(blob)
                        # if the model returned a JSON-encoded string, parse again
                        if isinstance(obj, str) and obj.strip().startswith("{"):
                            return json.loads(obj)
                        return obj
                    except Exception as e:
                        return {"error": f"json_parse_failed: {str(e)}", "raw": blob[:1200]}

    return {"error": "no_matching_close_brace", "raw": s[i:i+1200]}




def _hash_prompt(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]

def build_relation_prompt(anchor: dict, candidates: List[dict]) -> str:
    label = anchor.get("label", "")
    meta = anchor.get("meta", {}) or {}
    dom = meta.get("domain", "")
    rng = meta.get("range", "")
    gold_examples = anchor.get("members") or anchor.get("evidence") or ""

    lines = []
    lines.append(f"REF relation (anchor): {label}")
    lines.append(f"REF domain: {dom}")
    lines.append(f"REF range: {rng}")
    if gold_examples:
        lines.append(f"Gold examples (may be partial): {str(gold_examples)[:600]}")

    lines.append("\nTRACE candidates (ranked by similarity):")
    for i, c in enumerate(candidates, 1):
        meta_c = c.get("meta", {}) or {}
        lines.append(f"\n[{i}] trace_label: {c.get('label','')}")
        lines.append(f"    anchor_sim: {c.get('anchor_sim',0)}")
        lines.append(f"    type_hint: {c.get('type_hint','')}")
        lines.append(f"    desc: {str(c.get('desc',''))[:400]}")
        lines.append(f"    evidence: {str(c.get('evidence',''))[:400]}")
        # domain/range from TRACE relation meta
        lines.append(f"    trace_domain_classes: {meta_c.get('domain_classes', [])}")
        lines.append(f"    trace_range_classes: {meta_c.get('range_classes', [])}")

    lines.append(
        """
Task:
For EACH TRACE candidate, classify its semantic relation to the REF relation:

- Equivalent: same meaning AND appropriate domain/range behavior.
- Narrower: a correct refinement/sub-relation of REF (implies REF); may be more specific.
- Broader: more general than REF (REF is a special case).
- Unrelated: does not correspond.

Also set:
- usable_as_schema: true if this TRACE candidate is appropriate to map under the REF anchor in an evaluation table.
- suggested_action: Keep/Merge/Split/Reject
- confidence: 0..1
- note: short justification (1 sentence)

Return STRICT JSON ONLY:
{
  "ref_label": "<REF relation>",
  "ref_kind": "relation",
  "items": [
    {
      "trace_label": "...",
      "judgement": "Equivalent|Narrower|Broader|Unrelated",
      "usable_as_schema": true/false,
      "suggested_action": "Keep|Merge|Split|Reject",
      "confidence": 0.0,
      "note": "..."
    }
  ]
}
""".strip()
    )
    return "\n".join(lines)

def build_concept_prompt(anchor: dict, candidates: List[dict]) -> str:
    label = anchor.get("label", "")
    gold_instances = anchor.get("members") or anchor.get("evidence") or ""
    desc = anchor.get("desc") or ""

    lines = []
    lines.append(f"REF concept (anchor): {label}")
    if desc:
        lines.append(f"REF context: {str(desc)[:600]}")
    if gold_instances:
        lines.append(f"Gold instances (sample): {str(gold_instances)[:600]}")

    lines.append("\nTRACE candidates (ranked by similarity):")
    for i, c in enumerate(candidates, 1):
        meta_c = c.get("meta", {}) or {}
        lines.append(f"\n[{i}] trace_label: {c.get('label','')}")
        lines.append(f"    anchor_sim: {c.get('anchor_sim',0)}")
        lines.append(f"    type_hint: {c.get('type_hint','')}")
        lines.append(f"    desc: {str(c.get('desc',''))[:400]}")
        lines.append(f"    evidence: {str(c.get('evidence',''))[:400]}")
        lines.append(f"    members: {str(c.get('members',''))[:250]}")
        lines.append(f"    trace_meta: class_group={meta_c.get('class_group','')}, class_type_hint={meta_c.get('class_type_hint','')}")

    lines.append(
        """
Task:
For EACH TRACE candidate, classify its semantic relation to the REF concept:

- Equivalent: same concept/type.
- Narrower: valid subtype/refinement of REF.
- Broader: supertype/generalization.
- Unrelated: not the same concept.

Also set:
- usable_as_schema: true if this TRACE candidate is appropriate to map under the REF anchor in evaluation.
- suggested_action: Keep/Merge/Split/Reject
- confidence: 0..1
- note: short justification

Return STRICT JSON ONLY:
{
  "ref_label": "<REF concept>",
  "ref_kind": "concept",
  "items": [
    {
      "trace_label": "...",
      "judgement": "Equivalent|Narrower|Broader|Unrelated",
      "usable_as_schema": true/false,
      "suggested_action": "Keep|Merge|Split|Reject",
      "confidence": 0.0,
      "note": "..."
    }
  ]
}
""".strip()
    )
    return "\n".join(lines)

def normalize_llm_items(obj: dict, fallback_ref_label: str) -> List[dict]:
    """
    Extract standardized items:
      {trace_label, judgement, usable_as_schema, confidence, suggested_action, note}
    """
    items = obj.get("items", []) if isinstance(obj, dict) else []
    out = []
    for it in items:
        tl = (it.get("trace_label") or "").strip()
        jd = (it.get("judgement") or "").strip()
        ua = bool(it.get("usable_as_schema")) if "usable_as_schema" in it else False
        ac = (it.get("suggested_action") or "").strip()
        try:
            cf = float(it.get("confidence") or 0.0)
        except Exception:
            cf = 0.0
        note = (it.get("note") or "").strip()

        if not tl:
            continue
        if jd not in JUDGEMENTS:
            jd = "Unrelated"
        if ac not in ACTIONS:
            ac = "Reject"
        out.append({
            "ref_label": (obj.get("ref_label") or fallback_ref_label),
            "trace_label": tl,
            "judgement": jd,
            "usable_as_schema": ua,
            "suggested_action": ac,
            "confidence": cf,
            "note": note,
        })
    return out


# ============================================================
# LLM judging with caching (one call per anchor, top-K candidates together)
# ============================================================

def load_cached_judgements(jsonl_path: Path) -> Dict[str, dict]:
    """
    Cache key: ref_kind + "::" + ref_label
    Stored value: dict record with parsed "items"
    """
    cache = {}
    if not jsonl_path.exists():
        return cache
    for r in read_jsonl(jsonl_path):
        k = f"{r.get('ref_kind','')}::{r.get('ref_label','')}"
        if r.get("items"):
            cache[k] = r
    return cache

def judge_anchors(
    *,
    clusters: Dict[str, dict],
    kind: str,  # "relation" or "concept"
    active_weight: Dict[str, int],
    out_jsonl: Path,
    backend: Optional[LLMBackend],
    llm_cfg: LLMConfig,
) -> Tuple[List[dict], Dict[str, dict]]:
    """
    Returns:
      flat_items: list of normalized judgement items (one row per candidate)
      per_anchor: ref_label -> {covered_valid, covered_equiv, first_valid_rank, first_equiv_rank, ...}
    Writes a JSONL file with one record per anchor.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cached_judgements(out_jsonl) if llm_cfg.reuse_cached else {}

    flat_items: List[dict] = []
    per_anchor: Dict[str, dict] = {}

    system = "You are a precise ontology alignment judge. Return strict JSON only. No markdown."

    # Decide which anchors to judge
    def should_judge(ref_label: str) -> bool:
        if llm_cfg.judge_inactive_anchors:
            return True
        return active_weight.get(ref_label, 0) > 0

    records_to_write = []

    for ref_key, blk in clusters.items():
        ref_label = ref_key_to_label(ref_key)
        if not should_judge(ref_label):
            continue

        anchor = blk.get("anchor", {}) or {}
        # candidates always ranked by anchor_sim; no threshold
        cands = get_topk_members(blk, llm_cfg.top_k)

        # if no candidates, still produce an anchor record (covered=false)
        cache_key = f"{kind}::{ref_label}"
        if llm_cfg.reuse_cached and cache_key in cache:
            rec = cache[cache_key]
            items_norm = normalize_llm_items(rec, ref_label)
        else:
            if backend is None:
                # no LLM mode: empty judgement
                rec = {"ref_kind": kind, "ref_label": ref_label, "items": [], "raw": "no_llm_backend"}
                items_norm = []
            else:
                if kind == "relation":
                    prompt = build_relation_prompt(anchor, cands)
                else:
                    prompt = build_concept_prompt(anchor, cands)

                raw = backend.complete(system=system, user=prompt, max_tokens=llm_cfg.max_tokens)
                obj = _extract_json_obj(raw)

                items_norm = normalize_llm_items(obj, ref_label)

                rec = {
                    "ref_kind": kind,
                    "ref_label": ref_label,
                    "llm_model": llm_cfg.model,
                    "prompt_hash": _hash_prompt(prompt),
                    "top_k": llm_cfg.top_k,
                    "timestamp": time.time(),
                    "items": items_norm,   # normalized items only
                    "raw_error": obj.get("error", ""),
                }

            # polite tiny sleep for rate-limit safety (set to 0 if you prefer)
            # time.sleep(0.05)

        records_to_write.append(rec)

        # Build per-anchor stats from items_norm (rank is by candidate list order)
        # We need candidate rank order; reconstruct rank map from cands
        rank_map = { (c.get("label") or "").strip(): (i+1) for i, c in enumerate(cands) }

        def is_valid(it):
            return (it["judgement"] in ("Equivalent", "Narrower")) and bool(it["usable_as_schema"])

        def is_equiv(it):
            return (it["judgement"] == "Equivalent") and bool(it["usable_as_schema"])

        valid_items = [it for it in items_norm if is_valid(it)]
        equiv_items = [it for it in items_norm if is_equiv(it)]

        # ranks
        valid_ranks = sorted([rank_map.get(it["trace_label"], 10**9) for it in valid_items])
        equiv_ranks = sorted([rank_map.get(it["trace_label"], 10**9) for it in equiv_items])

        first_valid_rank = valid_ranks[0] if valid_ranks and valid_ranks[0] < 10**9 else None
        first_equiv_rank = equiv_ranks[0] if equiv_ranks and equiv_ranks[0] < 10**9 else None

        per_anchor[ref_label] = {
            "ref_label": ref_label,
            "active_weight_all": int(active_weight.get(ref_label, 0)),
            "n_candidates_present": len(blk.get("members", []) or []),
            "n_judged": len(items_norm),
            "n_valid": len(valid_items),
            "n_equiv": len(equiv_items),
            "covered_valid": int(len(valid_items) > 0),
            "covered_equiv": int(len(equiv_items) > 0),
            "first_valid_rank": first_valid_rank or "",
            "first_equiv_rank": first_equiv_rank or "",
        }

        flat_items.extend(items_norm)

    # write judgements
    write_jsonl(out_jsonl, records_to_write)
    return flat_items, per_anchor


# ============================================================
# Metrics aggregation (no similarity thresholds)
# ============================================================

def weighted_mean(values: List[float], weights: List[float]) -> float:
    denom = sum(weights) if weights else 0.0
    if denom <= 0:
        return 0.0
    return sum(v*w for v, w in zip(values, weights)) / denom

def compute_rank_metrics(
    per_anchor: Dict[str, dict],
    active_weight: Dict[str, int],
    k: int,
) -> dict:
    """
    Computes weighted coverage + MRR@K + Hits@K using LLM-validity.
    """
    anchors = [a for a,w in active_weight.items() if w > 0]
    if not anchors:
        return {
            "coverage_valid_weighted": 0.0,
            "coverage_equiv_weighted": 0.0,
            "hits_at_k_valid_weighted": 0.0,
            "mrr_at_k_valid_weighted": 0.0,
        }

    cov_v = []
    cov_e = []
    hitk = []
    mrr = []
    wts = []

    for a in anchors:
        w = float(active_weight.get(a, 0))
        rec = per_anchor.get(a, {})
        covered_v = float(rec.get("covered_valid", 0))
        covered_e = float(rec.get("covered_equiv", 0))
        r = rec.get("first_valid_rank", "")
        try:
            r = int(r) if r != "" else None
        except Exception:
            r = None

        hit = 1.0 if (r is not None and r <= k) else 0.0
        rr  = (1.0 / r) if (r is not None and r <= k and r > 0) else 0.0

        cov_v.append(covered_v)
        cov_e.append(covered_e)
        hitk.append(hit)
        mrr.append(rr)
        wts.append(w)

    return {
        "coverage_valid_weighted": weighted_mean(cov_v, wts),
        "coverage_equiv_weighted": weighted_mean(cov_e, wts),
        "hits_at_k_valid_weighted": weighted_mean(hitk, wts),
        "mrr_at_k_valid_weighted": weighted_mean(mrr, wts),
    }

def compute_candidate_precision_metrics(flat_items: List[dict]) -> dict:
    """
    Candidate-level precision (among judged candidates):
      valid = (Equivalent or Narrower) AND usable_as_schema
    """
    if not flat_items:
        return {
            "candidate_precision_valid": None,
            "candidate_narrower_rate_among_valid": None,
        }
    valid = [x for x in flat_items if (x["judgement"] in ("Equivalent","Narrower")) and bool(x["usable_as_schema"])]
    precision = (len(valid) / len(flat_items)) if flat_items else 0.0
    narrower = [x for x in valid if x["judgement"] == "Narrower"]
    narrower_rate = (len(narrower) / len(valid)) if valid else 0.0
    return {
        "candidate_precision_valid": precision,
        "candidate_narrower_rate_among_valid": narrower_rate,
    }

def compute_refinement_metrics(per_anchor: Dict[str, dict], active_weight: Dict[str,int]) -> dict:
    """
    Refinement = average number of VALID candidates per active anchor.
    """
    vals = []
    wts = []
    for ref, w in active_weight.items():
        if w <= 0:
            continue
        n_valid = float(per_anchor.get(ref, {}).get("n_valid", 0))
        vals.append(n_valid)
        wts.append(float(w))
    return {
        "refinement_mean_valid_candidates_weighted": weighted_mean(vals, wts) if wts else 0.0
    }


# ============================================================
# Similarity calibration (threshold-free): AP + PR curve (optional)
# ============================================================

def compute_similarity_calibration(
    clusters: Dict[str, dict],
    flat_items: List[dict],
    kind: str,
    out_csv_path: Optional[Path] = None,
) -> dict:
    """
    Builds dataset of (anchor_sim, is_valid) for judged candidates,
    then computes average precision + optional PR curve points.
    """
    if not SKLEARN_OK:
        return {"ap": None, "n_points": 0, "note": "sklearn_not_available"}

    # Map (ref_label, trace_label) -> (anchor_sim, is_valid)
    # We get anchor_sim from clusters; judgement from flat_items.
    # Build quick index from clusters for anchor_sim
    sim_index = {}
    for ref_key, blk in clusters.items():
        ref_label = ref_key_to_label(ref_key)
        for m in blk.get("members", []) or []:
            tl = (m.get("label") or "").strip()
            if not tl:
                continue
            try:
                s = float(m.get("anchor_sim") or 0.0)
            except Exception:
                s = 0.0
            sim_index[(ref_label, tl)] = s

    y_true = []
    y_score = []
    for it in flat_items:
        ref_label = it.get("ref_label","")
        tl = it.get("trace_label","")
        if (ref_label, tl) not in sim_index:
            continue
        s = sim_index[(ref_label, tl)]
        valid = (it["judgement"] in ("Equivalent","Narrower")) and bool(it["usable_as_schema"])
        y_true.append(1 if valid else 0)
        y_score.append(float(s))

    if not y_true:
        return {"ap": None, "n_points": 0, "note": "no_scored_pairs"}

    ap = float(average_precision_score(y_true, y_score))
    prec, rec, thr = precision_recall_curve(y_true, y_score)

    if out_csv_path is not None:
        rows = []
        # precision_recall_curve returns thr length = len(prec)-1
        for i in range(len(thr)):
            rows.append({
                "kind": kind,
                "threshold": float(thr[i]),
                "precision": float(prec[i]),
                "recall": float(rec[i]),
            })
        # final point (no threshold)
        rows.append({
            "kind": kind,
            "threshold": "",
            "precision": float(prec[-1]),
            "recall": float(rec[-1]),
        })
        write_csv(out_csv_path, rows)

    return {"ap": ap, "n_points": len(y_true), "note": ""}


# ============================================================
# Domain/Range consistency (uses best VALID relation candidate + concept mapping)
# ============================================================

def build_llm_valid_concept_map(flat_concept_items: List[dict]) -> Dict[str, str]:
    """
    From LLM-judged concept candidates:
      map TRACE class label -> REF concept anchor label
    Only for valid mappings (Equivalent/Narrower & usable_as_schema).
    If a trace label appears multiple times, keep the best by:
      Equivalent > Narrower, then higher confidence.
    """
    best = {}
    priority = {"Equivalent": 2, "Narrower": 1, "Broader": 0, "Unrelated": 0}
    for it in flat_concept_items:
        tl = it["trace_label"]
        ref = it["ref_label"]
        jd = it["judgement"]
        ua = bool(it["usable_as_schema"])
        if not ua or jd not in ("Equivalent","Narrower"):
            continue
        score = (priority.get(jd,0), float(it.get("confidence") or 0.0))
        if tl not in best or score > best[tl][0]:
            best[tl] = (score, ref)
    return {tl: ref for tl, (_, ref) in best.items()}

def evaluate_domain_range(
    ref_rels: List[RefRelation],
    rel_clusters: Dict[str, dict],
    rel_per_anchor: Dict[str, dict],
    rel_flat_items: List[dict],
    concept_map_llm: Dict[str, str],
    concept_map_auto: Dict[str, str],
    active_pred_freq: Dict[str,int],
    top_k: int,
) -> dict:
    """
    For each active REF predicate:
      - find best VALID TRACE candidate among top-K (using LLM judgements)
      - map its TRACE domain/range classes to REF concepts via concept_map_llm (fallback to auto)
      - domain_match = REF domain in mapped_domains
      - range_match  = REF range  in mapped_ranges
    Returns weighted accuracies.
    """
    pred2dr = {r.label: (r.domain, r.range) for r in ref_rels}

    # Build lookup of LLM judgements: ref_label -> {trace_label -> valid}
    judg_by_ref = defaultdict(dict)
    for it in rel_flat_items:
        ref = it["ref_label"]
        tl = it["trace_label"]
        valid = (it["judgement"] in ("Equivalent","Narrower")) and bool(it["usable_as_schema"])
        judg_by_ref[ref][tl] = valid

    total_w = sum(active_pred_freq.values()) or 1.0
    scores_llm_map = []
    scores_auto_map = []
    wts = []
    compared = 0

    for pred, w in active_pred_freq.items():
        if w <= 0:
            continue
        dom, rng = pred2dr.get(pred, ("",""))
        if pred not in rel_clusters:
            # rel_clusters keys are "REF::relation::X" not label; handle below
            pass

        # Find cluster block by searching key
        # (we keep it safe: O(#clusters) ok for small sizes)
        blk = None
        for ref_key, b in rel_clusters.items():
            if ref_key_to_label(ref_key) == pred:
                blk = b
                break
        if blk is None:
            continue

        # Candidate list ranked by sim; take top_k
        cands = get_topk_members(blk, top_k)
        # Find first VALID candidate according to LLM judgements
        best_cand = None
        for c in cands:
            tl = (c.get("label") or "").strip()
            if not tl:
                continue
            if judg_by_ref.get(pred, {}).get(tl, False):
                best_cand = c
                break
        if best_cand is None:
            continue

        meta = best_cand.get("meta", {}) or {}
        dom_trace = meta.get("domain_classes", []) or []
        rng_trace = meta.get("range_classes", []) or []

        def map_list(trace_list: List[str], primary: Dict[str,str], fallback: Dict[str,str]) -> List[str]:
            out = []
            for x in trace_list:
                x = (x or "").strip()
                if not x:
                    continue
                if x in primary:
                    out.append(primary[x])
                elif x in fallback:
                    out.append(fallback[x])
            return list(dict.fromkeys(out))

        dom_m_llm = map_list(dom_trace, concept_map_llm, concept_map_auto)
        rng_m_llm = map_list(rng_trace, concept_map_llm, concept_map_auto)
        dom_m_auto = map_list(dom_trace, concept_map_auto, concept_map_auto)
        rng_m_auto = map_list(rng_trace, concept_map_auto, concept_map_auto)

        dom_ok_llm = 1.0 if (dom and dom in dom_m_llm) else 0.0
        rng_ok_llm = 1.0 if (rng and rng in rng_m_llm) else 0.0
        dom_ok_auto = 1.0 if (dom and dom in dom_m_auto) else 0.0
        rng_ok_auto = 1.0 if (rng and rng in rng_m_auto) else 0.0

        score_llm = 0.5 * (dom_ok_llm + rng_ok_llm) if (dom or rng) else 0.0
        score_auto = 0.5 * (dom_ok_auto + rng_ok_auto) if (dom or rng) else 0.0

        scores_llm_map.append(score_llm)
        scores_auto_map.append(score_auto)
        wts.append(float(w))
        compared += 1

    return {
        "domain_range_acc_weighted_llmMap_fallbackAuto": weighted_mean(scores_llm_map, wts) if wts else 0.0,
        "domain_range_acc_weighted_autoMap": weighted_mean(scores_auto_map, wts) if wts else 0.0,
        "domain_range_n_compared": compared,
    }


# ============================================================
# Main evaluation routine (single ontology/run)
# ============================================================

def run_schema_evaluation(paths: EvalPaths, cfg: EvalConfig) -> Path:
    out_dir = paths.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load clusters
    ent_clusters, ent_global = load_anchored_clusters(paths.entity_clusters_path)
    rel_clusters, rel_global = load_anchored_clusters(paths.relation_clusters_path)

    # Load reference ontology + gold
    ref_concepts, ref_rels = load_ref_ontology(paths.ref_ontology_path)
    gold = load_gold_triples_any_format(paths.gold_triples_path)
    active_sets = compute_active_sets(ref_rels, gold)

    # Determine which split weights to use for judging (use "all" for broad reuse)
    active_pred_all = active_sets["all"]["active_pred_freq"]
    active_concept_all = active_sets["all"]["active_concept_weight"]

    # Intersect active concepts with available REF concept anchors (important!)
    anchored_ref_concepts = {ref_key_to_label(k) for k in ent_clusters.keys()}
    active_concept_all_anchored = {c: w for c, w in active_concept_all.items() if c in anchored_ref_concepts}
    missing_active_concepts = sorted([c for c in active_concept_all.keys() if c not in anchored_ref_concepts])

    # LLM backends (separate steps for better override compatibility)
    rel_backend = make_backend(cfg.llm, step="rel_res")
    ent_backend = make_backend(cfg.llm, step="class_res")

    # LLM judge anchors (one prompt per anchor, top-K candidates together)
    rel_flat, rel_per_anchor = judge_anchors(
        clusters=rel_clusters,
        kind="relation",
        active_weight=active_pred_all,
        out_jsonl=out_dir / "llm_rel_judgements.jsonl",
        backend=rel_backend,
        llm_cfg=cfg.llm,
    )

    ent_flat, ent_per_anchor = judge_anchors(
        clusters=ent_clusters,
        kind="concept",
        active_weight=active_concept_all_anchored,
        out_jsonl=out_dir / "llm_ent_judgements.jsonl",
        backend=ent_backend,
        llm_cfg=cfg.llm,
    )

    # Similarity calibration (optional, threshold-free)
    rel_cal = {"ap": None, "n_points": 0}
    ent_cal = {"ap": None, "n_points": 0}
    if cfg.compute_pr_curve:
        rel_cal = compute_similarity_calibration(
            clusters=rel_clusters,
            flat_items=rel_flat,
            kind="relation",
            out_csv_path=(out_dir / "sim_calibration_rel.csv") if SKLEARN_OK else None,
        )
        ent_cal = compute_similarity_calibration(
            clusters=ent_clusters,
            flat_items=ent_flat,
            kind="concept",
            out_csv_path=(out_dir / "sim_calibration_ent.csv") if SKLEARN_OK else None,
        )

    # Metrics per split (all + optional train/test)
    splits = ["all"]
    if cfg.compute_train_test:
        # only include if present
        for sp in ("train","valid","test"):
            if sp in active_sets:
                splits.append(sp)

    # Build deterministic auto mapping for concept labels (used for domain/range fallback)
    concept_map_auto = build_auto_entity_map(ent_clusters)
    # Build LLM-validated concept mapping (preferred)
    concept_map_llm = build_llm_valid_concept_map(ent_flat)

    summary_rows = []
    by_rel_rows = []
    by_con_rows = []

    # Per-anchor tables (include weights for each split)
    # Relations
    for ref_label, rec in rel_per_anchor.items():
        row = dict(rec)
        for sp in splits:
            row[f"active_weight_{sp}"] = int(active_sets[sp]["active_pred_freq"].get(ref_label, 0))
        by_rel_rows.append(row)

    # Concepts
    for ref_label, rec in ent_per_anchor.items():
        row = dict(rec)
        for sp in splits:
            row[f"active_weight_{sp}"] = int(active_sets[sp]["active_concept_weight"].get(ref_label, 0))
        by_con_rows.append(row)

    write_csv(out_dir / "by_relation.csv", by_rel_rows)
    write_csv(out_dir / "by_concept.csv", by_con_rows)

    # Split-level metrics
    for sp in splits:
        active_pred = active_sets[sp]["active_pred_freq"]
        active_concept = active_sets[sp]["active_concept_weight"]
        active_concept_anch = {c: w for c, w in active_concept.items() if c in anchored_ref_concepts}

        rel_rank = compute_rank_metrics(rel_per_anchor, active_pred, k=cfg.llm.top_k)
        con_rank = compute_rank_metrics(ent_per_anchor, active_concept_anch, k=cfg.llm.top_k)

        rel_prec = compute_candidate_precision_metrics([
            x for x in rel_flat if active_pred.get(x["ref_label"], 0) > 0 or cfg.llm.judge_inactive_anchors
        ])
        con_prec = compute_candidate_precision_metrics([
            x for x in ent_flat if active_concept_anch.get(x["ref_label"], 0) > 0 or cfg.llm.judge_inactive_anchors
        ])

        rel_refine = compute_refinement_metrics(rel_per_anchor, active_pred)
        con_refine = compute_refinement_metrics(ent_per_anchor, active_concept_anch)

        # Domain/range (relations only) using best VALID trace relation + concept maps
        dr = evaluate_domain_range(
            ref_rels=ref_rels,
            rel_clusters=rel_clusters,
            rel_per_anchor=rel_per_anchor,
            rel_flat_items=rel_flat,
            concept_map_llm=concept_map_llm,
            concept_map_auto=concept_map_auto,
            active_pred_freq=active_pred,
            top_k=cfg.llm.top_k,
        )

        summary_rows.append({
            "ontology_id": paths.ontology_id,
            "split": sp,

            "n_ref_concepts_total": len(ref_concepts),
            "n_ref_relations_total": len(ref_rels),

            "n_active_ref_relations": sum(1 for _,w in active_pred.items() if w > 0),
            "n_active_ref_concepts_total_from_DR": sum(1 for _,w in active_concept.items() if w > 0),
            "n_active_ref_concepts_anchored": sum(1 for _,w in active_concept_anch.items() if w > 0),
            "n_active_ref_concepts_missing_anchor": len([c for c,w in active_concept.items() if w > 0 and c not in anchored_ref_concepts]),

            # Relation headline metrics (LLM-validated, threshold-free)
            "rel_coverage_valid_weighted": rel_rank["coverage_valid_weighted"],
            "rel_coverage_equiv_weighted": rel_rank["coverage_equiv_weighted"],
            "rel_hits_at_k_valid_weighted": rel_rank["hits_at_k_valid_weighted"],
            "rel_mrr_at_k_valid_weighted": rel_rank["mrr_at_k_valid_weighted"],
            "rel_candidate_precision_valid": rel_prec["candidate_precision_valid"],
            "rel_candidate_narrower_rate_among_valid": rel_prec["candidate_narrower_rate_among_valid"],
            "rel_refinement_mean_valid_candidates_weighted": rel_refine["refinement_mean_valid_candidates_weighted"],

            # Concept headline metrics (LLM-validated, threshold-free)
            "concept_coverage_valid_weighted": con_rank["coverage_valid_weighted"],
            "concept_coverage_equiv_weighted": con_rank["coverage_equiv_weighted"],
            "concept_hits_at_k_valid_weighted": con_rank["hits_at_k_valid_weighted"],
            "concept_mrr_at_k_valid_weighted": con_rank["mrr_at_k_valid_weighted"],
            "concept_candidate_precision_valid": con_prec["candidate_precision_valid"],
            "concept_candidate_narrower_rate_among_valid": con_prec["candidate_narrower_rate_among_valid"],
            "concept_refinement_mean_valid_candidates_weighted": con_refine["refinement_mean_valid_candidates_weighted"],

            # Domain/Range (analysis)
            "rel_domain_range_acc_weighted_llmMap_fallbackAuto": dr["domain_range_acc_weighted_llmMap_fallbackAuto"],
            "rel_domain_range_acc_weighted_autoMap": dr["domain_range_acc_weighted_autoMap"],
            "rel_domain_range_n_compared": dr["domain_range_n_compared"],

            # Similarity calibration (threshold-free)
            "rel_sim_AP_validity": rel_cal.get("ap", None),
            "rel_sim_pairs_scored": rel_cal.get("n_points", 0),
            "concept_sim_AP_validity": ent_cal.get("ap", None),
            "concept_sim_pairs_scored": ent_cal.get("n_points", 0),
        })

    # Save summary
    write_csv(out_dir / "summary.csv", summary_rows)
    write_json(out_dir / "summary.json", {
        "ontology_id": paths.ontology_id,
        "paths": {
            "ref_ontology": str(paths.ref_ontology_path),
            "gold_triples": str(paths.gold_triples_path),
            "entity_clusters": str(paths.entity_clusters_path),
            "relation_clusters": str(paths.relation_clusters_path),
        },
        "cluster_globals": {"entity": ent_global, "relation": rel_global},
        "active_missing_concepts": missing_active_concepts[:50],
        "config": {
            "llm": vars(cfg.llm),
            "compute_train_test": cfg.compute_train_test,
            "compute_pr_curve": cfg.compute_pr_curve,
        },
        "notes": [
            "Coverage is determined by LLM validity (Equivalent/Narrower + usable_as_schema), not by similarity thresholds.",
            "Similarity is used only for ranking top-K and for threshold-free calibration (AP/PR curve).",
        ],
    })

    return out_dir / "summary.csv"


# ============================================================
# Optional: aggregate multiple ontology runs into one table
# ============================================================

def aggregate_summaries(summary_csv_paths: List[Path], out_csv: Path) -> None:
    rows = []
    for p in summary_csv_paths:
        rows.extend(read_csv(p))
    write_csv(out_csv, rows)

def read_csv(p: Path) -> List[dict]:
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(x) for x in r]


# ============================================================
# CLI entry (safe in Jupyter via parse_known_args)
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology-id", default="film")
    ap.add_argument("--ref-ontology", required=True)
    ap.add_argument("--gold-triples", required=True)
    ap.add_argument("--entity-clusters", required=True)
    ap.add_argument("--relation-clusters", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--llm-max-tokens", type=int, default=1400)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--prefer-dspy", action="store_true")
    ap.add_argument("--judge-inactive", action="store_true")
    ap.add_argument("--no-pr-curve", action="store_true")
    ap.add_argument("--no-train-test", action="store_true")

    args, _ = ap.parse_known_args()

    paths = EvalPaths(
        ontology_id=args.ontology_id,
        ref_ontology_path=Path(args.ref_ontology),
        gold_triples_path=Path(args.gold_triples),
        entity_clusters_path=Path(args.entity_clusters),
        relation_clusters_path=Path(args.relation_clusters),
        out_dir=Path(args.out_dir),
    )

    llm_cfg = LLMConfig(
        use_llm=bool(args.use_llm),
        model=args.llm_model,
        max_tokens=args.llm_max_tokens,
        top_k=args.top_k,
        prefer_dspy=bool(args.prefer_dspy),
        judge_inactive_anchors=bool(args.judge_inactive),
    )
    cfg = EvalConfig(
        llm=llm_cfg,
        compute_train_test=not bool(args.no_train_test),
        compute_pr_curve=not bool(args.no_pr_curve),
    )

    out = run_schema_evaluation(paths, cfg)
    print("[OK] wrote:", out)


# if __name__ == "__main__":
#     main()


from pathlib import Path

# =========================
# CONFIGURE THESE TWO ONLY
# =========================
DATA_ROOT = Path("Experiments/MYNE/Ex4_T2KGBench").resolve()
OUT_ROOT  = DATA_ROOT / "KGs_from_Essays" / "KG_Run_F3" / "OntCompResults" / "SchemaEvaL"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# DERIVED PATHS (FILM)
# =========================
ontology_id = "film"

ref_ontology_path = (
    DATA_ROOT
    / "dbpedia-webnlg"
    / "Raw"
    / "ontologies"
    / "19_film_ontology.json"
)

gold_triples_path = (
    DATA_ROOT
    / "KGs_from_Essays"
    / "KG_Run_F3"
    / "OntCompResults"
    / "gold_triples.jsonl"
)

entity_clusters_path = (
    DATA_ROOT
    / "KGs_from_Essays"
    / "KG_Run_F3"
    / "OntCompResults"
    / "AnchoredClusters"
    / "entity_anchored_clusters.json"
)

relation_clusters_path = (
    DATA_ROOT
    / "KGs_from_Essays"
    / "KG_Run_F3"
    / "OntCompResults"
    / "AnchoredClusters"
    / "relation_anchored_clusters.json"
)

# ===== build EvalPaths and config and run =====
paths = EvalPaths(
    ontology_id=ontology_id,
    ref_ontology_path=ref_ontology_path,
    gold_triples_path=gold_triples_path,
    entity_clusters_path=entity_clusters_path,
    relation_clusters_path=relation_clusters_path,
    out_dir=OUT_ROOT,
)

cfg = EvalConfig(
    llm=LLMConfig(
        use_llm=True,
        model="gpt-5.1",        # pick your preferred model
        max_tokens=1400,
        top_k=6,
        prefer_dspy=True,      # set False if you want OpenAI SDK fallback
        reuse_cached=False,
        judge_inactive_anchors=False,
    ),
    compute_train_test=True,
    compute_pr_curve=True,
)

# run (this returns path to summary.csv)
out_summary = run_schema_evaluation(paths, cfg)
print("Done →", out_summary)

#endregion#? Trace_ref_schema_eval - v1
#?#########################  End  ##########################




from pathlib import Path
import json

# Path to the relation anchored clusters you already generated
REL_CLUSTERS_PATH = Path(
    "Experiments/MYNE/Ex4_T2KGBench/KGs_from_Essays/KG_Run_F3/OntCompResults/AnchoredClusters/relation_anchored_clusters.json"
).resolve()

data = json.loads(REL_CLUSTERS_PATH.read_text(encoding="utf-8", errors="replace"))
rel_clusters = {k: v for k, v in data.items() if k != "_global"}   # drop global block
print("Loaded rel_clusters:", len(rel_clusters), "anchors")
print("Sample keys:", list(rel_clusters.keys())[:3])


# Make sure cfg + make_backend + build_relation_prompt + get_topk_members exist in the notebook
rel_backend = make_backend(cfg.llm, step="rel_res")

some_ref_key = next(iter(rel_clusters.keys()))
blk = rel_clusters[some_ref_key]
anchor = blk.get("anchor", {}) or {}
cands = get_topk_members(blk, cfg.llm.top_k)

prompt = build_relation_prompt(anchor, cands)
raw = rel_backend.complete(
    system="You are a precise ontology alignment judge. Return strict JSON only. No markdown.",
    user=prompt,
    max_tokens=cfg.llm.max_tokens,
)

print("=== RAW LLM OUTPUT (first 1200 chars) ===")
print(raw[:1200])
print("\n=== PARSED OBJECT ===")
print(_extract_json_obj(raw))


#?######################### Start ##########################
#region:#?   A print statement for improvement

from pathlib import Path
import json, csv
from collections import Counter

# =========================
# EDIT THIS ONLY (if needed)
# =========================
OUT_DIR = Path("Experiments/MYNE/Ex4_T2KGBench/KGs_from_Essays/KG_Run_F3/OntCompResults/SchemaEval").resolve()

# -------------------------
# Helpers
# -------------------------
def read_csv(p: Path):
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8", errors="replace"))

def read_jsonl(p: Path):
    rows = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows

def as_int(x, default=0):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default

def as_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

# -------------------------
# 0) List files
# -------------------------
print("\n==================")
print("OUT_DIR:", OUT_DIR)
print("==================")
if not OUT_DIR.exists():
    raise FileNotFoundError(f"OUT_DIR does not exist: {OUT_DIR}")

files = sorted([p for p in OUT_DIR.iterdir() if p.is_file()])
for p in files:
    print(f"- {p.name:28s}  size={p.stat().st_size}")

# -------------------------
# 1) summary.csv (core table)
# -------------------------
summary_path = OUT_DIR / "summary.csv"
summary = read_csv(summary_path)
print("\n==================")
print("summary.csv (ALL ROWS)")
print("==================")
for r in summary:
    # print only the headline columns first (still includes all rows)
    headline = {
        "ontology_id": r.get("ontology_id",""),
        "split": r.get("split",""),
        "n_active_ref_relations": r.get("n_active_ref_relations",""),
        "n_active_ref_concepts_anchored": r.get("n_active_ref_concepts_anchored",""),
        "rel_coverage_valid_weighted": r.get("rel_coverage_valid_weighted",""),
        "rel_coverage_equiv_weighted": r.get("rel_coverage_equiv_weighted",""),
        "rel_hits_at_k_valid_weighted": r.get("rel_hits_at_k_valid_weighted",""),
        "rel_mrr_at_k_valid_weighted": r.get("rel_mrr_at_k_valid_weighted",""),
        "concept_coverage_valid_weighted": r.get("concept_coverage_valid_weighted",""),
        "concept_coverage_equiv_weighted": r.get("concept_coverage_equiv_weighted",""),
        "concept_hits_at_k_valid_weighted": r.get("concept_hits_at_k_valid_weighted",""),
        "concept_mrr_at_k_valid_weighted": r.get("concept_mrr_at_k_valid_weighted",""),
        "rel_domain_range_acc_weighted_llmMap_fallbackAuto": r.get("rel_domain_range_acc_weighted_llmMap_fallbackAuto",""),
        "rel_sim_AP_validity": r.get("rel_sim_AP_validity",""),
        "concept_sim_AP_validity": r.get("concept_sim_AP_validity",""),
    }
    print(json.dumps(headline, indent=2))

# -------------------------
# 2) summary.json (config + missing anchors)
# -------------------------
sumjson_path = OUT_DIR / "summary.json"
if sumjson_path.exists():
    sumjson = read_json(sumjson_path)
    print("\n==================")
    print("summary.json (KEY EXCERPTS)")
    print("==================")
    print("ontology_id:", sumjson.get("ontology_id"))
    print("paths:", json.dumps(sumjson.get("paths", {}), indent=2)[:2000])
    print("cluster_globals:", json.dumps(sumjson.get("cluster_globals", {}), indent=2)[:2000])
    print("config:", json.dumps(sumjson.get("config", {}), indent=2)[:2000])
    miss = sumjson.get("active_missing_concepts", []) or []
    print("active_missing_concepts (first 50):", miss[:50])
else:
    print("\n(summary.json not found)")

# -------------------------
# 3) by_relation.csv + by_concept.csv (where the action is)
# -------------------------
def print_coverage_report(name, rows, weight_col="active_weight_all"):
    active = [r for r in rows if as_int(r.get(weight_col, 0)) > 0]
    covered = [r for r in active if as_int(r.get("covered_valid", 0)) == 1]
    uncovered = [r for r in active if as_int(r.get("covered_valid", 0)) == 0]

    total_w = sum(as_int(r.get(weight_col, 0)) for r in active) or 1
    covered_w = sum(as_int(r.get(weight_col, 0)) for r in covered)

    print("\n==================")
    print(f"{name} coverage report")
    print("==================")
    print(f"Active anchors: {len(active)} / total rows {len(rows)}")
    print(f"Covered(active): {len(covered)}   Uncovered(active): {len(uncovered)}")
    print(f"Weighted coverage (from file columns): {covered_w/total_w:.4f}")

    # Top uncovered by weight
    uncovered_sorted = sorted(uncovered, key=lambda r: as_int(r.get(weight_col, 0)), reverse=True)
    print("\nTop 15 uncovered ACTIVE anchors (by weight):")
    for r in uncovered_sorted[:15]:
        print({
            "ref_label": r.get("ref_label",""),
            weight_col: as_int(r.get(weight_col, 0)),
            "n_candidates_present": as_int(r.get("n_candidates_present",0)),
            "n_judged": as_int(r.get("n_judged",0)),
            "first_valid_rank": r.get("first_valid_rank",""),
            "n_valid": as_int(r.get("n_valid",0)),
        })

    # Top covered by weight
    covered_sorted = sorted(covered, key=lambda r: as_int(r.get(weight_col, 0)), reverse=True)
    print("\nTop 10 covered ACTIVE anchors (by weight):")
    for r in covered_sorted[:10]:
        print({
            "ref_label": r.get("ref_label",""),
            weight_col: as_int(r.get(weight_col, 0)),
            "first_valid_rank": r.get("first_valid_rank",""),
            "n_valid": as_int(r.get("n_valid",0)),
            "n_equiv": as_int(r.get("n_equiv",0)),
        })

by_rel_path = OUT_DIR / "by_relation.csv"
by_con_path = OUT_DIR / "by_concept.csv"
by_rel = read_csv(by_rel_path)
by_con = read_csv(by_con_path)

if by_rel:
    print_coverage_report("RELATIONS (by_relation.csv)", by_rel, weight_col="active_weight_all")
else:
    print("\n(by_relation.csv missing/empty)")

if by_con:
    print_coverage_report("CONCEPTS (by_concept.csv)", by_con, weight_col="active_weight_all")
else:
    print("\n(by_concept.csv missing/empty)")

# -------------------------
# 4) LLM judgement JSONLs (global stats + a few samples)
# -------------------------
def llm_stats(jsonl_path: Path, kind: str, n_show_anchors=5, n_show_items=10):
    rows = read_jsonl(jsonl_path)
    print("\n==================")
    print(f"{kind} LLM judgements: {jsonl_path.name}")
    print("==================")
    print("n_anchor_records:", len(rows))

    err = [r for r in rows if (r.get("raw_error") or "").strip()]
    print("anchors_with_raw_error:", len(err))
    if err[:5]:
        print("raw_error examples (up to 5):")
        for r in err[:5]:
            print({"ref_label": r.get("ref_label"), "raw_error": r.get("raw_error")})

    judg = Counter()
    action = Counter()
    valid_cnt = 0
    total_items = 0
    confs_valid = []

    for r in rows:
        items = r.get("items") or []
        for it in items:
            total_items += 1
            judg[it.get("judgement","")] += 1
            action[it.get("suggested_action","")] += 1
            is_valid = (it.get("judgement") in ("Equivalent","Narrower")) and bool(it.get("usable_as_schema"))
            if is_valid:
                valid_cnt += 1
                confs_valid.append(as_float(it.get("confidence"), 0.0))

    print("total_judged_items:", total_items)
    print("valid_items (Equiv/Narrower & usable):", valid_cnt)
    print("valid_rate:", (valid_cnt / total_items) if total_items else 0.0)
    print("judgement_dist:", dict(judg))
    print("action_dist:", dict(action))
    if confs_valid:
        confs_valid_sorted = sorted(confs_valid)
        print("valid_confidence: mean=", sum(confs_valid)/len(confs_valid),
              " median=", confs_valid_sorted[len(confs_valid_sorted)//2])

    # show a few anchor records (truncated)
    print(f"\nSample {n_show_anchors} anchors (first ones):")
    for r in rows[:n_show_anchors]:
        print({"ref_label": r.get("ref_label"), "n_items": len(r.get("items") or []), "raw_error": r.get("raw_error","")})
        for it in (r.get("items") or [])[:n_show_items]:
            print("  ", {
                "trace_label": it.get("trace_label"),
                "judgement": it.get("judgement"),
                "usable": it.get("usable_as_schema"),
                "conf": it.get("confidence"),
                "action": it.get("suggested_action"),
            })

llm_rel_path = OUT_DIR / "llm_rel_judgements.jsonl"
llm_ent_path = OUT_DIR / "llm_ent_judgements.jsonl"
if llm_rel_path.exists():
    llm_stats(llm_rel_path, "RELATION", n_show_anchors=3, n_show_items=8)
else:
    print("\n(llm_rel_judgements.jsonl not found)")

if llm_ent_path.exists():
    llm_stats(llm_ent_path, "CONCEPT", n_show_anchors=3, n_show_items=8)
else:
    print("\n(llm_ent_judgements.jsonl not found)")

# -------------------------
# 5) PR calibration CSVs (if present)
# -------------------------
def print_pr_head_tail(p: Path, n=5):
    if not p.exists():
        print(f"\n({p.name} not found)")
        return
    rows = read_csv(p)
    print("\n==================")
    print(p.name, f"(rows={len(rows)}) head/tail")
    print("==================")
    for r in rows[:n]:
        print(r)
    if len(rows) > n:
        print("...")
        for r in rows[-n:]:
            print(r)

print_pr_head_tail(OUT_DIR / "sim_calibration_rel.csv", n=5)
print_pr_head_tail(OUT_DIR / "sim_calibration_ent.csv", n=5)

print("\nDONE.")

#endregion#? A print statement for improvement
#?#########################  End  ##########################
 

#endregion#! Comparing our schema with Text2KG Benchmark Ontology
#!#############################################  End Chapter  ##################################################





#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################




#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################
