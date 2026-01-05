
#?######################### Start ##########################
#region:#?  Create KG for each Essay  - V3


"""
TRACE KG multi-essay runner

Drop this near the BOTTOM of your main .py file (AFTER all function definitions),
or put it in a separate script that imports those functions.

It will:

  - Read 100 essays from:
        SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json

  - For each essay i (1..N):
        * Reset the data pipeline folders (Chunks/Classes/Entities/KG/Relations)
        * Write that essay into Plain_Text.json (the chunking input)
        * Run, IN ORDER, the pipeline steps you specified:
              1) sentence_chunks_token_driven(...)
              2) embed_and_index_chunks(...)
              3) run_entity_extraction_on_chunks(...)
              4) iterative_resolution()
              5) produce_clean_jsonl(input_path, out_file)
              6) classrec_iterative_main()
              7) main_input_for_cls_res()
              8) run_pipeline_iteratively()
              9) run_rel_rec(...)           <-- needed to create relations_raw.jsonl
             10) run_relres_iteratively()
             11) export_relations_and_nodes_to_csv()

        * Copy the entire /data folder to:
              SGCE-KG/KGs_from_Essays/KG_Essay_<i>

        * Clear the pipeline data folders again so the next essay is independent

  - Use tqdm for progress
  - Record per-essay timings, success/fail, and basic counts to:
        SGCE-KG/KGs_from_Essays/trace_kg_essays_run_stats.json
"""

import json
import os
import shutil
import time
import traceback
from pathlib import Path

from tqdm import tqdm  # make sure `pip install tqdm`

# --------------------------------------------------------------------
# CONSTANT PATHS
# --------------------------------------------------------------------

BASE_ROOT = Path("SGCE-KG")
DATA_ROOT = BASE_ROOT / "data"
ESSAYS_JSON = BASE_ROOT / "Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json"
PLAIN_TEXT_JSON = DATA_ROOT / "pdf_to_json" / "Plain_Text.json"
KG_RUNS_ROOT = BASE_ROOT / "KGs_from_Essays"

CHUNKS_DIR = DATA_ROOT / "Chunks"
CLASSES_DIR = DATA_ROOT / "Classes"
ENTITIES_DIR = DATA_ROOT / "Entities"
KG_DIR = DATA_ROOT / "KG"
RELATIONS_DIR = DATA_ROOT / "Relations"

# Make sure base dirs exist
DATA_ROOT.mkdir(parents=True, exist_ok=True)
PLAIN_TEXT_JSON.parent.mkdir(parents=True, exist_ok=True)
KG_RUNS_ROOT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------

def clear_data_subfolders() -> None:
    """
    Remove EVERYTHING inside these pipeline folders (but keep the folders):
      - Chunks
      - Classes
      - Entities
      - KG
      - Relations

    This ensures each essay run is independent.
    """
    for d in [CHUNKS_DIR, CLASSES_DIR, ENTITIES_DIR, KG_DIR, RELATIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        for child in d.iterdir():
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except Exception as e:
                print(f"[warn] Failed to remove {child}: {e}")


def extract_essay_text(rec, idx: int) -> str:
    """
    Heuristic to get the essay text from one JSON record.

    Adjust if your Plain_Text_100_Essays.json has a different structure.
    """
    if isinstance(rec, str):
        return rec

    if isinstance(rec, dict):
        # Try common field names first
        for k in ["text", "essay_text", "content", "body", "answer", "Plain_Text"]:
            v = rec.get(k)
            if isinstance(v, str) and v.strip():
                return v

        # Fallback: choose the longest string field
        best = ""
        for v in rec.values():
            if isinstance(v, str) and len(v) > len(best):
                best = v
        if best:
            return best

    # Last resort
    return str(rec)


def load_essays():
    """
    Load essays from Plain_Text_100_Essays.json.

    Supports:
      - list of records
      - dict of key -> record
      - single string / other (treated as one essay)
    """
    if not ESSAYS_JSON.exists():
        raise FileNotFoundError(f"Essays file not found: {ESSAYS_JSON}")

    with ESSAYS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    essays = []

    if isinstance(data, list):
        for idx, rec in enumerate(data, start=1):  # 1-based index
            text = extract_essay_text(rec, idx)
            essays.append({
                "index": idx,
                "raw": rec,
                "text": text,
            })
    elif isinstance(data, dict):
        for idx, (key, rec) in enumerate(data.items(), start=1):
            text = extract_essay_text(rec, idx)
            essays.append({
                "index": idx,
                "key": key,
                "raw": rec,
                "text": text,
            })
    else:
        # Single essay case
        essays.append({
            "index": 1,
            "raw": data,
            "text": extract_essay_text(data, 1),
        })

    return essays


def write_plain_text_input(essay_meta: dict) -> None:
    """
    Overwrite Plain_Text.json with a single-doc JSON for the current essay.

    This is the ONLY input the chunker reads, and the path stays fixed.
    """
    essay_idx = essay_meta["index"]
    doc = {
        "id": f"essay_{essay_idx:03d}",
        "ref_index": essay_idx,
        "ref_title": f"Essay {essay_idx}",
        "text": essay_meta["text"],
    }
    with PLAIN_TEXT_JSON.open("w", encoding="utf-8") as f:
        json.dump([doc], f, ensure_ascii=False, indent=2)


def collect_chunk_ids():
    """
    After chunking, read chunks_sentence.jsonl and return list of chunk ids
    for run_entity_extraction_on_chunks.
    """
    chunks_path = CHUNKS_DIR / "chunks_sentence.jsonl"
    ids = []
    if not chunks_path.exists():
        return ids
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get("id")
            if cid is not None:
                ids.append(cid)
    return ids


def copy_data_for_essay(essay_index: int) -> Path:
    """
    Copy the entire /data folder to:

        /.../SGCE-KG/KGs_from_Essays/KG_Essay_<index>

    Returns the destination path.
    """
    dest = KG_RUNS_ROOT / f"KG_Essay_{essay_index:03d}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(DATA_ROOT, dest)
    return dest


def count_csv_rows(path: Path) -> int:
    """
    Count data rows in a CSV (minus header). Returns 0 if file doesn't exist.
    """
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        # subtract header if any lines
        n = sum(1 for _ in f)
    return max(n - 1, 0)


# --------------------------------------------------------------------
# MAIN MULTI-ESSAY RUNNER
# --------------------------------------------------------------------

def run_trace_kg_for_all_essays():
    essays = load_essays()
    run_stats = []

    print(f"[info] Loaded {len(essays)} essays from {ESSAYS_JSON}")

    for essay_meta in tqdm(essays, desc="TRACE KG essays"):
        idx = essay_meta["index"]
        label = f"Essay_{idx:03d}"

        essay_stat = {
            "essay_index": idx,
            "label": label,
            "success": False,
            "error": None,
            "traceback": None,
            "timings": {},
            "nodes_count": None,
            "relations_count": None,
            "data_snapshot_dir": None,
        }

        t_run_start = time.time()
        try:
            # --------------------------------------------------------
            # 0) RESET PIPELINE DIRECTORIES FOR THIS ESSAY
            # --------------------------------------------------------
            clear_data_subfolders()

            # --------------------------------------------------------
            # 1) WRITE INPUT FOR CHUNKING
            # --------------------------------------------------------
            write_plain_text_input(essay_meta)

            # --------------------------------------------------------
            # 2) CHUNKING
            # --------------------------------------------------------
            t0 = time.time()
            sentence_chunks_token_driven(
                "SGCE-KG/data/pdf_to_json/Plain_Text.json",
                "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                max_tokens_per_chunk=200,   # preferred upper bound (None to disable)
                min_tokens_per_chunk=100,   # expand small chunks to reach this minimum (None to disable)
                sentence_per_line=True,
                keep_ref_text=False,
                strip_leading_headings=True,
                force=True,
                debug=False,
            )
            essay_stat["timings"]["chunking"] = time.time() - t0

            # recompute chunk_ids for this essay
            chunk_ids = collect_chunk_ids()

            # --------------------------------------------------------
            # 3) embed_and_index_chunks
            # --------------------------------------------------------
            t0 = time.time()
            embed_and_index_chunks(
                "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                "SGCE-KG/data/Chunks/chunks_emb",
                "BAAI/bge-large-en-v1.5",
                "BAAI/bge-small-en-v1.5",
                False,   # use_small_model_for_dev
                32,      # batch_size
                None,    # device -> auto
                True,    # save_index
                True,    # force
            )
            essay_stat["timings"]["embed_and_index_chunks"] = time.time() - t0

            # --------------------------------------------------------
            # 4) Entity Recognition
            # --------------------------------------------------------
            t0 = time.time()
            run_entity_extraction_on_chunks(
                chunk_ids,
                prev_chunks=5,
                save_debug=False,
                model="gpt-5.1",
                max_tokens=8000,
            )
            essay_stat["timings"]["entity_recognition"] = time.time() - t0

            # --------------------------------------------------------
            # 5) Ent Resolution (Multi Run)
            # --------------------------------------------------------
            t0 = time.time()
            iterative_resolution()
            essay_stat["timings"]["entity_resolution"] = time.time() - t0

            # --------------------------------------------------------
            # 6) Cls Rec input producer
            # --------------------------------------------------------
            t0 = time.time()
            # input_path & out_file are already defined in your code
            produce_clean_jsonl(input_path, out_file)
            essay_stat["timings"]["cls_rec_input"] = time.time() - t0

            # --------------------------------------------------------
            # 7) Cls Recognition
            # --------------------------------------------------------
            t0 = time.time()
            classrec_iterative_main()
            essay_stat["timings"]["cls_recognition"] = time.time() - t0

            # --------------------------------------------------------
            # 8) Create input for Cls Res
            # --------------------------------------------------------
            t0 = time.time()
            main_input_for_cls_res()
            essay_stat["timings"]["cls_res_input"] = time.time() - t0

            # --------------------------------------------------------
            # 9) Cls Res Multi Run
            # --------------------------------------------------------
            t0 = time.time()
            run_pipeline_iteratively()
            essay_stat["timings"]["cls_res_multi_run"] = time.time() - t0

            # --------------------------------------------------------
            # 10) Relation Rec (single run)
            #     (Needed to create relations_raw.jsonl before Rel Res)
            # --------------------------------------------------------
            t0 = time.time()
            run_rel_rec(
                entities_path="SGCE-KG/data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl",
                chunks_path="SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                output_path="SGCE-KG/data/Relations/Rel Rec/relations_raw.jsonl",
                model="gpt-5.1",
            )
            essay_stat["timings"]["relation_recognition"] = time.time() - t0

            # --------------------------------------------------------
            # 11) Relation Res Multi Run
            # --------------------------------------------------------
            t0 = time.time()
            run_relres_iteratively()
            essay_stat["timings"]["relation_resolution"] = time.time() - t0

            # --------------------------------------------------------
            # 12) Export KG to CSVs
            # --------------------------------------------------------
            t0 = time.time()
            export_relations_and_nodes_to_csv()
            essay_stat["timings"]["export_kg"] = time.time() - t0

            # --------------------------------------------------------
            # 13) SIMPLE COUNTS (nodes / relations)
            # --------------------------------------------------------
            nodes_csv = KG_DIR / "nodes.csv"
            rels_csv = KG_DIR / "rels_fixed_no_raw.csv"
            essay_stat["nodes_count"] = count_csv_rows(nodes_csv)
            essay_stat["relations_count"] = count_csv_rows(rels_csv)

            essay_stat["success"] = True

        except Exception as e:
            essay_stat["error"] = str(e)
            essay_stat["traceback"] = traceback.format_exc()
            print(f"[error] Failure on {label}: {e}")

        finally:
            essay_stat["timings"]["total_seconds"] = time.time() - t_run_start

            # Snapshot /data (even on failure, for debugging that essay)
            try:
                dest_dir = copy_data_for_essay(idx)
                essay_stat["data_snapshot_dir"] = str(dest_dir)
            except Exception as e:
                print(f"[warn] Failed to snapshot data for {label}: {e}")

            # Clear pipeline dirs for NEXT essay
            clear_data_subfolders()

            run_stats.append(essay_stat)

    # ----------------------------------------------------------------
    # SAVE OVERALL RUN STATS
    # ----------------------------------------------------------------
    stats_path = KG_RUNS_ROOT / "trace_kg_essays_run_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote overall run stats to {stats_path}")


# --------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------

if __name__ == "__main__":
    run_trace_kg_for_all_essays()






#endregion#? Create KG for each Essay  - V3
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Create KG for each Essay  - V4



"""
TRACE KG multi-essay runner (fixed entity seed issue)

Runs the full pipeline independently for each essay in:
    SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json

For each essay i:
  1) Clear data/{Chunks,Classes,Entities,KG,Relations}
  2) Write that essay as the only document to data/pdf_to_json/Plain_Text.json
  3) Run, IN ORDER:

        sentence_chunks_token_driven(...)
        embed_and_index_chunks(...)
        run_entity_extraction_on_chunks(...)
        prepare_entity_seed_for_iterative_resolution()   <-- NEW helper
        iterative_resolution()
        produce_clean_jsonl(input_path, out_file)
        classrec_iterative_main()
        main_input_for_cls_res()
        run_pipeline_iteratively()
        run_rel_rec(...)
        run_relres_iteratively()
        export_relations_and_nodes_to_csv()

  4) Copy /data to KGs_from_Essays/KG_Essay_<i>
  5) Clear data/{Chunks,Classes,Entities,KG,Relations} again

Records timing + status per essay to:
    SGCE-KG/KGs_from_Essays/trace_kg_essays_run_stats.json
"""

import json
import os
import shutil
import time
import traceback
from pathlib import Path

from tqdm import tqdm  # pip install tqdm

# --------------------------------------------------------------------
# CONSTANT PATHS
# --------------------------------------------------------------------

BASE_ROOT = Path("SGCE-KG")
DATA_ROOT = BASE_ROOT / "data"
ESSAYS_JSON = BASE_ROOT / "Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json"
PLAIN_TEXT_JSON = DATA_ROOT / "pdf_to_json" / "Plain_Text.json"
KG_RUNS_ROOT = BASE_ROOT / "KGs_from_Essays"

CHUNKS_DIR = DATA_ROOT / "Chunks"
CLASSES_DIR = DATA_ROOT / "Classes"
ENTITIES_DIR = DATA_ROOT / "Entities"
KG_DIR = DATA_ROOT / "KG"
RELATIONS_DIR = DATA_ROOT / "Relations"

# Make sure base dirs exist
DATA_ROOT.mkdir(parents=True, exist_ok=True)
PLAIN_TEXT_JSON.parent.mkdir(parents=True, exist_ok=True)
KG_RUNS_ROOT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------

def clear_data_subfolders() -> None:
    """
    Remove EVERYTHING inside these pipeline folders (but keep the folders):
      - Chunks
      - Classes
      - Entities
      - KG
      - Relations
    """
    for d in [CHUNKS_DIR, CLASSES_DIR, ENTITIES_DIR, KG_DIR, RELATIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        for child in d.iterdir():
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except Exception as e:
                print(f"[warn] Failed to remove {child}: {e}")


def extract_essay_text(rec, idx: int) -> str:
    """
    Heuristic to get the essay text from one JSON record.
    Adjust this if your essay JSON schema differs.
    """
    if isinstance(rec, str):
        return rec

    if isinstance(rec, dict):
        # Try common field names first
        for k in ["text", "essay_text", "content", "body", "answer", "Plain_Text"]:
            v = rec.get(k)
            if isinstance(v, str) and v.strip():
                return v

        # Fallback: choose the longest string field
        best = ""
        for v in rec.values():
            if isinstance(v, str) and len(v) > len(best):
                best = v
        if best:
            return best

    # Last resort
    return str(rec)


def load_essays():
    """
    Load essays from Plain_Text_100_Essays.json.
    Supports list, dict, or single-object JSON.
    """
    if not ESSAYS_JSON.exists():
        raise FileNotFoundError(f"Essays file not found: {ESSAYS_JSON}")

    with ESSAYS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    essays = []

    if isinstance(data, list):
        for idx, rec in enumerate(data, start=1):  # 1-based index
            text = extract_essay_text(rec, idx)
            essays.append({
                "index": idx,
                "raw": rec,
                "text": text,
            })
    elif isinstance(data, dict):
        for idx, (key, rec) in enumerate(data.items(), start=1):
            text = extract_essay_text(rec, idx)
            essays.append({
                "index": idx,
                "key": key,
                "raw": rec,
                "text": text,
            })
    else:
        essays.append({
            "index": 1,
            "raw": data,
            "text": extract_essay_text(data, 1),
        })

    return essays


def write_plain_text_input(essay_meta: dict) -> None:
    """
    Overwrite Plain_Text.json with a single-doc JSON for the current essay.
    This is the ONLY input the chunker reads.
    """
    essay_idx = essay_meta["index"]
    doc = {
        "id": f"essay_{essay_idx:03d}",
        "ref_index": essay_idx,
        "ref_title": f"Essay {essay_idx}",
        "text": essay_meta["text"],
    }
    with PLAIN_TEXT_JSON.open("w", encoding="utf-8") as f:
        json.dump([doc], f, ensure_ascii=False, indent=2)


def collect_chunk_ids():
    """
    After chunking, read chunks_sentence.jsonl and return list of chunk ids
    for run_entity_extraction_on_chunks.
    """
    chunks_path = CHUNKS_DIR / "chunks_sentence.jsonl"
    ids = []
    if not chunks_path.exists():
        return ids
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get("id")
            if cid is not None:
                ids.append(cid)
    return ids


def copy_data_for_essay(essay_index: int) -> Path:
    """
    Copy the entire /data folder to:
        /.../SGCE-KG/KGs_from_Essays/KG_Essay_<index>
    Returns the destination path.
    """
    dest = KG_RUNS_ROOT / f"KG_Essay_{essay_index:03d}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(DATA_ROOT, dest)
    return dest


def count_csv_rows(path: Path) -> int:
    """
    Count data rows in a CSV (minus header). Returns 0 if file doesn't exist.
    """
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


# --------------------------------------------------------------------
# NEW: entity seed prep for iterative_resolution()
# --------------------------------------------------------------------

def _score_entity_candidate(path: Path) -> tuple:
    """
    Heuristic scoring to pick the best entity .jsonl file as a seed.
    Prefer names with 'entities_raw', then 'entities', and larger size.
    """
    name = path.name.lower()
    score = 0
    if "entities_raw" in name or "entity_raw" in name:
        score += 100
    elif "entities" in name or "entity" in name:
        score += 50
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    return (score, size)


def prepare_entity_seed_for_iterative_resolution():
    """
    Ensure that:
      data/Entities/iterative_runs/entities_raw_seed_backup.jsonl
      data/Entities/iterative_runs/entities_raw_seed.jsonl
    exist before calling iterative_resolution().

    Strategy:
      - Look for any *.jsonl under data/Entities (except inside iterative_runs).
      - Pick the best candidate by name + size.
      - Copy it as both seed and seed_backup.
    """
    iter_dir = ENTITIES_DIR / "iterative_runs"
    iter_dir.mkdir(parents=True, exist_ok=True)

    backup_path = iter_dir / "entities_raw_seed_backup.jsonl"
    seed_path = iter_dir / "entities_raw_seed.jsonl"

    # If both already exist, do nothing (e.g., if some other part of your code made them)
    if backup_path.exists() and seed_path.exists():
        return

    # Find candidate entity files produced by entity extraction for THIS essay
    candidates = []
    if ENTITIES_DIR.exists():
        for p in ENTITIES_DIR.rglob("*.jsonl"):
            # skip previous iterative_runs content (if any)
            try:
                if "iterative_runs" in p.parts:
                    continue
            except Exception:
                pass
            candidates.append(p)

    if not candidates:
        raise RuntimeError(
            "prepare_entity_seed_for_iterative_resolution: "
            "No .jsonl files found under data/Entities to use as seed."
        )

    best = max(candidates, key=_score_entity_candidate)
    # Copy to seed and backup
    shutil.copy2(best, backup_path)
    shutil.copy2(best, seed_path)
    print(f"[info] Seeded entity resolution from: {best.name}")


# --------------------------------------------------------------------
# MAIN MULTI-ESSAY RUNNER
# --------------------------------------------------------------------

def run_trace_kg_for_all_essays():
    essays = load_essays()
    run_stats = []

    print(f"[info] Loaded {len(essays)} essays from {ESSAYS_JSON}")

    for essay_meta in tqdm(essays, desc="TRACE KG essays"):
        idx = essay_meta["index"]
        label = f"Essay_{idx:03d}"

        essay_stat = {
            "index": idx,
            "essay_id": label,
            "t_chunking": None,
            "t_embed_index": None,
            "n_chunks": None,
            "t_build_chunk_ids": None,
            "t_entity_recognition": None,
            "t_entity_resolution": None,
            "t_cls_rec_input": None,
            "t_cls_recognition": None,
            "t_cls_res_input": None,
            "t_cls_res_multi_run": None,
            "t_rel_recognition": None,
            "t_rel_resolution": None,
            "t_export_kg": None,
            "success": False,
            "error": None,
            "traceback": None,
            "t_total": None,
            "nodes_count": None,
            "relations_count": None,
            "data_snapshot_dir": None,
        }

        t_run_start = time.time()
        try:
            # --------------------------------------------------------
            # 0) RESET PIPELINE DIRECTORIES FOR THIS ESSAY
            # --------------------------------------------------------
            clear_data_subfolders()

            # --------------------------------------------------------
            # 1) WRITE INPUT FOR CHUNKING
            # --------------------------------------------------------
            write_plain_text_input(essay_meta)

            # --------------------------------------------------------
            # 2) CHUNKING
            # --------------------------------------------------------
            t0 = time.time()
            sentence_chunks_token_driven(
                "SGCE-KG/data/pdf_to_json/Plain_Text.json",
                "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                max_tokens_per_chunk=200,   # preferred upper bound (None to disable)
                min_tokens_per_chunk=100,   # expand small chunks to reach this minimum (None to disable)
                sentence_per_line=True,
                keep_ref_text=False,
                strip_leading_headings=True,
                force=True,
                debug=False,
            )
            essay_stat["t_chunking"] = time.time() - t0

            # recompute chunk_ids for this essay
            t0 = time.time()
            chunk_ids = collect_chunk_ids()
            essay_stat["t_build_chunk_ids"] = time.time() - t0
            essay_stat["n_chunks"] = len(chunk_ids)

            # --------------------------------------------------------
            # 3) embed_and_index_chunks
            # --------------------------------------------------------
            t0 = time.time()
            embed_and_index_chunks(
                "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                "SGCE-KG/data/Chunks/chunks_emb",
                "BAAI/bge-large-en-v1.5",
                "BAAI/bge-small-en-v1.5",
                False,   # use_small_model_for_dev
                32,      # batch_size
                None,    # device -> auto
                True,    # save_index
                True,    # force
            )
            essay_stat["t_embed_index"] = time.time() - t0

            # --------------------------------------------------------
            # 4) Entity Recognition
            # --------------------------------------------------------
            t0 = time.time()
            run_entity_extraction_on_chunks(
                chunk_ids,
                prev_chunks=5,
                save_debug=False,
                model="gpt-5.1",
                max_tokens=8000,
            )
            essay_stat["t_entity_recognition"] = time.time() - t0

            # --------------------------------------------------------
            # 5) Seed for Ent Resolution (NEW helper)
            # --------------------------------------------------------
            prepare_entity_seed_for_iterative_resolution()

            # --------------------------------------------------------
            # 6) Ent Resolution (Multi Run)
            # --------------------------------------------------------
            t0 = time.time()
            iterative_resolution()
            essay_stat["t_entity_resolution"] = time.time() - t0

            # --------------------------------------------------------
            # 7) Cls Rec input producer
            # --------------------------------------------------------
            t0 = time.time()
            # input_path & out_file are defined in your existing code.
            produce_clean_jsonl(input_path, out_file)
            essay_stat["t_cls_rec_input"] = time.time() - t0

            # --------------------------------------------------------
            # 8) Cls Recognition
            # --------------------------------------------------------
            t0 = time.time()
            classrec_iterative_main()
            essay_stat["t_cls_recognition"] = time.time() - t0

            # --------------------------------------------------------
            # 9) Create input for Cls Res
            # --------------------------------------------------------
            t0 = time.time()
            main_input_for_cls_res()
            essay_stat["t_cls_res_input"] = time.time() - t0

            # --------------------------------------------------------
            # 10) Cls Res Multi Run
            # --------------------------------------------------------
            t0 = time.time()
            run_pipeline_iteratively()
            essay_stat["t_cls_res_multi_run"] = time.time() - t0

            # --------------------------------------------------------
            # 11) Relation Rec (single run)
            # --------------------------------------------------------
            t0 = time.time()
            run_rel_rec(
                entities_path="SGCE-KG/data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl",
                chunks_path="SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                output_path="SGCE-KG/data/Relations/Rel Rec/relations_raw.jsonl",
                model="gpt-5.1",
            )
            essay_stat["t_rel_recognition"] = time.time() - t0

            # --------------------------------------------------------
            # 12) Relation Res Multi Run
            # --------------------------------------------------------
            t0 = time.time()
            run_relres_iteratively()
            essay_stat["t_rel_resolution"] = time.time() - t0

            # --------------------------------------------------------
            # 13) Export KG to CSVs
            # --------------------------------------------------------
            t0 = time.time()
            export_relations_and_nodes_to_csv()
            essay_stat["t_export_kg"] = time.time() - t0

            # --------------------------------------------------------
            # 14) SIMPLE COUNTS (nodes / relations)
            # --------------------------------------------------------
            nodes_csv = KG_DIR / "nodes.csv"
            rels_csv = KG_DIR / "rels_fixed_no_raw.csv"
            essay_stat["nodes_count"] = count_csv_rows(nodes_csv)
            essay_stat["relations_count"] = count_csv_rows(rels_csv)

            essay_stat["success"] = True

        except Exception as e:
            essay_stat["error"] = str(e)
            essay_stat["traceback"] = traceback.format_exc()
            print(f"[error] Failure on {label}: {e}")

        finally:
            essay_stat["t_total"] = time.time() - t_run_start

            # Snapshot /data (even on failure, for debugging)
            try:
                dest_dir = copy_data_for_essay(idx)
                essay_stat["data_snapshot_dir"] = str(dest_dir)
            except Exception as e:
                print(f"[warn] Failed to snapshot data for {label}: {e}")

            # Clear pipeline dirs for NEXT essay
            clear_data_subfolders()

            run_stats.append(essay_stat)

    # ----------------------------------------------------------------
    # SAVE OVERALL RUN STATS
    # ----------------------------------------------------------------
    stats_path = KG_RUNS_ROOT / "trace_kg_essays_run_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote overall run stats to {stats_path}")


# --------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------

if __name__ == "__main__":
    run_trace_kg_for_all_essays()


#endregion#? Create KG for each Essay  - V4
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?   Create KG for each Essay  - V5



"""
TRACE KG multi-essay runner (fixed entity seed issue + defined input_path/out_file)

Runs the full pipeline independently for each essay in:
    SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json

For each essay i:
  1) Clear data/{Chunks,Classes,Entities,KG,Relations}
  2) Write that essay as the only document to data/pdf_to_json/Plain_Text.json
  3) Run, IN ORDER:

        sentence_chunks_token_driven(...)
        embed_and_index_chunks(...)
        run_entity_extraction_on_chunks(...)
        prepare_entity_seed_for_iterative_resolution()
        iterative_resolution()
        produce_clean_jsonl(input_path, out_file)
        classrec_iterative_main()
        main_input_for_cls_res()
        run_pipeline_iteratively()
        run_rel_rec(...)
        run_relres_iteratively()
        export_relations_and_nodes_to_csv()

  4) Copy /data to KGs_from_Essays/KG_Essay_<i>
  5) Clear data/{Chunks,Classes,Entities,KG,Relations} again

Records timing + status per essay to:
    SGCE-KG/KGs_from_Essays/trace_kg_essays_run_stats.json
"""

import json
import os
import shutil
import time
import traceback
from pathlib import Path

from tqdm import tqdm  # pip install tqdm

# --------------------------------------------------------------------
# CONSTANT PATHS
# --------------------------------------------------------------------

BASE_ROOT = Path("SGCE-KG")
DATA_ROOT = BASE_ROOT / "data"
ESSAYS_JSON = BASE_ROOT / "Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json"
PLAIN_TEXT_JSON = DATA_ROOT / "pdf_to_json" / "Plain_Text.json"
KG_RUNS_ROOT = BASE_ROOT / "KGs_from_Essays"

CHUNKS_DIR = DATA_ROOT / "Chunks"
CLASSES_DIR = DATA_ROOT / "Classes"
ENTITIES_DIR = DATA_ROOT / "Entities"
KG_DIR = DATA_ROOT / "KG"
RELATIONS_DIR = DATA_ROOT / "Relations"

# Make sure base dirs exist
DATA_ROOT.mkdir(parents=True, exist_ok=True)
PLAIN_TEXT_JSON.parent.mkdir(parents=True, exist_ok=True)
KG_RUNS_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# PATHS FOR Cls Rec INPUT (FIX FOR input_path / out_file)
# --------------------------------------------------------------------

# Directory where class-recognition expects its input entities
CLS_INPUT_DIR = DATA_ROOT / "Classes" / "Cls_Input"
CLS_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# This is the file classrec_iterative_main() reads:
#   INPUT_PATH = "/home/.../data/Classes/Cls_Input/cls_input_entities.jsonl"
out_file = CLS_INPUT_DIR / "cls_input_entities.jsonl"

# This must be the *final resolved entities* JSONL produced by iterative_resolution().
# Adjust the filename here if your entity-resolution code uses a different one.
input_path = (
    DATA_ROOT
    / "Entities"
    / "Ent_Res_IterativeRuns"
    / "overall_summary"
    / "entities_final.jsonl"
)
# ^^^^^ If your iterative_resolution() writes, for example, "final_entities_resolved.jsonl"
# instead of "entities_final.jsonl", just change that filename above.


# --------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------

def clear_data_subfolders() -> None:
    """
    Remove EVERYTHING inside these pipeline folders (but keep the folders):
      - Chunks
      - Classes
      - Entities
      - KG
      - Relations
    """
    for d in [CHUNKS_DIR, CLASSES_DIR, ENTITIES_DIR, KG_DIR, RELATIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        for child in d.iterdir():
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except Exception as e:
                print(f"[warn] Failed to remove {child}: {e}")


def extract_essay_text(rec, idx: int) -> str:
    """
    Heuristic to get the essay text from one JSON record.
    Adjust this if your essay JSON schema differs.
    """
    if isinstance(rec, str):
        return rec

    if isinstance(rec, dict):
        # Try common field names first
        for k in ["text", "essay_text", "content", "body", "answer", "Plain_Text"]:
            v = rec.get(k)
            if isinstance(v, str) and v.strip():
                return v

        # Fallback: choose the longest string field
        best = ""
        for v in rec.values():
            if isinstance(v, str) and len(v) > len(best):
                best = v
        if best:
            return best

    # Last resort
    return str(rec)


def load_essays():
    """
    Load essays from Plain_Text_100_Essays.json.
    Supports list, dict, or single-object JSON.
    """
    if not ESSAYS_JSON.exists():
        raise FileNotFoundError(f"Essays file not found: {ESSAYS_JSON}")

    with ESSAYS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    essays = []

    if isinstance(data, list):
        for idx, rec in enumerate(data, start=1):  # 1-based index
            text = extract_essay_text(rec, idx)
            essays.append({
                "index": idx,
                "raw": rec,
                "text": text,
            })
    elif isinstance(data, dict):
        for idx, (key, rec) in enumerate(data.items(), start=1):
            text = extract_essay_text(rec, idx)
            essays.append({
                "index": idx,
                "key": key,
                "raw": rec,
                "text": text,
            })
    else:
        essays.append({
            "index": 1,
            "raw": data,
            "text": extract_essay_text(data, 1),
        })

    return essays


def write_plain_text_input(essay_meta: dict) -> None:
    """
    Overwrite Plain_Text.json with a single-doc JSON for the current essay.
    This is the ONLY input the chunker reads.
    """
    essay_idx = essay_meta["index"]
    doc = {
        "id": f"essay_{essay_idx:03d}",
        "ref_index": essay_idx,
        "ref_title": f"Essay {essay_idx}",
        "text": essay_meta["text"],
    }
    with PLAIN_TEXT_JSON.open("w", encoding="utf-8") as f:
        json.dump([doc], f, ensure_ascii=False, indent=2)


def collect_chunk_ids():
    """
    After chunking, read chunks_sentence.jsonl and return list of chunk ids
    for run_entity_extraction_on_chunks.
    """
    chunks_path = CHUNKS_DIR / "chunks_sentence.jsonl"
    ids = []
    if not chunks_path.exists():
        return ids
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get("id")
            if cid is not None:
                ids.append(cid)
    return ids


def copy_data_for_essay(essay_index: int) -> Path:
    """
    Copy the entire /data folder to:
        /.../SGCE-KG/KGs_from_Essays/KG_Essay_<index>
    Returns the destination path.
    """
    dest = KG_RUNS_ROOT / f"KG_Essay_{essay_index:03d}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(DATA_ROOT, dest)
    return dest


def count_csv_rows(path: Path) -> int:
    """
    Count data rows in a CSV (minus header). Returns 0 if file doesn't exist.
    """
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


# --------------------------------------------------------------------
# NEW: entity seed prep for iterative_resolution()
# --------------------------------------------------------------------

def _score_entity_candidate(path: Path) -> tuple:
    """
    Heuristic scoring to pick the best entity .jsonl file as a seed.
    Prefer names with 'entities_raw', then 'entities', and larger size.
    """
    name = path.name.lower()
    score = 0
    if "entities_raw" in name or "entity_raw" in name:
        score += 100
    elif "entities" in name or "entity" in name:
        score += 50
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    return (score, size)


def prepare_entity_seed_for_iterative_resolution():
    """
    Ensure that:
      data/Entities/iterative_runs/entities_raw_seed_backup.jsonl
      data/Entities/iterative_runs/entities_raw_seed.jsonl
    exist before calling iterative_resolution().

    Strategy:
      - Look for any *.jsonl under data/Entities (except inside iterative_runs).
      - Pick the best candidate by name + size.
      - Copy it as both seed and seed_backup.
    """
    iter_dir = ENTITIES_DIR / "iterative_runs"
    iter_dir.mkdir(parents=True, exist_ok=True)

    backup_path = iter_dir / "entities_raw_seed_backup.jsonl"
    seed_path = iter_dir / "entities_raw_seed.jsonl"

    # If both already exist, do nothing (e.g., if some other part of your code made them)
    if backup_path.exists() and seed_path.exists():
        return

    # Find candidate entity files produced by entity extraction for THIS essay
    candidates = []
    if ENTITIES_DIR.exists():
        for p in ENTITIES_DIR.rglob("*.jsonl"):
            # skip previous iterative_runs content (if any)
            try:
                if "iterative_runs" in p.parts:
                    continue
            except Exception:
                pass
            candidates.append(p)

    if not candidates:
        raise RuntimeError(
            "prepare_entity_seed_for_iterative_resolution: "
            "No .jsonl files found under data/Entities to use as seed."
        )

    best = max(candidates, key=_score_entity_candidate)
    # Copy to seed and backup
    shutil.copy2(best, backup_path)
    shutil.copy2(best, seed_path)
    print(f"[info] Seeded entity resolution from: {best.name}")


# --------------------------------------------------------------------
# MAIN MULTI-ESSAY RUNNER
# --------------------------------------------------------------------

def run_trace_kg_for_all_essays():
    essays = load_essays()
    run_stats = []

    print(f"[info] Loaded {len(essays)} essays from {ESSAYS_JSON}")

    for essay_meta in tqdm(essays, desc="TRACE KG essays"):
        idx = essay_meta["index"]
        label = f"Essay_{idx:03d}"

        essay_stat = {
            "index": idx,
            "essay_id": label,
            "t_chunking": None,
            "t_embed_index": None,
            "n_chunks": None,
            "t_build_chunk_ids": None,
            "t_entity_recognition": None,
            "t_entity_resolution": None,
            "t_cls_rec_input": None,
            "t_cls_recognition": None,
            "t_cls_res_input": None,
            "t_cls_res_multi_run": None,
            "t_rel_recognition": None,
            "t_rel_resolution": None,
            "t_export_kg": None,
            "success": False,
            "error": None,
            "traceback": None,
            "t_total": None,
            "nodes_count": None,
            "relations_count": None,
            "data_snapshot_dir": None,
        }

        t_run_start = time.time()
        try:
            # --------------------------------------------------------
            # 0) RESET PIPELINE DIRECTORIES FOR THIS ESSAY
            # --------------------------------------------------------
            clear_data_subfolders()

            # --------------------------------------------------------
            # 1) WRITE INPUT FOR CHUNKING
            # --------------------------------------------------------
            write_plain_text_input(essay_meta)

            # --------------------------------------------------------
            # 2) CHUNKING
            # --------------------------------------------------------
            t0 = time.time()
            sentence_chunks_token_driven(
                "SGCE-KG/data/pdf_to_json/Plain_Text.json",
                "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                max_tokens_per_chunk=200,   # preferred upper bound (None to disable)
                min_tokens_per_chunk=100,   # expand small chunks to reach this minimum (None to disable)
                sentence_per_line=True,
                keep_ref_text=False,
                strip_leading_headings=True,
                force=True,
                debug=False,
            )
            essay_stat["t_chunking"] = time.time() - t0

            # recompute chunk_ids for this essay
            t0 = time.time()
            chunk_ids = collect_chunk_ids()
            essay_stat["t_build_chunk_ids"] = time.time() - t0
            essay_stat["n_chunks"] = len(chunk_ids)

            # --------------------------------------------------------
            # 3) embed_and_index_chunks
            # --------------------------------------------------------
            t0 = time.time()
            embed_and_index_chunks(
                "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                "SGCE-KG/data/Chunks/chunks_emb",
                "BAAI/bge-large-en-v1.5",
                "BAAI/bge-small-en-v1.5",
                False,   # use_small_model_for_dev
                32,      # batch_size
                None,    # device -> auto
                True,    # save_index
                True,    # force
            )
            essay_stat["t_embed_index"] = time.time() - t0

            # --------------------------------------------------------
            # 4) Entity Recognition
            # --------------------------------------------------------
            t0 = time.time()
            run_entity_extraction_on_chunks(
                chunk_ids,
                prev_chunks=5,
                save_debug=False,
                model="gpt-5.1",
                max_tokens=8000,
            )
            essay_stat["t_entity_recognition"] = time.time() - t0

            # --------------------------------------------------------
            # 5) Seed for Ent Resolution (NEW helper)
            # --------------------------------------------------------
            prepare_entity_seed_for_iterative_resolution()

            # --------------------------------------------------------
            # 6) Ent Resolution (Multi Run)
            # --------------------------------------------------------
            t0 = time.time()
            iterative_resolution()
            essay_stat["t_entity_resolution"] = time.time() - t0

            # --------------------------------------------------------
            # 7) Cls Rec input producer
            # --------------------------------------------------------
            t0 = time.time()
            # input_path & out_file now defined globally above.
            produce_clean_jsonl(input_path, out_file)
            essay_stat["t_cls_rec_input"] = time.time() - t0

            # --------------------------------------------------------
            # 8) Cls Recognition
            # --------------------------------------------------------
            t0 = time.time()
            classrec_iterative_main()
            essay_stat["t_cls_recognition"] = time.time() - t0

            # --------------------------------------------------------
            # 9) Create input for Cls Res
            # --------------------------------------------------------
            t0 = time.time()
            main_input_for_cls_res()
            essay_stat["t_cls_res_input"] = time.time() - t0

            # --------------------------------------------------------
            # 10) Cls Res Multi Run
            # --------------------------------------------------------
            t0 = time.time()
            run_pipeline_iteratively()
            essay_stat["t_cls_res_multi_run"] = time.time() - t0

            # --------------------------------------------------------
            # 11) Relation Rec (single run)
            # --------------------------------------------------------
            t0 = time.time()
            run_rel_rec(
                entities_path="SGCE-KG/data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl",
                chunks_path="SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                output_path="SGCE-KG/data/Relations/Rel Rec/relations_raw.jsonl",
                model="gpt-5.1",
            )
            essay_stat["t_rel_recognition"] = time.time() - t0

            # --------------------------------------------------------
            # 12) Relation Res Multi Run
            # --------------------------------------------------------
            t0 = time.time()
            run_relres_iteratively()
            essay_stat["t_rel_resolution"] = time.time() - t0

            # --------------------------------------------------------
            # 13) Export KG to CSVs
            # --------------------------------------------------------
            t0 = time.time()
            export_relations_and_nodes_to_csv()
            essay_stat["t_export_kg"] = time.time() - t0

            # --------------------------------------------------------
            # 14) SIMPLE COUNTS (nodes / relations)
            # --------------------------------------------------------
            nodes_csv = KG_DIR / "nodes.csv"
            rels_csv = KG_DIR / "rels_fixed_no_raw.csv"
            essay_stat["nodes_count"] = count_csv_rows(nodes_csv)
            essay_stat["relations_count"] = count_csv_rows(rels_csv)

            essay_stat["success"] = True

        except Exception as e:
            essay_stat["error"] = str(e)
            essay_stat["traceback"] = traceback.format_exc()
            print(f"[error] Failure on {label}: {e}")

        finally:
            essay_stat["t_total"] = time.time() - t_run_start

            # Snapshot /data (even on failure, for debugging)
            try:
                dest_dir = copy_data_for_essay(idx)
                essay_stat["data_snapshot_dir"] = str(dest_dir)
            except Exception as e:
                print(f"[warn] Failed to snapshot data for {label}: {e}")

            # Clear pipeline dirs for NEXT essay
            clear_data_subfolders()

            run_stats.append(essay_stat)

    # ----------------------------------------------------------------
    # SAVE OVERALL RUN STATS
    # ----------------------------------------------------------------
    stats_path = KG_RUNS_ROOT / "trace_kg_essays_run_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote overall run stats to {stats_path}")


# --------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------

if __name__ == "__main__":
    run_trace_kg_for_all_essays()


#endregion#? Create KG for each Essay  - V5
#?#########################  End  ##########################



