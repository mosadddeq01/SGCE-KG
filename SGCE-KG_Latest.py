








#!############################################# Start Chapter ##################################################
#region:#!   Entity Identification




#?######################### Start ##########################
#region:#?   Chunking v3

#!/usr/bin/env python3
"""
Chunking v3 — token-driven, sentence-preserving chunker

- Uses spaCy (en_core_web_sm) for sentence splitting and WILL RAISE if not installed.
- No fixed sentence-count control anymore.
- Two parameters control chunking:
    * max_tokens_per_chunk (int or None) : preferred upper bound for a chunk (won't split sentences)
    * min_tokens_per_chunk (int or None) : ensure small chunks are expanded to reach this minimum
- Behavior (simple and deterministic):
    - Walk sentences in order. Build a chunk by adding sentences until chunk_tokens >= min_tokens_per_chunk.
    - When adding a sentence would push tokens > max_tokens_per_chunk:
        • If chunk currently empty (single sentence > max): accept the single sentence.
        • Else, if removing the last sentence keeps tokens >= min_tokens_per_chunk, pop it and leave it for next chunk.
        • Otherwise keep the sentence (need to meet min requirement).
    - Never split sentences.
    - No overlap windows (keeps provenance simple).
- Output: JSONL file of chunks with fields similar to your previous format.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import re
from tqdm import tqdm

# ---------- simple token estimator (cheap heuristic) ----------
TOKENS_PER_WORD = 1.25
def estimate_tokens_fast(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * TOKENS_PER_WORD))

HEADING_RE = re.compile(r'^\s*\d+(\.\d+)*\s+[\w\-\(\) ]{1,80}$')

# -----------------------
# require spaCy + model
# -----------------------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm") from e
except Exception as e:
    raise RuntimeError("spaCy is required. Install with `pip install spacy` and the model `python -m spacy download en_core_web_sm`.") from e

def split_sentences_spacy(text: str) -> List[str]:
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# -----------------------
# main chunker
# -----------------------
def sentence_chunks_token_driven(
    sections_json_path: str,
    out_path: str = "data/chunks_sentence.jsonl",
    max_tokens_per_chunk: Optional[int] = 350,
    min_tokens_per_chunk: Optional[int] = 250,
    sentence_per_line: bool = True,
    keep_ref_text: bool = False,
    strip_leading_headings: bool = True,
    force: bool = False,
    debug: bool = False
) -> List[Dict]:
    """
    Token-driven chunker. See module docstring for behavior.
    """
    src = Path(sections_json_path)
    assert src.exists(), f"sections file not found: {src}"
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists() and not force:
        with open(outp, "r", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh]

    with open(src, "r", encoding="utf-8") as fh:
        sections = json.load(fh)

    chunks: List[Dict] = []
    global_gid = 1

    for sidx, sec in enumerate(tqdm(sections, desc="Chunking sections")):
        title = sec.get("title", f"section_{sidx}")
        text = sec.get("text", "") or ""
        start_page = sec.get("start_page")
        end_page = sec.get("end_page")

        sents = split_sentences_spacy(text)
        if not sents:
            continue

        n_sents = len(sents)
        if debug:
            print(f"[section {sidx}] title={title!r} n_sents={n_sents}")

        sent_idx = 0
        chunk_index_in_section = 0

        while sent_idx < n_sents:
            # Start a new chunk
            chunk_sentences: List[str] = []
            chunk_tokens = 0
            chunk_start_idx = sent_idx
            stripped_heading = None

            # If strip_leading_headings: check next sentence and remove it from chunk text if heading-like,
            # but keep it in provenance (we will still include it in 'sentences' for traceability).
            # Implementation: if heading, we will not remove it from chunk_sentences list but will not include in chunk_text.
            # Simpler: we detect heading and treat it as normal sentence but set stripped_heading field if popped later.
            # For simplicity, we will remove a leading heading-like sentence from chunk_text, but keep it as part of 'sentences' for provenance.
            # However, to keep behavior consistent, we'll implement: if the very next sentence looks like a heading, record and skip it from chunk_text,
            # and increment sent_idx by 1 (heading consumed and not part of token/accounting).
            if strip_leading_headings and sent_idx < n_sents:
                maybe = sents[sent_idx]
                if HEADING_RE.match(maybe) or (len(maybe) < 120 and maybe.strip().endswith("Mechanisms") and any(ch.isdigit() for ch in maybe)):
                    stripped_heading = maybe
                    # consume heading from stream (it won't count toward tokens or sentences in chunk text)
                    sent_idx += 1
                    # if we've consumed all sentences, break
                    if sent_idx >= n_sents:
                        # create an empty chunk that only contains heading? Better to emit a small chunk with heading text
                        chunk_sentences = []
                        chunk_text = maybe
                        chunk_tokens = estimate_tokens_fast(chunk_text)
                        # finalize single-heading chunk
                        chunk = {
                            "id": f"Ch_{global_gid:06d}",
                            "ref_index": sidx,
                            "ref_title": title,
                            "start_page": start_page,
                            "end_page": end_page,
                            "chunk_index_in_section": chunk_index_in_section,
                            "text": chunk_text,
                            "sentences": [maybe],
                            "span": [chunk_start_idx, chunk_start_idx],
                            "n_words": len(chunk_text.split()),
                            "n_tokens_est": chunk_tokens,
                            "stripped_heading": stripped_heading,
                            "reason": "heading_only"
                        }
                        if keep_ref_text:
                            chunk["ref_text"] = text
                        chunks.append(chunk)
                        global_gid += 1
                        chunk_index_in_section += 1
                        break  # move to next section

            # Now build chunk by adding sentences until we meet min_tokens_per_chunk (if set),
            # respecting max_tokens_per_chunk when possible.
            while sent_idx < n_sents:
                next_sent = sents[sent_idx]
                next_tok = estimate_tokens_fast(next_sent)

                # If chunk currently empty:
                if not chunk_sentences:
                    # tentatively add the sentence
                    chunk_sentences.append(next_sent)
                    chunk_tokens += next_tok
                    sent_idx += 1
                    # if this single sentence already exceeds max_tokens_per_chunk and min exists, we keep it
                    if max_tokens_per_chunk and chunk_tokens > max_tokens_per_chunk:
                        # single sentence exceeds max - we accept it per policy
                        break
                    # otherwise continue to accumulate until we hit min (if min set)
                    if min_tokens_per_chunk and chunk_tokens >= min_tokens_per_chunk:
                        break  # satisfied
                    # if no min is set and we have a sensible size, we can also break to avoid tiny chunks:
                    if not min_tokens_per_chunk and max_tokens_per_chunk and chunk_tokens >= max_tokens_per_chunk:
                        break
                    # else continue loop to add more
                    continue

                # chunk has content already; check if adding next sentence would exceed max
                would_be = chunk_tokens + next_tok
                if max_tokens_per_chunk and would_be > max_tokens_per_chunk:
                    # decide whether to keep next_sent in this chunk or postpone it
                    # compute tokens without last added sentence (i.e., current chunk_tokens)
                    if min_tokens_per_chunk and chunk_tokens >= min_tokens_per_chunk:
                        # we already satisfy min; leave next_sent for the next chunk
                        break
                    else:
                        # we do NOT satisfy min yet. Add the sentence even if it exceeds max (to meet min)
                        chunk_sentences.append(next_sent)
                        chunk_tokens = would_be
                        sent_idx += 1
                        break
                else:
                    # safe to add next sentence
                    chunk_sentences.append(next_sent)
                    chunk_tokens = would_be
                    sent_idx += 1
                    # if we have a min and satisfied it, we can stop
                    if min_tokens_per_chunk and chunk_tokens >= min_tokens_per_chunk:
                        break
                    # otherwise loop to add more (or until n_sents)
                    continue

            # finalize chunk_text (may exclude stripped_heading from text if present)
            chunk_text = ("\n" if sentence_per_line else " ").join(chunk_sentences).strip()
            n_tokens_est = chunk_tokens
            n_words = len(chunk_text.split()) if chunk_text else 0
            span_end = (chunk_start_idx + len(chunk_sentences) - 1) if chunk_sentences else chunk_start_idx

            # reason field for diagnostics
            reason = None
            if max_tokens_per_chunk and n_tokens_est > max_tokens_per_chunk:
                reason = "exceeded_max_kept_whole_or_to_meet_min"
            elif min_tokens_per_chunk and n_tokens_est < min_tokens_per_chunk:
                reason = "below_min_unable_to_expand"  # e.g., reached section end

            chunk = {
                "id": f"Ch_{global_gid:06d}",
                "ref_index": sidx,
                "ref_title": title,
                "start_page": start_page,
                "end_page": end_page,
                "chunk_index_in_section": chunk_index_in_section,
                "text": chunk_text,
                "sentences": chunk_sentences,
                "span": [chunk_start_idx, span_end],
                "n_words": n_words,
                "n_tokens_est": n_tokens_est,
                "stripped_heading": stripped_heading,
                "reason": reason
            }
            if keep_ref_text:
                chunk["ref_text"] = text

            chunks.append(chunk)
            global_gid += 1
            chunk_index_in_section += 1

        # end while sent_idx

    # write ndjson
    with open(outp, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    return chunks





# -----------------------
# Chunking  - Run statement
# -----------------------

# if __name__ == "__main__":
#     sentence_chunks_token_driven(
#         "SGCE-KG/data/pdf_to_json/Plain_Text.json",
#         "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
#         max_tokens_per_chunk=200,   # preferred upper bound (None to disable)
#         min_tokens_per_chunk=100,   # expand small chunks to reach this minimum (None to disable)
#         sentence_per_line=True,
#         keep_ref_text=False,
#         strip_leading_headings=True,
#         force=True,
#         debug=False
#     )


#endregion#? Chunking v3
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Embedding + FAISS Index

# ------------------------------
# Chunk Embedding + FAISS Index (fast, batched)
# ------------------------------
import os
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss

@torch.no_grad()
def embed_and_index_chunks(
    chunks_jsonl_path: str = "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
    output_prefix: str = "SGCE-KG/data/Chunks/chunks_emb",
    embed_model_large: str = "BAAI/bge-large-en-v1.5",
    embed_model_small: str = "BAAI/bge-small-en-v1.5",
    use_small_model_for_dev: bool = True,
    batch_size: int = 32,
    device: Optional[str] = None,
    save_index: bool = True,
    force: bool = False
):
    """
    Read NDJSON chunks, produce embeddings, save metadata + embeddings + optional FAISS index.
    Returns (meta_list, embeddings, faiss_index or None)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_path = Path(chunks_jsonl_path)
    assert in_path.exists(), f"Chunks file not found: {in_path}"

    out_meta_path = Path(f"{output_prefix}_meta.jsonl")
    out_vec_path = Path(f"{output_prefix}_vecs.npy")
    out_idx_path = Path(f"{output_prefix}_faiss.index")

    # idempotent shortcut
    if out_meta_path.exists() and out_vec_path.exists() and (out_idx_path.exists() or not save_index) and not force:
        # load
        meta = [json.loads(l) for l in open(out_meta_path, "r", encoding="utf-8")]
        vecs = np.load(out_vec_path)
        index = None
        if save_index and out_idx_path.exists():
            index = faiss.read_index(str(out_idx_path))
        return meta, vecs, index

    # choose model
    model_name = embed_model_small if use_small_model_for_dev else embed_model_large
    print(f"Loading tokenizer+model: {model_name} -> device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # load chunks
    with open(in_path, "r", encoding="utf-8") as fh:
        chunks = [json.loads(l) for l in fh]

    # prepare outputs
    metas = []
    vecs_list = []

    # batching
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i+batch_size]
        texts = [c.get("text","") for c in batch]
        # tokenize
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        # forward
        out = model(**enc)
        # pooling: mean-pool over valid tokens (attention mask)
        if hasattr(out, "last_hidden_state"):
            token_emb = out.last_hidden_state  # (B, T, D)
        elif hasattr(out, "hidden_states"):
            token_emb = out.hidden_states[-1]
        else:
            raise RuntimeError("Model output has no last_hidden_state or hidden_states")

        mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        token_emb = token_emb * mask  # zero out padded tokens
        sum_emb = token_emb.sum(dim=1)  # (B, D)
        denom = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_emb = sum_emb / denom  # (B, D)

        # normalize
        mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)

        emb_np = mean_emb.cpu().numpy().astype("float32")
        vecs_list.append(emb_np)

        for j, c in enumerate(batch):
            metas.append({
                "id": c.get("id"),
                "ref_index": c.get("ref_index"),
                "ref_title": c.get("ref_title"),
                "chunk_index_in_section": c.get("chunk_index_in_section"),
                "start_page": c.get("start_page"),
                "end_page": c.get("end_page"),
                "n_words": c.get("n_words"),
                # minimal provenance to keep meta small
            })

    all_vecs = np.vstack(vecs_list).astype("float32")

    # save metadata and vectors
    with open(out_meta_path, "w", encoding="utf-8") as fh:
        for m in metas:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    np.save(out_vec_path, all_vecs)

    index = None
    if save_index:
        dim = all_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(all_vecs)
        faiss.write_index(index, str(out_idx_path))

    return metas, all_vecs, index






# -----------------------
# embed_and_index_chunks  - Run statement
# -----------------------


# embed_and_index_chunks(
#     "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
#     "SGCE-KG/data/Chunks/chunks_emb",
#     "BAAI/bge-large-en-v1.5",
#     "BAAI/bge-small-en-v1.5",
#     False,   # use_small_model_for_dev
#     32,     # batch_size
#     None,   # device -> auto
#     True,   # save_index
#     True)  # force



#endregion#? Embedding + FAISS Index
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Entity Recognition v8 - Intrinsic properties added (Responses + gpt-5.1)

import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from datetime import datetime

# ---------- CONFIG: paths ----------
CHUNKS_JSONL =      "SGCE-KG/data/Chunks/chunks_sentence.jsonl"
ENTITIES_OUT =      "SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl"
DEFAULT_DEBUG_DIR = "SGCE-KG/data/Entities/Ent_Raw_0/entity_raw_debug_prompts_outputs"

# ---------- OPENAI client (load key from env or fallback file path) ----------
def _load_openai_key(
    envvar: str = "OPENAI_API_KEY",
    fallback_path: str = ".env"
) -> str:
    """
    Load OpenAI key from:
      1) Environment variable OPENAI_API_KEY
      2) Or, if not set, from a file at fallback_path (file content is the key).
    """
    key = os.getenv(envvar)
    if key:
        return key

    # If env var not set, try reading key directly from fallback_path file
    p = Path(fallback_path)
    if p.exists():
        try:
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass

    return ""

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️  OPENAI API key not found or seems invalid. "
          "Set OPENAI_API_KEY env or place key text in the fallback file path.")
client = OpenAI(api_key=OPENAI_KEY)

# ---------- Utility: load chunks ----------
def load_chunks(chunks_jsonl_path: str = CHUNKS_JSONL) -> List[Dict]:
    p = Path(chunks_jsonl_path)
    if not p.exists():
        # don't assert at import time; raise only when the function is actually called
        raise FileNotFoundError(
            f"Chunks file not found: {p}\n"
            f"Please run your chunking step first to create this file, e.g. sentence_chunks_token_driven(...)\n"
            f"Expected path: {chunks_jsonl_path}"
        )
    with open(p, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]

# ---------- Save helper (append-safe) ----------
def save_entities(entities: List[Dict], out_path: str = ENTITIES_OUT):
    if not entities:
        print("save_entities: nothing to save.")
        return
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "a", encoding="utf-8") as fh:
        for e in entities:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Saved {len(entities)} entities to {out_path}")

# ---------- Small helper to find a chunk by id ----------
def get_chunk_by_id(chunk_id: str, chunks: List[Dict]) -> Dict:
    for c in chunks:
        if c.get("id") == chunk_id:
            return c
    raise ValueError(f"chunk_id {chunk_id} not found")

# ---------- Helper: get previous chunks from same section ----------
def get_previous_chunks(chunk: Dict, chunks: List[Dict], prev_n: int = 1) -> List[Dict]:
    """
    Return up to prev_n previous chunks from the same ref_index (by chunk_index_in_section order).
    Preserves chronological order (oldest -> nearest previous).
    """
    if prev_n <= 0:
        return []
    ref_index = chunk.get("ref_index")
    idx_in_section = chunk.get("chunk_index_in_section", 0)
    same_sec = [c for c in chunks if c.get("ref_index") == ref_index]
    same_sec_sorted = sorted(same_sec, key=lambda x: x.get("chunk_index_in_section", 0))
    prevs = []
    for c in same_sec_sorted:
        if c.get("chunk_index_in_section", 0) < idx_in_section:
            prevs.append(c)
    # take at most prev_n from the end (nearest previous), then return in chronological order
    prevs = prevs[-prev_n:] if prevs else []
    return prevs

# ---------- Prompt builder (chunk + optional prev_chunks) ----------
def build_entity_prompt_with_context(chunk: Dict, prev_chunks: Optional[List[Dict]] = None) -> str:
    
    """
    Build a prompt that includes the focus chunk and n previous chunk(s) as explicit context.
    Previous chunks are concatenated as plain text (no IDs) to form CONTEXT text.
    The LLM's task: extract entities from the FOCUS chunk only. Previous chunks are provided
    as context for disambiguation. The prompt encourages adding low-confidence new entities if
    uncertain (they will be resolved later).

    Notes:
    - The suggested type hints are *recommendations* only. The model is explicitly allowed
      to propose any other, more specific, or domain-appropriate type strings.
    - This version enforces: do NOT extract relation-level qualifiers here (postpone to Rel Rec).
    - The only allowed "properties" now are truly intrinsic node properties — very rare.
    """
    focus_text = chunk.get("text", "") or ""

    # Suggested (preferred) type hints — not exhaustive and NOT binding.
    suggested_types = [
        "Component",
        "Material",
        "DamageMechanism",
        "FailureEvent",
        "Symptom",
        "Action",
        "FunctionalUnit",
        "OperatingCondition",
        "Environment",
        "InspectionMethod",
        "MitigationAction",
        "Location",
        "ProcessUnit",
        "YOU MAY USE ANY OTHER TYPE THAT FITS BETTER"]

    parts = [
        "GOAL: We are creating a context-enriched knowledge graph (KG) from textual documents.",
        "YOUR TASK (THIS STEP ONLY): Extract entity mentions from the FOCUS chunk ONLY. Relation-level qualifiers, conditions, and other contextual information will be extracted later in the RELATION EXTRACTION step (Rel Rec).",
        "",
        "PRINCIPLES (read carefully):",
        "- Extract broadly: prefer recall (extract candidate mentions). Do NOT be conservative. When in doubt, include the candidate mention. Later stages will cluster, canonicalize, and resolve.",
        "- Ground every output in the FOCUS chunk. You may CONSULT CONTEXT (previous chunks concatenated) only for disambiguation/pronoun resolution.",
        "- DO NOT output relation-level qualifiers, situational context, or evidential/epistemic markers in this step. The ONLY exception is truly intrinsic node properties (see 'INTRINSIC NODE PROPERTIES' below).",
        "- The suggested type hints below are guidance — you may propose more specific domain-appropriate types.",
        "",
        "CORE INSTRUCTION FOR CONCEPTUAL ENTITIES (ENTITY-LEVEL, NOT CLASSES):",
        "- For recurring concepts (phenomena, processes, failure modes, behaviors, conditions, states, methods), extract a SHORT, STABLE, REUSABLE entity-level label as `entity_name`.",
        "- `entity_name` is a canonical mention-level surface form (normalized for this mention). It is NOT an ontology or Schema class label. If you think a class is relevant, place it in `entity_type_hint`.",
        "- If the text describes the concept indirectly (e.g., 'this type of …', 'loss of … under conditions'), infer the best short label (e.g., 'graphitization') and put evidence in `entity_description` and `resolution_context`.",
        "- If unsure of the label, still propose it and lower the `confidence_score` (e.g., 0.5–0.7). We prefer 'extract first, judge later'.",
        "",
        "INTRINSIC NODE PROPERTIES:",
        "- You MUST include `node_properties` when the property is identity-defining for the entity (removing it would change what the entity fundamentally is) and they are MANDATORY when present in FOCUS chunk.",
        "- Intrinsic property means ANY stable attributes that define identity no mather what (removing it would change what the entity fundamentally is) e.g material_grade='304', e.g., chemical formula, etc.",
        "- Expectation: IF the property is defined in relation to other entities, that is not intrinsic, therefore we postpone them to Rel Rec.",
        "",
        "CONFIDENCE GUIDELINES:",
        "- 0.90 - 1.00 : Certain — explicit mention in FOCUS chunk, clear support.",
        "- 0.70 - 0.89 : Likely — supported by FOCUS or resolved by CONTEXT.",
        "- 0.40 - 0.69 : Possible — plausible inference; partial support.",
        "- 0.00 - 0.39 : Speculative — weakly supported; include only if likely useful.",
        "",
        "SUGGESTED TYPE HINTS (Just to give you an idea! You may propose any other type):",
        f"- {', '.join(suggested_types)}",
        "",
        "OUTPUT FORMAT INSTRUCTIONS (REVISED — REQUIRED):",
        "- Return ONLY a single JSON array (no extra commentary, no markdown fences).",
        "- Each element must be an object with the following keys (exact names):",
        "   * entity_name (string) — short canonical surface label for the mention (mention-level, NOT class).",
        "   * entity_description (string) — 10–25 word description derived from the FOCUS chunk (and CONTEXT if needed).",
        "   * entity_type_hint (string) — suggested type (from list or a better string).",
        "   * context_phrase (string) — short (3–10 word) excerpt from the FOCUS chunk that PROVES the mention provenance (required when possible).",
        "   * resolution_context (string) — minimal 20–120 word excerpt that best explains WHY this mention maps to `entity_name`. Prefer the sentence containing the mention and at most one neighbor sentence; if CONTEXT was required, include up to one supporting sentence from CONTEXT. This is used for clustering/resolution — make it disambiguating (co-mentions, verbs, numerics).",
        "   * confidence_score (float) — 0.0–1.0.",
        "   * node_properties (array of objects, Include ONLY if an intrinsic property is present in the focus chunk) — Include for any clearly stated attribute. Each: { 'prop_name': str, 'prop_value': str|num, 'justification': str }."
        "",
        "IMPORTANT:",
        "- DO NOT list entities that appear only in CONTEXT. Only extract mentions present in the FOCUS chunk.",
        "- DO NOT output relation qualifiers, situational context, causal hints, or uncertainty markers here. Postpone them to Rel Rec.",
        "- Do not output ontology-level class names as `entity_name`. If relevant, place such information in `entity_type_hint` and keep `entity_name` a mention-level label.",
        "- For conceptual entities that are described indirectly, prefer a short canonical mention and keep the descriptive evidence in `entity_description` and `resolution_context`.",
        "",
        "EMBEDDING WEIGHT NOTE (for clustering later):",
        "WEIGHTS = {\"name\": 0.45, \"desc\": 0.25, \"resolution_context\": 0.25, \"type\": 0.05}",
        "Build resolution_context precisely — it is the second-most important signal after name and description.",
        "",
        "EXAMPLES (follow JSON shape exactly):"
    ]

    # include previous context if provided
    if prev_chunks:
        ctx_texts = [pc.get("text", "").strip() for pc in prev_chunks if pc.get("text", "").strip()]
        if ctx_texts:
            parts.append("=== CONTEXT (previous chunks concatenated for disambiguation) ===\n")
            parts.append("\n\n".join(ctx_texts))
        else:
            parts.append("NO PREVIOUS CONTEXT PROVIDED.\n---")
    else:
        parts.append("NO PREVIOUS CONTEXT PROVIDED.\n---")

    parts.append("")
    parts.append("=== FOCUS CHUNK (extract from here) ===")
    parts.append(f"FOCUS_CHUNK_ID: {chunk.get('id')}")
    parts.append(focus_text)
    parts.append("")
    parts.append("EXAMPLE OUTPUT (three diverse examples — strictly follow JSON shape):")

    examples = [
      {
        "entity_name": "graphitization",
        "entity_description": "Formation of graphite in steel that reduces ductility and increases brittleness near weld regions.",
        "entity_type_hint": "DamageMechanism",
        "context_phrase": "this type of graphitization",
        "resolution_context": "this type of graphitization observed in low-alloy steels near welds, indicating localized graphite formation associated with embrittlement and loss of toughness.",
        "confidence_score": 0.85
      },
      {
        "entity_name": "authentication failure",
        "entity_description": "A software event where credentials are rejected repeatedly, often producing error 401 in logs.",
        "entity_type_hint": "SoftwareEvent",
        "context_phrase": "repeated authentication failure",
        "resolution_context": "repeated authentication failure for user 'svc_backup' recorded in the log with multiple 401 entries, indicating failed credential validation.",
        "confidence_score": 0.88
      },
      {
        "entity_name": "austenitic stainless steel",
        "entity_description": "A stainless steel family with austenitic crystal structure, commonly designated by grades like 304 or 316.",
        "entity_type_hint": "Material",
        "context_phrase": "austenitic stainless steel (304)",
        "resolution_context": "austenitic stainless steel (304) explicitly mentioned in FOCUS, parenthetical grade '304' identifies the material grade.",
        "confidence_score": 0.95,
        "node_properties": [
          {
            "prop_name": "material_grade",
            "prop_value": "304",
            "justification": "explicit explanation of inclusion in the FOCUS chunk"
          }
        ]
      }
    ]
    parts.append(json.dumps(examples, ensure_ascii=False, indent=2))

    # final join
    return "\n\n".join(parts)

# ---------- OpenAI call (wrapper) ----------
def call_openai(
    prompt: str,
    model: str = "gpt-5.1",
    max_tokens: int = 2000,
    # temperature: float = 0.0,
    reasoning_effort: str = "low"
) -> str:
    """
    Wrapper using the Responses API (works with gpt-5.1 and others).

    - For gpt-5.x models, we pass `reasoning={'effort': reasoning_effort}`.
    - `max_tokens` is mapped to Responses' `max_output_tokens`.
    """
    try:
        print(
            f"[call_openai] model={model} "
            f"max_output_tokens={max_tokens} "
            # f"temperature={temperature} "
            f"reasoning_effort={reasoning_effort}"
        )

        kwargs: Dict[str, object] = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # "temperature": temperature,
        }

        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        # Only attach reasoning config for gpt-5.x models
        if reasoning_effort and isinstance(reasoning_effort, str) and model.startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": reasoning_effort}

        response = client.responses.create(**kwargs)

        # Convenient helper property (joined assistant text)
        txt = response.output_text or ""
        print(
            f"[call_openai] received response (len={len(txt)} chars) "
            f"status={getattr(response, 'status', None)}"
        )

        # Optional: warn if response is incomplete due to max_output_tokens
        if getattr(response, "status", None) == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            print(f"[call_openai] WARNING: response incomplete: {incomplete_details}")

        return txt
    except Exception as e:
        print("OpenAI call error:", e)
        return ""

# ---------- Debug file writer (single clean implementation) ----------
def write_debug_file(
    debug_dir: str,
    chunk: Dict,
    prev_ctx: List[Dict],
    prompt: str,
    llm_output: str,
    parsed_entities: Optional[List[Dict]] = None,
    error: Optional[str] = None
) -> str:
    """
    Write a JSON file containing:
      - explicit focus_chunk (full chunk dict with id, ref_index, chunk_index_in_section, ref_title, text),
      - explicit context_chunks (list of {id, ref_index, chunk_index_in_section, ref_title, text}),
      - prompt_full (string),
      - llm_output_full (string),
      - parsed_entities (list),
      - error (string or None), metadata.
    Returns the path to the file created.
    """
    debug_dir_path = Path(debug_dir)
    debug_dir_path.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex
    ts = datetime.utcnow().isoformat() + "Z"
    fname = f"{chunk.get('id','unknown')}_{ts.replace(':','-').replace('.','-')}_{run_id[:8]}.json"
    out_path = debug_dir_path / fname

    # prepare context chunks minimal view (id + text) to avoid huge nested objects
    context_min = []
    for pc in (prev_ctx or []):
        context_min.append({
            "id": pc.get("id"),
            "ref_index": pc.get("ref_index"),
            "chunk_index_in_section": pc.get("chunk_index_in_section"),
            "ref_title": pc.get("ref_title"),
            "text": pc.get("text")
        })

    payload = {
        "run_id": run_id,
        "timestamp_utc": ts,
        "chunk_id": chunk.get("id"),
        "focus_chunk": {
            "id": chunk.get("id"),
            "ref_index": chunk.get("ref_index"),
            "chunk_index_in_section": chunk.get("chunk_index_in_section"),
            "ref_title": chunk.get("ref_title"),
            "text": chunk.get("text")
        },
        "context_chunks": context_min,
        "prompt_full": prompt,
        "llm_output_full": llm_output,
        "parsed_entities": parsed_entities or [],
        "error": error
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return str(out_path)

# ---------- Main: extract_entities_from_chunk (with optional prev context + debug saving) ----------
def extract_entities_from_chunk(
    chunk_id: str,
    chunks_path: str = CHUNKS_JSONL,
    prev_chunks: int = 2,       # how many previous chunks to include as CONTEXT (default 1). Set 0 to disable.
    model: str = "gpt-5.1",
    max_tokens: int = 16000,
    save_debug: bool = False,   # if True, write full prompt+output+parsed to a debug JSON file
    debug_dir: str = DEFAULT_DEBUG_DIR
) -> List[Dict]:
    """
    Extract entities from the specified focus chunk, optionally including up to `prev_chunks`
    previous chunks from the same section as disambiguating CONTEXT.

    Uses the Responses API (via call_openai) and supports gpt-5.1 reasoning models.

    If save_debug=True, a structured JSON file containing the full prompt and full LLM output
    (and the focus/context text) will be written to `debug_dir` for later inspection.
    """
    chunks = load_chunks(chunks_path)
    try:
        chunk = get_chunk_by_id(chunk_id, chunks)
    except ValueError as e:
        print(e)
        return []

    prev_ctx = get_previous_chunks(chunk, chunks, prev_n=prev_chunks)

    prompt = build_entity_prompt_with_context(chunk, prev_ctx)

    # debug: show short prompt preview in console (we keep this to avoid huge console dumps)
    p_shown = prompt if len(prompt) <= 2000 else prompt[:2000] + "\n\n...[TRUNCATED PROMPT]"
    print(f"\n--- ENTITY EXTRACTION PROMPT for {chunk_id} (prev_ctx={len(prev_ctx)}) ---\n{p_shown}\n{'-'*80}")

    raw = call_openai(prompt, model=model, max_tokens=max_tokens, reasoning_effort="low")
    if not raw:
        print("Empty LLM response.")
        # optionally save debug with empty llm_output
        if save_debug:
            dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, "", [], error="Empty LLM response")
            print(f"Debug file written to: {dbg_path}")
        return []

    txt = raw.strip()
    # unwrap markdown fences if present (be liberal)
    if txt.startswith("```") and txt.endswith("```"):
        # Strip surrounding backticks and optional 'json' language tag
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()

    # console preview of LLM output (short)
    preview = txt if len(txt) <= 4000 else txt[:4000] + "\n\n...[TRUNCATED OUTPUT]"
    print(f"[LLM raw output preview]\n{preview}\n{'-'*80}")

    parsed = []
    error_msg = None
    try:
        parsed = json.loads(txt)
        if not isinstance(parsed, list):
            raise ValueError("Parsed JSON is not a list/array")
        print(f"Parsed JSON array with {len(parsed)} items")
    except Exception as e:
        error_msg = str(e)
        print("Failed to parse JSON from model output:", e)
        print("Model raw output (truncated):", txt[:2000])
        # if debug saving enabled, still save the raw output and error (including focus/context)
        if save_debug:
            dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, txt, [], error=error_msg)
            print(f"Debug file written to: {dbg_path}")
        return []

    results = []
    for e in parsed:
        # robust field extraction with fallbacks
        name = e.get("entity_name") or e.get("name") or e.get("label")
        if not name:
            continue

        # required fields mapping
        entity_description = e.get("entity_description") or e.get("description") or ""
        entity_type_hint = e.get("entity_type_hint") or e.get("type") or "Other"
        context_phrase = e.get("context_phrase") or ""
        resolution_context = e.get("resolution_context") or e.get("used_context_excerpt") or ""
        confidence_raw = e.get("confidence_score") if e.get("confidence_score") is not None else e.get("confidence")
        try:
            confidence_score = float(confidence_raw) if confidence_raw is not None else None
        except Exception:
            confidence_score = None

        node_props_raw = e.get("node_properties") or []
        # normalize node_properties if present
        node_properties = []
        if isinstance(node_props_raw, list):
            for np in node_props_raw:
                if isinstance(np, dict):
                    node_properties.append({
                        "prop_name": np.get("prop_name") or np.get("name"),
                        "prop_value": np.get("prop_value") or np.get("value"),
                        "justification": np.get("justification", ""),
                        # "confidence": float(np.get("confidence")) if np.get("confidence") is not None else None
                    })

        ent = {
            "id": f"En_{uuid.uuid4().hex[:8]}",
            "flag": "entity_raw",
            "chunk_id": chunk_id,
            "ref_index": chunk.get("ref_index"),
            "chunk_index_in_section": chunk.get("chunk_index_in_section"),
            "ref_title": chunk.get("ref_title"),
            "text_span": context_phrase,
            "entity_name": name,
            "entity_description": entity_description,
            "entity_type_hint": entity_type_hint,
            "confidence_score": confidence_score,
            "resolution_context": resolution_context,
            "node_properties": node_properties
        }
        results.append(ent)

    print(f"extract_entities_from_chunk: extracted {len(results)} canonical entity records (will be saved).")

    # SAVE results into canonical NDJSON file (append-safe)
    save_entities(results, out_path=ENTITIES_OUT)

    # If debug saving is requested, write the full prompt + output + parsed entities to file (including focus+context)
    if save_debug:
        dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, txt, parsed, error=None)
        print(f"Debug file written to: {dbg_path}")

    return results

# ---------- Driver example ----------
# chunk_ids = ["Ch_000120"]  # modify as needed
# chunk_ids = [f"Ch_{i:06d}" for i in range(0, 224)]

# ---------- Driver: run on all chunks in the chunks file ----------
# chunks = load_chunks(CHUNKS_JSONL)
# chunk_ids = [c["id"] for c in chunks]

# chunk_ids = [
#     "Ch_000001"]
# ]
# "Ch_000001", "Ch_000002", "Ch_000003", "Ch_000004", "Ch_000005", "Ch_000006",
# "Ch_000007", "Ch_000008", "Ch_000009", "Ch_000010",
# "Ch_000011", "Ch_000012", "Ch_000013", "Ch_000014", "Ch_000015", "Ch_000016",
# "Ch_000017", "Ch_000018", "Ch_000119", "Ch_000020", "Ch_000021"
# # "Ch_000138", "Ch_000139", "Ch_000140", "Ch_000141", "Ch_000142", "Ch_000143",


def run_entity_extraction_on_chunks(
    chunk_ids: List[str] = None,
    prev_chunks: int = 3,
    save_debug: bool = False,
    debug_dir: str = DEFAULT_DEBUG_DIR,
    model: str = "gpt-5.1",
    max_tokens: int = 16000
):
    """
    Convenience driver to run entity extraction over a list of chunk_ids
    using gpt-5.1 + Responses API.
    If chunk_ids is None or empty, derive them from CHUNKS_JSONL.
    """
    # load all chunks into a list (materialize generator if needed)
    chunks = list(load_chunks(CHUNKS_JSONL))

    # if caller didn't pass chunk_ids, derive from chunks
    if not chunk_ids:
        chunk_ids = [c["id"] for c in chunks]

    all_results: List[Dict] = []
    for cid in chunk_ids:
        res = extract_entities_from_chunk(
            cid,
            chunks_path=CHUNKS_JSONL,
            prev_chunks=prev_chunks,
            model=model,
            max_tokens=max_tokens,
            save_debug=save_debug,
            debug_dir=debug_dir
        )
        if res:
            all_results.extend(res)
    return all_results



# # -----------------------
# # Entity Recognition  - Run statement
# # -----------------------

# # Example run:
# if __name__ == "__main__":
#     # set save_debug=True to persist full prompt+llm output (and focus/context text)
#     # to files in DEFAULT_DEBUG_DIR
#     run_entity_extraction_on_chunks(
#         chunk_ids,
#         prev_chunks=5,
#         save_debug=False,
#         model="gpt-5.1",
#         max_tokens=8000
#     )

#endregion#? Entity Recognition v8 - Intrinsic properties added (Responses + gpt-5.1)
#?#########################  End  ##########################





#*######################### Start ##########################
#region:#?   Embedding and clustering recognized entities    -  V3- Forced HDBSCAN - Resolution + Properties Added (type folded into ctx)

"""
embed_and_cluster_entities_force_hdbscan.py

Changes from previous: the entity_type_hint is injected into the `ctx` text
so context = [TYPE:<type>] + resolution_context + serialized node_properties (if present).
We removed the standalone "type" embedding field and only embed: name, desc, ctx.
Everything else left intact.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# --- required clustering lib ---
try:
    import hdbscan
except Exception as e:
    raise RuntimeError("hdbscan is required for this script. Install with `pip install hdbscan`") from e

# optional but recommended
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ------------------ Config / Hyperparams ------------------
# Note: type field removed and its signal is folded into ctx text.
# Remaining weights have been rescaled to sum to 1.
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35 }
EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # change if needed
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Force HDBSCAN params
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"   # we will normalize vectors so euclidean ~ cosine

# UMAP (optional)
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8  # you see over-merged clusters → decrease N_NEIGHBORS
UMAP_MIN_DIST = 0.0

# ------------------ Helpers ------------------
def load_entities(path: str) -> List[Dict]:
    p = Path(path)
    assert p.exists(), f"entities file not found: {p}"
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# ------------------ Embedder ------------------
class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
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
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ------------------ Build fields & combine embeddings ------------------
def build_field_texts(entities: List[Dict]):
    """
    Build text lists for fields:
      - name: entity_name (preferred)
      - desc: entity_description
      - ctx: [TYPE:<entity_type_hint>] + resolution_context (preferred) + serialized node_properties if present
    Node properties are appended into ctx so they contribute to the 'ctx' embedding (no extra weight).
    """
    names, descs, ctxs = [], [], []
    for e in entities:
        # name & desc
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")

        # resolution_context preferred; fall back to older fields
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or safe_text(e, "used_context_excerpt") or ""

        # entity type (folded into ctx as a hint)
        etype = safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or ""

        # serialize node_properties if present and append to resolution context (so it contributes to ctx embedding)
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    pieces.append(f"{pname}:{pval}" if pname and pval else (pname or pval))
            if pieces:
                node_props_text = " | ".join(pieces)

        # build ctx: type hint first (if present), then resolution, then props
        ctx_parts = []
        if etype:
            ctx_parts.append(f"[TYPE:{etype}]")
        if resolution:
            ctx_parts.append(resolution)
        if node_props_text:
            ctx_parts.append(node_props_text)

        combined_ctx = " ; ".join([p for p in ctx_parts if p])
        ctxs.append(combined_ctx)

    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS):
    names, descs, ctxs = build_field_texts(entities)
    D_ref = None
    # encode each field (if empty, make zeros)
    print("[compute] encoding name field ...")
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    D_ref = emb_name.shape[1] if emb_name is not None else None

    print("[compute] encoding desc field ...")
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    if D_ref is None and emb_desc is not None:
        D_ref = emb_desc.shape[1]

    print("[compute] encoding ctx field ...")
    emb_ctx = embedder.encode_batch(ctxs) if any(t.strip() for t in ctxs) else None
    if D_ref is None and emb_ctx is not None:
        D_ref = emb_ctx.shape[1]

    if D_ref is None:
        raise ValueError("All fields empty — cannot embed")

    # helper to make arrays shape (N, D_ref)
    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D_ref))
        if arr.shape[1] != D_ref:
            raise ValueError("embedding dimension mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx  = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0:
        raise ValueError("Sum of weights must be > 0")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

# ------------------ Forced HDBSCAN clustering ------------------
def run_hdbscan(embeddings: np.ndarray,
                min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples: int = HDBSCAN_MIN_SAMPLES,
                metric: str = HDBSCAN_METRIC,
                use_umap: bool = USE_UMAP):
    print(f"[cluster] forcing HDBSCAN min_cluster_size={min_cluster_size} min_samples={min_samples} metric={metric} use_umap={use_umap}")
    X = embeddings
    # if use_umap:
    #     if not UMAP_AVAILABLE:
    #         print("[cluster] WARNING: UMAP not available — running HDBSCAN on original embeddings")
    #     else:
    #         print("[cluster] running UMAP reduction ->", UMAP_N_COMPONENTS, "dims")
    #         reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
    #                             min_dist=UMAP_MIN_DIST, metric='cosine', random_state=42)
    #         X = reducer.fit_transform(X)
    #         print("[cluster] UMAP done, X.shape=", X.shape)

    # inside run_hdbscan (embed_and_cluster_entities_force_hdbscan.py)
    if use_umap:
        if not UMAP_AVAILABLE:
            print("[cluster] WARNING: UMAP not available — running HDBSCAN on original embeddings")
        else:
            n_samples = X.shape[0]
            # ensure n_components < n_samples and at least 2 dims
            n_components = min(UMAP_N_COMPONENTS, max(2, n_samples - 1))
            # ensure n_neighbors < n_samples and at least 2
            n_neighbors = min(UMAP_N_NEIGHBORS, max(2, n_samples - 1))
            print("[cluster] running UMAP reduction ->", n_components, "dims (n_samples=", n_samples, ", n_neighbors=", n_neighbors, ")")
            try:
                reducer = umap.UMAP(n_components=n_components,
                                    n_neighbors=n_neighbors,
                                    min_dist=UMAP_MIN_DIST,
                                    metric='cosine',
                                    random_state=42)
                X = reducer.fit_transform(X)
                print("[cluster] UMAP done, X.shape=", X.shape)
            except Exception as e:
                print(f"[cluster] UMAP reduction failed (n_samples={n_samples}, n_components={n_components}, n_neighbors={n_neighbors}) -> continuing without UMAP. Err: {e}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X)
    probs = getattr(clusterer, "probabilities_", None)
    return labels, probs, clusterer

# def save_entities_with_clusters(entities: List[Dict], labels: np.ndarray, out_jsonl: str, clusters_summary_path: str):
#     outp = Path(out_jsonl)
#     outp.parent.mkdir(parents=True, exist_ok=True)
#     with open(outp, "w", encoding="utf-8") as fh:
#         for e, lab in zip(entities, labels):
#             out = dict(e)
#             out["_cluster_id"] = int(lab)
#             fh.write(json.dumps(out, ensure_ascii=False) + "\n")
#     # summary
#     summary = {}
#     for idx, lab in enumerate(labels):
#         summary.setdefault(int(lab), []).append(entities[idx].get("entity_name") or f"En_{idx}")
#     with open(clusters_summary_path, "w", encoding="utf-8") as fh:
#         json.dump({"n_entities": len(entities), "n_clusters": len(summary), "clusters": {str(k): v for k, v in summary.items()}}, fh, ensure_ascii=False, indent=2)
#     print(f"[save] wrote {out_jsonl} and summary {clusters_summary_path}")

def save_entities_with_clusters(entities: List[Dict],
                                labels: np.ndarray,
                                out_jsonl: str,
                                clusters_summary_path: str,
                                include_fields: List[str] = None,
                                max_field_chars: int = 240):
    """
    Writes:
      - out_jsonl : entity JSONL annotated with _cluster_id (same as before)
      - clusters_summary_path : richer summary where each entity object in a cluster is written
        as a compact single-line JSON (one line per entity object).

    include_fields: list of keys to include in each entity object (defaults below)
    """
    if include_fields is None:
        include_fields = ["entity_name", "entity_type_hint", "entity_description", "resolution_context"]

    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # write JSONL with cluster ids (same as before)
    with open(outp, "w", encoding="utf-8") as fh:
        for e, lab in zip(entities, labels):
            out = dict(e)
            out["_cluster_id"] = int(lab)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Build cluster mapping with compact per-entity JSON strings
    clusters = {}
    for idx, lab in enumerate(labels):
        lab_int = int(lab)
        ent = entities[idx] if idx < len(entities) else {}
        # helper to safely retrieve + trim fields
        def _get_trim(key):
            v = ent.get(key) or ""
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            s = str(v)
            if len(s) > max_field_chars:
                return s[:max_field_chars-3] + "..."
            return s

        obj = {}
        # entity_name (fallback to placeholder)
        obj["entity_name"] = _get_trim("entity_name") or f"En_{idx}"
        if "entity_type_hint" in include_fields:
            obj["entity_type_hint"] = _get_trim("entity_type_hint") or _get_trim("entity_type") or ""
        if "entity_description" in include_fields:
            obj["entity_description"] = _get_trim("entity_description") or _get_trim("description") or ""
        if "resolution_context" in include_fields:
            ctx = ent.get("resolution_context") or ent.get("text_span") or ent.get("context_phrase") or ent.get("used_context_excerpt") or ""
            if isinstance(ctx, (list, dict)):
                ctx = json.dumps(ctx, ensure_ascii=False)
            ctx = str(ctx)
            if len(ctx) > max_field_chars:
                ctx = ctx[:max_field_chars-3] + "..."
            obj["resolution_context"] = ctx

        clusters.setdefault(lab_int, []).append(obj)

    # Now write clusters_summary.json but ensure each entity object is on a single line.
    meta = {"n_entities": len(entities), "n_clusters": len(clusters)}
    # We'll stream-write the JSON to control formatting: pretty top-level, but compact objects
    with open(clusters_summary_path, "w", encoding="utf-8") as fh:
        fh.write("{\n")
        fh.write(f'  "n_entities": {meta["n_entities"]},\n')
        fh.write(f'  "n_clusters": {meta["n_clusters"]},\n')
        fh.write('  "clusters": {\n')

        # iterate clusters in sorted order of cluster id for stability
        cluster_items = sorted(clusters.items(), key=lambda x: x[0])
        for ci, (lab_int, objs) in enumerate(cluster_items):
            fh.write(f'    "{lab_int}": [\n')
            for oi, obj in enumerate(objs):
                # compact JSON for the object (single line)
                obj_json = json.dumps(obj, ensure_ascii=False, separators=(",", ": "))
                # indent two levels inside cluster array
                fh.write(f'      {obj_json}')
                # trailing comma except last object
                if oi < len(objs) - 1:
                    fh.write(",\n")
                else:
                    fh.write("\n")
            # close this cluster array
            fh.write("    ]")
            # trailing comma except last cluster
            if ci < len(cluster_items) - 1:
                fh.write(",\n")
            else:
                fh.write("\n")

        # close clusters and top-level
        fh.write("  }\n")
        fh.write("}\n")

    print(f"[save] wrote {out_jsonl} and summary {clusters_summary_path} (compact entity objects per line)")


# ------------------ Main entry (CLI + notebook-safe) ------------------
def main_cli(args):
    entities = load_entities(args.entities_in)
    print(f"Loaded {len(entities)} entities from {args.entities_in}")

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined.shape)

    # force HDBSCAN (ignore args.use_method)
    labels, probs, clusterer = run_hdbscan(combined,
                                          min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                          min_samples=HDBSCAN_MIN_SAMPLES,
                                          metric=HDBSCAN_METRIC,
                                          use_umap=USE_UMAP if args.use_umap else False)

    # diagnostics
    import numpy as np
    from collections import Counter
    labels_arr = np.array(labels)
    n = len(labels_arr)
    n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    n_noise = int((labels_arr == -1).sum())
    print(f"[diagnostic] clusters (excl -1): {n_clusters}  noise: {n_noise} ({n_noise/n*100:.1f}%)")
    counts = Counter(labels_arr)
    top = sorted(((lab, sz) for lab, sz in counts.items() if lab != -1), key=lambda x: x[1], reverse=True)[:10]
    print("[diagnostic] top cluster sizes:", top)

    save_entities_with_clusters(entities, labels_arr, args.out_jsonl, args.clusters_summary)
    print("Clustering finished.")




#endregion#? Embedding and clustering recognized entities    -  V3- Forced HDBSCAN - Resolution + Properties Added (type folded into ctx)
#*#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Embedding and clustering recognized entities    -  V4


"""
embed_and_cluster_entities_force_hdbscan.py

Changes from previous: the entity_type_hint is injected into the `ctx` text
so context = [TYPE:<type>] + resolution_context + serialized node_properties (if present).
We removed the standalone "type" embedding field and only embed: name, desc, ctx.
Everything else left intact.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# --- required clustering lib ---
try:
    import hdbscan
except Exception as e:
    raise RuntimeError("hdbscan is required for this script. Install with `pip install hdbscan`") from e

# optional but recommended
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ------------------ Config / Hyperparams ------------------
# Note: type field removed and its signal is folded into ctx text.
# Remaining weights have been rescaled to sum to 1.
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35 }
EMBED_MODEL = "BAAI/bge-large-en-v1.5"   # change if needed
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Force HDBSCAN params
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"   # we will normalize vectors so euclidean ~ cosine

# UMAP (optional)
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8  # you see over-merged clusters → decrease N_NEIGHBORS
UMAP_MIN_DIST = 0.0

# ------------------ Helpers ------------------
def load_entities(path: str) -> List[Dict]:
    p = Path(path)
    assert p.exists(), f"entities file not found: {p}"
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# ------------------ Embedder ------------------
class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
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
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ------------------ Build fields & combine embeddings ------------------
def build_field_texts(entities: List[Dict]):
    """
    Build text lists for fields:
      - name: entity_name (preferred)
      - desc: entity_description
      - ctx: [TYPE:<entity_type_hint>] + resolution_context (preferred) + serialized node_properties if present
    Node properties are appended into ctx so they contribute to the 'ctx' embedding (no extra weight).
    """
    names, descs, ctxs = [], [], []
    for e in entities:
        # name & desc
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")

        # resolution_context preferred; fall back to older fields
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or safe_text(e, "used_context_excerpt") or ""

        # entity type (folded into ctx as a hint)
        etype = safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or ""

        # serialize node_properties if present and append to resolution context (so it contributes to ctx embedding)
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    pieces.append(f"{pname}:{pval}" if pname and pval else (pname or pval))
            if pieces:
                node_props_text = " | ".join(pieces)

        # build ctx: type hint first (if present), then resolution, then props
        ctx_parts = []
        if etype:
            ctx_parts.append(f"[TYPE:{etype}]")
        if resolution:
            ctx_parts.append(resolution)
        if node_props_text:
            ctx_parts.append(node_props_text)

        combined_ctx = " ; ".join([p for p in ctx_parts if p])
        ctxs.append(combined_ctx)

    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS):
    names, descs, ctxs = build_field_texts(entities)
    D_ref = None
    # encode each field (if empty, make zeros)
    print("[compute] encoding name field ...")
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    D_ref = emb_name.shape[1] if emb_name is not None else None

    print("[compute] encoding desc field ...")
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    if D_ref is None and emb_desc is not None:
        D_ref = emb_desc.shape[1]

    print("[compute] encoding ctx field ...")
    emb_ctx = embedder.encode_batch(ctxs) if any(t.strip() for t in ctxs) else None
    if D_ref is None and emb_ctx is not None:
        D_ref = emb_ctx.shape[1]

    if D_ref is None:
        raise ValueError("All fields empty — cannot embed")

    # helper to make arrays shape (N, D_ref)
    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D_ref))
        if arr.shape[1] != D_ref:
            raise ValueError("embedding dimension mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx  = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0:
        raise ValueError("Sum of weights must be > 0")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

# ------------------ Forced HDBSCAN clustering ------------------
def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    use_umap: bool = USE_UMAP,
) -> Tuple[np.ndarray, object]:
    """
    Wrapper around HDBSCAN with optional UMAP reduction.

    - Uses UMAP if available and n_samples >= 5.
    - If UMAP fails, falls back to the original embeddings.
    - Returns (labels, clusterer).
    """
    X = embeddings
    n_samples = X.shape[0]

    if use_umap and UMAP_AVAILABLE and n_samples >= 5:
        n_components = min(UMAP_N_COMPONENTS, max(2, n_samples - 2))
        n_neighbors = min(UMAP_N_NEIGHBORS, max(2, n_samples - 2))
        try:
            print(
                f"[cluster] running UMAP reduction -> {n_components} dims "
                f"(n_samples={n_samples}, n_neighbors={n_neighbors})"
            )
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=UMAP_MIN_DIST,
                metric="cosine",
                random_state=42,
            )
            X = reducer.fit_transform(X)
        except Exception as e:
            print(
                f"[cluster] UMAP reduction failed (n_samples={n_samples}, "
                f"n_components={n_components}, n_neighbors={n_neighbors}) "
                f"-> continuing without UMAP. Err: {e}"
            )
            X = embeddings  # fall back

    print(
        f"[cluster] forcing HDBSCAN min_cluster_size={min_cluster_size} "
        f"min_samples={min_samples} metric={metric} use_umap={use_umap}"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer


# def save_entities_with_clusters(entities: List[Dict], labels: np.ndarray, out_jsonl: str, clusters_summary_path: str):
#     outp = Path(out_jsonl)
#     outp.parent.mkdir(parents=True, exist_ok=True)
#     with open(outp, "w", encoding="utf-8") as fh:
#         for e, lab in zip(entities, labels):
#             out = dict(e)
#             out["_cluster_id"] = int(lab)
#             fh.write(json.dumps(out, ensure_ascii=False) + "\n")
#     # summary
#     summary = {}
#     for idx, lab in enumerate(labels):
#         summary.setdefault(int(lab), []).append(entities[idx].get("entity_name") or f"En_{idx}")
#     with open(clusters_summary_path, "w", encoding="utf-8") as fh:
#         json.dump({"n_entities": len(entities), "n_clusters": len(summary), "clusters": {str(k): v for k, v in summary.items()}}, fh, ensure_ascii=False, indent=2)
#     print(f"[save] wrote {out_jsonl} and summary {clusters_summary_path}")

def save_entities_with_clusters(entities: List[Dict],
                                labels: np.ndarray,
                                out_jsonl: str,
                                clusters_summary_path: str,
                                include_fields: List[str] = None,
                                max_field_chars: int = 240):
    """
    Writes:
      - out_jsonl : entity JSONL annotated with _cluster_id (same as before)
      - clusters_summary_path : richer summary where each entity object in a cluster is written
        as a compact single-line JSON (one line per entity object).

    include_fields: list of keys to include in each entity object (defaults below)
    """
    if include_fields is None:
        include_fields = ["entity_name", "entity_type_hint", "entity_description", "resolution_context"]

    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # write JSONL with cluster ids (same as before)
    with open(outp, "w", encoding="utf-8") as fh:
        for e, lab in zip(entities, labels):
            out = dict(e)
            out["_cluster_id"] = int(lab)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Build cluster mapping with compact per-entity JSON strings
    clusters = {}
    for idx, lab in enumerate(labels):
        lab_int = int(lab)
        ent = entities[idx] if idx < len(entities) else {}
        # helper to safely retrieve + trim fields
        def _get_trim(key):
            v = ent.get(key) or ""
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            s = str(v)
            if len(s) > max_field_chars:
                return s[:max_field_chars-3] + "..."
            return s

        obj = {}
        # entity_name (fallback to placeholder)
        obj["entity_name"] = _get_trim("entity_name") or f"En_{idx}"
        if "entity_type_hint" in include_fields:
            obj["entity_type_hint"] = _get_trim("entity_type_hint") or _get_trim("entity_type") or ""
        if "entity_description" in include_fields:
            obj["entity_description"] = _get_trim("entity_description") or _get_trim("description") or ""
        if "resolution_context" in include_fields:
            ctx = ent.get("resolution_context") or ent.get("text_span") or ent.get("context_phrase") or ent.get("used_context_excerpt") or ""
            if isinstance(ctx, (list, dict)):
                ctx = json.dumps(ctx, ensure_ascii=False)
            ctx = str(ctx)
            if len(ctx) > max_field_chars:
                ctx = ctx[:max_field_chars-3] + "..."
            obj["resolution_context"] = ctx

        clusters.setdefault(lab_int, []).append(obj)

    # Now write clusters_summary.json but ensure each entity object is on a single line.
    meta = {"n_entities": len(entities), "n_clusters": len(clusters)}
    # We'll stream-write the JSON to control formatting: pretty top-level, but compact objects
    with open(clusters_summary_path, "w", encoding="utf-8") as fh:
        fh.write("{\n")
        fh.write(f'  "n_entities": {meta["n_entities"]},\n')
        fh.write(f'  "n_clusters": {meta["n_clusters"]},\n')
        fh.write('  "clusters": {\n')

        # iterate clusters in sorted order of cluster id for stability
        cluster_items = sorted(clusters.items(), key=lambda x: x[0])
        for ci, (lab_int, objs) in enumerate(cluster_items):
            fh.write(f'    "{lab_int}": [\n')
            for oi, obj in enumerate(objs):
                # compact JSON for the object (single line)
                obj_json = json.dumps(obj, ensure_ascii=False, separators=(",", ": "))
                # indent two levels inside cluster array
                fh.write(f'      {obj_json}')
                # trailing comma except last object
                if oi < len(objs) - 1:
                    fh.write(",\n")
                else:
                    fh.write("\n")
            # close this cluster array
            fh.write("    ]")
            # trailing comma except last cluster
            if ci < len(cluster_items) - 1:
                fh.write(",\n")
            else:
                fh.write("\n")

        # close clusters and top-level
        fh.write("  }\n")
        fh.write("}\n")

    print(f"[save] wrote {out_jsonl} and summary {clusters_summary_path} (compact entity objects per line)")


# ------------------ Main entry (CLI + notebook-safe) ------------------
def main_cli(args):
    entities = load_entities(args.entities_in)
    print(f"Loaded {len(entities)} entities from {args.entities_in}")

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined.shape)

    # force HDBSCAN (ignore args.use_method)
    labels, probs, clusterer = run_hdbscan(combined,
                                          min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                          min_samples=HDBSCAN_MIN_SAMPLES,
                                          metric=HDBSCAN_METRIC,
                                          use_umap=USE_UMAP if args.use_umap else False)

    # diagnostics
    import numpy as np
    from collections import Counter
    labels_arr = np.array(labels)
    n = len(labels_arr)
    n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    n_noise = int((labels_arr == -1).sum())
    print(f"[diagnostic] clusters (excl -1): {n_clusters}  noise: {n_noise} ({n_noise/n*100:.1f}%)")
    counts = Counter(labels_arr)
    top = sorted(((lab, sz) for lab, sz in counts.items() if lab != -1), key=lambda x: x[1], reverse=True)[:10]
    print("[diagnostic] top cluster sizes:", top)

    save_entities_with_clusters(entities, labels_arr, args.out_jsonl, args.clusters_summary)
    print("Clustering finished.")



#endregion#? Embedding and clustering recognized entities    -  V4
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   Final Ent Res - (aligned with EntityRec v7 & gpt-5.1 Responses, embedding pipeline With SubCluster json)

# orchestrator_with_chunk_texts_v101_gpt5.py
"""
Entity resolution orchestrator aligned with Entity Recognition v7 and the updated embedding
pipeline (name/desc/ctx). Performs local sub-clustering, chunk-text inclusion, token safety guard,
and tqdm progress bars.

Changes (Dec 2025):
 - stricter input validation (requires _cluster_id + context fields)
 - saves local-subcluster summaries per coarse cluster for debugging
 - fallback to "no local sub-clustering" when fragmentation is excessive (keeps members together)
 - more robust min_cluster_size handling and explanatory logging

Updated (Jan 2026):
 - switch LLM backend from Chat Completions (gpt-4o) to Responses API with gpt-5.1
 - use `max_output_tokens` instead of `max_tokens`
 - no `temperature` parameter (not supported by gpt-5.1)
 - optional reasoning config for gpt-5.x models
"""
import os
import json
import uuid
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np
from tqdm import tqdm

# transformers embedder
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan is required. Install with `pip install hdbscan`")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# OpenAI client loader (reuses your pattern)
from openai import OpenAI

def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = ".env") -> str:
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️  OPENAI API key not found or seems invalid. Set OPENAI_API_KEY env or place key in fallback file path.")
client = OpenAI(api_key=OPENAI_KEY)

# ---------------- Paths & config ----------------
CLUSTERED_IN = Path("SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")   # input (from previous clustering)
CHUNKS_JSONL = Path("SGCE-KG/data/Chunks/chunks_sentence.jsonl")

ENT_OUT = Path("SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")
CANON_OUT = Path("SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
LOG_OUT = Path("SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/resolution_log.jsonl")

# NOTE: weights changed to match embed_and_cluster V3 (name, desc, ctx) — type is folded into ctx.
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35}
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# local HDBSCAN
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1
LOCAL_HDBSCAN_METRIC = "euclidean"

# UMAP options
LOCAL_USE_UMAP = False   # default OFF for robustness; enable via CLI --use_umap
UMAP_DIMS = 32
UMAP_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0
UMAP_MIN_SAMPLES_TO_RUN = 25  # only run UMAP when cluster size >= this

# LLM / prompt (now via Responses API + gpt-5.1)
MODEL = "gpt-5.1"
MAX_TOKENS = 16000               # mapped to max_output_tokens
REASONING_EFFORT = "low"       # for gpt-5.x models

# orchestration thresholds (as requested)
MAX_CLUSTER_PROMPT = 11       # coarse cluster size threshold to trigger local sub-clustering
MAX_MEMBERS_PER_PROMPT = 10    # <= 10 entities per LLM call
TRUNC_CHUNK_CHARS = 1000
INCLUDE_PREV_CHUNKS = 0

# token safety
PROMPT_TOKEN_LIMIT = 8000  # rough char/4 estimate threshold

# fallback fragmentation rule (tunable)
# if local sub-clustering produces more than len(members)/FALLBACK_FRAG_THRESHOLD_FACTOR non-noise subclusters,
# we will FALLBACK to _not_ using local_subcluster and instead chunk the original coarse cluster directly.
FALLBACK_FRAG_THRESHOLD_FACTOR = 2   # e.g., if > len(members)/2 non-noise subclusters -> fallback

# ---------------- Utility functions ----------------
def load_chunks(chunks_jsonl_path: Path) -> List[Dict]:
    assert chunks_jsonl_path.exists(), f"Chunks file not found: {chunks_jsonl_path}"
    chunks = []
    with open(chunks_jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def safe_text(e: Dict, key: str) -> str:
    v = e.get(key)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

# ---------------- HF embedder ----------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE):
        print(f"[Embedder] loading model {model_name} on {device} ...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        # return shape (N, D) where D = model hidden size
        if len(texts) == 0:
            # fallback: use model.config.hidden_size if available
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------- field builder aligned with EntityRec v7 ----------------
def build_field_texts(entities: List[Dict]):
    """
    Build text lists for fields:
      - name: entity_name (preferred)
      - desc: entity_description
      - ctx: [TYPE:<entity_type_hint>] + resolution_context (preferred) + serialized node_properties (if present)
    This matches the embedding pipeline in embed_and_cluster V3.
    """
    names, descs, ctxs = [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or safe_text(e, "entity_name_original") or "")
        descs.append(safe_text(e, "entity_description") or "")

        # resolution_context preferred (new schema) — fallback chain below
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or safe_text(e, "used_context_excerpt") or ""

        # fold type and node_properties into ctx (as text hints)
        etype = safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or ""
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    if pname and pval:
                        pieces.append(f"{pname}:{pval}")
                    elif pname:
                        pieces.append(pname)
                    elif pval:
                        pieces.append(str(pval))
            if pieces:
                node_props_text = " | ".join(pieces)

        ctx_parts = []
        if etype:
            ctx_parts.append(f"[TYPE:{etype}]")
        if resolution:
            ctx_parts.append(resolution)
        if node_props_text:
            ctx_parts.append(node_props_text)

        combined_ctx = " ; ".join([p for p in ctx_parts if p])
        ctxs.append(combined_ctx)
    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS) -> np.ndarray:
    """
    Compute combined normalized embeddings from name, desc, ctx only.
    """
    names, descs, ctxs = build_field_texts(entities)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx  = embedder.encode_batch(ctxs)  if any(t.strip() for t in ctxs) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]
            break
    if D is None:
        raise ValueError("No textual field produced embeddings; check your entity fields")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx  = _ensure(emb_ctx)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx  = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0:
        raise ValueError("invalid weights")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

# ---------------- robust local_subcluster ----------------
def local_subcluster(cluster_entities: List[Dict],
                     entity_id_to_index: Dict[str, int],
                     all_embeddings: np.ndarray,
                     min_cluster_size: int = LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                     min_samples: int = LOCAL_HDBSCAN_MIN_SAMPLES,
                     use_umap: bool = LOCAL_USE_UMAP,
                     umap_dims: int = UMAP_DIMS):
    """
    Returns: dict[label] -> list[entity_dict]
    Notes:
      - min_cluster_size is treated as a true minimum (HDBSCAN may still mark small noise).
      - We protect against trivial inputs and return fallback single cluster if hdbscan fails.
    """
    from collections import defaultdict
    from sklearn.preprocessing import normalize as _normalize

    idxs = [entity_id_to_index[e["id"]] for e in cluster_entities]
    X = all_embeddings[idxs]
    X = _normalize(X, axis=1)
    n = X.shape[0]

    if n <= 1:
        return {0: list(cluster_entities)} if n == 1 else {-1: []}

    # ensure min_cluster_size is sensible (it is a minimum)
    min_cluster_size = max(2, int(min_cluster_size))
    min_cluster_size = min(min_cluster_size, max(2, n))  # cannot be larger than cluster size

    if min_samples is None:
        min_samples = max(1, int(min_cluster_size * 0.1))
    else:
        min_samples = max(1, int(min_samples))

    X_sub = X
    if use_umap and UMAP_AVAILABLE and n >= UMAP_MIN_SAMPLES_TO_RUN:
        n_components = min(umap_dims, max(2, n - 4))  # keep k <= n-4 for stability
        try:
            reducer = umap.UMAP(n_components=n_components,
                                n_neighbors=min(UMAP_NEIGHBORS, max(2, n-1)),
                                min_dist=UMAP_MIN_DIST,
                                metric='cosine',
                                random_state=42)
            X_sub = reducer.fit_transform(X)
        except Exception as e:
            print(f"[local_subcluster] UMAP failed for n={n}, n_components={n_components} -> fallback without UMAP. Err: {e}")
            X_sub = X
    else:
        if use_umap and UMAP_AVAILABLE and n < UMAP_MIN_SAMPLES_TO_RUN:
            print(f"[local_subcluster] skipping UMAP for n={n} (threshold {UMAP_MIN_SAMPLES_TO_RUN})")

    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric=LOCAL_HDBSCAN_METRIC,
                                    cluster_selection_method='eom')
        labels = clusterer.fit_predict(X_sub)
    except Exception as e:
        print(f"[local_subcluster] HDBSCAN failed for n={n} -> fallback single cluster. Err: {e}")
        return {0: list(cluster_entities)}

    groups = defaultdict(list)
    for ent, lab in zip(cluster_entities, labels):
        groups[int(lab)].append(ent)
    return groups

# ------------------ LLM helpers (Responses API + gpt-5.1) ------------------
def call_llm_with_prompt(
    prompt: str,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    reasoning_effort: str = REASONING_EFFORT
) -> str:
    """
    Call the LLM using the Responses API (compatible with gpt-5.1).

    - Uses `max_output_tokens` instead of `max_tokens`.
    - Does NOT send `temperature` (unsupported by gpt-5.1).
    - Adds `reasoning={'effort': reasoning_effort}` for gpt-5.x models.
    """
    try:
        print(f"[call_llm_with_prompt] model={model} max_output_tokens={max_tokens} reasoning_effort={reasoning_effort}")
        kwargs: Dict[str, object] = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        if model.startswith("gpt-5") and reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        response = client.responses.create(**kwargs)
        txt = response.output_text or ""

        status = getattr(response, "status", None)
        if status == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            print(f"[call_llm_with_prompt] WARNING: response incomplete: {incomplete_details}")

        print(f"[call_llm_with_prompt] received response (len={len(txt)} chars, status={status})")
        return txt
    except Exception as e:
        print("LLM call error:", e)
        return ""

def extract_json_array(text: str):
    if not text:
        return None
    text = text.strip()
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

# ---------------- Prompt building ----------------
PROMPT_TEMPLATE = """You are a careful knowledge-graph resolver.
Given the following small cohesive group of candidate entity mentions, decide which ones to MERGE into a single canonical entity, which to MODIFY, and which to KEEP.

Return ONLY a JSON ARRAY. Each element must be one of:
- MergeEntities: {{ "action":"MergeEntities", "entity_ids":[...], "canonical_name":"...", "canonical_description":"...", "canonical_type":"...", "rationale":"..." }}
- ModifyEntity: {{ "action":"ModifyEntity", "entity_id":"...", "new_name":"...", "new_description":"...", "new_type_hint":"...", "rationale":"..." }}
- KeepEntity: {{ "action":"KeepEntity", "entity_id":"...", "rationale":"..." }}

Rules:
- Use ONLY the provided information (name/desc/type_hint/confidence/resolution_context/text_span/chunk_text).
- Be conservative: if unsure, KEEP rather than MERGE.
- If merging, ensure merged items truly refer to the same concept.
- Provide short rationale for each action (1-2 sentences).

Group members (id | name | type_hint | confidence | desc | text_span | chunk_text [truncated]):
{members_json}

Return JSON array only (no commentary).
"""

def build_member_with_chunk(m: Dict, chunks_index: Dict[str, Dict]) -> Dict:
    """
    Build the member record included in the LLM prompt.
    Uses resolution_context (preferred) as part of text_span fallback and includes truncated chunk_text if available.
    """
    chunk_text = ""
    chunk_id = m.get("chunk_id")
    if chunk_id:
        ch = chunks_index.get(chunk_id)
        if ch:
            ct = ch.get("text","")
            # optionally include previous chunks? currently disabled (INCLUDE_PREV_CHUNKS)
            chunk_text = " ".join(ct.split())
            if len(chunk_text) > TRUNC_CHUNK_CHARS:
                chunk_text = chunk_text[:TRUNC_CHUNK_CHARS].rsplit(" ",1)[0] + "..."
    # text_span should show the precise mention proof; prefer context_phrase then resolution_context
    text_span = m.get("context_phrase") or m.get("resolution_context") or m.get("text_span") or ""
    return {
        "id": m.get("id"),
        "name": m.get("entity_name"),
        "type_hint": m.get("entity_type_hint"),
        "confidence": m.get("confidence_score"),
        "desc": m.get("entity_description"),
        "text_span": text_span,
        "chunk_text": chunk_text
    }

# ------------------ apply actions ----------------
def apply_actions(members: List[Dict], actions: List[Dict], entities_by_id: Dict[str, Dict],
                  canonical_store: List[Dict], log_entries: List[Dict]):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for act in (actions or []):
        typ = act.get("action")
        if typ == "MergeEntities":
            ids = act.get("entity_ids", [])
            canonical_name = act.get("canonical_name")
            canonical_desc = act.get("canonical_description", "")
            canonical_type = act.get("canonical_type", "")
            rationale = act.get("rationale", "")
            can_id = "Can_" + uuid.uuid4().hex[:8]
            canonical = {
                "canonical_id": can_id,
                "canonical_name": canonical_name,
                "canonical_description": canonical_desc,
                "canonical_type": canonical_type,
                "source": "LLM_resolution_v101_gpt5",
                "rationale": rationale,
                "timestamp": ts
            }
            canonical_store.append(canonical)
            for eid in ids:
                ent = entities_by_id.get(eid)
                if ent:
                    ent["canonical_id"] = can_id
                    ent["resolved_action"] = "merged"
                    ent["resolution_rationale"] = rationale
                    ent["resolved_time"] = ts
            log_entries.append({"time": ts, "action": "merge", "canonical_id": can_id, "merged_ids": ids, "rationale": rationale})
        elif typ == "ModifyEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            if ent:
                new_name = act.get("new_name")
                new_desc = act.get("new_description")
                new_type = act.get("new_type_hint")
                rationale = act.get("rationale","")
                if new_name:
                    ent["entity_name"] = new_name
                if new_desc:
                    ent["entity_description"] = new_desc
                if new_type:
                    ent["entity_type_hint"] = new_type
                ent["resolved_action"] = "modified"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append({"time": ts, "action": "modify", "entity_id": eid, "rationale": rationale})
        elif typ == "KeepEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            rationale = act.get("rationale","")
            if ent:
                ent["resolved_action"] = "kept"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append({"time": ts, "action": "keep", "entity_id": eid, "rationale": rationale})
        else:
            log_entries.append({"time": ts, "action": "unknown", "payload": act})

# ------------------ Orchestration main ----------------
def validate_entities_schema(entities: List[Dict]):
    """
    Ensure every entity has the minimum required fields expected by the new pipeline.
    Raise ValueError with a helpful message listing offending entries if not.
    """
    required = ["id", "entity_name", "_cluster_id"]  # require cluster id for grouping
    problems = []
    for i, e in enumerate(entities):
        missing = [k for k in required if k not in e]
        # validate that at least one context field exists (resolution_context | context_phrase | text_span | used_context_excerpt)
        context_present = any(k in e and e.get(k) not in (None, "") for k in ("resolution_context","context_phrase","text_span","used_context_excerpt"))
        if missing or not context_present:
            problems.append({"index": i, "id": e.get("id"), "missing_keys": missing, "has_context": context_present, "sample": {k: e.get(k) for k in ["entity_name","entity_type_hint","resolution_context","context_phrase","text_span","_cluster_id"]}})
        # ensure confidence_score exists (may be None but field should be present to avoid KeyError later)
        if "confidence_score" not in e:
            e["confidence_score"] = None

    if problems:
        msg_lines = ["Entities schema validation failed — some entries are missing required keys or have no context field:"]
        for p in problems[:20]:
            msg_lines.append(f" - idx={p['index']} id={p['id']} missing={p['missing_keys']} has_context={p['has_context']} sample={p['sample']}")
        if len(problems) > 20:
            msg_lines.append(f" - ... and {len(problems)-20} more problematic entries")
        raise ValueError("\n".join(msg_lines))

def write_local_subcluster_summary(out_dir: Path, cid: int, subgroups: Dict[int, List[Dict]]):
    """
    Save a compact summary JSON of local subclusters for debugging.
    Format:
      { "cluster_id": cid, "n_entities": N, "n_subclusters": K, "clusters": {"0": [ {id,name,type_hint,desc}, ...], "-1": [...] } }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"cluster_id": cid, "n_entities": sum(len(v) for v in subgroups.values()), "n_subclusters": len(subgroups)}
    clusters = {}
    for lab, members in sorted(subgroups.items(), key=lambda x: x[0]):
        clusters[str(lab)] = [{"id": m.get("id"), "entity_name": m.get("entity_name"), "entity_type_hint": m.get("entity_type_hint"), "entity_description": (m.get("entity_description") or "")[:200]} for m in members]
    summary["clusters"] = clusters
    path = out_dir / f"cluster_{cid}_subclusters.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return path

def orchestrate():
    print("Loading clustered entities from:", CLUSTERED_IN)
    entities = []
    with open(CLUSTERED_IN, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entities.append(json.loads(line))
    n_entities = len(entities)
    print("Loaded entities:", n_entities)

    # validate schema early and loudly
    validate_entities_schema(entities)

    print("Loading chunks from:", CHUNKS_JSONL)
    chunks = load_chunks(CHUNKS_JSONL) if CHUNKS_JSONL.exists() else []
    chunks_index = {c.get("id"): c for c in chunks}
    print("Loaded chunks:", len(chunks))

    # ensure ids are unique
    ids = [e.get("id") for e in entities]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate entity ids found in clustered input — ids must be unique.")

    entities_by_id = {e["id"]: e for e in entities}
    entity_id_to_index = {e["id"]: i for i, e in enumerate(entities)}

    # embedder + combined embeddings (name/desc/ctx)
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_embeddings = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("Combined embeddings shape:", combined_embeddings.shape)

    # group by coarse cluster
    by_cluster = defaultdict(list)
    for e in entities:
        by_cluster[e.get("_cluster_id")].append(e)

    canonical_store = []
    log_entries = []

    cluster_ids = sorted([k for k in by_cluster.keys() if k != -1])
    noise_count = len(by_cluster.get(-1, []))
    print("Clusters to resolve (excluding noise):", len(cluster_ids), "noise_count:", noise_count)

    local_subclusters_dir = ENT_OUT.parent / "local_subclusters"
    local_subclusters_dir.mkdir(parents=True, exist_ok=True)

    # outer progress bar over clusters
    with tqdm(cluster_ids, desc="Clusters", unit="cluster") as pbar_clusters:
        for cid in pbar_clusters:
            members = by_cluster[cid]
            size = len(members)
            pbar_clusters.set_postfix(cluster=cid, size=size)

            # decide path: direct prompt chunks OR local sub-cluster
            if size <= MAX_CLUSTER_PROMPT:
                # number of prompts for this coarse cluster
                n_prompts = math.ceil(size / MAX_MEMBERS_PER_PROMPT)
                with tqdm(range(n_prompts), desc=f"Cluster {cid} prompts", leave=False, unit="prompt") as pbar_prompts:
                    for i in pbar_prompts:
                        s = i * MAX_MEMBERS_PER_PROMPT
                        chunk = members[s:s+MAX_MEMBERS_PER_PROMPT]
                        payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                        members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                        prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                        est_tokens = max(1, int(len(prompt) / 4))
                        pbar_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                        if est_tokens > PROMPT_TOKEN_LIMIT:
                            # skip LLM call, conservative
                            for m in chunk:
                                log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                    "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                    "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                m["resolved_action"] = "kept_skipped_prompt"
                                m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                            continue
                        llm_out = call_llm_with_prompt(prompt)
                        actions = extract_json_array(llm_out)
                        if actions is None:
                            actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                        apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)
            else:
                # large cluster -> local sub-cluster
                subgroups = local_subcluster(members, entity_id_to_index, combined_embeddings,
                                            min_cluster_size=LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                                            min_samples=LOCAL_HDBSCAN_MIN_SAMPLES,
                                            use_umap=LOCAL_USE_UMAP, umap_dims=UMAP_DIMS)

                # ALWAYS persist subcluster summary for inspection (even if we fallback)
                subcluster_summary_path = write_local_subcluster_summary(local_subclusters_dir, cid, subgroups)
                print(f"[info] wrote local subcluster summary: {subcluster_summary_path}")

                # compute fragmentation statistic: number of non-noise subclusters
                nonnoise_clusters = [lab for lab in subgroups.keys() if lab != -1]
                num_nonnoise = len(nonnoise_clusters)
                # fallback condition: too fragmented -> prefer sequential chunking of original coarse cluster
                fallback_threshold = max(1, int(len(members) / FALLBACK_FRAG_THRESHOLD_FACTOR))
                fallback = num_nonnoise > fallback_threshold

                if fallback:
                    # FALLBACK: do not use subgroups; process original members sequentially (keeps original local ordering)
                    print(f"[fallback] cluster {cid} had {num_nonnoise} non-noise subclusters for {len(members)} members; falling back to sequential chunking (preserves grouping).")
                    log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "action":"fallback_no_subcluster", "cluster": cid, "n_members": len(members), "n_nonnoise_subclusters": num_nonnoise})
                    # chunk original members iteratively
                    n_prompts = math.ceil(len(members) / MAX_MEMBERS_PER_PROMPT)
                    with tqdm(range(n_prompts), desc=f"Cluster {cid} fallback prompts", leave=False, unit="prompt") as pbar_prompts:
                        for i in pbar_prompts:
                            s = i * MAX_MEMBERS_PER_PROMPT
                            chunk = members[s:s+MAX_MEMBERS_PER_PROMPT]
                            payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                            members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                            prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                            est_tokens = max(1, int(len(prompt) / 4))
                            pbar_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                            if est_tokens > PROMPT_TOKEN_LIMIT:
                                for m in chunk:
                                    log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                        "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                        "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                    m["resolved_action"] = "kept_skipped_prompt"
                                    m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                                continue
                            llm_out = call_llm_with_prompt(prompt)
                            actions = extract_json_array(llm_out)
                            if actions is None:
                                actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                            apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)
                else:
                    # proceed with subgroups as planned
                    sub_items = sorted(subgroups.items(), key=lambda x: -len(x[1]))
                    with tqdm(sub_items, desc=f"Cluster {cid} subclusters", leave=False, unit="sub") as pbar_subs:
                        for sublab, submembers in pbar_subs:
                            subsize = len(submembers)
                            pbar_subs.set_postfix(sublab=sublab, subsize=subsize)
                            if sublab == -1:
                                for m in submembers:
                                    m["resolved_action"] = "kept_noise_local"
                                    m["resolution_rationale"] = "Local-subcluster noise preserved"
                                    log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                        "action":"keep_noise_local", "entity_id": m["id"], "cluster": cid})
                                continue
                            # prompts for this subcluster
                            n_prompts = math.ceil(subsize / MAX_MEMBERS_PER_PROMPT)
                            with tqdm(range(n_prompts), desc=f"Sub {sublab} prompts", leave=False, unit="prompt") as pbar_sub_prompts:
                                for i in pbar_sub_prompts:
                                    s = i * MAX_MEMBERS_PER_PROMPT
                                    chunk = submembers[s:s+MAX_MEMBERS_PER_PROMPT]
                                    payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                                    members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                                    prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                                    est_tokens = max(1, int(len(prompt) / 4))
                                    pbar_sub_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                                    if est_tokens > PROMPT_TOKEN_LIMIT:
                                        for m in chunk:
                                            log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                                "action":"skip_large_prompt_keep", "entity_id": m["id"],
                                                                "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}"})
                                            m["resolved_action"] = "kept_skipped_prompt"
                                            m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                                        continue
                                    llm_out = call_llm_with_prompt(prompt)
                                    actions = extract_json_array(llm_out)
                                    if actions is None:
                                        actions = [{"action":"KeepEntity","entity_id": m["id"], "rationale":"LLM parse failed; conservatively kept"} for m in chunk]
                                    apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)

    # global noise handling
    for nent in by_cluster.get(-1, []):
        ent = entities_by_id[nent["id"]]
        ent["resolved_action"] = "kept_noise_global"
        ent["resolution_rationale"] = "Global noise preserved for manual review"
        log_entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "action":"keep_noise_global", "entity_id": ent["id"]})

    # write outputs
    final_entities = list(entities_by_id.values())
    ENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ENT_OUT, "w", encoding="utf-8") as fh:
        for e in final_entities:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(CANON_OUT, "w", encoding="utf-8") as fh:
        for c in canonical_store:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(LOG_OUT, "a", encoding="utf-8") as fh:
        for lg in log_entries:
            fh.write(json.dumps(lg, ensure_ascii=False) + "\n")

    print("\nResolution finished. Wrote:", ENT_OUT, CANON_OUT, LOG_OUT)
    print(f"[info] local subcluster summaries (if any) are under: {local_subclusters_dir}")




#endregion#? Final Ent Res - (aligned with EntityRec v7 & gpt-5.1 Responses, embedding pipeline With SubCluster json)
#?#########################  End  ##########################





#*######################### Start ##########################
#region:#?   Analyze_entity_resolution


# #!/usr/bin/env python3
# """
# analyze_entity_resolution.py

# Creates entResAnalysis/ with:
#  - merged_groups.json     : mapping canonical_id -> list of member entities (full objects)
#  - merged_groups.csv      : flat CSV: canonical_id, member_id, member_name, desc, type, confidence, _cluster_id, resolved_action
#  - canonical_summary.csv  : canonical_id, canonical_name, canonical_type, n_members, example_members
#  - actions_summary.json   : counts per resolved_action
#  - type_distribution.csv  : counts per entity_type_hint (for merged vs kept)
#  - charts: merges_hist.png, actions_pie.png

# Usage:
#   python analyze_entity_resolution.py
# """

# import json
# from pathlib import Path
# from collections import defaultdict, Counter
# import csv
# import pandas as pd
# import matplotlib.pyplot as plt

# # --------- Config / paths ----------
# ENT_RES_FILE = Path("SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")
# CANON_FILE = Path("SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
# OUT_DIR = Path("SGCE-KG/data/Entities/Ent_1st/Ent_1st_Analysis")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # --------- Helpers ----------
# def load_jsonl(path):
#     data = []
#     with open(path, "r", encoding="utf-8") as fh:
#         for ln in fh:
#             ln = ln.strip()
#             if not ln:
#                 continue
#             try:
#                 data.append(json.loads(ln))
#             except Exception as e:
#                 print("skip line (json error):", e)
#     return data

# # --------- Load data ----------
# print("Loading files...")
# entities = load_jsonl(ENT_RES_FILE) if ENT_RES_FILE.exists() else []
# canons = load_jsonl(CANON_FILE) if CANON_FILE.exists() else []

# print(f"Loaded {len(entities)} entities, {len(canons)} canonical records")

# # index canonical metadata if present
# canon_meta = {c.get("canonical_id"): c for c in canons}

# # --------- Build merged groups ----------
# merged = defaultdict(list)       # canonical_id -> list of entity dicts
# unmerged = []                    # entities without canonical_id
# for e in entities:
#     cid = e.get("canonical_id")
#     if cid:
#         merged[cid].append(e)
#     else:
#         unmerged.append(e)

# # Save merged_groups.json
# with open(OUT_DIR / "merged_groups.json", "w", encoding="utf-8") as fh:
#     json.dump({k: v for k,v in merged.items()}, fh, ensure_ascii=False, indent=2)

# # Save merged_groups.csv (flat)
# csv_fields = ["canonical_id", "canonical_name", "member_id", "member_name", "member_desc",
#               "member_type", "confidence_score", "_cluster_id", "resolved_action", "resolution_rationale"]
# with open(OUT_DIR / "merged_groups.csv", "w", newline='', encoding="utf-8") as fh:
#     writer = csv.DictWriter(fh, fieldnames=csv_fields)
#     writer.writeheader()
#     for cid, members in merged.items():
#         canon_name = canon_meta.get(cid, {}).get("canonical_name", "")
#         for m in members:
#             writer.writerow({
#                 "canonical_id": cid,
#                 "canonical_name": canon_name,
#                 "member_id": m.get("id"),
#                 "member_name": m.get("entity_name"),
#                 "member_desc": m.get("entity_description"),
#                 "member_type": m.get("entity_type_hint"),
#                 "confidence_score": m.get("confidence_score"),
#                 "_cluster_id": m.get("_cluster_id"),
#                 "resolved_action": m.get("resolved_action"),
#                 "resolution_rationale": m.get("resolution_rationale","")
#             })

# # Save canonical_summary.csv
# canon_rows = []
# for cid, members in merged.items():
#     row = {
#         "canonical_id": cid,
#         "canonical_name": canon_meta.get(cid, {}).get("canonical_name", ""),
#         "canonical_type": canon_meta.get(cid, {}).get("canonical_type", ""),
#         "n_members": len(members),
#         "example_members": " | ".join([m.get("entity_name","") for m in members[:5]])
#     }
#     canon_rows.append(row)
# canon_df = pd.DataFrame(canon_rows).sort_values("n_members", ascending=False)
# canon_df.to_csv(OUT_DIR / "canonical_summary.csv", index=False)

# # Save actions summary
# action_counts = Counter([e.get("resolved_action","<none>") for e in entities])
# with open(OUT_DIR / "actions_summary.json", "w", encoding="utf-8") as fh:
#     json.dump(action_counts, fh, ensure_ascii=False, indent=2)

# # Save type distribution (merged vs unmerged)
# def type_counter(list_of_entities):
#     c = Counter()
#     for e in list_of_entities:
#         t = e.get("entity_type_hint") or "<unknown>"
#         c[t] += 1
#     return c

# merged_types = type_counter([m for members in merged.values() for m in members])
# unmerged_types = type_counter(unmerged)

# # write as CSV table
# all_types = sorted(set(list(merged_types.keys()) + list(unmerged_types.keys())))
# with open(OUT_DIR / "type_distribution.csv", "w", newline='', encoding="utf-8") as fh:
#     w = csv.writer(fh)
#     w.writerow(["type", "merged_count", "unmerged_count"])
#     for t in all_types:
#         w.writerow([t, merged_types.get(t,0), unmerged_types.get(t,0)])

# # --------- Quick stats & charts ----------
# # 1) Histogram of canonical cluster sizes
# sizes = [len(members) for members in merged.values()]
# if len(sizes) == 0:
#     print("No merged canonical groups found. Exiting chart generation.")
# else:
#     plt.figure(figsize=(6,4))
#     plt.hist(sizes, bins=range(1, max(sizes)+2), edgecolor='black')
#     plt.title("Distribution of canonical cluster sizes (# members)")
#     plt.xlabel("members per canonical entity")
#     plt.ylabel("count")
#     plt.tight_layout()
#     plt.savefig(OUT_DIR / "merges_hist.png", dpi=150)
#     plt.close()

# # 2) Pie chart of resolved_action counts (top actions)
# actions_df = pd.DataFrame(action_counts.items(), columns=["action","count"]).sort_values("count", ascending=False)
# plt.figure(figsize=(6,6))
# top_actions = actions_df.head(6)
# plt.pie(top_actions["count"], labels=top_actions["action"], autopct="%1.1f%%", startangle=140)
# plt.title("Top resolved_action distribution")
# plt.tight_layout()
# plt.savefig(OUT_DIR / "actions_pie.png", dpi=150)
# plt.close()

# # 3) Top canonical groups CSV (top 50)
# canon_df.head(50).to_csv(OUT_DIR / "top50_canonical.csv", index=False)

# # 4) A simple mapping file for quick inspection: canonical_id -> [member names]
# simple_map = {cid: [m.get("entity_name") for m in members] for cid,members in merged.items()}
# with open(OUT_DIR / "canonical_to_members_sample.json", "w", encoding="utf-8") as fh:
#     json.dump(simple_map, fh, ensure_ascii=False, indent=2)

# # 5) Save unmerged entities (for manual review)
# with open(OUT_DIR / "unmerged_entities.jsonl", "w", encoding="utf-8") as fh:
#     for e in unmerged:
#         fh.write(json.dumps(e, ensure_ascii=False) + "\n")

# # ---------- Print short summary ----------
# print("Analysis saved to:", OUT_DIR)
# print("Counts:")
# print(" - total entities:", len(entities))
# print(" - canonical groups (merged):", len(merged))
# print(" - unmerged entities:", len(unmerged))
# print("Top 10 canonical groups (id, size):")
# for cid, members in canon_df[["canonical_id","n_members"]].head(10).itertuples(index=False, name=None):
#     print(" ", cid, members)

# # optional: show top 10 actions
# print("Top actions:")
# for a,cnt in action_counts.most_common(10):
#     print(" ", a, cnt)

# print("\nFiles produced (entResAnalysis/):")
# for p in sorted(OUT_DIR.iterdir()):
#     print(" ", p.name)


#endregion#? Analyze_entity_resolution
#*#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Iterative Ent Res with_per_run_outputs




# ============================================================
# Iterative Embedding + Clustering + Resolution Driver (Robust)
# ============================================================

from pathlib import Path
import json
import shutil
import time
from typing import List, Dict

# ---------------- Iteration control ----------------
DEFAULT_MAX_ITERS = 5
MIN_MERGES_TO_CONTINUE = 1

# ---------------- Paths ----------------
ENT_RAW_SEED = Path("SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl")

CLUSTERED_PATH = Path("SGCE-KG/data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")
CANONICAL_PATH = Path("SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
RESOLVED_PATH = Path("SGCE-KG/data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")

ITER_DIR = (ENT_RAW_SEED.parent).parent / "iterative_runs"
ITER_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Utilities ----------------
def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out

def write_jsonl(path: Path, objs: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for o in objs:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")

def normalize_entity_schema(e: Dict) -> Dict:
    """
    Force a consistent schema/order for ALL entities.
    """
    return {
        "id": e.get("id"),
        "entity_name": e.get("entity_name"),
        "entity_description": e.get("entity_description"),
        "entity_type_hint": e.get("entity_type_hint"),
        "confidence_score": e.get("confidence_score"),
        "resolution_context": e.get("resolution_context"),
        "node_properties": e.get("node_properties", []),
        "merged_from": e.get("merged_from", [e.get("id")]),
        "members": e.get("members", [e]),
        "merge_history": e.get("merge_history", [])
    }

# ---------------- Build next-iteration entities ----------------
def build_next_entities(resolved_entities: List[Dict],
                        canonical_entities: List[Dict],
                        iteration: int):

    canon_map = {c["canonical_id"]: c for c in canonical_entities if "canonical_id" in c}

    by_canon = {}
    survivors = []

    for e in resolved_entities:
        cid = e.get("canonical_id")
        if cid:
            by_canon.setdefault(cid, []).append(e)
        else:
            survivors.append(e)

    next_entities = []
    n_merges = 0

    # ---- merged groups ----
    for cid, members in by_canon.items():
        canon = canon_map.get(cid, {})
        n_merges += 1

        merged_from = []
        merge_history = []

        for m in members:
            merged_from.extend(m.get("merged_from", [m["id"]]))
            merge_history.extend(m.get("merge_history", []))

        merge_history.append({
            "iteration": iteration,
            "canonical_id": cid,
            "merged_ids": list(set(merged_from))
        })

        # aggregate node properties
        props = []
        seen = set()
        for m in members:
            for p in m.get("node_properties", []) or []:
                k = (p.get("prop_name"), p.get("prop_value"))
                if k not in seen:
                    seen.add(k)
                    props.append(p)

        merged_entity = {
            "id": cid,
            "entity_name": canon.get("canonical_name", members[0]["entity_name"]),
            "entity_description": canon.get("canonical_description", members[0].get("entity_description")),
            "entity_type_hint": canon.get("canonical_type", members[0].get("entity_type_hint")),
            "confidence_score": max(
                [m.get("confidence_score") for m in members if m.get("confidence_score") is not None],
                default=None
            ),
            "resolution_context": canon.get("rationale", ""),
            "node_properties": props,
            "merged_from": sorted(set(merged_from)),
            "members": members,
            "merge_history": merge_history
        }

        next_entities.append(normalize_entity_schema(merged_entity))

    # ---- unchanged entities ----
    for e in survivors:
        e2 = dict(e)
        e2.setdefault("merged_from", [e["id"]])
        e2.setdefault("members", [e])
        e2.setdefault("merge_history", [])
        next_entities.append(normalize_entity_schema(e2))

    return next_entities, n_merges

# ---------------- Main iterative loop ----------------
def iterative_resolution():

    current_input = ENT_RAW_SEED

    # backup seed once
    seed_backup = ITER_DIR / "entities_raw_seed_backup.jsonl"
    if not seed_backup.exists():
        shutil.copy2(ENT_RAW_SEED, seed_backup)

    for it in range(1, DEFAULT_MAX_ITERS + 1):
        print(f"\n================ ITERATION {it} ================")

        # ---- 1) Embedding + clustering ----
        print("[1] Embedding + clustering")
        main_cli(type("Args", (), {
            "entities_in": Path(current_input),
            "out_jsonl": str(CLUSTERED_PATH),
            "clusters_summary": str(CLUSTERED_PATH.parent / f"clusters_summary_iter{it}.json"),
            "use_umap": True
        })())
        

        # ---- 2) Resolution ----
        print("[2] Entity resolution")
        orchestrate()

        # ---- 3) Build next iteration ----
        resolved = load_jsonl(RESOLVED_PATH)
        canon = load_jsonl(CANONICAL_PATH)

        next_entities, n_merges = build_next_entities(resolved, canon, iteration=it)

        print(f"[3] Merges in iteration {it}: {n_merges}")

        # save per-iteration output
        out_path = ITER_DIR / f"entities_iter{it}.jsonl"
        write_jsonl(out_path, next_entities)

        # ---- stopping conditions ----
        if n_merges < MIN_MERGES_TO_CONTINUE:
            print("✔ Stopping: merges below threshold.")
            break

        if it >= DEFAULT_MAX_ITERS:
            print("✔ Stopping: reached max iterations.")
            break

        current_input = out_path
        time.sleep(0.3)

    print("\n=== Iterative resolution COMPLETE ===")
    print(f"All intermediate iterations saved under: {ITER_DIR}")





# -----------------------
# Ent Res Multi Run  - Run statement
# -----------------------

# if __name__ == "__main__":
#     iterative_resolution()





#endregion#? Iterative Ent Res with_per_run_outputs
#?#########################  End  ##########################








#*######################### Start ##########################
#region:#?   Produce_class_input_from_iter K - V2


# #!/usr/bin/env python3
# """
# Produce a clean JSONL for class-identification with only the requested fields.

# This version is more robust in how it recovers `chunk_id`:
# - It searches recursively for any `chunk_id` field inside each record, so
#   different internal structures (e.g. nested in members) are handled.
# - It also looks up chunk_ids for any IDs listed in `merged_from` by using
#   the original raw entities file.
# """

# import sys, json
# from pathlib import Path
# from typing import Dict, List

# # ---------- CONFIG: adjust if needed ----------

# # Directory with per-iteration entity files
# iter_path = Path("SGCE-KG/data/Entities/iterative_runs/")
# input_files = sorted(iter_path.glob("entities_iter*.jsonl"))
# if not input_files:
#     raise FileNotFoundError(f"No iteration files found in: {iter_path}")

# latest_file = input_files[-1]
# print(f"Using latest iteration file: {latest_file}")
# input_path = latest_file

# # Original raw entities (used to recover chunk_ids via merged_from)
# RAW_SEED_PATH = Path("SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl")

# # Output file for class-identification input
# out_dir = Path("SGCE-KG/data/Classes/Cls_Input")
# out_file = out_dir / "cls_input_entities.jsonl"

# # ---------- guard against ipykernel injected args ----------
# if any(a.startswith("--f=") or a.startswith("--ipykernel") for a in sys.argv[1:]) or "ipykernel" in sys.argv[0]:
#     sys.argv = [sys.argv[0]]

# # ---------- helpers ----------

# def load_jsonl(path: Path) -> List[Dict]:
#     if not path.exists():
#         raise FileNotFoundError(f"Input not found: {path}")
#     out = []
#     with path.open("r", encoding="utf-8") as fh:
#         for line in fh:
#             line = line.strip()
#             if not line:
#                 continue
#             out.append(json.loads(line))
#     return out

# def ensure_list(x):
#     if x is None:
#         return []
#     if isinstance(x, list):
#         return x
#     return [x]

# def synth_id(base_name: str, idx: int):
#     safe = (base_name or "no_name").strip().replace(" ", "_")[:40]
#     return f"Tmp_{safe}_{idx}"

# def dedupe_preserve_order(values: List) -> List[str]:
#     """
#     Deduplicate while preserving order; normalize everything to str and
#     drop falsy values (None, "", etc.).
#     """
#     seen = set()
#     out: List[str] = []
#     for v in values:
#         if not v:
#             continue
#         sv = str(v)
#         if sv not in seen:
#             seen.add(sv)
#             out.append(sv)
#     return out

# def find_chunk_ids_anywhere(obj) -> List:
#     """
#     Recursively find all values of any key named 'chunk_id' in a nested
#     dict/list structure.
#     """
#     found: List = []
#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             if k == "chunk_id" and v is not None:
#                 found.extend(ensure_list(v))
#             else:
#                 found.extend(find_chunk_ids_anywhere(v))
#     elif isinstance(obj, list):
#         for item in obj:
#             found.extend(find_chunk_ids_anywhere(item))
#     return found

# def build_raw_chunk_index(seed_path: Path) -> Dict[str, List[str]]:
#     """
#     Build a mapping from raw entity id -> list of chunk_ids from the
#     original entities_raw.jsonl file.
#     """
#     id_to_chunks: Dict[str, List[str]] = {}
#     if not seed_path.exists():
#         print(f"WARNING: raw seed file not found; cannot augment chunk_ids from raw entities: {seed_path}")
#         return id_to_chunks

#     raw_recs = load_jsonl(seed_path)
#     for r in raw_recs:
#         rid = r.get("id") or r.get("entity_id") or None
#         if not rid:
#             continue
#         chunks = find_chunk_ids_anywhere(r)
#         if chunks:
#             id_to_chunks[rid] = dedupe_preserve_order(chunks)
#     return id_to_chunks

# # ---------- main ----------

# def produce_clean_jsonl(inp: Path, outp: Path):
#     # Build lookup from raw entity id -> chunk_ids, so we can use merged_from
#     raw_id_to_chunks = build_raw_chunk_index(RAW_SEED_PATH)

#     recs = load_jsonl(inp)
#     outp.parent.mkdir(parents=True, exist_ok=True)

#     cleaned = []
#     for i, r in enumerate(recs):
#         # pick id (prefer top-level id or canonical_id); otherwise synth
#         rid = r.get("id") or r.get("canonical_id") or r.get("canonical") or None
#         if not rid:
#             rid = synth_id(r.get("entity_name"), i)

#         # ---- chunk_id gathering ----
#         chunk_ids: List[str] = []

#         # 1) Any chunk_id present anywhere in this record (robust to structure)
#         direct_chunks = find_chunk_ids_anywhere(r)
#         chunk_ids.extend(direct_chunks)

#         # 2) Augment with chunk_ids from raw entities referenced in merged_from
#         for src_id in ensure_list(r.get("merged_from")):
#             if not src_id:
#                 continue
#             src_chunks = raw_id_to_chunks.get(str(src_id))
#             if src_chunks:
#                 chunk_ids.extend(src_chunks)

#         # final dedupe + normalization
#         chunk_ids = dedupe_preserve_order(chunk_ids)

#         # ---- node_properties normalization ----
#         node_props = r.get("node_properties") or []
#         if not isinstance(node_props, list):
#             node_props = [node_props]

#         cleaned_rec = {
#             "id": rid,
#             "entity_name": r.get("entity_name") or r.get("canonical_name") or "",
#             "entity_description": r.get("entity_description") or r.get("canonical_description") or "",
#             "entity_type_hint": r.get("entity_type_hint") or r.get("canonical_type") or r.get("entity_type") or "",
#             "confidence_score": r.get("confidence_score") if r.get("confidence_score") is not None else None,
#             "resolution_context": r.get("resolution_context") or r.get("text_span") or r.get("context_phrase") or "",
#             "flag": r.get("flag") or "entity_raw",
#             "chunk_id": chunk_ids,
#             "node_properties": node_props,
#         }
#         cleaned.append(cleaned_rec)

#     with outp.open("w", encoding="utf-8") as fh:
#         for rec in cleaned:
#             fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

#     print(f"Wrote {len(cleaned)} records -> {outp}")


# -----------------------
# Cls Rec input producer - Run statement
# -----------------------

# if __name__ == "__main__":
#     produce_clean_jsonl(input_path, out_file)


#endregion#? Produce_class_input_from_iter K - V2
#*#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Produce_class_input_from_iter K - V3


#!/usr/bin/env python3
"""
Produce a clean JSONL for class-identification with only the requested fields.

This version is more robust in how it recovers `chunk_id`:
- It searches recursively for any `chunk_id` field inside each record, so
  different internal structures (e.g. nested in members) are handled.
- It also looks up chunk_ids for any IDs listed in `merged_from` by using
  the original raw entities file.
"""

import sys, json
from pathlib import Path
from typing import Dict, List, Optional

# ---------- CONFIG: adjust if needed ----------

# Original raw entities (used to recover chunk_ids via merged_from)
RAW_SEED_PATH = Path("SGCE-KG/data/Entities/Ent_Raw_0/entities_raw.jsonl")

# Default locations for latest iteration entities and class-identification output
DEFAULT_ITER_DIR = Path("SGCE-KG/data/Entities/iterative_runs/")
DEFAULT_CLS_OUT_DIR = Path("SGCE-KG/data/Classes/Cls_Input")
DEFAULT_CLS_OUT_FILE = DEFAULT_CLS_OUT_DIR / "cls_input_entities.jsonl"

# ---------- guard against ipykernel injected args ----------
if any(a.startswith("--f=") or a.startswith("--ipykernel") for a in sys.argv[1:]) or "ipykernel" in sys.argv[0]:
    sys.argv = [sys.argv[0]]

# ---------- helpers ----------




def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def synth_id(base_name: str, idx: int):
    safe = (base_name or "no_name").strip().replace(" ", "_")[:40]
    return f"Tmp_{safe}_{idx}"

def dedupe_preserve_order(values: List) -> List[str]:
    """
    Deduplicate while preserving order; normalize everything to str and
    drop falsy values (None, "", etc.).
    """
    seen = set()
    out: List[str] = []
    for v in values:
        if not v:
            continue
        sv = str(v)
        if sv not in seen:
            seen.add(sv)
            out.append(sv)
    return out

def find_chunk_ids_anywhere(obj) -> List:
    """
    Recursively find all values of any key named 'chunk_id' in a nested
    dict/list structure.
    """
    found: List = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "chunk_id" and v is not None:
                found.extend(ensure_list(v))
            else:
                found.extend(find_chunk_ids_anywhere(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_chunk_ids_anywhere(item))
    return found

def build_raw_chunk_index(seed_path: Path) -> Dict[str, List[str]]:
    """
    Build a mapping from raw entity id -> list of chunk_ids from the
    original entities_raw.jsonl file.
    """
    id_to_chunks: Dict[str, List[str]] = {}
    if not seed_path.exists():
        print(f"WARNING: raw seed file not found; cannot augment chunk_ids from raw entities: {seed_path}")
        return id_to_chunks

    raw_recs = load_jsonl(seed_path)
    for r in raw_recs:
        rid = r.get("id") or r.get("entity_id") or None
        if not rid:
            continue
        chunks = find_chunk_ids_anywhere(r)
        if chunks:
            id_to_chunks[rid] = dedupe_preserve_order(chunks)
    return id_to_chunks


# ---------- main ----------

def produce_clean_jsonl(inp: Optional[Path] = None, outp: Optional[Path] = None):
    """Produce a cleaned JSONL for class-identification.

    If *inp* is None, the latest iteration file in DEFAULT_ITER_DIR is used.
    If *outp* is None, DEFAULT_CLS_OUT_FILE is used.
    """

    # Determine input/output paths lazily so nothing runs on import.
    if inp is None:
        iter_path = DEFAULT_ITER_DIR
        input_files = sorted(iter_path.glob("entities_iter*.jsonl"))
        if not input_files:
            raise FileNotFoundError(f"No iteration files found in: {iter_path}")
        latest_file = input_files[-1]
        print(f"Using latest iteration file: {latest_file}")
        inp = latest_file

    if outp is None:
        outp = DEFAULT_CLS_OUT_FILE

    # Build lookup from raw entity id -> chunk_ids, so we can use merged_from
    raw_id_to_chunks = build_raw_chunk_index(RAW_SEED_PATH)

    recs = load_jsonl(inp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cleaned = []
    for i, r in enumerate(recs):
        # pick id (prefer top-level id or canonical_id); otherwise synth
        rid = r.get("id") or r.get("canonical_id") or r.get("canonical") or None
        if not rid:
            rid = synth_id(r.get("entity_name"), i)

        # ---- chunk_id gathering ----
        chunk_ids: List[str] = []

        # 1) Any chunk_id present anywhere in this record (robust to structure)
        direct_chunks = find_chunk_ids_anywhere(r)
        chunk_ids.extend(direct_chunks)

        # 2) Augment with chunk_ids from raw entities referenced in merged_from
        for src_id in ensure_list(r.get("merged_from")):
            if not src_id:
                continue
            src_chunks = raw_id_to_chunks.get(str(src_id))
            if src_chunks:
                chunk_ids.extend(src_chunks)

        # final dedupe + normalization
        chunk_ids = dedupe_preserve_order(chunk_ids)

        # ---- node_properties normalization ----
        node_props = r.get("node_properties") or []
        if not isinstance(node_props, list):
            node_props = [node_props]

        cleaned_rec = {
            "id": rid,
            "entity_name": r.get("entity_name") or r.get("canonical_name") or "",
            "entity_description": r.get("entity_description") or r.get("canonical_description") or "",
            "entity_type_hint": r.get("entity_type_hint") or r.get("canonical_type") or r.get("entity_type") or "",
            "confidence_score": r.get("confidence_score") if r.get("confidence_score") is not None else None,
            "resolution_context": r.get("resolution_context") or r.get("text_span") or r.get("context_phrase") or "",
            "flag": r.get("flag") or "entity_raw",
            "chunk_id": chunk_ids,
            "node_properties": node_props,
        }
        cleaned.append(cleaned_rec)

    with outp.open("w", encoding="utf-8") as fh:
        for rec in cleaned:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(cleaned)} records -> {outp}")


# -----------------------
# Cls Rec input producer - Run statement
# -----------------------

# if __name__ == "__main__":
#     produce_clean_jsonl(input_path, out_file)


#endregion#? Produce_class_input_from_iter K - V3
#?#########################  End  ##########################




#endregion#! Entity Identification
#!#############################################  End Chapter  ##################################################







#!############################################# Start Chapter ##################################################
#region:#!   Class Identification









#?######################### Start ##########################
#region:#?   Cls Rec V4 - Class hint type included  - Gpt 5.1 fix

#!/usr/bin/env python3
"""
classrec_iterative_v4.py

Iterative Class Recognition (ClassRec) with per-iteration class outputs
matching the cluster-file visual/JSON format and including class metadata
(label/desc/confidence/evidence + source cluster id + class_type_hint).

This is a fix for KeyError caused by unescaped braces in the prompt template.
All literal braces in the prompt are escaped ({{ and }}), except {members_block}.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# OpenAI client
from openai import OpenAI

# ----------------------------- CONFIG -----------------------------
INPUT_PATH = Path("SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("SGCE-KG/data/Classes/Cls_Rec")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_CANDIDATES_OUT = OUT_DIR / "class_candidates.jsonl"
INITIAL_CLUSTER_OUT = OUT_DIR / "initial_cluster_entities.json"
RECLUSTER_PREFIX = OUT_DIR / "recluster_round_"
CLASSES_PREFIX = OUT_DIR / "classes_round_"
REMAINING_OUT = OUT_DIR / "remaining_entities.jsonl"

# embedder / model
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHTS = {"name": 0.40, "desc": 0.25, "ctx": 0.35}

# HDBSCAN + UMAP
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0

# local subcluster
MAX_CLUSTER_SIZE_FOR_LOCAL = 30
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1

# prompt and LLM / limits
OPENAI_MODEL = "gpt-5.1"
OPENAI_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 8000
MAX_MEMBERS_PER_PROMPT = 10
PROMPT_CHAR_PER_TOKEN = 4          # crude estimate
MAX_PROMPT_TOKENS_EST = 8000

# iteration control
MAX_RECLUSTER_ROUNDS = 12  # safety cap
VERBOSE = False

# ------------------------ OpenAI client loader -----------------------
def _load_openai_key(envvar: str = "OPENAI_API_KEY", fallback_path: str = ".env"):
    key = os.getenv(envvar, fallback_path)
    if isinstance(key, str) and Path(key).exists():
        try:
            txt = Path(key).read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
if not OPENAI_KEY or not isinstance(OPENAI_KEY, str) or len(OPENAI_KEY) < 10:
    print("⚠️ OPENAI key missing or short. Set OPENAI_API_KEY or put key in fallback file.")
client = OpenAI(api_key=OPENAI_KEY)

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_completion_tokens: int = LLM_MAX_TOKENS) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=LLM_MAX_TOKENS
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ------------------------- HF Embedder ------------------------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE: print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers ----------------------------------
def load_entities(path: Path) -> List[Dict]:
    assert path.exists(), f"Input not found: {path}"
    ents = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ents.append(json.loads(line))
    return ents

def safe_text(e: Dict, k: str) -> str:
    v = e.get(k)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def build_field_texts(entities: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    names, descs, ctxs = [], [], []
    for e in entities:
        names.append(safe_text(e, "entity_name") or "")
        descs.append(safe_text(e, "entity_description") or "")
        resolution = safe_text(e, "resolution_context") or safe_text(e, "text_span") or safe_text(e, "context_phrase") or ""
        et = safe_text(e, "entity_type_hint") or ""
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np in node_props:
                if isinstance(np, dict):
                    pname = np.get("prop_name") or np.get("name") or ""
                    pval = np.get("prop_value") or np.get("value") or ""
                    if pname and pval:
                        pieces.append(f"{pname}:{pval}")
                    elif pname:
                        pieces.append(pname)
            if pieces:
                node_props_text = " | ".join(pieces)
        parts = []
        if et:
            parts.append(f"[TYPE:{et}]")
        if resolution:
            parts.append(resolution)
        if node_props_text:
            parts.append(node_props_text)
        ctxs.append(" ; ".join(parts))
    return names, descs, ctxs

def compute_combined_embeddings(embedder: HFEmbedder, entities: List[Dict], weights=WEIGHTS) -> np.ndarray:
    names, descs, ctxs = build_field_texts(entities)
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_ctx  = embedder.encode_batch(ctxs)  if any(t.strip() for t in ctxs) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]; break
    if D is None:
        raise ValueError("No textual field produced embeddings")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name); emb_desc = _ensure(emb_desc); emb_ctx = _ensure(emb_ctx)
    w_name = weights.get("name", 0.0); w_desc = weights.get("desc", 0.0); w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0: raise ValueError("invalid weights")
    w_name /= Wsum; w_desc /= Wsum; w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined

def run_hdbscan(embeddings: np.ndarray, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC, use_umap=USE_UMAP) -> Tuple[np.ndarray, object]:
    X = embeddings
    if use_umap and UMAP_AVAILABLE and X.shape[0] >= 5:
        reducer = umap.UMAP(n_components=min(UMAP_N_COMPONENTS, max(2, X.shape[0]-1)),
                            n_neighbors=min(UMAP_N_NEIGHBORS, max(2, X.shape[0]-1)),
                            min_dist=UMAP_MIN_DIST,
                            metric='cosine', random_state=42)
        X = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X)
    return labels, clusterer


# ------------------ Prompt (REVISED to be MUST) -----------------------
# NOTE: all literal braces are escaped ({{ and }}) except {members_block}
CLASS_PROMPT_TEMPLATE = """
You are a careful schema / ontology suggester.
Your job is to induce a meaningful, reusable schema from entity-level evidence.
Be conservative, precise, and resist over-generalization.

=====================
INPUT YOU ARE GIVEN
=====================

Each entity you see is already a resolved entity from a previous pipeline stage.
For each entity, you are given the following fields:

- entity_id: (string) unique id for the entity.
- entity_name: (string) short canonical entity-level name.
- entity_description: (string) concise explanation of the entity.
- resolution_context: (string) a 20–120 word excerpt explaining why this entity was named this way;
  this is the PRIMARY semantic evidence.
- entity_type_hint: (string) a weak, local hint about entity role (e.g., Material, Component, FailureMechanism).
  This is a suggestion only and may be incorrect.
- node_properties: (optional) small list/dict of intrinsic properties (e.g., grade:304).

=====================
YOUR TASK
=====================

Given a small group of entities, suggest ZERO or more classes that group entities
which genuinely belong together in a practical, reusable way.

Important rules (ENFORCED):
- If the input contains MULTIPLE entities:
  -> You MUST ONLY create classes that contain TWO OR MORE members.
  -> You MUST NOT create a single-member class when multiple entities are present.
- The ONLY situation where a single-member class is allowed:
  -> When the input contains EXACTLY ONE entity (single-entity mode).

Do NOT force entities into classes by broadening or renaming a class to "fit" them.
If an entity does not clearly belong, omit it — it will be revisited later.

SINGLE-ENTITY MODE (IMPORTANT)
- You are receiving EXACTLY ONE entity in this prompt. In this case you SHOULD produce exactly one class that contains that entity (a single-member class). 
- The single-member class must include: class_label, class_description, class_type_hint (if possible), member_ids (use the provided entity_id), and a confidence value.  
- Do NOT invent other entity_ids; use the entity_id exactly as provided. If you judge that no sensible class exists, still return a short single-member class using a conservative label like "Misc: <entity_name>" with low confidence (e.g., 0.10) rather than returning an empty array. This helps downstream experiments while keeping the class low-weight.


=====================
TWO-LEVEL SCHEMA
=====================

We are building a TWO-LEVEL schema:

Level 1 (Classes): groups of entities (e.g., "High-Temperature Corrosion")
Level 2 (Class_Type_Hint): an upper-level connector that groups classes (e.g., "Failure Mechanism")

- Class_Type_Hint is NOT the same as entity_type_hint.
- Infer Class_Type_Hint from the class members; do NOT blindly copy entity_type_hint.

=====================
OUTPUT FORMAT (REQUIRED)
=====================

Return ONLY a JSON ARRAY.

Each element must have:
- class_label (string): short canonical name (1-3 words)
- class_description (string): 1–2 sentences explaining membership & distinction
- class_type_hint (string): upper-level family (e.g., "Failure Mechanism")
- member_ids (array[string]): entity_ids from the input that belong to this class
- confidence (float): 0.0–1.0 confidence estimate
- evidence_excerpt (string, optional): brief excerpt (5–30 words) that supports the grouping

HARD OUTPUT RULES:
- Use ONLY provided entity_ids.
- member_ids MUST be from the input.
- Prefer non-overlapping classes; small overlap allowed only if justified.
- If no sensible class, return [].

=====================
EXAMPLES
=====================

GOOD:
Input entities:
- graphitization (En_1)
- sulfidation    (En_2)

Output:
[
  {{
    "class_label": "High-Temperature Degradation",
    "class_description": "Material degradation mechanisms at elevated temperatures (graphitization, sulfidation).",
    "class_type_hint": "Failure Mechanism",
    "member_ids": ["En_1","En_2"],
    "confidence": 0.87
  }}
]

BAD (DO NOT DO):
Entities: graphitization, pressure gauge
-> Do NOT output a broad "Equipment Issue" that forces both into one class. Prefer [].

=====================
ENTITIES
=====================

Each entity below is provided as:
- entity_id
- entity_name
- entity_description
- resolution_context
- entity_type_hint
- node_properties

Entities:
{members_block}

Return JSON array only.
"""

def build_members_block(members: List[Dict]) -> str:
    rows = []
    for m in members:
        eid = m.get("id", "")
        name = (m.get("entity_name") or "")[:120].replace("\n", " ")
        desc = (m.get("entity_description") or "")[:300].replace("\n", " ")
        res = (m.get("resolution_context") or m.get("context_phrase") or "")[:400].replace("\n", " ")
        et = (m.get("entity_type_hint") or "")[:80].replace("\n", " ")
        node_props = m.get("node_properties") or []
        np_txt = json.dumps(node_props, ensure_ascii=False) if node_props else ""
        rows.append(f"{eid} | {name} | {desc} | {res} | {et} | {np_txt}")
    return "\n".join(rows)

def parse_json_array_from_text(txt: str):
    if not txt:
        return None
    s = txt.strip()
    if s.startswith("```"):
        s = s.strip("`")
    start = s.find('[')
    end = s.rfind(']')
    if start != -1 and end != -1 and end > start:
        cand = s[start:end+1]
        try:
            return json.loads(cand)
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        return None

# ------------------- Worker: process a chunk of members --------------------
def process_member_chunk_llm(members: List[Dict], single_entity_mode: bool = False) -> List[Dict]:
    members_block = build_members_block(members)
    prompt = CLASS_PROMPT_TEMPLATE.format(members_block=members_block)
    est_tokens = max(1, int(len(prompt) / PROMPT_CHAR_PER_TOKEN))
    if est_tokens > MAX_PROMPT_TOKENS_EST:
        if VERBOSE: print(f"[warning] prompt too large (est_tokens={est_tokens}) -> skipping chunk of size {len(members)}")
        return []
    llm_out = call_llm(prompt)
    arr = parse_json_array_from_text(llm_out)
    if not arr:
        return []
    candidates = []
    provided_ids = {m.get("id") for m in members}
    for c in arr:
        label = c.get("class_label") or c.get("label") or c.get("name")
        if not label:
            continue
        member_ids = c.get("member_ids") or c.get("members") or []
        member_ids = [mid for mid in member_ids if mid in provided_ids]
        if not member_ids:
            continue
        if not single_entity_mode and len(members) > 1 and len(member_ids) < 2:
            continue
        confidence = float(c.get("confidence", 0.0)) if c.get("confidence") is not None else 0.0
        desc = c.get("class_description") or c.get("description") or ""
        ev = c.get("evidence_excerpt") or ""
        class_type = c.get("class_type_hint") or c.get("class_type") or ""
        candidate = {
            "candidate_id": "ClsC_" + uuid.uuid4().hex[:8],
            "class_label": label,
            "class_description": desc,
            "class_type_hint": class_type,
            "member_ids": member_ids,
            "confidence": confidence,
            "evidence_excerpt": ev,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        candidates.append(candidate)
    return candidates

# -------------------- Utility: write cluster files (full entity objects) -----------
def write_cluster_summary(path: Path, cluster_map: Dict[int, List[int]], entities: List[Dict]):
    n_entities = len(entities)
    clusters = {}
    for k, idxs in sorted(cluster_map.items(), key=lambda x: x[0]):
        arr = []
        for i in idxs:
            ent = entities[i]
            arr.append(ent)
        clusters[str(k)] = arr
    meta = {"n_entities": n_entities, "n_clusters": len(clusters), "clusters": clusters}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{\n')
        fh.write(f'  "n_entities": {meta["n_entities"]},\n')
        fh.write(f'  "n_clusters": {meta["n_clusters"]},\n')
        fh.write('  "clusters": {\n')
        cluster_items = list(clusters.items())
        for ci, (k, ents) in enumerate(cluster_items):
            fh.write(f'    "{k}": [\n')
            for ei, ent in enumerate(ents):
                ent_json = json.dumps(ent, ensure_ascii=False, separators=(",", ": "))
                fh.write(f'      {ent_json}')
                if ei < len(ents) - 1:
                    fh.write(',\n')
                else:
                    fh.write('\n')
            fh.write('    ]')
            if ci < len(cluster_items) - 1:
                fh.write(',\n')
            else:
                fh.write('\n')
        fh.write('  }\n')
        fh.write('}\n')

def write_classes_round(path: Path, candidates: List[Dict], entities: List[Dict], id_to_index: Dict[str,int]):
    classes_map = {}
    total_members = 0
    for c in candidates:
        cid = c.get("candidate_id") or ("ClsC_" + uuid.uuid4().hex[:8])
        mids = c.get("member_ids", [])
        member_objs = []
        for mid in mids:
            if mid in id_to_index:
                member_objs.append(entities[id_to_index[mid]])
        if not member_objs:
            continue
        meta = {
            "class_label": c.get("class_label", ""),
            "class_description": c.get("class_description", ""),
            "class_type_hint": c.get("class_type_hint", ""),
            "confidence": float(c.get("confidence", 0.0)),
            "evidence_excerpt": c.get("evidence_excerpt", ""),
            "source_cluster_id": c.get("source_cluster_id", None),
            "members": member_objs
        }
        classes_map[cid] = meta
        total_members += len(member_objs)
    meta = {"n_classes": len(classes_map), "n_members_total": total_members, "classes": classes_map}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{\n')
        fh.write(f'  "n_classes": {meta["n_classes"]},\n')
        fh.write(f'  "n_members_total": {meta["n_members_total"]},\n')
        fh.write('  "classes": {\n')
        items = list(classes_map.items())
        for ci, (k, cls_meta) in enumerate(items):
            fh.write(f'    "{k}": {{\n')
            fh.write(f'      "class_label": {json.dumps(cls_meta["class_label"], ensure_ascii=False)},\n')
            fh.write(f'      "class_description": {json.dumps(cls_meta["class_description"], ensure_ascii=False)},\n')
            fh.write(f'      "class_type_hint": {json.dumps(cls_meta["class_type_hint"], ensure_ascii=False)},\n')
            fh.write(f'      "confidence": {json.dumps(cls_meta["confidence"], ensure_ascii=False)},\n')
            fh.write(f'      "evidence_excerpt": {json.dumps(cls_meta["evidence_excerpt"], ensure_ascii=False)},\n')
            fh.write(f'      "source_cluster_id": {json.dumps(cls_meta["source_cluster_id"], ensure_ascii=False)},\n')
            fh.write(f'      "members": [\n')
            for ei, ent in enumerate(cls_meta["members"]):
                ent_json = json.dumps(ent, ensure_ascii=False, separators=(",", ": "))
                fh.write(f'        {ent_json}')
                if ei < len(cls_meta["members"]) - 1:
                    fh.write(',\n')
                else:
                    fh.write('\n')
            fh.write('      ]\n')
            fh.write('    }')
            if ci < len(items) - 1:
                fh.write(',\n')
            else:
                fh.write('\n')
        fh.write('  }\n')
        fh.write('}\n')

# -------------------- Main iterative orchestration -----------------------
def classrec_iterative_main():
    entities = load_entities(INPUT_PATH)
    print(f"[start] loaded {len(entities)} entities from {INPUT_PATH}")

    for e in entities:
        if "id" not in e:
            e["id"] = "En_" + uuid.uuid4().hex[:8]

    id_to_index = {e["id"]: i for i, e in enumerate(entities)}

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_emb = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("[info] embeddings computed, shape:", combined_emb.shape)

    labels, _ = run_hdbscan(combined_emb, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, use_umap=USE_UMAP)
    print("[info] initial clustering done. unique labels:", len(set(labels)))

    cluster_to_indices = {}
    for idx, lab in enumerate(labels):
        cluster_to_indices.setdefault(int(lab), []).append(idx)

    write_cluster_summary(INITIAL_CLUSTER_OUT, cluster_to_indices, entities)
    if VERBOSE: print(f"[write] initial cluster file -> {INITIAL_CLUSTER_OUT}")

    seen_by_llm = set()
    assigned_entity_ids = set()
    all_candidates = []

    def call_and_record(members_indices: List[int], source_cluster: Optional[object]=None, single_entity_mode: bool=False) -> List[Dict]:
        nonlocal seen_by_llm, assigned_entity_ids, all_candidates
        if not members_indices:
            return []
        members = [entities[i] for i in members_indices]
        results = []
        for i in range(0, len(members), MAX_MEMBERS_PER_PROMPT):
            chunk = members[i:i+MAX_MEMBERS_PER_PROMPT]
            for m in chunk:
                if m.get("id"):
                    seen_by_llm.add(m["id"])
            candidates = process_member_chunk_llm(chunk, single_entity_mode=single_entity_mode)
            for c in candidates:
                mids = c.get("member_ids", [])
                member_entities = [entities[id_to_index[mid]] for mid in mids if mid in id_to_index]
                if not member_entities:
                    continue
                c["member_entities"] = member_entities
                c["source_cluster_id"] = source_cluster
                all_candidates.append(c)
                for mid in mids:
                    assigned_entity_ids.add(mid)
                results.append(c)
        return results

    round0_candidates = []
    if VERBOSE: print("[round0] processing coarse non-noise clusters")
    for lab, idxs in sorted(cluster_to_indices.items(), key=lambda x: x[0]):
        if lab == -1:
            continue
        if VERBOSE: print(f"[round0] cluster {lab} size={len(idxs)}")
        if len(idxs) > MAX_CLUSTER_SIZE_FOR_LOCAL:
            try:
                sub_emb = combined_emb[idxs]
                local_clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, LOCAL_HDBSCAN_MIN_CLUSTER_SIZE),
                                                  min_samples=LOCAL_HDBSCAN_MIN_SAMPLES, metric='euclidean', cluster_selection_method='eom')
                local_labels = local_clusterer.fit_predict(sub_emb)
            except Exception:
                local_labels = np.zeros(len(idxs), dtype=int)
            local_map = {}
            for i_local, lab_local in enumerate(local_labels):
                global_idx = idxs[i_local]
                local_map.setdefault(int(lab_local), []).append(global_idx)
            for sublab, subidxs in local_map.items():
                if sublab == -1:
                    continue
                source_id = {"coarse_cluster": int(lab), "local_subcluster": int(sublab)}
                cand = call_and_record(subidxs, source_cluster=source_id, single_entity_mode=(len(subidxs) == 1))
                round0_candidates.extend(cand)
        else:
            source_id = {"coarse_cluster": int(lab), "local_subcluster": None}
            cand = call_and_record(idxs, source_cluster=source_id, single_entity_mode=(len(idxs) == 1))
            round0_candidates.extend(cand)

    classes_round0_path = Path(f"{CLASSES_PREFIX}0.json")
    write_classes_round(classes_round0_path, round0_candidates, entities, id_to_index)
    if VERBOSE: print(f"[write] classes round 0 -> {classes_round0_path}")

    original_noise_indices = cluster_to_indices.get(-1, [])
    round_num = 0
    while round_num < MAX_RECLUSTER_ROUNDS:
        round_num += 1
        seen_but_unassigned = list(seen_by_llm - assigned_entity_ids)
        pool_ids = {entities[i]["id"] for i in original_noise_indices}
        pool_ids.update(seen_but_unassigned)
        pool_ids = [pid for pid in pool_ids if pid not in assigned_entity_ids]
        if not pool_ids:
            if VERBOSE: print(f"[reclust {round_num}] pool empty -> stopping")
            break
        pool_indices = [id_to_index[pid] for pid in pool_ids if pid in id_to_index]
        if not pool_indices:
            if VERBOSE: print(f"[reclust {round_num}] no valid pool indices -> stopping")
            break

        if VERBOSE: print(f"[reclust {round_num}] reclustering pool size={len(pool_indices)}")
        try:
            sub_emb = combined_emb[pool_indices]
            labels_sub, _ = run_hdbscan(sub_emb, min_cluster_size=2, min_samples=1, use_umap=False)
        except Exception:
            labels_sub = np.zeros(len(pool_indices), dtype=int)

        sub_cluster_map = {}
        for local_i, lab_sub in enumerate(labels_sub):
            global_idx = pool_indices[local_i]
            sub_cluster_map.setdefault(int(lab_sub), []).append(global_idx)

        recluster_path = Path(f"{RECLUSTER_PREFIX}{round_num}.json")
        write_cluster_summary(recluster_path, sub_cluster_map, entities)
        if VERBOSE: print(f"[write] recluster round {round_num} -> {recluster_path}")

        round_candidates = []
        new_classes_count = 0
        for lab_sub, gidxs in sorted(sub_cluster_map.items(), key=lambda x: (x[0]==-1, x[0])):
            if lab_sub == -1:
                continue
            if VERBOSE: print(f"[reclust {round_num}] processing subcluster {lab_sub} size={len(gidxs)}")
            source_id = {"recluster_round": int(round_num), "subcluster": int(lab_sub)}
            cand = call_and_record(gidxs, source_cluster=source_id, single_entity_mode=(len(gidxs) == 1))
            round_candidates.extend(cand)
            new_classes_count += len(cand)

        classes_round_path = Path(f"{CLASSES_PREFIX}{round_num}.json")
        write_classes_round(classes_round_path, round_candidates, entities, id_to_index)
        if VERBOSE: print(f"[write] classes round {round_num} -> {classes_round_path}  (new_classes={new_classes_count})")

        if new_classes_count == 0:
            if VERBOSE: print(f"[reclust {round_num}] no new classes -> stopping recluster loop")
            break

    remaining_after_reclustering = [e for e in entities if e["id"] not in assigned_entity_ids]
    if VERBOSE: print(f"[single pass] remaining entities (before single-entity pass): {len(remaining_after_reclustering)}")

    single_candidates = []
    for e in remaining_after_reclustering:
        source_id = {"single_pass": True}
        cand = call_and_record([id_to_index[e["id"]]], source_cluster=source_id, single_entity_mode=True)
        single_candidates.extend(cand)

    classes_single_path = Path(f"{CLASSES_PREFIX}single.json")
    write_classes_round(classes_single_path, single_candidates, entities, id_to_index)
    if VERBOSE: print(f"[write] classes round single -> {classes_single_path}")

    with open(CLASS_CANDIDATES_OUT, "w", encoding="utf-8") as fh:
        for c in all_candidates:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    if VERBOSE: print(f"[write] cumulative class_candidates -> {CLASS_CANDIDATES_OUT} (count={len(all_candidates)})")

    final_remaining = [e for e in entities if e["id"] not in assigned_entity_ids]
    with open(REMAINING_OUT, "w", encoding="utf-8") as fh:
        for e in final_remaining:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    if VERBOSE: print(f"[write] final remaining entities -> {REMAINING_OUT} (count={len(final_remaining)})")

    print("[done] ClassRec iterative v4 finished.")



# -----------------------
# Cls Recognition  - Run statement
# -----------------------



# if __name__ == "__main__":
#     classrec_iterative_main()


#endregion#? Cls Rec V4 - Class hint type included
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Create input for Cls Res from per-round classes




#!/usr/bin/env python3
"""
merge_classes_for_cls_res.py

Merge per-round classes files (classes_round_*.json) into a single
JSONL + JSON file suitable as input to the next step (Cls Res).

Output:
 - SGCE-KG/data/Classes/Cls_Res/Cls_Res_input/classes_for_cls_res.jsonl
 - SGCE-KG/data/Classes/Cls_Res/Cls_Res_input/classes_for_cls_res.json
"""

import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("SGCE-KG/data/Classes/Cls_Rec")
PATTERN = "classes_round_*.json"

OUTPUT_ROOT = Path("SGCE-KG/data/Classes/Cls_Res/Cls_Res_input")
OUT_JSONL = OUTPUT_ROOT / "classes_for_cls_res.jsonl"
OUT_JSON  = OUTPUT_ROOT / "classes_for_cls_res.json"

def read_classes_file(p: Path):
    """
    Reads a classes_round file in the format produced by write_classes_round.
    Returns dict candidate_id -> class_meta (with members list of full entity objects)
    """
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] failed to parse {p}: {e}")
        return {}
    classes = j.get("classes") or {}
    out = {}
    for cid, meta in classes.items():
        # meta expected to contain class_label, class_description, class_type_hint, confidence, evidence_excerpt, source_cluster_id, members
        # ensure members is list
        members = meta.get("members") or []
        # normalize member objects: ensure each has an "id"
        members_norm = []
        for m in members:
            if isinstance(m, dict) and "id" in m:
                members_norm.append(m)
            else:
                # skip bad members
                continue
        out[cid] = {
            "candidate_id": cid,
            "class_label": meta.get("class_label", "") or "",
            "class_description": meta.get("class_description", "") or "",
            "class_type_hint": meta.get("class_type_hint", "") or "",
            "confidence": float(meta.get("confidence", 0.0) or 0.0),
            "evidence_excerpt": meta.get("evidence_excerpt", "") or "",
            "source_cluster_id": meta.get("source_cluster_id", None),
            "members": members_norm,
            # provenance: where we loaded it from
            "_source_file": str(p)
        }
    return out

def merge_classes(all_classes_by_file):
    """
    Merge classes by (label, class_type_hint) normalized key.
    If same key seen multiple times:
      - union members (unique by id)
      - take class_description from the instance with highest confidence (tie -> latest file appearance)
      - confidence = max(confidences)
      - evidence_excerpt: prefer higher confidence's excerpt
      - source_files: list
      - candidate_ids: list of contributing candidate ids
    """
    merged = {}
    key_to_ids = defaultdict(list)  # for debugging
    for cid, c in all_classes_by_file.items():
        key = (c["class_label"].strip().lower(), (c.get("class_type_hint") or "").strip().lower())
        if key not in merged:
            merged[key] = {
                "canonical_class_label": c["class_label"],
                "class_type_hint": c.get("class_type_hint", ""),
                "class_description": c.get("class_description", ""),
                "confidence": c.get("confidence", 0.0),
                "evidence_excerpt": c.get("evidence_excerpt", "") or "",
                "members_by_id": {m["id"]: m for m in c.get("members", [])},
                "candidate_ids": [c["candidate_id"]],
                "source_files": [c.get("_source_file")],
            }
        else:
            cur = merged[key]
            # union members
            for m in c.get("members", []):
                cur["members_by_id"][m["id"]] = m
            # update confidence & description/evidence if this c has higher confidence
            if c.get("confidence", 0.0) > cur["confidence"]:
                cur["confidence"] = c.get("confidence", 0.0)
                # replace description/evidence with higher-confidence one
                if c.get("class_description"):
                    cur["class_description"] = c.get("class_description", cur["class_description"])
                if c.get("evidence_excerpt"):
                    cur["evidence_excerpt"] = c.get("evidence_excerpt", cur["evidence_excerpt"])
            # append provenance
            cur["candidate_ids"].append(c["candidate_id"])
            sf = c.get("_source_file")
            if sf and sf not in cur["source_files"]:
                cur["source_files"].append(sf)
        key_to_ids[key].append(cid)

    # convert merged to output list
    out_list = []
    for key, v in merged.items():
        members = list(v["members_by_id"].values())
        out_obj = {
            "class_label": v["canonical_class_label"],
            "class_type_hint": v["class_type_hint"],
            "class_description": v.get("class_description",""),
            "confidence": float(v.get("confidence",0.0)),
            "evidence_excerpt": v.get("evidence_excerpt",""),
            "member_ids": [m["id"] for m in members],
            "members": members,
            "candidate_ids": v.get("candidate_ids", []),
            "source_files": v.get("source_files", [])
        }
        out_list.append(out_obj)
    return out_list

def main_input_for_cls_res():
    files = sorted(ROOT.glob(PATTERN))
    if not files:
        print(f"[error] no files found matching {PATTERN} in {ROOT}")
        return
    print(f"[info] found {len(files)} files. Reading...")
    all_classes = {}
    total_classes = 0
    for f in files:
        cls_map = read_classes_file(f)
        if not cls_map:
            continue
        for cid, c in cls_map.items():
            # ensure candidate id unique by prefixing file if collided
            if cid in all_classes:
                cid_new = f"{Path(c['_source_file']).stem}__{cid}"
                c["candidate_id"] = cid_new
                all_classes[cid_new] = c
            else:
                all_classes[cid] = c
        total_classes += len(cls_map)
    print(f"[info] collected {len(all_classes)} raw class entries (from {total_classes} source classes). Merging...")

    merged = merge_classes(all_classes)
    print(f"[info] merged -> {len(merged)} classes. Writing outputs...")

    # write JSONL (one merged class per line) and JSON (array)
    with open(OUT_JSONL, "w", encoding="utf-8") as fh_jsonl, open(OUT_JSON, "w", encoding="utf-8") as fh_json:
        for c in merged:
            fh_jsonl.write(json.dumps(c, ensure_ascii=False) + "\n")
        fh_json.write(json.dumps(merged, ensure_ascii=False, indent=2))

    # summary
    total_members = sum(len(c.get("members", [])) for c in merged)
    print(f"[done] wrote {OUT_JSONL} and {OUT_JSON}.")
    print(f"       classes: {len(merged)}, total member objects (post-merge, possibly duplicated across classes): {total_members}")




# # -----------------------
# # Create input for Cls Res  - Run statement
# # -----------------------


# if __name__ == "__main__":
#     main_input_for_cls_res()


#endregion#? Create input for Cls Res from per-round classes
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Cls Res V8  - Split + Remark + Summary (gpt-5.1 Responses)

#!/usr/bin/env python3
"""
classres_iterative_v8_gpt5.py

Class Resolution (Cls Res) — cluster class candidates, ask LLM to
order a sequence of functions (merge/create/reassign/modify/split) for each cluster,
then execute those functions locally and produce final resolved classes.

Key features:
- TWO-LAYER SCHEMA: Class_Group -> Classes -> Entities
- class_label treated as provisional; may be revised if evidence suggests a clearer name.
- LLM orders structural + schema actions using a small function vocabulary.
- Provisional IDs for newly created/merged/split classes so later steps can refer to them.
- 'remarks' channel so the LLM can flag out-of-scope or higher-level concerns without
  misusing structural functions.
- Conservative behavior with strong validation and logging.
- Summary folder with aggregated decisions and useful statistics.

Input:
  SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res.json

Output (written under OUT_DIR):
  - per-cluster decisions: cluster_<N>_decisions.json
  - per-cluster raw llm output: llm_raw/cluster_<N>_llm_raw.txt
  - per-cluster prompts: llm_raw/cluster_<N>_prompt.txt
  - cumulative action log: cls_res_action_log.jsonl
  - final resolved classes: final_classes_resolved.json and .jsonl
  - summary/all_clusters_decisions.json (aggregated decisions)
  - summary/stats_summary.json (aggregate statistics)

Updated for gpt-5.1:
- Uses OpenAI Responses API (client.responses.create) instead of Chat Completions.
- Uses max_output_tokens instead of max_tokens.
- Does not send temperature (unsupported by gpt-5.1).
- Optional reasoning={'effort': 'low'} for gpt-5.x models.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# transformers embedder (reuse same embedder pattern as ClassRec)
from transformers import AutoTokenizer, AutoModel

# OpenAI client (same style as your previous script)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------- CONFIG -----------------------------
INPUT_CLASSES =     Path("SGCE-KG/data/Classes/Cls_Res/Cls_Res_input/classes_for_cls_res.json")
#INPUT_CLASSES = Path("SGCE-KG/data/Classes/Cls_Rec/classes_for_cls_res-Wrong.json")
SRC_ENTITIES_PATH = Path("SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("SGCE-KG/data/Classes/Cls_Res")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_LLM_DIR = OUT_DIR / "llm_raw"
RAW_LLM_DIR.mkdir(exist_ok=True)

# Embedding model (changeable)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Weights for fields used to build class text for embeddings
CLASS_EMB_WEIGHTS = {
    "label": 0.30,
    "desc": 0.25,
    "type_hint": 0.10,
    "evidence": 0.05,
    "members": 0.30
}

# clustering params
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# LLM / OpenAI (updated for gpt-5.1 + Responses)
OPENAI_MODEL = "gpt-5.1"          # updated to gpt-5.1
OPENAI_TEMPERATURE = 0.0          # kept for signature compatibility; NOT sent to API
LLM_MAX_TOKENS = 16000             # mapped to max_output_tokens
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
REASONING_EFFORT = "low"          # for gpt-5.x models

# behavioral flags
VERBOSE = False
WRITE_INTERMEDIATE = True

# ---------------------- Helpers: OpenAI key loader ---------------------
def _load_openai_key(envvar: str = OPENAI_API_KEY_ENV, fallback_path: str = ".env"):
    key = os.getenv(envvar, None)
    if key:
        return key
    # fallback: try file
    if Path(fallback_path).exists():
        txt = Path(fallback_path).read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None

OPENAI_KEY = _load_openai_key()
if OpenAI is not None and OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None
    if VERBOSE:
        print("⚠️ OpenAI client not initialized (missing package or API key). LLM calls will fail unless OpenAI client is available.")

def call_llm(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = OPENAI_TEMPERATURE,  # unused for gpt-5.1; kept for compatibility
    max_tokens: int = LLM_MAX_TOKENS,
    reasoning_effort: str = REASONING_EFFORT,
) -> str:
    """
    Call the LLM using the Responses API (compatible with gpt-5.1).

    - Uses `max_output_tokens` instead of `max_tokens`.
    - Does NOT send `temperature` (unsupported by gpt-5.1).
    - Adds `reasoning={'effort': reasoning_effort}` for gpt-5.x models.
    """
    if client is None:
        raise RuntimeError("OpenAI client not available. Set OPENAI_API_KEY and install openai package.")
    try:
        if VERBOSE:
            print(f"[call_llm] model={model} max_output_tokens={max_tokens} reasoning_effort={reasoning_effort}")
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "user", "content": prompt}
            ],
        }
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        if model.startswith("gpt-5") and reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        resp = client.responses.create(**kwargs)
        txt = resp.output_text or ""
        status = getattr(resp, "status", None)
        if status == "incomplete":
            incomplete_details = getattr(resp, "incomplete_details", None)
            print(f"[call_llm] WARNING: response incomplete: {incomplete_details}")
        return txt
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ---------------------- HF Embedder (same style as ClassRec) -------------
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE:
            print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        if len(texts) == 0:
            D = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 return_tensors="pt", max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers -------------------------------------
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def safe_str(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).replace("\n", " ").strip()

def compact_member_info(member: Dict) -> Dict:
    # Only pass id, name, desc, entity_type_hint to LLM prompt
    return {
        "id": member.get("id"),
        "entity_name": safe_str(member.get("entity_name", ""))[:180],
        "entity_description": safe_str(member.get("entity_description", ""))[:400],
        "entity_type_hint": safe_str(member.get("entity_type_hint", ""))[:80]
    }

# ---------------------- Build class texts & embeddings ------------------
def build_class_texts(classes: List[Dict]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    labels, descs, types, evids, members_agg = [], [], [], [], []
    for c in classes:
        labels.append(safe_str(c.get("class_label", ""))[:120])
        descs.append(safe_str(c.get("class_description", ""))[:300])
        types.append(safe_str(c.get("class_type_hint", ""))[:80])
        evids.append(safe_str(c.get("evidence_excerpt", ""))[:200])
        mems = c.get("members", []) or []
        mem_texts = []
        for m in mems:
            name = safe_str(m.get("entity_name", ""))
            desc = safe_str(m.get("entity_description", ""))
            etype = safe_str(m.get("entity_type_hint", ""))
            mem_texts.append(f"{name} ({etype}) - {desc[:120]}")
        members_agg.append(" ; ".join(mem_texts)[:1000])
    return labels, descs, types, evids, members_agg

def compute_class_embeddings(embedder: HFEmbedder, classes: List[Dict], weights: Dict[str, float]) -> np.ndarray:
    labels, descs, types, evids, members_agg = build_class_texts(classes)
    emb_label = embedder.encode_batch(labels) if any(t.strip() for t in labels) else None
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None
    emb_type = embedder.encode_batch(types) if any(t.strip() for t in types) else None
    emb_evid = embedder.encode_batch(evids) if any(t.strip() for t in evids) else None
    emb_mem = embedder.encode_batch(members_agg) if any(t.strip() for t in members_agg) else None

    # determine D
    D = None
    for arr in (emb_label, emb_desc, emb_type, emb_evid, emb_mem):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]
            break
    if D is None:
        raise ValueError("No textual fields produced embeddings for classes")

    def ensure(arr):
        if arr is None:
            return np.zeros((len(classes), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_label = ensure(emb_label)
    emb_desc = ensure(emb_desc)
    emb_type = ensure(emb_type)
    emb_evid = ensure(emb_evid)
    emb_mem = ensure(emb_mem)

    w_label = weights.get("label", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_type = weights.get("type_hint", 0.0)
    w_evid = weights.get("evidence", 0.0)
    w_mem = weights.get("members", 0.0)
    W = w_label + w_desc + w_type + w_evid + w_mem
    if W <= 0:
        raise ValueError("invalid class emb weights")
    w_label /= W
    w_desc /= W
    w_type /= W
    w_evid /= W
    w_mem /= W

    combined = (
        w_label * emb_label
        + w_desc * emb_desc
        + w_type * emb_type
        + w_evid * emb_evid
        + w_mem * emb_mem
    )
    combined = normalize(combined, axis=1)
    return combined

# ---------------------- clustering -------------------------------------
def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    use_umap: bool = USE_UMAP
) -> Tuple[np.ndarray, object]:
    X = embeddings
    N = X.shape[0]
    # Decide whether to attempt UMAP
    if use_umap and UMAP_AVAILABLE and N >= 6:
        safe_n_components = min(UMAP_N_COMPONENTS, max(2, N - 2))
        safe_n_neighbors = min(UMAP_N_NEIGHBORS, max(2, N - 1))
        try:
            reducer = umap.UMAP(
                n_components=safe_n_components,
                n_neighbors=safe_n_neighbors,
                min_dist=UMAP_MIN_DIST,
                metric="cosine",
                random_state=42
            )
            X_reduced = reducer.fit_transform(X)
            if X_reduced is not None and X_reduced.shape[0] == N:
                X = X_reduced
            else:
                if VERBOSE:
                    print(f"[warn] UMAP returned invalid shape {None if X_reduced is None else X_reduced.shape}; skipping UMAP")
        except Exception as e:
            if VERBOSE:
                print(f"[warn] UMAP failed (N={N}, n_comp={safe_n_components}, n_nei={safe_n_neighbors}): {e}. Proceeding without UMAP.")
            X = embeddings
    else:
        if use_umap and UMAP_AVAILABLE and VERBOSE:
            print(f"[info] Skipping UMAP (N={N} < 6) to avoid unstable spectral computations.")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer


# ---------------------- LLM prompt template (extended) ------------------
CLSRES_PROMPT_TEMPLATE = """
You are a very proactive class-resolution assistant.
You are given a set of candidate CLASSES that appear to belong, or may plausibly belong, to the same semantic cluster.
Your job is to produce a clear, *actionable* ordered list of schema edits for the given cluster.
Do NOT be passive. If the evidence supports structural change (merge / split / reassign / create), you MUST propose it.

Only return [] if there is extremely strong evidence that NO change is needed (rare).

The cluster grouping you are given is only *suggestive* and may be incorrect — your job is to resolve, correct, and produce a coherent schema from the evidence.

This is an iterative process — act now with well-justified structural corrections (include a short justification), rather than deferring small but meaningful fixes.
This step will be repeated in later iterations; reasonable but imperfect changes can be corrected later.
It is worse to miss a necessary change than to propose a well-justified change that might be slightly adjusted later.

Your CRUCIAL task is to refine the schema using this tentative grouping.

========================
SCHEMA STRUCTURE (CRITICAL)
========================

We are building a TWO-LAYER SCHEMA over entities:

Level 0: Class_Group        (connects related classes)
Level 1: Classes            (group entities)
Level 2: Entities

Structure:
Class_Group
  └── Class
        └── Entity

- Class_Group is the PRIMARY mechanism for connecting related classes.
- Classes that share a Class_Group are considered semantically related.
- This relationship propagates to their entities.

========================
IMPORTANT FIELD DISTINCTIONS
========================

- class_type_hint (existing field):
  A local, descriptive hint assigned to each class in isolation.
  It is often noisy, incomplete, and inconsistent across classes.
  Do NOT assume it is globally correct or reusable.

- class_label:
  Existing class_label values are PROVISIONAL names.
  You MAY revise them when entity evidence suggests a clearer or a better canonical label.

- Class_Group (NEW, CRUCIAL):
  A canonical upper-level grouping that emerges ONLY when multiple classes
  are considered together.
  It is used to connect related classes into a coherent schema.
  Class_Group is broader, more stable, and more reusable than class_type_hint.

- remarks (optional, internal):
  Free-text notes attached to a class, used to flag important issues that are
  outside the scope of structural changes (e.g., upstream entity resolution that you suspect is wrong, or entities that
  look identical but should not be changed here), DO NOT try to fix them via merge or reassign.
   Instead, attach a human-facing remark via modify_class (using the 'remark' field)
   so that a human can review it later.

Class_Group is NOT a synonym of class_type_hint.

========================
YOUR PRIMARY TASK
========================

Note: the provided cluster grouping is tentative and may be wrong — 
you must correct it as needed to produce a coherent Class_Group → Class → Entity schema.

For the given cluster of classes:

1) Assess whether any structural changes are REQUIRED:
   - merge duplicate or near-duplicate classes
   - reassign clearly mis-assigned entities
   - create a new class when necessary
   - split an overloaded class into more coherent subclasses when justified
   - modify class metadata when meaningfully incorrect

2) ALWAYS assess and assign an appropriate Class_Group:
   - If Class_Group is missing, null, or marked as TBD → you MUST assign it.
   - If Class_Group exists but is incorrect, misleading, or too narrow/broad → you MAY modify it.
   - If everything else is correct (which is not the case most of the time), assigning or confirming Class_Group ALONE is sufficient.

3) If you notice important issues that are OUTSIDE the scope of these functions
   (e.g., upstream entity resolution that you suspect is wrong, or entities that
   look identical but should not be changed here), DO NOT try to fix them via merge or reassign.
   Instead, attach a human-facing remark via modify_class (using the 'remark' field)
   so that a human can review it later.

========================
SOME CONSERVATISM RULES (They should not make you passive)
========================

- Always try to attempt structural proposals (merge/split/reassign/create) unless the cluster truly is already optimal.
- Do NOT perform cosmetic edits or unnecessary normalization.

You MAY perform multiple structural actions in one cluster
(e.g., merge + rename + reassign + split), when needed.

========================
MERGING & OVERLAP HEURISTICS
========================

- Entity overlap alone does not automatically require merging.
- Merge when evidence indicates the SAME underlying concepts
  (e.g., near-identical semantics, interchangeable usage, or redundant distinctions).
- Reassignment is appropriate when overlap reveals mis-typed or mis-scoped entities,
  even if classes should remain separate.

Quick heuristic:
- Same concept → merge.
- Different concept, same domain → keep separate classes with the SAME Class_Group.
- Different domain → different Class_Group.

Avoid vague Class_Group names (e.g., "Misc", "General", "Other").
Prefer domain-meaningful groupings that help connect related classes.

If a class should be collapsed or weakened but has no clear merge partner, DO NOT use merge_classes;
instead, reassign its entities to better classes and/or add a remark for human review.

IMPORTANT:
- You MUST NOT call merge_classes with only one class_id.
- merge_classes is ONLY for merging TWO OR MORE existing classes into ONE new class.
- If you only want to update or clarify a single class (for example, its label, description,
  type hint, class_group, or remarks), use modify_class instead.
- Do NOT use merge_classes to clean up or deduplicate entities inside one class.

========================
AVAILABLE FUNCTIONS
========================

Return ONLY a JSON ARRAY of ordered function calls.

Each object must have:
- "function": one of
  ["merge_classes", "create_class", "reassign_entities", "modify_class", "split_class"]
- "args": arguments as defined below.

ID HANDLING RULES
- You MUST NOT invent real class IDs.
- You MUST use ONLY class_ids that appear in the input CLASSES (candidate_id values),
  except when referring to newly merged/created/split classes.
- When you need to refer to a newly merged/created/split class in later steps,
  you MUST assign a provisional_id (any consistent string).
- Use the same provisional_id whenever referencing that new class again.
- After you merge classes into a new class, you should NOT continue to treat the original
  class_ids as separate entities; refer to the new merged class via its provisional_id.

Example (pattern, not required verbatim):

{
  "function": "merge_classes",
  "args": {
    "class_ids": ["ClsC_da991b68", "ClsC_e32f4a47"],
    "provisional_id": "MERGE(ClsC_da991b68|ClsC_e32f4a47)",
    "new_name": "...",
    "new_description": "...",
    "new_class_type_hint": "Standard",
    "justification": "One-line reason citing entity overlap and semantic equivalence.",
    "remark": null,
    "confidence": 0.95
  }
}

Later:

{
  "function": "reassign_entities",
  "args": {
    "entity_ids": ["En_xxx"],
    "from_class_id": "ClsC_e32f4a47",
    "to_class_id": "MERGE(ClsC_da991b68|ClsC_e32f4a47)",
    "justification": "Why this entity fits better in the merged class.",
    "remark": null,
    "confidence": 0.9
  }
}

We will internally map provisional_id → real class id.

------------------------
Function definitions
------------------------

JUSTIFICATION REQUIREMENT
- Every function call MUST include:
    "justification": "<one-line reason>"
  explaining why the action is necessary.
- This justification should cite concrete evidence (entity overlap, conflicting descriptions,
  mis-scoped members, missing Class_Group, overloaded classes, etc.).
- You MAY also include "confidence": <0.0–1.0> to indicate your belief in the action.
- You MAY include "remark" to provide a short human-facing note.

1) merge_classes
args = {
  "class_ids": [<existing_class_ids>],   # MUST contain at least 2 valid ids
  "provisional_id": <string or null>,    # how you will refer to the new class later
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

2) create_class
args = {
  "name": <string>,
  "description": <string or null>,
  "class_type_hint": <string or null>,
  "member_ids": [<entity_ids>],          # optional, must be from provided entities
  "provisional_id": <string or null>,    # how you will refer to this new class later
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

3) reassign_entities
args = {
  "entity_ids": [<entity_ids>],
  "from_class_id": <existing_class_id or provisional_id or null>,
  "to_class_id": <existing_class_id or provisional_id>,
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

4) modify_class
args = {
  "class_id": <existing_class_id or provisional_id>,
  "new_name": <string or null>,
  "new_description": <string or null>,
  "new_class_type_hint": <string or null>,
  "new_class_group": <string or null>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "justification": <string>,
  "confidence": <number between 0 and 1, optional>
}

- Use modify_class with 'remark' (and no structural change) when you want to flag
  issues that are outside the scope of this step (e.g., suspected entity-level duplicates).

5) split_class
args = {
  "source_class_id": <existing_class_id or provisional_id>,
  "splits": [
    {
      "name": <string or null>,
      "description": <string or null>,
      "class_type_hint": <string or null>,
      "member_ids": [<entity_ids>],      # must be from source_class member_ids
      "provisional_id": <string or null> # how you will refer to this new split class
    },
    ...
  ],
  "justification": <string>,
  "remark": <string or null>,            # optional: attach a human-facing remark/flag
  "confidence": <number between 0 and 1, optional>
}

Semantics of split_class:
- Use split_class when a single class is overloaded and should be divided into
  narrower, more coherent classes.
- Entities listed in splits[*].member_ids MUST come from the source_class's member_ids.
- The specified entities are REMOVED from the source_class and grouped into new classes.
- Any members not mentioned in any split remain in the source_class.

NOTE:
- Class_Group is normally set or updated via modify_class.
- Assigning or confirming Class_Group is also REQUIRED for every cluster unless it is already clearly correct.

========================
VALIDATION RULES
========================

- Use ONLY provided entity_ids and class_ids (candidate_id values) for existing classes.
- For new classes, use provisional_id handles and be consistent.
- Order matters: later steps may depend on earlier ones.
- merge_classes with fewer than 2 valid class_ids will be ignored.

========================
STRATEGY GUIDANCE
========================

- Prefer assigning/adjusting Class_Group over heavy structural changes when both are valid.
- Merge classes ONLY when they are genuinely redundant (same concept).
- Use split_class when a class clearly bundles multiple distinct concepts that should be separated.
- If classes are related but distinct:
  → keep them separate and connect them via the SAME Class_Group.
- Think in terms of schema connectivity and meaningful structure, not cosmetic cleanup.
- If in doubt about structural change but you see a potential issue, use modify_class with a 'remark'
  rather than forcing an uncertain structural edit.

========================
INPUT CLASSES
========================

Each class includes:
- candidate_id
- class_label
- class_description
- class_type_hint
- class_group (may be null or "TBD")
- confidence
- evidence_excerpt
- member_ids
- members (entity id, name, description, type)
- remarks (optional list of prior remarks)

{cluster_block}

========================
OUTPUT
========================

Return ONLY the JSON array of ordered function calls.
Return [] only if you are highly confident (> very strong evidence) that no change is needed.
If any ambiguous or conflicting evidence exists, return a concrete ordered action list
(with justifications and, optionally, confidence scores).
You are a very proactive class-resolution assistant. You are not called very proactive only by using modify_class for new_class_group. You must use other functions as well to be called a proactive class-resolution assistant.

"""

def sanitize_json_like(text: str) -> Optional[Any]:
    # crude sanitizer: extract first [...] region and try loads. Fix common trailing commas and smart quotes.
    if not text or not text.strip():
        return None
    s = text.strip()
    # replace smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # find first [ ... ] block
    start = s.find("[")
    end = s.rfind("]")
    cand = s
    if start != -1 and end != -1 and end > start:
        cand = s[start:end + 1]
    # remove trailing commas before closing braces/brackets
    cand = re.sub(r",\s*([\]}])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        return None

# ---------------------- Action executors --------------------------------
def execute_merge_classes(
    all_classes: Dict[str, Dict],
    class_ids: List[str],
    new_name: Optional[str],
    new_desc: Optional[str],
    new_type: Optional[str]
) -> str:
    # validate class ids
    class_ids = [cid for cid in class_ids if cid in all_classes]
    if len(class_ids) < 2:
        raise ValueError("merge_classes: need at least 2 valid class_ids")
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    members_map: Dict[str, Dict] = {}
    confidence = 0.0
    evidence = ""
    desc_choice = None
    type_choice = new_type or ""
    class_group_choice = None

    for cid in class_ids:
        c = all_classes[cid]
        confidence = max(confidence, float(c.get("confidence", 0.0)))
        if c.get("evidence_excerpt") and not evidence:
            evidence = c.get("evidence_excerpt")
        if c.get("class_description") and desc_choice is None:
            desc_choice = c.get("class_description")
        cg = c.get("class_group")
        if cg and cg not in ("", "TBD", None):
            if class_group_choice is None:
                class_group_choice = cg
            elif class_group_choice != cg:
                # conflicting groups -> keep first; fixable later by modify_class
                class_group_choice = class_group_choice
        for m in c.get("members", []):
            members_map[m["id"]] = m

    if new_desc:
        desc_choice = new_desc
    new_label = new_name or all_classes[class_ids[0]].get("class_label", "MergedClass")
    if not new_type:
        for cid in class_ids:
            if all_classes[cid].get("class_type_hint"):
                type_choice = all_classes[cid].get("class_type_hint")
                break

    merged_obj = {
        "candidate_id": new_cid,
        "class_label": new_label,
        "class_description": desc_choice or "",
        "class_type_hint": type_choice or "",
        "class_group": class_group_choice or "TBD",
        "confidence": float(confidence),
        "evidence_excerpt": evidence or "",
        "member_ids": list(members_map.keys()),
        "members": list(members_map.values()),
        "candidate_ids": class_ids,
        "merged_from": class_ids,
        "remarks": [],
        "_merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    for cid in class_ids:
        all_classes.pop(cid, None)
    all_classes[new_cid] = merged_obj
    return new_cid

def execute_create_class(
    all_classes: Dict[str, Dict],
    name: str,
    description: Optional[str],
    class_type_hint: Optional[str],
    member_ids: Optional[List[str]],
    id_to_entity: Dict[str, Dict]
) -> str:
    if not name:
        raise ValueError("create_class: 'name' is required")
    new_cid = "ClsR_" + uuid.uuid4().hex[:8]
    members = []
    mids = member_ids or []
    for mid in mids:
        ent = id_to_entity.get(mid)
        if ent:
            members.append(ent)
    obj = {
        "candidate_id": new_cid,
        "class_label": name,
        "class_description": description or "",
        "class_type_hint": class_type_hint or "",
        "class_group": "TBD",
        "confidence": 0.5,
        "evidence_excerpt": "",
        "member_ids": [m["id"] for m in members],
        "members": members,
        "candidate_ids": [],
        "remarks": [],
        "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    all_classes[new_cid] = obj
    return new_cid

def execute_reassign_entities(
    all_classes: Dict[str, Dict],
    entity_ids: List[str],
    from_class_id: Optional[str],
    to_class_id: str,
    id_to_entity: Dict[str, Dict]
):
    # remove from source(s)
    for cid, c in list(all_classes.items()):
        if from_class_id and cid != from_class_id:
            continue
        new_members = [m for m in c.get("members", []) if m["id"] not in set(entity_ids)]
        new_member_ids = [m["id"] for m in new_members]
        c["members"] = new_members
        c["member_ids"] = new_member_ids
        all_classes[cid] = c
    # add to destination
    if to_class_id not in all_classes:
        raise ValueError(f"reassign_entities: to_class_id {to_class_id} not found")
    dest = all_classes[to_class_id]
    existing = {m["id"] for m in dest.get("members", [])}
    for eid in entity_ids:
        if eid in existing:
            continue
        ent = id_to_entity.get(eid)
        if ent:
            dest.setdefault("members", []).append(ent)
            dest.setdefault("member_ids", []).append(eid)
    dest["confidence"] = max(dest.get("confidence", 0.0), 0.4)
    all_classes[to_class_id] = dest

def execute_modify_class(
    all_classes: Dict[str, Dict],
    class_id: str,
    new_name: Optional[str],
    new_desc: Optional[str],
    new_type: Optional[str],
    new_class_group: Optional[str],
    new_remark: Optional[str]
):
    if class_id not in all_classes:
        raise ValueError(f"modify_class: class_id {class_id} not found")
    c = all_classes[class_id]
    if new_name:
        c["class_label"] = new_name
    if new_desc:
        c["class_description"] = new_desc
    if new_type:
        c["class_type_hint"] = new_type
    if new_class_group:
        c["class_group"] = new_class_group
    if new_remark:
        existing = c.get("remarks")
        if existing is None:
            existing = []
        elif not isinstance(existing, list):
            existing = [str(existing)]
        existing.append(str(new_remark))
        c["remarks"] = existing
    all_classes[class_id] = c

def execute_split_class(
    all_classes: Dict[str, Dict],
    source_class_id: str,
    splits_specs: List[Dict[str, Any]]
) -> List[Tuple[str, Optional[str], List[str]]]:
    """
    Split a source class into several new classes.
    Returns list of (new_class_id, provisional_id, member_ids_used) for each created class.
    """
    if source_class_id not in all_classes:
        raise ValueError(f"split_class: source_class_id {source_class_id} not found")
    src = all_classes[source_class_id]
    src_members = src.get("members", []) or []
    src_member_map = {m["id"]: m for m in src_members if isinstance(m, dict) and m.get("id")}
    used_ids: set = set()
    created: List[Tuple[str, Optional[str], List[str]]] = []

    for spec in splits_specs:
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        desc = spec.get("description")
        th = spec.get("class_type_hint") or src.get("class_type_hint", "")
        mids_raw = spec.get("member_ids", []) or []
        prov_id = spec.get("provisional_id")
        # use only members that belong to source and are not already used
        valid_mids = []
        for mid in mids_raw:
            if mid in src_member_map and mid not in used_ids:
                valid_mids.append(mid)
        if not valid_mids:
            continue
        members = [src_member_map[mid] for mid in valid_mids]
        new_cid = "ClsR_" + uuid.uuid4().hex[:8]
        obj = {
            "candidate_id": new_cid,
            "class_label": name or src.get("class_label", "SplitClass"),
            "class_description": desc or src.get("class_description", ""),
            "class_type_hint": th or src.get("class_type_hint", ""),
            "class_group": src.get("class_group", "TBD"),
            "confidence": src.get("confidence", 0.5),
            "evidence_excerpt": src.get("evidence_excerpt", ""),
            "member_ids": valid_mids,
            "members": members,
            "candidate_ids": [],
            "remarks": [],
            "_split_from": source_class_id,
            "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        all_classes[new_cid] = obj
        used_ids.update(valid_mids)
        created.append((new_cid, prov_id, valid_mids))

    if used_ids:
        remaining_members = [m for m in src_members if m["id"] not in used_ids]
        src["members"] = remaining_members
        src["member_ids"] = [m["id"] for m in remaining_members]
        all_classes[source_class_id] = src

    return created

# helper to resolve real class ID (handles provisional IDs and retired IDs)
def resolve_class_id(
    raw_id: Optional[str],
    all_classes: Dict[str, Dict],
    provisional_to_real: Dict[str, str],
    allow_missing: bool = False
) -> Optional[str]:
    if raw_id is None:
        return None
    real = provisional_to_real.get(raw_id, raw_id)
    if real not in all_classes and not allow_missing:
        raise ValueError(f"resolve_class_id: {raw_id} (resolved to {real}) not found in all_classes")
    return real

# ---------------------- Main orchestration ------------------------------
def classres_main():
    # load classes
    if not INPUT_CLASSES.exists():
        raise FileNotFoundError(f"Input classes file not found: {INPUT_CLASSES}")
    classes_list = load_json(INPUT_CLASSES)
    print(f"[start] loaded {len(classes_list)} merged candidate classes from {INPUT_CLASSES}")

    # build id->entity map (from members)
    id_to_entity: Dict[str, Dict] = {}
    for c in classes_list:
        for m in c.get("members", []):
            if isinstance(m, dict) and m.get("id"):
                id_to_entity[m["id"]] = m

    # ensure classes have candidate_id keys, class_group field, and remarks field
    all_classes: Dict[str, Dict] = {}
    for c in classes_list:
        cid = c.get("candidate_id") or ("ClsC_" + uuid.uuid4().hex[:8])
        members = c.get("members", []) or []
        mids = [m["id"] for m in members if isinstance(m, dict) and m.get("id")]
        c["member_ids"] = mids
        c["members"] = members
        if "class_group" not in c or c.get("class_group") in (None, ""):
            c["class_group"] = "TBD"
        # normalize remarks
        if "remarks" not in c or c["remarks"] is None:
            c["remarks"] = []
        elif not isinstance(c["remarks"], list):
            c["remarks"] = [str(c["remarks"])]
        c["candidate_id"] = cid
        all_classes[cid] = c

    # embedder
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    class_objs = list(all_classes.values())
    class_ids_order = list(all_classes.keys())
    combined_emb = compute_class_embeddings(embedder, class_objs, CLASS_EMB_WEIGHTS)
    print("[info] class embeddings computed shape:", combined_emb.shape)

    # clustering
    labels, clusterer = run_hdbscan(
        combined_emb,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        use_umap=USE_UMAP
    )
    print("[info] clustering done. unique labels:", set(labels))

    # map cluster -> class ids
    cluster_to_classids: Dict[int, List[str]] = {}
    for idx, lab in enumerate(labels):
        cid = class_ids_order[idx]
        cluster_to_classids.setdefault(int(lab), []).append(cid)

    # prepare action log
    action_log_path = OUT_DIR / "cls_res_action_log.jsonl"
    if action_log_path.exists():
        action_log_path.unlink()

    # iterate clusters (skip -1 initially)
    cluster_keys = sorted([k for k in cluster_to_classids.keys() if k != -1])
    cluster_keys += [-1]  # append noise at end

    for cluster_label in cluster_keys:
        class_ids = cluster_to_classids.get(cluster_label, [])
        if not class_ids:
            continue
        print(f"[cluster] {cluster_label} -> {len(class_ids)} classes")

        # build cluster block to pass to LLM
        cluster_classes = []
        for cid in class_ids:
            c = all_classes.get(cid)
            if not c:
                continue
            members_compact = [compact_member_info(m) for m in c.get("members", [])]
            cluster_classes.append({
                "candidate_id": cid,
                "class_label": c.get("class_label", ""),
                "class_description": c.get("class_description", ""),
                "class_type_hint": c.get("class_type_hint", ""),
                "class_group": c.get("class_group", "TBD"),
                "confidence": float(c.get("confidence", 0.0)),
                "evidence_excerpt": c.get("evidence_excerpt", ""),
                "member_ids": c.get("member_ids", []),
                "members": members_compact,
                "remarks": c.get("remarks", [])
            })

        cluster_block = json.dumps(cluster_classes, ensure_ascii=False, indent=2)
        prompt = CLSRES_PROMPT_TEMPLATE.replace("{cluster_block}", cluster_block)

        # log prompt
        prompt_path = RAW_LLM_DIR / f"cluster_{cluster_label}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # call LLM
        raw_out = ""
        try:
            raw_out = call_llm(prompt)
        except Exception as e:
            print(f"[warning] LLM call failed for cluster {cluster_label}: {e}")
            raw_out = ""

        # write raw output
        raw_path = RAW_LLM_DIR / f"cluster_{cluster_label}_llm_raw.txt"
        raw_path.write_text(raw_out, encoding="utf-8")

        # try parse/sanitize
        parsed = sanitize_json_like(raw_out)
        if parsed is None:
            print(f"[warn] failed to parse LLM output for cluster {cluster_label}; skipping automated actions for this cluster.")
            dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
            dec_path.write_text(
                json.dumps({"cluster_label": cluster_label, "raw_llm": raw_out}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            continue

        # mapping from provisional_id (and retired ids) -> real class id (per cluster)
        provisional_to_real: Dict[str, str] = {}
        decisions: List[Dict[str, Any]] = []

        # execute parsed function list in order
        for step in parsed:
            if not isinstance(step, dict):
                continue
            fn = step.get("function")
            args = step.get("args", {}) or {}

            justification = args.get("justification")
            confidence_val = args.get("confidence", None)
            remark_val = args.get("remark")

            try:
                if fn == "merge_classes":
                    cids_raw = args.get("class_ids", []) or []
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    prov_id = args.get("provisional_id")

                    # resolve any potential provisional IDs in class_ids
                    cids_real = [provisional_to_real.get(cid, cid) for cid in cids_raw]
                    valid_cids = [cid for cid in cids_real if cid in all_classes]

                    if len(valid_cids) < 2:
                        decisions.append({
                            "action": "merge_skip_too_few",
                            "requested_class_ids": cids_raw,
                            "valid_class_ids": valid_cids,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val
                        })
                        continue

                    new_cid = execute_merge_classes(all_classes, valid_cids, new_name, new_desc, new_type)

                    # map provisional id -> new class id
                    if prov_id:
                        provisional_to_real[prov_id] = new_cid
                    # also map old real ids -> new id so later references can still resolve
                    for old in valid_cids:
                        provisional_to_real.setdefault(old, new_cid)

                    decisions.append({
                        "action": "merge_classes",
                        "input_class_ids": valid_cids,
                        "result_class_id": new_cid,
                        "provisional_id": prov_id,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                elif fn == "create_class":
                    name = args.get("name")
                    desc = args.get("description")
                    t = args.get("class_type_hint")
                    mids = args.get("member_ids", []) or []
                    prov_id = args.get("provisional_id")

                    mids_valid = [m for m in mids if m in id_to_entity]
                    new_cid = execute_create_class(all_classes, name, desc, t, mids_valid, id_to_entity)

                    if prov_id:
                        provisional_to_real[prov_id] = new_cid

                    decisions.append({
                        "action": "create_class",
                        "result_class_id": new_cid,
                        "provisional_id": prov_id,
                        "member_ids_added": mids_valid,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                elif fn == "reassign_entities":
                    eids = args.get("entity_ids", []) or []
                    from_c_raw = args.get("from_class_id")
                    to_c_raw = args.get("to_class_id")

                    eids_valid = [e for e in eids if e in id_to_entity]

                    from_c = resolve_class_id(from_c_raw, all_classes, provisional_to_real, allow_missing=True)
                    to_c = resolve_class_id(to_c_raw, all_classes, provisional_to_real, allow_missing=False)

                    execute_reassign_entities(all_classes, eids_valid, from_c, to_c, id_to_entity)

                    decisions.append({
                        "action": "reassign_entities",
                        "entity_ids": eids_valid,
                        "from": from_c_raw,
                        "from_resolved": from_c,
                        "to": to_c_raw,
                        "to_resolved": to_c,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                elif fn == "modify_class":
                    cid_raw = args.get("class_id")
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    new_group = args.get("new_class_group")
                    new_remark = args.get("remark")

                    cid_real = resolve_class_id(cid_raw, all_classes, provisional_to_real, allow_missing=False)
                    execute_modify_class(all_classes, cid_real, new_name, new_desc, new_type, new_group, new_remark)

                    decisions.append({
                        "action": "modify_class",
                        "class_id": cid_raw,
                        "class_id_resolved": cid_real,
                        "new_name": new_name,
                        "new_description": new_desc,
                        "new_class_type_hint": new_type,
                        "new_class_group": new_group,
                        "remark": new_remark,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                elif fn == "split_class":
                    source_raw = args.get("source_class_id")
                    splits_specs = args.get("splits", []) or []

                    source_real = resolve_class_id(source_raw, all_classes, provisional_to_real, allow_missing=False)
                    created = execute_split_class(all_classes, source_real, splits_specs)

                    created_summary = []
                    for new_cid, prov_id, mids_used in created:
                        if prov_id:
                            provisional_to_real[prov_id] = new_cid
                        created_summary.append({
                            "new_class_id": new_cid,
                            "provisional_id": prov_id,
                            "member_ids": mids_used
                        })

                    decisions.append({
                        "action": "split_class",
                        "source_class_id": source_raw,
                        "source_class_id_resolved": source_real,
                        "created_classes": created_summary,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                else:
                    decisions.append({
                        "action": "skip_unknown_function",
                        "raw": step,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

            except Exception as e:
                decisions.append({
                    "action": "error_executing",
                    "function": fn,
                    "error": str(e),
                    "input": step,
                    "justification": justification,
                    "remark": remark_val,
                    "confidence": confidence_val
                })

        # write decisions file for this cluster
        dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
        dec_obj = {
            "cluster_label": cluster_label,
            "cluster_classes": cluster_classes,
            "llm_raw": raw_out,
            "parsed_steps": parsed,
            "executed_decisions": decisions,
            "provisional_to_real": provisional_to_real,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        dec_path.write_text(json.dumps(dec_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # append to action log
        with open(action_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dec_obj, ensure_ascii=False) + "\n")

    # After all clusters processed: write final classes output
    final_classes = list(all_classes.values())
    out_json = OUT_DIR / "final_classes_resolved.json"
    out_jsonl = OUT_DIR / "final_classes_resolved.jsonl"
    out_json.write_text(json.dumps(final_classes, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for c in final_classes:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[done] wrote final resolved classes -> {out_json}  (count={len(final_classes)})")
    print(f"[done] action log -> {action_log_path}")

    # ---------------------- SUMMARY AGGREGATION -------------------------
    summary_dir = OUT_DIR / "summary"
    summary_dir.mkdir(exist_ok=True)

    # Aggregate all per-cluster decision files
    cluster_decisions: List[Dict[str, Any]] = []
    for path in sorted(OUT_DIR.glob("cluster_*_decisions.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            cluster_decisions.append(obj)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}")

    # Write aggregated decisions (same per-cluster structure, just as a list)
    all_clusters_decisions_path = summary_dir / "all_clusters_decisions.json"
    all_clusters_decisions_path.write_text(
        json.dumps(cluster_decisions, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Compute statistics
    total_clusters = len(cluster_decisions)
    actions_by_type: Dict[str, int] = {}
    total_remarks = 0
    clusters_with_any_decisions = 0
    clusters_with_structural = 0
    clusters_only_classgroup = 0

    structural_actions = {"merge_classes", "create_class", "reassign_entities", "split_class"}

    for cd in cluster_decisions:
        decs = cd.get("executed_decisions", [])
        if not decs:
            continue
        clusters_with_any_decisions += 1
        has_structural = False
        only_classgroup = True

        for d in decs:
            act = d.get("action")
            actions_by_type[act] = actions_by_type.get(act, 0) + 1

            # count remarks if present
            rem = d.get("remark")
            if rem:
                total_remarks += 1

            if act in structural_actions:
                has_structural = True
                only_classgroup = False
            elif act == "modify_class":
                new_group = d.get("new_class_group")
                new_name = d.get("new_name")
                new_desc = d.get("new_description")
                new_type = d.get("new_class_type_hint")
                # "only class group" means the only change is class_group (no name/desc/type/remark change)
                # remark alone is allowed
                if not new_group or new_name or new_desc or new_type:
                    only_classgroup = False
            else:
                # merge_skip_too_few, error_executing, skip_unknown_function, etc.
                only_classgroup = False

        if has_structural:
            clusters_with_structural += 1
        if only_classgroup:
            clusters_only_classgroup += 1

    total_structural_actions = sum(
        count for act, count in actions_by_type.items() if act in structural_actions
    )
    total_errors = actions_by_type.get("error_executing", 0)

    stats = {
        "total_clusters": total_clusters,
        "total_clusters_with_any_decisions": clusters_with_any_decisions,
        "total_clusters_with_structural_changes": clusters_with_structural,
        "total_clusters_only_class_group_updates": clusters_only_classgroup,
        "total_actions_by_type": actions_by_type,
        "total_structural_actions": total_structural_actions,
        "total_errors": total_errors,
        "total_remarks_logged": total_remarks,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    stats_path = summary_dir / "stats_summary.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] summary decisions -> {all_clusters_decisions_path}")
    print(f"[done] summary stats -> {stats_path}")




# # -----------------------
# # Cls Res  - Run statement
# # -----------------------

# if __name__ == "__main__":
#     classres_main()

#endregion#? Cls Res V8  - Split + Remark + Summary (gpt-5.1 Responses)
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Multi Run Cls Res

# =========================
# Iterative runner for ClassRes (UPDATED final-summary exports)
# Paste this into the same file after V7 (replaces prior runner's final summary block)
# =========================

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------
# CONFIG - reuse or override
# -----------------------
BASE_INPUT_CLASSES = Path("SGCE-KG/data/Classes/Cls_Res/Cls_Res_input/classes_for_cls_res.json")
EXPERIMENT_ROOT = Path("SGCE-KG/data/Classes/Cls_Res/Cls_Res_IterativeRuns")

MAX_RUNS: int = 4
STRUCTURAL_CHANGE_THRESHOLD: Optional[int] = 0
TOTAL_ACTIONS_THRESHOLD: Optional[int] = None
MAX_NO_CHANGE_RUNS: Optional[int] = 2

FINAL_CLASSES_FILENAME = "final_classes_resolved.json"
ACTION_LOG_FILENAME = "cls_res_action_log.jsonl"

# -----------------------
# Helpers (same as before)
# -----------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _safe_json_load_line(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None

def compute_run_summary_from_action_log(action_log_path: Path) -> Dict[str, Any]:
    summary = {
        "total_clusters": 0,
        "total_clusters_with_any_decisions": 0,
        "total_clusters_with_structural_changes": 0,
        "total_clusters_only_class_group_updates": 0,
        "total_actions_by_type": {},
        "total_structural_actions": 0,
        "total_errors": 0,
        "total_remarks_logged": 0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    if not action_log_path.exists():
        return summary

    with action_log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = _safe_json_load_line(line)
            if not obj:
                continue
            summary["total_clusters"] += 1
            executed = obj.get("executed_decisions", []) or []
            if executed:
                summary["total_clusters_with_any_decisions"] += 1

            structural_here = 0
            only_class_group_updates = True
            remarks_here = 0

            for entry in executed:
                action = entry.get("action")
                summary["total_actions_by_type"].setdefault(action, 0)
                summary["total_actions_by_type"][action] += 1

                if action in ("merge_classes", "create_class", "reassign_entities", "split_class"):
                    structural_here += 1
                    only_class_group_updates = False
                elif action == "modify_class":
                    new_name = entry.get("new_name")
                    new_desc = entry.get("new_description")
                    new_type = entry.get("new_class_type_hint")
                    new_group = entry.get("new_class_group")
                    remark = entry.get("remark") or None
                    if remark:
                        remarks_here += 1
                    if any([new_name, new_desc, new_type]):
                        only_class_group_updates = False
                elif action == "error_executing":
                    summary["total_errors"] += 1
                    only_class_group_updates = False
                # count explicit remark fields
                if isinstance(entry, dict) and entry.get("remark"):
                    remarks_here += 1

            summary["total_structural_actions"] += structural_here
            if structural_here > 0:
                summary["total_clusters_with_structural_changes"] += 1
            if only_class_group_updates and executed:
                summary["total_clusters_only_class_group_updates"] += 1
            summary["total_remarks_logged"] += remarks_here

    return summary

# -----------------------
# New: overall export helpers
# -----------------------
def _build_entity_lookup_from_input(input_path: Path) -> Dict[str, Dict]:
    """
    Build map entity_id -> full entity object from the original input classes file.
    This lets us recover fields like chunk_id, resolution_context when final classes may
    only contain compact member info.
    """
    entity_map: Dict[str, Dict] = {}
    if not input_path.exists():
        return entity_map
    try:
        arr = _load_json(input_path)
        for c in arr:
            for m in c.get("members", []) or []:
                if isinstance(m, dict) and m.get("id"):
                    entity_map[m["id"]] = m
    except Exception:
        # best-effort: ignore on error
        pass
    return entity_map

def _aggregate_final_classes_and_entities(final_classes_path: Path, base_input_path: Path) -> Dict[str, Any]:
    """
    Return dict with:
      - classes: final classes array (as in final_classes_resolved.json)
      - entities_map: map entity_id -> merged full entity info (pull fields from classes and original input)
    """
    classes = []
    entities_map: Dict[str, Dict] = {}
    if final_classes_path.exists():
        try:
            classes = _load_json(final_classes_path)
        except Exception:
            classes = []

    # build lookup from original input to backfill entity fields
    input_entity_lookup = _build_entity_lookup_from_input(base_input_path)

    # iterate classes and collect members, merge any available entity fields
    for c in classes:
        members = c.get("members", []) or []
        for m in members:
            mid = m.get("id") if isinstance(m, dict) else None
            if not mid:
                continue
            # start with the member object from final classes (may be compact)
            merged = dict(m) if isinstance(m, dict) else {"id": mid}
            # if original input had richer info, copy missing fields
            original = input_entity_lookup.get(mid)
            if original:
                for k, v in original.items():
                    if k not in merged or merged.get(k) in (None, "", []):
                        merged[k] = v
            # attach class pointers (class id, label, class_group)
            merged["_class_id"] = c.get("candidate_id") or c.get("candidate_id") or None
            merged["_class_label"] = c.get("class_label")
            merged["_class_group"] = c.get("class_group")
            entities_map[mid] = merged

    # also include any entities that were in the input but not present in final classes
    for mid, original in input_entity_lookup.items():
        if mid not in entities_map:
            merged = dict(original)
            merged["_class_id"] = None
            merged["_class_label"] = None
            merged["_class_group"] = None
            entities_map[mid] = merged

    return {"classes": classes, "entities_map": entities_map}

def _write_entities_with_class_jsonl(entities_map: Dict[str, Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for eid, ent in entities_map.items():
            rec = {
                "entity_id": eid,
                "entity": ent,
                "class_id": ent.get("_class_id"),
                "class_label": ent.get("_class_label"),
                "class_group": ent.get("_class_group")
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -----------------------
# Main runner (same loop as before, but calls enhanced final exporter)
# -----------------------
def run_pipeline_iteratively():
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    overall_runs: List[Dict[str, Any]] = []
    no_change_streak = 0
    current_input_path = BASE_INPUT_CLASSES

    for run_idx in range(MAX_RUNS):
        print("\n" + "=" * 36)
        print(f"=== RUN {run_idx:02d} ===")
        print("=" * 36)

        run_dir = EXPERIMENT_ROOT / f"run_{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # set globals used by your pipeline
        globals()["INPUT_CLASSES"] = current_input_path
        globals()["OUT_DIR"] = run_dir
        globals()["RAW_LLM_DIR"] = run_dir / "llm_raw"
        globals()["RAW_LLM_DIR"].mkdir(parents=True, exist_ok=True)

        # call the pipeline's main function (assumes classres_main defined already)
        print(f"[run {run_idx}] calling classres_main() with INPUT_CLASSES={current_input_path} OUT_DIR={run_dir}")
        classres_main()

        final_classes_path = run_dir / FINAL_CLASSES_FILENAME
        action_log_path = run_dir / ACTION_LOG_FILENAME

        run_summary = compute_run_summary_from_action_log(action_log_path)
        run_summary["run_index"] = run_idx
        run_summary["run_path"] = str(run_dir)
        run_summary["final_classes_path"] = str(final_classes_path) if final_classes_path.exists() else None

        summary_dir = run_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        run_summary_path = summary_dir / "cls_res_summary.json"
        _write_json(run_summary_path, run_summary)

        overall_runs.append({"run_index": run_idx, "run_dir": str(run_dir), "summary_path": str(run_summary_path)})

        total_structural = int(run_summary.get("total_structural_actions", 0))
        total_actions = int(sum(run_summary.get("total_actions_by_type", {}).values() or []))

        print(f"[run {run_idx}] total_structural_actions = {total_structural}")
        print(f"[run {run_idx}] total_actions = {total_actions}")

        is_no_change = False
        if STRUCTURAL_CHANGE_THRESHOLD is not None and total_structural <= STRUCTURAL_CHANGE_THRESHOLD:
            is_no_change = True
        if TOTAL_ACTIONS_THRESHOLD is not None and total_actions <= TOTAL_ACTIONS_THRESHOLD:
            is_no_change = True

        if MAX_NO_CHANGE_RUNS is not None:
            if is_no_change:
                no_change_streak += 1
            else:
                no_change_streak = 0

            print(f"[run {run_idx}] no_change_streak = {no_change_streak} (threshold {MAX_NO_CHANGE_RUNS})")
            if no_change_streak >= MAX_NO_CHANGE_RUNS:
                print(f"[stop] Convergence achieved after run {run_idx} (no_change_streak={no_change_streak}).")
                current_input_path = final_classes_path if final_classes_path.exists() else current_input_path
                break

        if final_classes_path.exists():
            current_input_path = final_classes_path
        else:
            print(f"[warn] final classes file not found for run {run_idx}. Stopping iterative runs.")
            break

    # -----------------------
    # ENHANCED FINAL EXPORTS
    # -----------------------
    overall_dir = EXPERIMENT_ROOT / "overall_summary"
    overall_dir.mkdir(parents=True, exist_ok=True)

    # collect per-run summaries (read the run summary JSON files)
    per_run_stats: List[Dict[str, Any]] = []
    for r in overall_runs:
        sp = Path(r["summary_path"])
        if sp.exists():
            try:
                per_run_stats.append(_load_json(sp))
            except Exception:
                per_run_stats.append({"run_index": r["run_index"], "error": "failed to load summary"})


    # aggregate overall stats (summing counts)
    aggregated = {
        "total_runs_executed": len(per_run_stats),
        "sum_total_clusters": sum([p.get("total_clusters", 0) for p in per_run_stats]),
        "sum_structural_actions": sum([p.get("total_structural_actions", 0) for p in per_run_stats]),
        "sum_errors": sum([p.get("total_errors", 0) for p in per_run_stats]),
        "sum_remarks": sum([p.get("total_remarks_logged", 0) for p in per_run_stats]),
        "by_run": per_run_stats,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    stats_path = overall_dir / "stats.json"
    _write_json(stats_path, aggregated)

    # Build classes + entities bundle for downstream relation stage
    final_classes = []
    if current_input_path.exists():
        try:
            final_classes = _load_json(current_input_path)
        except Exception:
            final_classes = []

    classes_and_entities = _aggregate_final_classes_and_entities(current_input_path, BASE_INPUT_CLASSES)
    classes_and_entities_path = overall_dir / "classes_and_entities.json"
    _write_json(classes_and_entities_path, classes_and_entities)

    # write entities+class mapping as jsonl for downstream tools
    entities_with_class_path = overall_dir / "entities_with_class.jsonl"
    _write_entities_with_class_jsonl(classes_and_entities.get("entities_map", {}), entities_with_class_path)

    # also write the final_classes array alone for convenience
    final_classes_path_out = overall_dir / "final_classes_resolved.json"
    _write_json(final_classes_path_out, final_classes)

    print(f"\n[done] Overall stats written to: {stats_path}")
    print(f"[done] Classes+entities bundle written to: {classes_and_entities_path}")
    print(f"[done] Entities jsonl (entity + class info) written to: {entities_with_class_path}")
    print(f"[done] Final classes (copy) written to: {final_classes_path_out}")





# # -----------------------
# # Cls Res Multi Run - Run statement
# # -----------------------
# if __name__ == "__main__":
#     run_pipeline_iteratively() 




#endregion#? Multi Run Cls Res
#?#########################  End  ##########################










#endregion#! Class Identification
#!#############################################  End Chapter  ##################################################












#!############################################# Start Chapter ##################################################
#region:#!   Relation Identification





#?######################### Start ##########################
#region:#?   Rel Rec v4 - Prompt change (Better Rel Naming)

#!/usr/bin/env python3
"""
Relation Recognition (Rel Rec) — Context-Enriched KG

- Reads:
    - entities_with_class.jsonl
    - chunks_sentence.jsonl
- For each chunk, finds entities that occur in that chunk.
- Calls an LLM (OpenAI Responses API) to extract directed relations between those entities.
- Writes:
    - relations_raw.jsonl  (one JSON object per relation instance)

Key design:
- Entities are ALREADY resolved and guaranteed to belong to their chunks.
- This is the LAST time we look at the raw chunk text.
- We aim for HIGH RECALL and rich QUALIFIERS (context-enriched KG).
- Intrinsic node properties were already handled in entity stages; here we treat
  almost everything else as relation-level context.

Requirements:
    pip install openai

Environment:
    export OPENAI_API_KEY="sk-..."

Adjust paths & MODEL_NAME as needed.
"""

import argparse
import json
import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from openai import OpenAI

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL_NAME = "gpt-5.1"   # or any other model you use

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_entities_by_chunk(
    entities_path: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """
    Load entities_with_class.jsonl and build:

    - entities_by_chunk:  chunk_id -> list of entity dicts (for that chunk)
    - entities_by_id:     entity_id -> entity dict (global)

    Each entity dict contains:
        entity_id, entity_name, entity_description, 
        class_id, class_label, class_group, chunk_ids, node_properties
    """
    entities_by_chunk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    entities_by_id: Dict[str, Dict[str, Any]] = {}

    logger.info("Loading entities from %s", entities_path)
    with open(entities_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            entity_id = rec["entity_id"]
            ent = rec["entity"]

            entity_record = {
                "entity_id": entity_id,
                "entity_name": ent.get("entity_name"),
                "entity_description": ent.get("entity_description"),
                # "entity_type_hint": ent.get("entity_type_hint"),
                "class_id": rec.get("class_id"),
                "class_label": rec.get("class_label"),
                "class_group": rec.get("class_group"),
                "chunk_ids": ent.get("chunk_id", []),
                "node_properties": ent.get("node_properties", []),
            }

            entities_by_id[entity_id] = entity_record

            for ch_id in entity_record["chunk_ids"]:
                entities_by_chunk[ch_id].append(entity_record)

    logger.info(
        "Loaded %d entities, mapped to %d chunks",
        len(entities_by_id),
        len(entities_by_chunk),
    )
    return entities_by_chunk, entities_by_id


def iter_chunks(chunks_path: str) -> Iterable[Dict[str, Any]]:
    """
    Yield chunks from chunks_sentence.jsonl.

    Expected fields include (example):
        {
          "id": "Ch_000001",
          "ref_index": 0,
          "ref_title": "2.1 Standards",
          "text": "...",
          ...
        }
    """
    logger.info("Streaming chunks from %s", chunks_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -----------------------------------------------------------------------------
# LLM Prompt & Call
# -----------------------------------------------------------------------------

REL_REC_INSTRUCTIONS = """
You are an expert relation extractor for a CONTEXT-ENRICHED knowledge graph.

High-level setting:
- The entities you see are ALREADY resolved and guaranteed to belong to this chunk.
  They may not always appear with exactly the same surface form in the text,
  but you must **trust** that they are conceptually present in this chunk.
- This is the FINAL stage that has direct access to the raw chunk text.
  After this, we will NOT revisit the chunk for more information.
- Our goal is to capture as MUCH context as possible:
  relations PLUS rich qualifiers (conditions, constraints, uncertainty, etc.).

Your inputs:
- A single text chunk from a technical document.
- A list of resolved entities that appear in that chunk.
- Each entity has: entity_id, entity_name, entity_description, class_label, class_group.

Your main task:
- Identify ZERO or MORE **directed** relations between the entities in this chunk.
- A relation is a meaningful, text-supported connection between a HEAD (subject) entity
  and a TAIL (object) entity.
- The graph is DIRECTED. You must choose subject_entity_id and object_entity_id
  based on how the text expresses the relationship.
  Note: Some relations may be conceptually symmetric or bidirectional; in such cases,
  choose a reasonable subject → object direction for representation purposes,
  and explain any ambiguity in justification or remark.


Very important principles:

1) TRUST THE ENTITIES (we performed entity resolution)
   - We have already run entity recognition & resolution upstream; the provided entities are canonical / resolved mentions derived from the document.
   - Do NOT question whether an entity belongs to this chunk. The exact surface string may not appear verbatim in the chunk because:
       * the entity was canonicalized (normalized) during entity resolution,
       * the chunk uses synonyms, abbreviations, pronouns, or implicit references,
       * the entity was merged from multiple mentions across the section.
   - Use the provided entity_name, entity_description,
     class_label, and class_group to recognize mentions and evidence even when the exact surface form is absent.
   - You may create relations between any pair of provided entities if the chunk text supports the relation conceptually — prefer recall over rejecting a plausible relation solely because the surface token doesn't match exactly.

2) HIGH RECALL & COVERAGE (discover first, refine later)
   - Assume that almost all informational content in the chunk should end up in the KG
     either as a relation or as qualifiers on relations.
   - Do NOT limit discovery based on naming concerns.
   - It is BETTER to propose a relation with lower confidence (and explain your doubts)
     than to miss a real relation or important qualifier.
   - If you are unsure, still output the relation with lower `confidence` and explain
     your uncertainty in `justification` and/or `remark`.

3) QUALIFIERS LIVE ON RELATIONS
   - Intrinsic node properties were (mostly) already captured during entity recognition/resolution.
   - Here, we treat almost all remaining contextual information as relation-level qualifiers.
   - Do NOT try to create or modify entity properties.
   - If you spot something that looks like a missing intrinsic property,
     mention it in `remark` instead of modeling it as a property.

4) CAPTURE QUALIFIERS EVEN IF THE RELATION IS UNCERTAIN
   - If you clearly see a contextual piece (condition, constraint, etc.) that SHOULD be attached
     to some relation but you struggle to identify the exact relation semantics:
       * Make a best-effort guess for the relation_name between the most relevant entities.
       * Set a lower `confidence`.
       * In `remark`, clearly state that this relation was primarily created to capture that qualifier
         and describe the issue (e.g., "relation semantics unclear", "heuristic relation choice").
   - This ensures we do not lose important context even when relation semantics are fuzzy.

5) DO NOT BE OVERLY BIASED BY EXAMPLES
   - Any examples of relation names, relation types, or qualifier values in this prompt are
     **illustrative, not exhaustive**.
   - You are free to discover ANY relation that is supported by the text.
   - Guidance below affects how relations should be *expressed*, not which relations you should find.


-------------------------------------------
Relation naming (VERY IMPORTANT):
-------------------------------------------
- relation_name:
    - A short, normalized phrase describing the CORE semantic relation between subject and object.
    - The goal is NOT to restrict which relations you discover,
      but to express discovered relations in a reusable, abstract form.
    - relation_name SHOULD be something that could reasonably apply to many different
      entity pairs across the corpus, even if the surface wording differs.
    - If the relation meaning is specific to this instance,
      capture that specificity in rel_desc, qualifiers, or remark — not in relation_name.
    - Examples (NOT exhaustive): "causes", "occurs_in", "is_part_of", "used_for",
      "located_in", "prevents", "requires", "correlates_with", "associated_with".
    - Do NOT include qualifiers like "at high temperature", "during startup", "may"
      in relation_name.

- relation_surface:
    - The exact phrase or minimal text span from the chunk that expresses the relation.
    - This MAY be instance-specific and does NOT need to be reusable.
    - Examples (NOT exhaustive): "leads to", "results in", "is part of",
      "used for controlling", "associated with".


-------------------------------------------
Relation hint type (rel_hint_type):
-------------------------------------------
- A short free-form hint describing the nature of the relation
  (e.g., causal, dependency, containment, usage, constraint, correlation, requirement, etc.).
- This is ONLY a hint to help later relation resolution and grouping.
- There is NO fixed list; choose whatever best fits the relation.
- If unsure, still provide your best guess and explain uncertainty in justification.
- It must be a short phrase (1–3 words), not a sentence.


-------------------------------------------
Qualifiers:
-------------------------------------------
For each relation, fill this dict (values are strings or null):

  "qualifiers": {
    "TemporalQualifier": "... or null",
    "SpatialQualifier": "... or null",
    "OperationalConstraint": "... or null",
    "ConditionExpression": "... or null",
    "UncertaintyQualifier": "... or null",
    "CausalHint": "... or null",
    "LogicalMarker": "... or null",
    "OtherQualifier": "expectedType: value or null"
  }

Meanings (examples are illustrative, NOT exhaustive):
- TemporalQualifier:
    when something holds (e.g., "during heating", "after 1000h", "at 25°C").
- SpatialQualifier:
    where something holds (e.g., "near weld", "in heat-affected zone").
- OperationalConstraint:
    operating or environmental conditions (e.g., "elevated temperature", "high load").
- ConditionExpression:
    explicit conditional or threshold clauses (e.g., "temperature > 450°C").
- UncertaintyQualifier:
    modality or hedging (e.g., "may", "likely", "suspected").
- CausalHint:
    lexical causal cues beyond the main verb (e.g., "due to", "caused by").
- LogicalMarker:
    discourse or logic markers (e.g., "if", "when", "unless").
- OtherQualifier:
    use when a qualifier does not fit the above categories
    (encode both expected type and value, e.g., "MeasurementContext: laboratory test").

If there are multiple "other" qualifiers, you may combine them in a single string.


-------------------------------------------
Other fields (and how to use them):
-------------------------------------------
- confidence:
    - float between 0 and 1 (your estimated confidence that this relation is correctly captured).
    - When in doubt, still output the relation but with a lower confidence (e.g., 0.2–0.4).

- rel_desc:
    - A brief instance-level explanation of how the subject and object are related here.
    - This is evidence-oriented and may mention the specific entities involved.

- resolution_context:
    - Short text intended to help later relation resolution / canonicalization
      (e.g., why a certain direction was chosen, or how this phrasing compares to others).

- justification:
    - Explanation of how the chunk text supports this relation.
    - Also state doubts here if semantics are ambiguous.

- remark:
    - Free-text notes for meta-issues or edge cases:
        * "relation created mainly to capture qualifier X"
        * "may reflect document organization rather than domain semantics"
        * "possible intrinsic property not captured upstream"


-------------------------------------------
Output format:
-------------------------------------------
- You MUST output a single JSON object with exactly one top-level key "relations":
  { "relations": [ ... ] }

- Each relation object MUST have this exact shape (all keys present, use null if not applicable):

  {
    "subject_entity_id": "En_...",
    "object_entity_id": "En_...",
    "relation_surface": "string",
    "relation_name": "string",
    "rel_desc": "string",
    "rel_hint_type": "string",
    "confidence": 0.0,
    "resolution_context": "string or null",
    "justification": "string or null",
    "remark": "string or null",
    "qualifiers": {
      "TemporalQualifier": "string or null",
      "SpatialQualifier": "string or null",
      "OperationalConstraint": "string or null",
      "ConditionExpression": "string or null",
      "UncertaintyQualifier": "string or null",
      "CausalHint": "string or null",
      "LogicalMarker": "string or null",
      "OtherQualifier": "string or null"
    },
    "evidence_excerpt": "short excerpt from the chunk text (<= 40 words)"
  }

- subject_entity_id and object_entity_id MUST be chosen from the provided entities.
- You may propose relations between ANY pair of provided entities if the chunk supports it.
- If there are no relations at all, return: { "relations": [] }
- rel_hint_type must be a short phrase (1–3 words), not a sentence.

Coverage:
- Try to cover ALL meaningful connections and contextual information that can be attached
  to those connections, even if it means producing low-confidence relations with detailed remarks.

Return ONLY valid JSON. No extra commentary.
"""




def call_llm_extract_relations_for_chunk(
    client: OpenAI,
    model: str,
    chunk: Dict[str, Any],
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Call the LLM using the Responses API to extract relations for a single chunk.

    Returns a list of relation dicts (without relation_id, chunk_id, or class info added yet).
    """
    if not entities or len(entities) < 2:
        return []

    payload = {
        "chunk_id": chunk["id"],
        "chunk_text": chunk.get("text", ""),
        "entities": [
            {
                "entity_id": e["entity_id"],
                "entity_name": e["entity_name"],
                "entity_description": e["entity_description"],
                "class_label": e["class_label"],
                "class_group": e["class_group"],
                "node_properties": e.get("node_properties", []),
            }
            for e in entities
        ],
    }

    try:
        response = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            input=[
                {
                    "role": "developer",
                    "content": REL_REC_INSTRUCTIONS,
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
        )
    except Exception as e:
        logger.error("LLM call failed for chunk %s: %s", chunk["id"], e)
        return []

    raw_text = response.output_text
    if not raw_text:
        logger.warning("Empty response for chunk %s", chunk["id"])
        return []

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error(
            "Failed to parse JSON for chunk %s. Raw response:\n%s",
            chunk["id"],
            raw_text,
        )
        return []

    relations = parsed.get("relations", [])
    if not isinstance(relations, list):
        logger.error(
            "Expected 'relations' to be a list for chunk %s. Got: %r",
            chunk["id"],
            type(relations),
        )
        return []

    return relations


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_rel_rec(
    entities_path: str,
    chunks_path: str,
    output_path: str,
    model: str = MODEL_NAME,
) -> None:
    """
    Full Relation Recognition pipeline:

    - load entities_by_chunk, entities_by_id
    - iterate over chunks
    - for each chunk, call LLM to extract relations
    - enrich relations with relation_id, chunk_id, subject/object class info
    - write to relations_raw.jsonl (streaming, flushed after each chunk)
    """
    entities_by_chunk, entities_by_id = load_entities_by_chunk(entities_path)
    client = OpenAI()

    n_chunks = 0
    n_chunks_called = 0
    n_relations = 0

    logger.info("Writing relations to %s", output_path)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for chunk in iter_chunks(chunks_path):
            n_chunks += 1
            chunk_id = chunk["id"]
            chunk_entities = entities_by_chunk.get(chunk_id, [])

            if len(chunk_entities) < 2:
                continue  # cannot form relations

            n_chunks_called += 1
            logger.info(
                "Chunk %s: %d entities -> calling LLM", chunk_id, len(chunk_entities)
            )

            relations = call_llm_extract_relations_for_chunk(
                client=client,
                model=model,
                chunk=chunk,
                entities=chunk_entities,
            )

            for rel in relations:
                # Basic validation of required LLM fields
                subj_id = rel.get("subject_entity_id")
                obj_id = rel.get("object_entity_id")
                if subj_id not in entities_by_id or obj_id not in entities_by_id:
                    logger.warning(
                        "Skipping relation with unknown entity ids in chunk %s: %s",
                        chunk_id,
                        rel,
                    )
                    continue

                # Normalize qualifiers structure to always have all keys
                expected_qual_keys = [
                    "TemporalQualifier",
                    "SpatialQualifier",
                    "OperationalConstraint",
                    "ConditionExpression",
                    "UncertaintyQualifier",
                    "CausalHint",
                    "LogicalMarker",
                    "OtherQualifier",
                ]
                q = rel.get("qualifiers") or {}
                if not isinstance(q, dict):
                    q = {}
                for k in expected_qual_keys:
                    q.setdefault(k, None)
                rel["qualifiers"] = q

                # Enrich with KG-level metadata
                rel["relation_id"] = f"RelR_{uuid.uuid4().hex[:12]}"
                rel["chunk_id"] = chunk_id

                subj = entities_by_id[subj_id]
                obj = entities_by_id[obj_id]

                # Add class/group metadata
                rel.setdefault("subject_class_group", subj.get("class_group"))
                rel.setdefault("subject_class_label", subj.get("class_label"))
                rel.setdefault("object_class_group", obj.get("class_group"))
                rel.setdefault("object_class_label", obj.get("class_label"))

                # Add entity names for convenience in relations_raw.jsonl
                rel.setdefault("subject_entity_name", subj.get("entity_name"))
                rel.setdefault("object_entity_name", obj.get("entity_name"))

                out_f.write(json.dumps(rel, ensure_ascii=False) + "\n")
                n_relations += 1

            # flush after each chunk so data is written incrementally
            # out_f.flush()

    logger.info(
        "Rel Rec done. Chunks: %d, chunks_with_LLM: %d, relations: %d",
        n_chunks,
        n_chunks_called,
        n_relations,
    )



# -----------------------------------------------------------------------------
# CLI (optional for terminal use; in notebooks call run_rel_rec(...) directly)
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relation Recognition (Rel Rec)")
    p.add_argument(
        "--entities",
        required=True,
        help="Path to entities_with_class.jsonl",
    )
    p.add_argument(
        "--chunks",
        required=True,
        help="Path to chunks_sentence.jsonl",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to output relations_raw.jsonl",
    )
    p.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})",
    )
    return p.parse_args()

#make output path directory if it doesn't exist
output_dir = "SGCE-KG/data/Relations/Rel Rec"
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 



# -----------------------
# Relation Rec - Run statement
# -----------------------


# run_rel_rec(
#     entities_path="SGCE-KG/data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl",
#     chunks_path="SGCE-KG/data/Chunks/chunks_sentence.jsonl",
#     output_path="SGCE-KG/data/Relations/Rel Rec/relations_raw.jsonl",
#     model="gpt-5.1"
# )




#endregion#? Rel Rec v4 - Prompt change (Better Rel Naming)
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Rel Res V4  - Canonical + RelCls + RelClsGroup + Schema + LocalSubcluster

#!/usr/bin/env python3
"""
relres_iterative_v4.py

Relation Resolution (Rel Res) — cluster relation instances, ask LLM to
assign/normalize:

  - canonical_rel_name    (normalized predicate used on KG edges)
  - canonical_rel_desc    (reusable description of that predicate)
  - rel_cls               (relation class, groups canonical_rel_names)
  - rel_cls_group         (broad group like COMPOSITION, CAUSALITY, ...)

while preserving:

  - relation_name         (raw name from Rel Rec)
  - rel_desc              (instance-level description)
  - qualifiers, head/tail, etc.

Key properties:
- We NEVER remove relation instances; we only enrich them with schema.
- canonical_rel_name is what will be used as edge label in the KG.
- rel_cls / rel_cls_group give you a 2-layer schema for relations.
- Multi-run friendly: TBD fields can be filled in the first run, refined later.
- Uses global HDBSCAN + optional local subclustering + MAX_MEMBERS_PER_PROMPT
  so LLM chunks stay reasonably small.

Input:
  SGCE-KG/data/Relations/Rel Rec/relations_raw.jsonl

Output (under OUT_DIR):
  - per-(cluster,local,part) decisions: cluster_<ID>_decisions.json
  - per-(cluster,local,part) raw llm output: llm_raw/cluster_<ID>_llm_raw.txt
  - per-(cluster,local,part) prompts: llm_raw/cluster_<ID>_prompt.txt
  - cumulative action log: rel_res_action_log.jsonl
  - final resolved relations: relations_resolved.json and .jsonl
  - summary/all_clusters_decisions.json
  - summary/stats_summary.json
  - summary/canonical_rel_schema.json
  - summary/rel_cls_schema.json
  - summary/rel_cls_group_schema.json
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import normalize

# clustering libs
try:
    import hdbscan
except Exception:
    raise RuntimeError("hdbscan required: pip install hdbscan")
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# transformers embedder
from transformers import AutoTokenizer, AutoModel

# OpenAI client (Responses API)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------- CONFIG -----------------------------

INPUT_RELATIONS = Path("SGCE-KG/data/Relations/Rel Rec/relations_raw.jsonl")
OUT_DIR = Path("SGCE-KG/data/Relations/Rel Res")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_LLM_DIR = OUT_DIR / "llm_raw"
RAW_LLM_DIR.mkdir(exist_ok=True)

# Embedding model
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Weights for buckets used to build relation embeddings
# Buckets:
#   name            = relation_name
#   desc            = rel_desc
#   head_tail       = subject/object names + class info
#   hint_canonical  = rel_hint_type + canonical_rel_name + canonical_rel_desc + rel_cls
REL_EMB_WEIGHTS = {
    "name": 0.25,
    "desc": 0.15,
    "head_tail": 0.20,
    "hint_canonical": 0.40,
}

# Global clustering params
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# Local subcluster params (when a cluster is very large)
MAX_CLUSTER_SIZE_FOR_LOCAL = 30
LOCAL_HDBSCAN_MIN_CLUSTER_SIZE = 2
LOCAL_HDBSCAN_MIN_SAMPLES = 1

# LLM prompt chunking
MAX_MEMBERS_PER_PROMPT = 10  # max relation instances per LLM call

# LLM / OpenAI
OPENAI_MODEL = "gpt-5.1"
OPENAI_TEMPERATURE = 0.0
LLM_MAX_OUTPUT_TOKENS = 16000
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

VERBOSE = False

# ---------------------- Helpers: OpenAI client ---------------------

def _load_openai_key(envvar: str = OPENAI_API_KEY_ENV, fallback_path: str = ".env"):
    key = os.getenv(envvar, None)
    if key:
        return key
    if Path(fallback_path).exists():
        txt = Path(fallback_path).read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None

OPENAI_KEY = _load_openai_key()
if OpenAI is not None and OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None
    if VERBOSE:
        print("⚠️ OpenAI client not initialized (missing package or API key). LLM calls will fail unless OpenAI client is available.")

def call_llm(prompt: str, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE, max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS) -> str:
    """
    Use OpenAI Responses API with a single developer prompt and a small user nudge.
    """
    if client is None:
        raise RuntimeError("OpenAI client not available. Set OPENAI_API_KEY and install openai package.")
    try:
        resp = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            max_output_tokens=max_output_tokens,
            input=[
                {"role": "developer", "content": prompt},
                {"role": "user", "content": "Return ONLY the JSON array of function calls now."},
            ],
        )
        return resp.output_text or ""
    except Exception as e:
        print("LLM call error:", e)
        return ""

# ---------------------- HF Embedder --------------------------------

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class HFEmbedder:
    def __init__(self, model_name=EMBED_MODEL, device=DEVICE):
        if VERBOSE:
            print(f"[embedder] loading {model_name} on {device}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def dim(self) -> int:
        return getattr(self.model.config, "hidden_size", 1024)

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized embeddings.
        """
        if len(texts) == 0:
            D = self.dim
            return np.zeros((0, D))
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            embs.append(pooled.cpu().numpy())
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs

# ---------------------- IO helpers ---------------------------------

def load_relations(path: Path) -> List[Dict[str, Any]]:
    """
    Load relations_raw.jsonl and ensure schema-related fields exist:

      - canonical_rel_name (default "TBD")
      - canonical_rel_desc (default "")
      - rel_cls (default "TBD")
      - rel_cls_group (default "TBD")
      - remarks (list of strings)
    """
    rels: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Relations file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # ensure relation_id exists
            rid = obj.get("relation_id") or ("RelR_" + uuid.uuid4().hex[:8])
            obj["relation_id"] = rid

            # ensure schema fields
            if "canonical_rel_name" not in obj or str(obj.get("canonical_rel_name", "")).strip() == "":
                obj["canonical_rel_name"] = "TBD"
            if "canonical_rel_desc" not in obj or obj.get("canonical_rel_desc") is None:
                obj["canonical_rel_desc"] = ""
            if "rel_cls" not in obj or str(obj.get("rel_cls", "")).strip() == "":
                obj["rel_cls"] = "TBD"
            if "rel_cls_group" not in obj or str(obj.get("rel_cls_group", "")).strip() == "":
                obj["rel_cls_group"] = "TBD"

            # normalize remarks: merge original remark + remarks list into a 'remarks' list
            initial_remark = obj.get("remark")
            remarks = obj.get("remarks")
            norm_remarks: List[str] = []
            if isinstance(remarks, list):
                norm_remarks.extend([str(r) for r in remarks if r])
            elif isinstance(remarks, str) and remarks.strip():
                norm_remarks.append(remarks.strip())
            if isinstance(initial_remark, str) and initial_remark.strip():
                norm_remarks.append(initial_remark.strip())
            obj["remarks"] = norm_remarks

            rels.append(obj)
    if VERBOSE:
        print(f"[start] loaded {len(rels)} relations from {path}")
    return rels

def safe_str(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).replace("\n", " ").strip()

# ---------------------- Build relation texts & embeddings ----------

def build_relation_texts(
    relations: List[Dict[str, Any]]
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Build four text buckets for each relation:

    - name_texts:       relation_name
    - desc_texts:       rel_desc
    - head_tail_texts:  head/tail entity + class info
    - hint_texts:       rel_hint_type + canonical_rel_name + canonical_rel_desc + rel_cls
    """
    name_texts, desc_texts, head_tail_texts, hint_texts = [], [], [], []

    for r in relations:
        # name
        rname = safe_str(r.get("relation_name", ""))
        name_texts.append(rname[:256])

        # desc
        rdesc = safe_str(r.get("rel_desc", ""))
        desc_texts.append(rdesc[:512])

        # head_tail
        subj_name = safe_str(r.get("subject_entity_name", ""))
        subj_cl = safe_str(r.get("subject_class_label", ""))
        subj_cg = safe_str(r.get("subject_class_group", ""))
        obj_name = safe_str(r.get("object_entity_name", ""))
        obj_cl = safe_str(r.get("object_class_label", ""))
        obj_cg = safe_str(r.get("object_class_group", ""))

        head_tail = f"{subj_name} ({subj_cl}, {subj_cg}) -> {obj_name} ({obj_cl}, {obj_cg})"
        head_tail_texts.append(head_tail[:512])

        # hint_canonical
        hint_parts = []
        for key in ["rel_hint_type", "canonical_rel_name", "canonical_rel_desc", "rel_cls"]:
            val = safe_str(r.get(key, ""))
            if val and val.upper() != "TBD":
                hint_parts.append(val)
        hint_text = " ; ".join(hint_parts)
        hint_texts.append(hint_text[:512])

    return name_texts, desc_texts, head_tail_texts, hint_texts

def any_nonempty(lst: List[str]) -> bool:
    return any(safe_str(t) for t in lst)

def compute_relation_embeddings(
    embedder: HFEmbedder,
    relations: List[Dict[str, Any]],
    weights: Dict[str, float]
) -> np.ndarray:
    """
    Compute combined embeddings for relations using four buckets with fixed weights.
    """
    N = len(relations)
    if N == 0:
        raise ValueError("No relations to embed")

    name_texts, desc_texts, head_tail_texts, hint_texts = build_relation_texts(relations)

    emb_name = embedder.encode_batch(name_texts) if any_nonempty(name_texts) else None
    emb_desc = embedder.encode_batch(desc_texts) if any_nonempty(desc_texts) else None
    emb_ht = embedder.encode_batch(head_tail_texts) if any_nonempty(head_tail_texts) else None
    emb_hint = embedder.encode_batch(hint_texts) if any_nonempty(hint_texts) else None

    D = embedder.dim
    combined = np.zeros((N, D), dtype=np.float32)

    def add_bucket(emb: Optional[np.ndarray], weight: float):
        nonlocal combined
        if emb is None:
            return
        if emb.shape[0] != N:
            raise ValueError("embedding row mismatch")
        combined += weight * emb  # emb already row-normalized

    add_bucket(emb_name, weights.get("name", 0.0))
    add_bucket(emb_desc, weights.get("desc", 0.0))
    add_bucket(emb_ht, weights.get("head_tail", 0.0))
    add_bucket(emb_hint, weights.get("hint_canonical", 0.0))

    combined = normalize(combined, axis=1)
    return combined

# ---------------------- clustering ---------------------------------

def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    use_umap: bool = USE_UMAP
) -> Tuple[np.ndarray, object]:
    X = embeddings
    N = X.shape[0]
    if use_umap and UMAP_AVAILABLE and N >= 6:
        safe_n_components = min(UMAP_N_COMPONENTS, max(2, N - 2))
        safe_n_neighbors = min(UMAP_N_NEIGHBORS, max(2, N - 1))
        try:
            reducer = umap.UMAP(
                n_components=safe_n_components,
                n_neighbors=safe_n_neighbors,
                min_dist=UMAP_MIN_DIST,
                metric="cosine",
                random_state=42
            )
            X_reduced = reducer.fit_transform(X)
            if X_reduced is not None and X_reduced.shape[0] == N:
                X = X_reduced
            else:
                if VERBOSE:
                    print(f"[warn] UMAP returned invalid shape; skipping UMAP")
        except Exception as e:
            if VERBOSE:
                print(f"[warn] UMAP failed (N={N}): {e}. Proceeding without UMAP.")
            X = embeddings
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer

# ---------------------- LLM prompt template ------------------------

RELRES_PROMPT_TEMPLATE = """
You are a very proactive RELATION-RESOLUTION assistant, and an expert in knowledge graphs (KGs) and schema (ontology) design.

You are given a CLUSTER of relation INSTANCES from a context-enriched technical KG.
Each relation instance already connects specific subject and object entities, and has:

- relation_id
- relation_name        (raw, from Relation Recognition; may be noisy)
- rel_desc             (instance-level description; may be verbose)
- rel_hint_type        (short free-form hint; may be noisy)
- canonical_rel_name   (often "TBD" initially)
- canonical_rel_desc   (often empty initially)
- rel_cls              (often "TBD" initially)
- rel_cls_group        (often "TBD" initially)
- subject_entity_name, object_entity_name
- subject_class_label, subject_class_group
- object_class_label, object_class_group
- qualifiers (temporal, spatial, etc.)
- confidence, remarks, etc.

IMPORTANT CONTEXT ABOUT THE PIPELINE
------------------------------------
- ALL these fields (especially relation_name, rel_desc, and rel_hint_type) were
  generated by previous LLM stages. They are **approximate** and may be:
    - awkward in wording,
    - too specific,
    - too generic,
    - or slightly wrong.
- This step follows a "generate first, refine later" philosophy.
  Your job is to **refine, disambiguate, normalize, and improve human readability as a KG expert**, not to copy the wording blindly.
- Use upstream fields (especially relation_name, rel_desc, and rel_hint_type) as **semantic hints**, not as final label candidates AT ALL!

MULTI-RUN / REFINEMENT INSTRUCTIONS (READ ONLY WHEN THE FOLLOWING FIELDS ARE ALREADY FILLED: canonical_rel_name, canonical_rel_desc, rel_cls, rel_cls_group)
- Only read/apply the following guidance when at least one of the fields above is NOT "TBD" for relations in this chunk. It means we are in a refinement run.
- Do NOT be passive: if any pre-filled value is inconsistent, ambiguous, or improvable, you MUST propose a correction (use modify_rel_schema) with a concise justification.
- Only return [] if there is extremely strong evidence that no change is required (this is rare).
- This is an iterative process — act now with well-justified corrections using modify_rel_schema (include a short justification), rather than deferring small but meaningful fixes.
- This step will be repeated in later iterations; reasonable but imperfect changes can be corrected later. It is worse to miss a necessary change than to propose a well-justified change that might be slightly adjusted later.


ABOUT CLUSTERS (CRITICAL)
-------------------------
- The cluster you see is produced automatically from embeddings.
- The cluster is **only suggestive**, NOT a hard class:
    - It may contain multiple different canonical relations.
    - It may contain multiple different relation classes.
    - It may contain multiple different relation class groups.
- Your task is **NOT**:
    - "Find ONE canonical_rel_name that covers (almost) all relations in the cluster."
    - "Force all relations in the cluster into the same rel_cls or rel_cls_group."
- Your task **IS**:
    - For EACH relation instance, decide what canonical_rel_name, rel_cls,
      and rel_cls_group are appropriate (especially from the lens of a KG and Schema expert).
    - If some instances do NOT naturally share semantics, assign them different
      canonical_rel_name / rel_cls / rel_cls_group, even though they are
      in the same cluster.
    - If some other instances in the cluster naturally share exactly the same
      semantics, group them together in the same function call.

CRUCIAL DISTINCTION (SCOPE)
---------------------------
We use a 3-layer abstraction for relations:

1) canonical_rel_name
   - Fine-grained, predicate-level.
   - Groups relations that use essentially the SAME predicate meaning and direction.
   - This is the predicate that will be used as the edge label in the KG.
   - Examples: "occurs_in", "resists", "provides_resistance_to", "is_subtype_of",
     "has_metallurgical_structure", "grouped_with".

2) rel_cls
   - A broader relation CLASS that may cover multiple canonical_rel_name values.
   - Think of this as a family of similar predicates.
   - Examples (illustrative): "alloying_element_relation", "microstructure_relation",
     "hierarchy_relation", "corrosion_resistance_relation", "document_series_relation".
   - IMPORTANT: rel_cls should be more specific than broad groups, but more general
     than a single canonical_rel_name.

3) rel_cls_group
   - Very broad semantic DIMENSIONS:
       COMPOSITION, CAUSALITY, IDENTITY, TEMPORALITY, SPATIALITY, AGENCY,
       ASSOCIATION, MODIFICATION, USAGE, REQUIREMENT, etc.
   - This list is illustrative, NOT exhaustive.
   - Example:
       canonical_rel_name: "contains_alloying_element"
       rel_cls:            "alloying_element_relation"
       rel_cls_group:      "COMPOSITION"

RELATION-BY-RELATION THINKING (SUPER IMPORTANT)
-----------------------------------------------
For every relation instance in the cluster, conceptually ask:

1) What is the **best canonical predicate** (canonical_rel_name) for THIS instance,
   ignoring awkward wording in relation_name / rel_hint_type?
2) Which broader **relation class** (rel_cls) best describes this connection?
3) Which **rel_cls_group** (COMPOSITION, CAUSALITY, etc.) does it belong to?

Then:
- If you see other instances in the cluster with the **same semantics**, you may
  include them in the same function call (same canonical_rel_name / rel_cls / rel_cls_group).
- Do NOT try to find one label that covers ALL instances in the cluster.

NOISY HINTS AND HOW TO USE THEM
-------------------------------
- relation_name, rel_desc, rel_hint_type are hints, not perfect labels.
- Examples of bad patterns that you should NOT copy directly:
    - rel_hint_type = "co-classification" → this is awkward wording; interpret it
      as "these things are grouped together as co-equal categories" and choose
      a better canonical name / class (e.g., "grouped_with" with rel_cls
      "coequal_category_relation", rel_cls_group "ASSOCIATION" or "IDENTITY").
    - rel_hint_type = "causal" → this suggests rel_cls_group "CAUSALITY",
      but is NOT a good canonical_rel_name or rel_cls by itself.
- Also avoid redundant patterns like:
    - rel_cls = "composition_relation" when rel_cls_group = "COMPOSITION".
      In that case, choose a more specific rel_cls label, e.g.,
      "alloying_element_relation" or "microstructure_relation".

DEDUPLICATION & NORMALIZATION
-----------------------------
- Within a cluster, if you see multiple predicate variants that are semantically
  the same (e.g., "has_subtype", "is_subtype_of"), you MUST normalize them
  to **one** canonical_rel_name and apply it consistently:
    - For hierarchical relations, prefer a single consistent style, e.g., "is_subtype_of".
- Avoid creating trivial variants that only differ in small wording:
    - Do NOT keep both "provides_resistance_to" and "improves_resistance_to"
      if they are used identically for the same semantic link; pick one.
- Avoid rel_cls that simply repeats rel_cls_group with "_relation" added
  (e.g., "composition_relation" when rel_cls_group is "COMPOSITION").

WHAT YOU NEVER CHANGE
----------------------
- SUBJECT and OBJECT entities are fixed; you MUST NOT change them.
- You NEVER delete relation instances.
- You NEVER move qualifiers from relations onto nodes; qualifiers stay on the
  relation instance.

YOUR FUNCTION VOCABULARY
------------------------
You must return an ordered JSON ARRAY of function calls using ONLY:

1) set_canonical_rel
2) set_rel_cls
3) set_rel_cls_group
4) modify_rel_schema
5) add_rel_remark

Think RELATION-BY-RELATION first, and only group multiple relation_ids into the
same function call when you are genuinely sure they share the same semantics.

--------------------------------
FUNCTION DEFINITIONS
--------------------------------

1) set_canonical_rel
   Use this when you want to SET or ALIGN canonical_rel_name / canonical_rel_desc
   for one or more relation instances (especially when values are "TBD").

   args = {
     "relation_ids": [<relation_id>...],        # REQUIRED, at least 1
     "canonical_rel_name": <string>,           # REQUIRED, normalized predicate for KG edge
     "canonical_rel_desc": <string or null>,   # OPTIONAL, reusable description of this predicate
     "justification": <string>,                # REQUIRED: why these instances share this canonical predicate
     "remark": <string or null>,               # optional human-facing notes
     "confidence": <number between 0 and 1, optional>
   }

   Notes:
   - canonical_rel_name should be a short predicate-style phrase, e.g. "occurs_in",
     "resists", "provides_resistance_to", "is_subtype_of", "has_metallurgical_structure".
   - Do NOT include qualifiers (e.g., "at high temperature") here.
   - Use this on relatively TIGHT groups of semantically equivalent predicates.
   - It is perfectly fine to call set_canonical_rel with a **single** relation_id
     when others in the cluster do not share the semantics.


2) set_rel_cls
   Use this when you want to SET or ALIGN rel_cls for one or more instances,
   grouping them into a broader relation class (family).

   args = {
     "relation_ids": [<relation_id>...],        # REQUIRED, at least 1
     "rel_cls": <string>,                      # REQUIRED: class name (e.g., "structure_relation")
     "justification": <string>,                # REQUIRED: why these instances belong to this class
     "remark": <string or null>,               # optional notes
     "confidence": <number between 0 and 1, optional>
   }

   Notes:
   - A rel_cls usually covers MULTIPLE canonical_rel_name values, not just one.
   - Think of rel_cls as a conceptual family: e.g. "alloying_element_relation",
     "microstructure_relation", "document_series_relation".
   - Do NOT simply repeat rel_cls_group (e.g. avoid "composition_relation" when
     rel_cls_group is "COMPOSITION") unless there is truly no more specific
     and meaningful class you can provide.


3) set_rel_cls_group
   Use this when you want to SET or ALIGN rel_cls_group for one or more instances
   (or their classes), using a very broad semantic category.

   args = {
     "relation_ids": [<relation_id>...],        # REQUIRED, at least 1
     "rel_cls_group": <string>,                # REQUIRED: broad group (e.g., "COMPOSITION")
     "justification": <string>,                # REQUIRED: why this group fits
     "remark": <string or null>,               # optional notes
     "confidence": <number between 0 and 1, optional>
   }

   Notes:
   - rel_cls_group is broad and somewhat orthogonal.
   - Typical groups: COMPOSITION, CAUSALITY, IDENTITY, TEMPORALITY, SPATIALITY,
     AGENCY, INTERACTION, ASSOCIATION, MODIFICATION, USAGE, REQUIREMENT, etc.
   - General hints like rel_hint_type="causal" are better mapped to rel_cls_group
     (CAUSALITY) than to canonical_rel_name.


4) modify_rel_schema
   Use this to REFINE or CORRECT existing schema fields for one or more relations
   (canonical_rel_name / canonical_rel_desc / rel_cls / rel_cls_group).

   args = {
     "relation_ids": [<relation_id>...],            # REQUIRED, at least 1
     "canonical_rel_name": <string or null>,        # OPTIONAL: new canonical name
     "canonical_rel_desc": <string or null>,        # OPTIONAL: new canonical description
     "rel_cls": <string or null>,                   # OPTIONAL: new relation class
     "rel_cls_group": <string or null>,             # OPTIONAL: new broad group
     "justification": <string>,                     # REQUIRED: why this modification is needed
     "remark": <string or null>,                    # optional notes / flags
     "confidence": <number between 0 and 1, optional>
   }

   Notes:
   - Do NOT be passive: if any pre-filled value is inconsistent, ambiguous, or improvable, you MUST propose a correction using modify_rel_schema.
   - This is iterative: make well-justified corrections now (include a short justification); later runs may further refine them. 
   - It is worse to miss a necessary change than to propose a justified change that might be slightly adjusted later.
   - You MUST NOT try to change (poor) relation_names. As long as the canonical_rel_name, rel_cls, and rel_cls_group are fine, we are good. 
     But if they need correction, YOU MUST use modify_rel_schema to correct them. We will not use raw relation name in the KG. 
   - You MAY use this to normalize even minor variants (e.g., consolidate "has_subtype"
   and "is_subtype_of" to "is_subtype_of") when needed. We want consistent schema.
   


5) add_rel_remark
   Use this when you ONLY want to attach a human-facing remark / caveat / TODO
   without changing any schema fields.

   args = {
     "relation_ids": [<relation_id>...],        # REQUIRED, at least 1
     "remark": <string>,                        # REQUIRED: the remark text
     "justification": <string>,                # REQUIRED: why this remark is useful
     "confidence": <number between 0 and 1, optional>
   }

   Notes:
   - Use this for issues outside scope (e.g., upstream entity resolution suspicion), or
     to flag ambiguous or borderline cases for later human review.

--------------------------------
INPUT RELATIONS (THIS CHUNK)
--------------------------------

Below is the JSON array of relation instances in THIS PART of a cluster
(we split very large clusters into smaller chunks just to keep the prompt size manageable):

{cluster_block}

--------------------------------
OUTPUT
--------------------------------

Return ONLY the JSON ARRAY of function calls, e.g.:

[
  {
    "function": "set_canonical_rel",
    "args": { ... }
  },
  {
    "function": "set_rel_cls",
    "args": { ... }
  },
  ...
]

- If this chunk is already perfect (rare), you MAY return [].
- Every call MUST include a "justification".
- Use "remark" (or add_rel_remark) to flag issues or uncertainties that are outside
  the scope of these functions.
"""

def sanitize_json_array(text: str) -> Optional[Any]:
    """
    Extract and parse the first JSON array from the text.
    Grab [ ... ] block, fix simple trailing commas, and json.loads.
    """
    if not text or not text.strip():
        return None
    s = text.strip()
    # replace smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    cand = s[start:end + 1]
    # remove trailing commas before closing braces/brackets
    cand = re.sub(r",\s*([\]}])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        return None

# ---------------------- Action executors ----------------------------

def execute_set_canonical_rel(
    rel_by_id: Dict[str, Dict[str, Any]],
    relation_ids: List[str],
    canonical_rel_name: Optional[str],
    canonical_rel_desc: Optional[str],
):
    if canonical_rel_name is None:
        return
    canon_name = canonical_rel_name.strip()
    if not canon_name:
        return
    for rid in relation_ids:
        if rid not in rel_by_id:
            continue
        r = rel_by_id[rid]
        r["canonical_rel_name"] = canon_name
        if canonical_rel_desc is not None:
            r["canonical_rel_desc"] = canonical_rel_desc.strip()
        rel_by_id[rid] = r

def execute_set_rel_cls(
    rel_by_id: Dict[str, Dict[str, Any]],
    relation_ids: List[str],
    rel_cls: Optional[str],
):
    if rel_cls is None:
        return
    cls_name = rel_cls.strip()
    if not cls_name:
        return
    for rid in relation_ids:
        if rid not in rel_by_id:
            continue
        r = rel_by_id[rid]
        r["rel_cls"] = cls_name
        rel_by_id[rid] = r

def execute_set_rel_cls_group(
    rel_by_id: Dict[str, Dict[str, Any]],
    relation_ids: List[str],
    rel_cls_group: Optional[str],
):
    if rel_cls_group is None:
        return
    grp_name = rel_cls_group.strip()
    if not grp_name:
        return
    for rid in relation_ids:
        if rid not in rel_by_id:
            continue
        r = rel_by_id[rid]
        r["rel_cls_group"] = grp_name
        rel_by_id[rid] = r

def execute_modify_rel_schema(
    rel_by_id: Dict[str, Dict[str, Any]],
    relation_ids: List[str],
    canonical_rel_name: Optional[str],
    canonical_rel_desc: Optional[str],
    rel_cls: Optional[str],
    rel_cls_group: Optional[str],
    new_relation_name: Optional[str],
    original_relation_name: Optional[str],
):
    for rid in relation_ids:
        if rid not in rel_by_id:
            continue
        r = rel_by_id[rid]

        if canonical_rel_name is not None and canonical_rel_name.strip():
            r["canonical_rel_name"] = canonical_rel_name.strip()
        if canonical_rel_desc is not None:
            r["canonical_rel_desc"] = canonical_rel_desc.strip()
        if rel_cls is not None and rel_cls.strip():
            r["rel_cls"] = rel_cls.strip()
        if rel_cls_group is not None and rel_cls_group.strip():
            r["rel_cls_group"] = rel_cls_group.strip()

        if new_relation_name is not None and new_relation_name.strip():
            if "original_relation_name" not in r:
                if original_relation_name:
                    r["original_relation_name"] = original_relation_name
                else:
                    r["original_relation_name"] = r.get("relation_name", "")
            r["relation_name"] = new_relation_name.strip()

        rel_by_id[rid] = r

def execute_add_rel_remark(
    rel_by_id: Dict[str, Dict[str, Any]],
    relation_ids: List[str],
    remark: Optional[str],
):
    if remark is None:
        return
    txt = remark.strip()
    if not txt:
        return
    for rid in relation_ids:
        if rid not in rel_by_id:
            continue
        r = rel_by_id[rid]
        existing = r.get("remarks")
        if existing is None:
            existing = []
        elif not isinstance(existing, list):
            existing = [str(existing)]
        existing.append(txt)
        r["remarks"] = existing
        rel_by_id[rid] = r

# ---------------------- Main orchestration -------------------------

def relres_main():
    # load relations
    relations = load_relations(INPUT_RELATIONS)
    rel_by_id: Dict[str, Dict[str, Any]] = {r["relation_id"]: r for r in relations}

    # embedder
    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    rel_ids_order = [r["relation_id"] for r in relations]
    rel_id_to_index = {rid: i for i, rid in enumerate(rel_ids_order)}
    combined_emb = compute_relation_embeddings(embedder, relations, REL_EMB_WEIGHTS)
    print("[info] relation embeddings computed shape:", combined_emb.shape)

    # global clustering
    labels, clusterer = run_hdbscan(
        combined_emb,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        use_umap=USE_UMAP
    )
    unique_labels = sorted(set(labels))
    print("[info] global clustering done. unique labels:", unique_labels)

    # map cluster -> relation ids
    cluster_to_relids: Dict[int, List[str]] = {}
    for idx, lab in enumerate(labels):
        rid = rel_ids_order[idx]
        cluster_to_relids.setdefault(int(lab), []).append(rid)

    # action log
    action_log_path = OUT_DIR / "rel_res_action_log.jsonl"
    if action_log_path.exists():
        action_log_path.unlink()

    # helper to process a subset of relation ids with LLM (one prompt)
    def run_llm_on_subset(sub_rel_ids: List[str], cluster_label_str: str):
        if not sub_rel_ids:
            return

        # build compact representation for this chunk
        cluster_relations = []
        for rid in sub_rel_ids:
            r = rel_by_id.get(rid)
            if not r:
                continue
            cluster_relations.append({
                "relation_id": r["relation_id"],
                "relation_name": safe_str(r.get("relation_name", "")),
                "rel_desc": safe_str(r.get("rel_desc", "")),
                "rel_hint_type": safe_str(r.get("rel_hint_type", "")),
                "canonical_rel_name": safe_str(r.get("canonical_rel_name", "")),
                "canonical_rel_desc": safe_str(r.get("canonical_rel_desc", "")),
                "rel_cls": safe_str(r.get("rel_cls", "")),
                "rel_cls_group": safe_str(r.get("rel_cls_group", "")),
                "subject_entity_name": safe_str(r.get("subject_entity_name", "")),
                "object_entity_name": safe_str(r.get("object_entity_name", "")),
                "subject_class_label": safe_str(r.get("subject_class_label", "")),
                "subject_class_group": safe_str(r.get("subject_class_group", "")),
                "object_class_label": safe_str(r.get("object_class_label", "")),
                "object_class_group": safe_str(r.get("object_class_group", "")),
                "qualifiers": r.get("qualifiers", {}),
                "confidence": float(r.get("confidence", 0.0)),
                "remarks": r.get("remarks", [])
            })

        if not cluster_relations:
            return

        cluster_block = json.dumps(cluster_relations, ensure_ascii=False, indent=2)
        prompt = RELRES_PROMPT_TEMPLATE.replace("{cluster_block}", cluster_block)

        # log prompt
        prompt_path = RAW_LLM_DIR / f"cluster_{cluster_label_str}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # call LLM
        raw_out = ""
        try:
            raw_out = call_llm(prompt)
        except Exception as e:
            print(f"[warning] LLM call failed for {cluster_label_str}: {e}")
            raw_out = ""

        # write raw output
        raw_path = RAW_LLM_DIR / f"cluster_{cluster_label_str}_llm_raw.txt"
        raw_path.write_text(raw_out, encoding="utf-8")

        parsed = sanitize_json_array(raw_out)
        if parsed is None:
            print(f"[warn] failed to parse LLM output for chunk {cluster_label_str}; skipping automated actions for this chunk.")
            dec_path = OUT_DIR / f"cluster_{cluster_label_str}_decisions.json"
            dec_path.write_text(
                json.dumps({"cluster_label": cluster_label_str, "raw_llm": raw_out}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            return

        decisions: List[Dict[str, Any]] = []

        # execute parsed function list in order
        for step in parsed:
            if not isinstance(step, dict):
                continue
            fn = step.get("function")
            args = step.get("args", {}) or {}
            justification = args.get("justification")
            confidence_val = args.get("confidence", None)
            remark_val = args.get("remark")

            try:
                if fn == "set_canonical_rel":
                    rel_ids_raw = args.get("relation_ids", []) or []
                    canon_name = args.get("canonical_rel_name")
                    canon_desc = args.get("canonical_rel_desc")
                    rel_ids_valid = [rid for rid in rel_ids_raw if rid in rel_by_id]

                    if not rel_ids_valid or canon_name is None:
                        decisions.append({
                            "action": "set_canonical_rel_skip",
                            "requested_relation_ids": rel_ids_raw,
                            "canonical_rel_name": canon_name,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val
                        })
                        continue

                    execute_set_canonical_rel(rel_by_id, rel_ids_valid, canon_name, canon_desc)

                    decisions.append({
                        "action": "set_canonical_rel",
                        "relation_ids": rel_ids_valid,
                        "canonical_rel_name": canon_name,
                        "canonical_rel_desc": canon_desc,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                elif fn == "set_rel_cls":
                    rel_ids_raw = args.get("relation_ids", []) or []
                    rel_cls = args.get("rel_cls")
                    rel_ids_valid = [rid for rid in rel_ids_raw if rid in rel_by_id]

                    if not rel_ids_valid or rel_cls is None:
                        decisions.append({
                            "action": "set_rel_cls_skip",
                            "requested_relation_ids": rel_ids_raw,
                            "rel_cls": rel_cls,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val
                        })
                        continue

                    execute_set_rel_cls(rel_by_id, rel_ids_valid, rel_cls)

                    decisions.append({
                        "action": "set_rel_cls",
                        "relation_ids": rel_ids_valid,
                        "rel_cls": rel_cls,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                elif fn == "set_rel_cls_group":
                    rel_ids_raw = args.get("relation_ids", []) or []
                    rel_cls_group = args.get("rel_cls_group")
                    rel_ids_valid = [rid for rid in rel_ids_raw if rid in rel_by_id]

                    if not rel_ids_valid or rel_cls_group is None:
                        decisions.append({
                            "action": "set_rel_cls_group_skip",
                            "requested_relation_ids": rel_ids_raw,
                            "rel_cls_group": rel_cls_group,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val
                        })
                        continue

                    execute_set_rel_cls_group(rel_by_id, rel_ids_valid, rel_cls_group)

                    decisions.append({
                        "action": "set_rel_cls_group",
                        "relation_ids": rel_ids_valid,
                        "rel_cls_group": rel_cls_group,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                elif fn == "modify_rel_schema":
                    rel_ids_raw = args.get("relation_ids", []) or []
                    canon_name = args.get("canonical_rel_name")
                    canon_desc = args.get("canonical_rel_desc")
                    rel_cls = args.get("rel_cls")
                    rel_cls_group = args.get("rel_cls_group")
                    new_rel_name = args.get("new_relation_name")
                    orig_rel_name = args.get("original_relation_name")

                    rel_ids_valid = [rid for rid in rel_ids_raw if rid in rel_by_id]
                    if not rel_ids_valid:
                        decisions.append({
                            "action": "modify_rel_schema_skip_no_valid_relations",
                            "requested_relation_ids": rel_ids_raw,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val
                        })
                        continue

                    execute_modify_rel_schema(
                        rel_by_id,
                        rel_ids_valid,
                        canon_name,
                        canon_desc,
                        rel_cls,
                        rel_cls_group,
                        new_rel_name,
                        orig_rel_name
                    )

                    decisions.append({
                        "action": "modify_rel_schema",
                        "relation_ids": rel_ids_valid,
                        "canonical_rel_name": canon_name,
                        "canonical_rel_desc": canon_desc,
                        "rel_cls": rel_cls,
                        "rel_cls_group": rel_cls_group,
                        "new_relation_name": new_rel_name,
                        "original_relation_name": orig_rel_name,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

                elif fn == "add_rel_remark":
                    rel_ids_raw = args.get("relation_ids", []) or []
                    rel_ids_valid = [rid for rid in rel_ids_raw if rid in rel_by_id]
                    remark_text = args.get("remark")

                    if not rel_ids_valid or not remark_text:
                        decisions.append({
                            "action": "add_rel_remark_skip",
                            "requested_relation_ids": rel_ids_raw,
                            "remark": remark_text,
                            "justification": justification,
                            "confidence": confidence_val
                        })
                        continue

                    execute_add_rel_remark(rel_by_id, rel_ids_valid, remark_text)

                    decisions.append({
                        "action": "add_rel_remark",
                        "relation_ids": rel_ids_valid,
                        "remark": remark_text,
                        "justification": justification,
                        "confidence": confidence_val
                    })

                else:
                    decisions.append({
                        "action": "skip_unknown_function",
                        "function": fn,
                        "raw": step,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val
                    })

            except Exception as e:
                decisions.append({
                    "action": "error_executing",
                    "function": fn,
                    "error": str(e),
                    "input": step,
                    "justification": justification,
                    "remark": remark_val,
                    "confidence": confidence_val
                })

        # write decisions file for this chunk
        dec_path = OUT_DIR / f"cluster_{cluster_label_str}_decisions.json"
        dec_obj = {
            "cluster_label": cluster_label_str,
            "cluster_relations": cluster_relations,
            "llm_raw": raw_out,
            "parsed_steps": parsed,
            "executed_decisions": decisions,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        dec_path.write_text(json.dumps(dec_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # append to global action log
        with action_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dec_obj, ensure_ascii=False) + "\n")

    # iterate clusters (skip noise -1 first, then noise)
    cluster_keys = [k for k in cluster_to_relids.keys() if k != -1]
    cluster_keys = sorted(cluster_keys)
    if -1 in cluster_to_relids:
        cluster_keys.append(-1)

    for cluster_label in cluster_keys:
        rel_ids_global = cluster_to_relids.get(cluster_label, [])
        if not rel_ids_global:
            continue
        print(f"[cluster] {cluster_label} -> {len(rel_ids_global)} relations")

        # local subclustering for large clusters
        if len(rel_ids_global) > MAX_CLUSTER_SIZE_FOR_LOCAL:
            print(f"[cluster] {cluster_label}: size {len(rel_ids_global)} > {MAX_CLUSTER_SIZE_FOR_LOCAL}, running local HDBSCAN")
            idxs = [rel_id_to_index[rid] for rid in rel_ids_global]
            try:
                sub_emb = combined_emb[idxs]
                local_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(2, LOCAL_HDBSCAN_MIN_CLUSTER_SIZE),
                    min_samples=LOCAL_HDBSCAN_MIN_SAMPLES,
                    metric="euclidean",
                    cluster_selection_method="eom"
                )
                local_labels = local_clusterer.fit_predict(sub_emb)
            except Exception as e:
                print(f"[warn] local HDBSCAN failed for cluster {cluster_label}: {e}. Treating as single group.")
                local_labels = np.zeros(len(idxs), dtype=int)

            local_map: Dict[int, List[str]] = {}
            for i_local, lab_local in enumerate(local_labels):
                rid = rel_ids_global[i_local]
                local_map.setdefault(int(lab_local), []).append(rid)

            for lab_local, local_rel_ids in sorted(local_map.items(), key=lambda x: x[0]):
                label_prefix = f"{cluster_label}_loc{lab_local}"
                # split into chunks for LLM
                for part_idx in range(0, len(local_rel_ids), MAX_MEMBERS_PER_PROMPT):
                    part_rel_ids = local_rel_ids[part_idx:part_idx + MAX_MEMBERS_PER_PROMPT]
                    chunk_label = f"{label_prefix}_p{part_idx//MAX_MEMBERS_PER_PROMPT}"
                    print(f"[cluster] {chunk_label}: processing {len(part_rel_ids)} relations")
                    run_llm_on_subset(part_rel_ids, chunk_label)

        else:
            label_prefix = str(cluster_label)
            for part_idx in range(0, len(rel_ids_global), MAX_MEMBERS_PER_PROMPT):
                part_rel_ids = rel_ids_global[part_idx:part_idx + MAX_MEMBERS_PER_PROMPT]
                chunk_label = f"{label_prefix}_p{part_idx//MAX_MEMBERS_PER_PROMPT}"
                print(f"[cluster] {chunk_label}: processing {len(part_rel_ids)} relations")
                run_llm_on_subset(part_rel_ids, chunk_label)

    # After all chunks processed: write final relations output
    final_relations = list(rel_by_id.values())
    out_json = OUT_DIR / "relations_resolved.json"
    out_jsonl = OUT_DIR / "relations_resolved.jsonl"
    out_json.write_text(json.dumps(final_relations, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in final_relations:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] wrote final resolved relations -> {out_json}  (count={len(final_relations)})")
    print(f"[done] action log -> {action_log_path}")

    # ---------------------- SUMMARY AGGREGATION ---------------------

    summary_dir = OUT_DIR / "summary"
    summary_dir.mkdir(exist_ok=True)

    # Aggregate per-chunk decision files
    cluster_decisions: List[Dict[str, Any]] = []
    for path in sorted(OUT_DIR.glob("cluster_*_decisions.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            cluster_decisions.append(obj)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}")

    all_clusters_decisions_path = summary_dir / "all_clusters_decisions.json"
    all_clusters_decisions_path.write_text(
        json.dumps(cluster_decisions, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    total_clusters = len(cluster_decisions)  # here "cluster" means one LLM chunk
    actions_by_type: Dict[str, int] = {}
    total_remarks = 0
    clusters_with_any_decisions = 0

    for cd in cluster_decisions:
        decs = cd.get("executed_decisions", [])
        if not decs:
            continue
        clusters_with_any_decisions += 1
        for d in decs:
            act = d.get("action")
            actions_by_type[act] = actions_by_type.get(act, 0) + 1
            rem = d.get("remark")
            if rem:
                total_remarks += 1

    total_errors = actions_by_type.get("error_executing", 0)

    stats = {
        "total_chunks": total_clusters,
        "total_chunks_with_any_decisions": clusters_with_any_decisions,
        "total_actions_by_type": actions_by_type,
        "total_errors": total_errors,
        "total_remarks_logged": total_remarks,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    stats_path = summary_dir / "stats_summary.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] summary decisions -> {all_clusters_decisions_path}")
    print(f"[done] summary stats -> {stats_path}")

    # ---------------------- SCHEMA AGGREGATION ----------------------

    # 1) Canonical relation schema
    canonical_map: Dict[str, Dict[str, Any]] = {}
    for r in final_relations:
        cname = safe_str(r.get("canonical_rel_name", ""))
        if not cname or cname.upper() == "TBD":
            continue
        cdesc = safe_str(r.get("canonical_rel_desc", ""))
        rel_cls = safe_str(r.get("rel_cls", ""))
        rel_grp = safe_str(r.get("rel_cls_group", ""))
        rid = r.get("relation_id")

        entry = canonical_map.setdefault(cname, {
            "canonical_rel_name": cname,
            "canonical_rel_desc_candidates": set(),
            "rel_cls_set": set(),
            "rel_cls_group_set": set(),
            "relation_ids": []
        })
        if cdesc:
            entry["canonical_rel_desc_candidates"].add(cdesc)
        if rel_cls and rel_cls.upper() != "TBD":
            entry["rel_cls_set"].add(rel_cls)
        if rel_grp and rel_grp.upper() != "TBD":
            entry["rel_cls_group_set"].add(rel_grp)
        if rid:
            entry["relation_ids"].append(rid)

    canonical_schema = []
    for cname, info in canonical_map.items():
        desc_candidates = list(info["canonical_rel_desc_candidates"])
        chosen_desc = desc_candidates[0] if desc_candidates else ""
        canonical_schema.append({
            "canonical_rel_name": cname,
            "canonical_rel_desc": chosen_desc,
            "rel_cls": sorted(info["rel_cls_set"]),
            "rel_cls_group": sorted(info["rel_cls_group_set"]),
            "instance_count": len(info["relation_ids"]),
            "example_relation_ids": info["relation_ids"][:10]
        })

    canonical_schema_path = summary_dir / "canonical_rel_schema.json"
    canonical_schema_path.write_text(
        json.dumps(canonical_schema, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 2) Relation class schema
    cls_map: Dict[str, Dict[str, Any]] = {}
    for r in final_relations:
        cls_name = safe_str(r.get("rel_cls", ""))
        if not cls_name or cls_name.upper() == "TBD":
            continue
        grp_name = safe_str(r.get("rel_cls_group", ""))
        cname = safe_str(r.get("canonical_rel_name", ""))
        rid = r.get("relation_id")

        entry = cls_map.setdefault(cls_name, {
            "rel_cls": cls_name,
            "rel_cls_group_set": set(),
            "canonical_rel_names": set(),
            "relation_ids": []
        })
        if grp_name and grp_name.upper() != "TBD":
            entry["rel_cls_group_set"].add(grp_name)
        if cname and cname.upper() != "TBD":
            entry["canonical_rel_names"].add(cname)
        if rid:
            entry["relation_ids"].append(rid)

    cls_schema = []
    for cls_name, info in cls_map.items():
        cls_schema.append({
            "rel_cls": cls_name,
            "rel_cls_group": sorted(info["rel_cls_group_set"]),
            "canonical_rel_names": sorted(info["canonical_rel_names"]),
            "instance_count": len(info["relation_ids"]),
            "example_relation_ids": info["relation_ids"][:10]
        })

    cls_schema_path = summary_dir / "rel_cls_schema.json"
    cls_schema_path.write_text(
        json.dumps(cls_schema, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 3) Relation class group schema
    grp_map: Dict[str, Dict[str, Any]] = {}
    for r in final_relations:
        grp_name = safe_str(r.get("rel_cls_group", ""))
        if not grp_name or grp_name.upper() == "TBD":
            continue
        cls_name = safe_str(r.get("rel_cls", ""))
        cname = safe_str(r.get("canonical_rel_name", ""))
        rid = r.get("relation_id")

        entry = grp_map.setdefault(grp_name, {
            "rel_cls_group": grp_name,
            "rel_cls_set": set(),
            "canonical_rel_names": set(),
            "relation_ids": []
        })
        if cls_name and cls_name.upper() != "TBD":
            entry["rel_cls_set"].add(cls_name)
        if cname and cname.upper() != "TBD":
            entry["canonical_rel_names"].add(cname)
        if rid:
            entry["relation_ids"].append(rid)

    grp_schema = []
    for grp_name, info in grp_map.items():
        grp_schema.append({
            "rel_cls_group": grp_name,
            "rel_cls": sorted(info["rel_cls_set"]),
            "canonical_rel_names": sorted(info["canonical_rel_names"]),
            "instance_count": len(info["relation_ids"]),
            "example_relation_ids": info["relation_ids"][:10]
        })

    grp_schema_path = summary_dir / "rel_cls_group_schema.json"
    grp_schema_path.write_text(
        json.dumps(grp_schema, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[done] canonical relation schema -> {canonical_schema_path}")
    print(f"[done] relation class schema -> {cls_schema_path}")
    print(f"[done] relation class group schema -> {grp_schema_path}")

# if __name__ == "__main__":
#     relres_main()

#endregion#?   Rel Res V4  - Canonical + RelCls + RelClsGroup + Schema + LocalSubcluster
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Multi Run Rel Res

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------
# CONFIG - Rel Res iterative
# -----------------------

# First-run input: raw relations from Rel Rec
BASE_INPUT_RELATIONS = Path("SGCE-KG/data/Relations/Rel Rec/relations_raw.jsonl")

# Root for iterative runs; each run gets its own subfolder
EXPERIMENT_ROOT = Path("SGCE-KG/data/Relations/Rel Res_IterativeRuns")

MAX_RUNS: int = 4

# If total schema-modifying actions in a run <= SCHEMA_CHANGE_THRESHOLD,
# that run is considered "no-change" w.r.t schema.
SCHEMA_CHANGE_THRESHOLD: Optional[int] = 0

# Optional: if total actions (including skips, remarks, etc.) <= this threshold,
# you may also treat the run as "no-change".
TOTAL_ACTIONS_THRESHOLD: Optional[int] = None

# Stop after this many consecutive "no-change" runs (if not None)
MAX_NO_CHANGE_RUNS: Optional[int] = 1

FINAL_RELATIONS_FILENAME_JSON = "relations_resolved.json"
FINAL_RELATIONS_FILENAME_JSONL = "relations_resolved.jsonl"
ACTION_LOG_FILENAME = "rel_res_action_log.jsonl"

# -----------------------
# Helpers
# -----------------------

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _safe_json_load_line(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None

def compute_run_summary_from_action_log(action_log_path: Path) -> Dict[str, Any]:
    """
    Summarize a single Rel Res run based on rel_res_action_log.jsonl.

    We treat the following actions as "schema-modifying":
      - set_canonical_rel
      - set_rel_cls
      - set_rel_cls_group
      - modify_rel_schema

    Remarks-only:
      - add_rel_remark

    Everything else (skips, errors, unknown) is counted but not schema-changing.
    """
    summary = {
        "total_chunks": 0,                          # each line ~ one LLM chunk
        "total_chunks_with_any_decisions": 0,
        "total_chunks_with_schema_changes": 0,
        "total_actions_by_type": {},
        "total_schema_actions": 0,
        "total_remark_actions": 0,                  # add_rel_remark calls
        "total_errors": 0,
        "total_remarks_logged": 0,                  # remark fields in actions
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    if not action_log_path.exists():
        return summary

    schema_actions = {"set_canonical_rel", "set_rel_cls", "set_rel_cls_group", "modify_rel_schema"}

    with action_log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = _safe_json_load_line(line)
            if not obj:
                continue
            summary["total_chunks"] += 1
            executed = obj.get("executed_decisions", []) or []
            if executed:
                summary["total_chunks_with_any_decisions"] += 1

            schema_here = 0
            remark_actions_here = 0
            remarks_logged_here = 0

            for entry in executed:
                action = entry.get("action")
                summary["total_actions_by_type"].setdefault(action, 0)
                summary["total_actions_by_type"][action] += 1

                if action in schema_actions:
                    schema_here += 1
                elif action == "add_rel_remark":
                    remark_actions_here += 1
                elif action == "error_executing":
                    summary["total_errors"] += 1

                # count explicit remark fields
                if isinstance(entry, dict) and entry.get("remark"):
                    remarks_logged_here += 1

            summary["total_schema_actions"] += schema_here
            summary["total_remark_actions"] += remark_actions_here
            if schema_here > 0:
                summary["total_chunks_with_schema_changes"] += 1
            summary["total_remarks_logged"] += remarks_logged_here

    return summary

# -----------------------
# Main iterative runner for Rel Res
# -----------------------

def run_relres_iteratively():
    """
    Run relres_main() multiple times, feeding the previous run's
    relations_resolved.jsonl as the next run's input, until convergence
    or MAX_RUNS is reached.

    Convergence is defined via:
      - SCHEMA_CHANGE_THRESHOLD
      - TOTAL_ACTIONS_THRESHOLD
      - MAX_NO_CHANGE_RUNS
    """
    from pathlib import Path  # ensure Path is in local scope if importing externally

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

    overall_runs: List[Dict[str, Any]] = []
    no_change_streak = 0

    # First run input is the raw relations from Rel Rec
    current_input_path = BASE_INPUT_RELATIONS
    last_run_dir: Optional[Path] = None

    for run_idx in range(MAX_RUNS):
        print("\n" + "=" * 36)
        print(f"=== REL RES RUN {run_idx:02d} ===")
        print("=" * 36)

        run_dir = EXPERIMENT_ROOT / f"run_{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        last_run_dir = run_dir

        # Set globals used by Rel Res V4 pipeline
        globals()["INPUT_RELATIONS"] = current_input_path
        globals()["OUT_DIR"] = run_dir
        globals()["RAW_LLM_DIR"] = run_dir / "llm_raw"
        globals()["RAW_LLM_DIR"].mkdir(parents=True, exist_ok=True)

        # Call the pipeline's main function (assumes relres_main defined already)
        print(f"[run {run_idx}] calling relres_main() with INPUT_RELATIONS={current_input_path} OUT_DIR={run_dir}")
        relres_main()

        final_relations_json = run_dir / FINAL_RELATIONS_FILENAME_JSON
        final_relations_jsonl = run_dir / FINAL_RELATIONS_FILENAME_JSONL
        action_log_path = run_dir / ACTION_LOG_FILENAME

        run_summary = compute_run_summary_from_action_log(action_log_path)
        run_summary["run_index"] = run_idx
        run_summary["run_path"] = str(run_dir)
        run_summary["final_relations_json"] = str(final_relations_json) if final_relations_json.exists() else None
        run_summary["final_relations_jsonl"] = str(final_relations_jsonl) if final_relations_jsonl.exists() else None

        summary_dir = run_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        run_summary_path = summary_dir / "rel_res_summary.json"
        _write_json(run_summary_path, run_summary)

        overall_runs.append({
            "run_index": run_idx,
            "run_dir": str(run_dir),
            "summary_path": str(run_summary_path)
        })

        total_schema = int(run_summary.get("total_schema_actions", 0))
        total_actions = int(sum(run_summary.get("total_actions_by_type", {}).values() or []))

        print(f"[run {run_idx}] total_schema_actions = {total_schema}")
        print(f"[run {run_idx}] total_actions = {total_actions}")

        # Determine if this run counts as "no-change"
        is_no_change = False
        if SCHEMA_CHANGE_THRESHOLD is not None and total_schema <= SCHEMA_CHANGE_THRESHOLD:
            is_no_change = True
        if TOTAL_ACTIONS_THRESHOLD is not None and total_actions <= TOTAL_ACTIONS_THRESHOLD:
            is_no_change = True

        if MAX_NO_CHANGE_RUNS is not None:
            if is_no_change:
                no_change_streak += 1
            else:
                no_change_streak = 0

            print(f"[run {run_idx}] no_change_streak = {no_change_streak} (threshold {MAX_NO_CHANGE_RUNS})")
            if no_change_streak >= MAX_NO_CHANGE_RUNS:
                print(f"[stop] Rel Res convergence achieved after run {run_idx} (no_change_streak={no_change_streak}).")
                # still update current_input_path for downstream use
                if final_relations_jsonl.exists():
                    current_input_path = final_relations_jsonl
                break

        # Prepare next-run input
        if final_relations_jsonl.exists():
            current_input_path = final_relations_jsonl
        else:
            print(f"[warn] final relations jsonl not found for run {run_idx}. Stopping iterative runs.")
            break

    # -----------------------
    # OVERALL SUMMARY EXPORTS
    # -----------------------
    overall_dir = EXPERIMENT_ROOT / "overall_summary"
    overall_dir.mkdir(parents=True, exist_ok=True)

    # collect per-run summaries
    per_run_stats: List[Dict[str, Any]] = []
    for r in overall_runs:
        sp = Path(r["summary_path"])
        if sp.exists():
            try:
                per_run_stats.append(_load_json(sp))
            except Exception:
                per_run_stats.append({"run_index": r["run_index"], "error": "failed to load summary"})

    aggregated = {
        "total_runs_executed": len(per_run_stats),
        "sum_total_chunks": sum([p.get("total_chunks", 0) for p in per_run_stats]),
        "sum_total_schema_actions": sum([p.get("total_schema_actions", 0) for p in per_run_stats]),
        "sum_total_errors": sum([p.get("total_errors", 0) for p in per_run_stats]),
        "sum_total_remarks_logged": sum([p.get("total_remarks_logged", 0) for p in per_run_stats]),
        "by_run": per_run_stats,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    stats_path = overall_dir / "stats.json"
    _write_json(stats_path, aggregated)

    # Copy final relations + schema from the last run
    if last_run_dir is not None:
        final_relations_json = last_run_dir / FINAL_RELATIONS_FILENAME_JSON
        final_relations_jsonl = last_run_dir / FINAL_RELATIONS_FILENAME_JSONL
        final_summary_dir = last_run_dir / "summary"

        final_relations = []
        if final_relations_json.exists():
            try:
                final_relations = _load_json(final_relations_json)
            except Exception:
                final_relations = []

        # write a copy of final relations into overall_summary
        final_relations_path_out = overall_dir / "relations_resolved.json"
        final_relations_jsonl_path_out = overall_dir / "relations_resolved.jsonl"
        _write_json(final_relations_path_out, final_relations)

        if final_relations_jsonl.exists():
            # copy jsonl line-by-line
            with final_relations_jsonl.open("r", encoding="utf-8") as src, \
                 final_relations_jsonl_path_out.open("w", encoding="utf-8") as dst:
                for line in src:
                    dst.write(line)

        # also copy the latest schema files for convenience
        canonical_schema_src = final_summary_dir / "canonical_rel_schema.json"
        rel_cls_schema_src = final_summary_dir / "rel_cls_schema.json"
        rel_cls_group_schema_src = final_summary_dir / "rel_cls_group_schema.json"

        if canonical_schema_src.exists():
            canonical_schema = _load_json(canonical_schema_src)
            _write_json(overall_dir / "canonical_rel_schema.json", canonical_schema)

        if rel_cls_schema_src.exists():
            cls_schema = _load_json(rel_cls_schema_src)
            _write_json(overall_dir / "rel_cls_schema.json", cls_schema)

        if rel_cls_group_schema_src.exists():
            grp_schema = _load_json(rel_cls_group_schema_src)
            _write_json(overall_dir / "rel_cls_group_schema.json", grp_schema)

        print(f"\n[done] Overall stats written to: {stats_path}")
        print(f"[done] Final relations (copy) written to: {final_relations_path_out}")
        print(f"[done] Final relations jsonl (copy) written to: {final_relations_jsonl_path_out}")
        print(f"[done] Final canonical_rel_schema copied to overall_summary (if present)")
        print(f"[done] Final rel_cls_schema copied to overall_summary (if present)")
        print(f"[done] Final rel_cls_group_schema copied to overall_summary (if present)")
    else:
        print("[warn] No runs executed; nothing to export to overall_summary.")




# # # -----------------------
# # Relation Res Multi Run - Run statement
# # -----------------------

# run_relres_iteratively() 



#endregion#?   Multi Run Rel Res
#?#########################  End  ##########################







#endregion#! Relation Identification
#!#############################################  End Chapter  ##################################################








#!############################################# Start Chapter ##################################################
#region:#!   KG Assembly







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
#region:#?   CSV relations + nodes for Neo4j KG import - V4




import json, csv
from pathlib import Path

# Source JSONL (same as your V2/V3)
relations_jl = Path("SGCE-KG/data/Relations/Rel Res_IterativeRuns/overall_summary/relations_resolved.jsonl")

# Outputs
rels_out_csv  = Path("SGCE-KG/data/KG/rels_fixed_no_raw.csv")
nodes_out_csv = Path("SGCE-KG/data/KG/nodes.csv")

def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)

def first_non_empty(obj, keys):
    """Return the first non-empty value among obj[key] for keys."""
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return None

# --- Load relation objects (same robustness as your code) ---
rows = []
if not relations_jl.exists():
    raise FileNotFoundError(relations_jl)

with relations_jl.open("r", encoding="utf-8") as fh:
    for ln in fh:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            # line might be a JSON array
            try:
                arr = json.loads(ln)
                if isinstance(arr, list):
                    rows.extend(arr)
                continue
            except Exception:
                raise
        # If wrapper has "relations" list
        if isinstance(obj, dict) and "relations" in obj and isinstance(obj["relations"], list):
            rows.extend(obj["relations"])
        elif isinstance(obj, list):
            rows.extend(obj)
        else:
            rows.append(obj)

print(f"Loaded {len(rows)} relation objects.")

# --- Build relations CSV + collect entity info at the same time ---

rels_rows = []
entities = {}  # entity_id -> {entity_id, entity_name, entity_description, class_label, class_group, seen_in_chunks:set}

for r in rows:
    # IDs for subject and object
    subj = r.get("subject_entity_id") or r.get("subject_id") or r.get("subject_entity") or r.get("subject_entity_name") or r.get("subject")
    objt = r.get("object_entity_id")  or r.get("object_id")  or r.get("object_entity")  or r.get("object_entity_name") or r.get("object")
    subj_id = safe_str(subj)
    objt_id = safe_str(objt)

    # chunk_id (used later to build seen_in_chunks for nodes)
    chunk_id_val = r.get("chunk_id") or r.get("source_chunk") or r.get("context_chunk") or None
    chunk_id = safe_str(chunk_id_val)

    # --- relation row (this is your V3 structure, no raw_relation_object) ---
    relation_row = {
        "relation_id": safe_str(r.get("relation_id") or r.get("id") or r.get("rid") or ""),
        "start_id":    subj_id,
        "end_id":      objt_id,
        "raw_relation_name":   safe_str(r.get("relation_name") or r.get("rel_name") or ""),
        "canonical_rel_name":  safe_str(r.get("canonical_rel_name") or r.get("canonical") or ""),
        "canonical_rel_desc":  safe_str(r.get("canonical_rel_desc") or r.get("canonical_desc") or ""),
        "rel_cls":             safe_str(r.get("rel_cls") or r.get("relation_class") or ""),
        "rel_cls_group":       safe_str(r.get("rel_cls_group") or r.get("relation_class_group") or r.get("rel_group") or ""),
        "rel_hint_type":       safe_str(r.get("rel_hint_type") or r.get("hint") or ""),
        "confidence":          safe_str(r.get("confidence") if r.get("confidence") not in (None, "") else ""),
        "resolution_context":  safe_str(r.get("resolution_context") or r.get("resolution") or ""),
        "justification":       safe_str(r.get("justification") or ""),
        "remark":              safe_str(r.get("remark") or r.get("remarks") or ""),
        "evidence_excerpt":    safe_str(r.get("evidence_excerpt") or r.get("evidence") or ""),
        "chunk_id":            chunk_id,
        "qualifiers":          json.dumps(r.get("qualifiers") or r.get("qualifier") or {}, ensure_ascii=False),
        "rel_desc":            safe_str(r.get("rel_desc") or r.get("relation_description") or "")
    }
    rels_rows.append(relation_row)

    # --- helper: update one entity record from this relation row ---
    def update_entity(side_prefix, entity_id):
        if not entity_id:
            return

        ent = entities.get(entity_id)
        if ent is None:
            ent = {
                "entity_id":          entity_id,
                "entity_name":        "",
                "entity_description": "",
                "class_label":        "",
                "class_group":        "",
                "seen_in_chunks":     set()
            }
            entities[entity_id] = ent

        if side_prefix == "subject":
            name_keys        = ["subject_entity_name", "subject_name", "subject"]
            desc_keys        = ["subject_entity_description", "subject_description", "subject_desc", "subject_entity_desc"]
            class_label_keys = ["subject_class_label", "subject_entity_class", "subject_entity_type", "subject_label", "subject_cls"]
            class_group_keys = ["subject_class_group", "subject_entity_group", "subject_group", "subject_cls_group"]
        else:
            name_keys        = ["object_entity_name", "object_name", "object"]
            desc_keys        = ["object_entity_description", "object_description", "object_desc", "object_entity_desc"]
            class_label_keys = ["object_class_label", "object_entity_class", "object_entity_type", "object_label", "object_cls"]
            class_group_keys = ["object_class_group", "object_entity_group", "object_group", "object_cls_group"]

        if not ent["entity_name"]:
            name_val = first_non_empty(r, name_keys)
            if name_val is not None:
                ent["entity_name"] = safe_str(name_val)

        if not ent["entity_description"]:
            desc_val = first_non_empty(r, desc_keys)
            if desc_val is not None:
                ent["entity_description"] = safe_str(desc_val)

        if not ent["class_label"]:
            cls_val = first_non_empty(r, class_label_keys)
            if cls_val is not None:
                ent["class_label"] = safe_str(cls_val)

        if not ent["class_group"]:
            grp_val = first_non_empty(r, class_group_keys)
            if grp_val is not None:
                ent["class_group"] = safe_str(grp_val)

        if chunk_id:
            ent["seen_in_chunks"].add(chunk_id)

    # update both ends
    update_entity("subject", subj_id)
    update_entity("object", objt_id)

print(f"Collected {len(rels_rows)} relation rows.")
print(f"Collected {len(entities)} unique entities from relations.")

# --- Write relations CSV (rels_fixed_no_raw.csv) ---

rel_fieldnames = [
    "relation_id","start_id","end_id","raw_relation_name","canonical_rel_name","canonical_rel_desc",
    "rel_cls","rel_cls_group","rel_hint_type","confidence","resolution_context","justification",
    "remark","evidence_excerpt","chunk_id","qualifiers","rel_desc"
]

rels_out_csv.parent.mkdir(parents=True, exist_ok=True)
with rels_out_csv.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=rel_fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for rrow in rels_rows:
        writer.writerow(rrow)

print("Wrote relations CSV:", rels_out_csv)

# --- Write nodes CSV (nodes.csv) ---

node_fieldnames = [
    "entity_id","entity_name","entity_description","class_label","class_group","seen_in_chunks"
]

nodes_out_csv.parent.mkdir(parents=True, exist_ok=True)
with nodes_out_csv.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=node_fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for ent in entities.values():
        row = {
            "entity_id":          ent["entity_id"],
            "entity_name":        ent["entity_name"],
            "entity_description": ent["entity_description"],
            "class_label":        ent["class_label"],
            "class_group":        ent["class_group"],
            # store as JSON list string, so Neo4j can parse with apoc.convert.fromJsonList
            "seen_in_chunks":     json.dumps(sorted(ent["seen_in_chunks"]), ensure_ascii=False),
        }
        writer.writerow(row)

print("Wrote nodes CSV:", nodes_out_csv)



#endregion#? CSV relations + nodes for Neo4j KG import - V4
#*#########################  End  ##########################




#*######################### Start ##########################
#region:#?   Cypher Queries - V4

MATCH (n)
DETACH DELETE n;










LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CALL (row) {
  WITH row
  // skip empty rows
  WITH row WHERE row.entity_id IS NOT NULL AND row.entity_id <> ''
  MERGE (e:Entity {entity_id: row.entity_id})
  SET
    e.name              = row.entity_name,
    e.description       = row.entity_description,
    e.class_label       = row.class_label,
    e.class_group       = row.class_group,
    e.seen_in_chunks    = apoc.convert.fromJsonList(row.seen_in_chunks)
} IN TRANSACTIONS OF 1000 ROWS;







LOAD CSV WITH HEADERS FROM 'file:///rels_fixed_no_raw.csv' AS row
CALL (row) {
  WITH row
  // skip rows missing endpoints
  WITH row WHERE row.start_id IS NOT NULL AND row.start_id <> '' AND row.end_id IS NOT NULL AND row.end_id <> ''
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
    r.confidence         = CASE row.confidence WHEN "" THEN NULL ELSE toFloat(row.confidence) END,
    r.description        = row.rel_desc,
    r.resolution_context = row.resolution_context,
    r.justification      = row.justification,
    r.remark             = row.remark,
    r.evidence_excerpt   = row.evidence_excerpt,
    r.chunk_id           = row.chunk_id,
    r.qualifiers_json    = row.qualifiers
  WITH r, qmap
  FOREACH (k IN keys(qmap) |
    SET r['qual_' + k] = toString(qmap[k])
  )
} IN TRANSACTIONS OF 1000 ROWS;






MATCH (n)
RETURN n





MATCH (n)-[r]->(m)
RETURN n, r, m










MATCH (e:Entity)
WHERE e.class_group IS NOT NULL
CALL apoc.create.addLabels(e, [e.class_group]) YIELD node
RETURN count(node);




// remove Entity label from nodes that have a viz label
MATCH (e:Entity)
WHERE e._viz_label IS NOT NULL AND e._viz_label <> ''
REMOVE e:Entity
RETURN count(e) AS entity_labels_removed;
















#?######################### Start ##########################
#region:#?   See classes as colors

// create a sanitized label string in property `_viz_label`
MATCH (e:Entity)
WHERE e.class_group IS NOT NULL AND e.class_group <> ''
WITH e,
     replace(replace(replace(e.class_group, ' ', '_'), '-', '_'), '/', '_') AS lab0
WITH e,
     // ensure label has only letters/digits/underscores at start (basic sanitization)
     apoc.text.regreplace(lab0, '[^A-Za-z0-9_]', '_') AS lab1
WITH e,
     CASE WHEN lab1 =~ '^[A-Za-z].*' THEN lab1 ELSE 'C_' + lab1 END AS vizlabel
SET e._viz_label = vizlabel
RETURN count(e) AS nodes_updated;




// add labels from _viz_label (one label per node)
MATCH (e:Entity)
WHERE e._viz_label IS NOT NULL AND e._viz_label <> ''
CALL apoc.create.addLabels(e, [e._viz_label]) YIELD node
RETURN count(node) AS labels_added;




// remove Entity label from nodes that have a viz label
MATCH (e:Entity)
WHERE e._viz_label IS NOT NULL AND e._viz_label <> ''
REMOVE e:Entity
RETURN count(e) AS entity_labels_removed;




#endregion#? See classes as colors
#?#########################  End  ##########################







#endregion#? Cypher Queries
#*#########################  End  ##########################



#*######################### Start ##########################
#region:#?   CSV relations + nodes for Neo4j KG import - V6


import json, csv
from pathlib import Path

# Sources
relations_jl = Path(
    "SGCE-KG/data/Relations/Rel Res_IterativeRuns/overall_summary/relations_resolved.jsonl"
)
entities_cls_jl = Path(
    "SGCE-KG/data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl"
)

# Outputs
rels_out_csv  = Path("SGCE-KG/data/KG/rels_fixed_no_raw.csv")
nodes_out_csv = Path("SGCE-KG/data/KG/nodes.csv")


def sanitize_string_for_csv_json(s):
    """
    For content that will be embedded inside JSON *inside* CSV (e.g., qualifiers),
    remove problematic double quotes that produce \" sequences which break Neo4j's
    CSV parser. Replace them with single quotes.
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.replace('"', "'")


def sanitize_for_json_in_csv(obj):
    """
    Recursively sanitize dict/list structures so that any string values
    no longer contain raw double quotes. Keys almost never contain quotes,
    but we treat them too for safety.
    """
    if isinstance(obj, dict):
        return {
            sanitize_string_for_csv_json(k) if isinstance(k, str) else k:
                sanitize_for_json_in_csv(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [sanitize_for_json_in_csv(v) for v in obj]
    elif isinstance(obj, str):
        return sanitize_string_for_csv_json(obj)
    else:
        return obj


def safe_str(x):
    """
    Safe string conversion for scalar values.
    For dict/list, we JSON-encode after sanitizing for CSV safety.
    """
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        return json.dumps(sanitize_for_json_in_csv(x), ensure_ascii=False)
    return str(x)


def first_non_empty(obj, keys):
    """Return the first non-empty value among obj[key] for keys."""
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return None


def new_entity_record(entity_id):
    """Initialize a canonical entity record in our in-memory table."""
    return {
        "entity_id":               entity_id,
        "entity_name":             "",
        "entity_description":      "",
        "entity_type_hint":        "",
        "entity_confidence":       "",  # stored as string in CSV
        "entity_resolution_context": "",
        "entity_flag":             "",
        "class_id":                "",
        "class_label":             "",
        "class_group":             "",
        "node_properties":         [],
        "seen_in_chunks":          set(),  # internal; becomes chunk_ids in CSV
    }


# --- 1) Load entities from entities_with_class.jsonl (primary source of node info) ---

entities = {}  # entity_id -> entity_record

if not entities_cls_jl.exists():
    raise FileNotFoundError(entities_cls_jl)

with entities_cls_jl.open("r", encoding="utf-8") as fh:
    for ln in fh:
        ln = ln.strip()
        if not ln:
            continue
        obj = json.loads(ln)

        # entity_id is top-level, but we also fall back to nested entity.id
        eid = obj.get("entity_id")
        nested_ent = obj.get("entity") or {}
        if eid is None:
            eid = nested_ent.get("id")
        if not eid:
            continue

        ent = entities.get(eid)
        if ent is None:
            ent = new_entity_record(eid)
            entities[eid] = ent

        # --- Fill from nested "entity" object first ---
        # Name & description
        if not ent["entity_name"]:
            val = nested_ent.get("entity_name") or nested_ent.get("name") or obj.get("entity_name")
            if val is not None:
                ent["entity_name"] = safe_str(val)

        if not ent["entity_description"]:
            val = nested_ent.get("entity_description") or nested_ent.get("description") or obj.get("entity_description")
            if val is not None:
                ent["entity_description"] = safe_str(val)

        # Type hint
        if not ent["entity_type_hint"]:
            val = nested_ent.get("entity_type_hint") or obj.get("entity_type_hint")
            if val is not None:
                ent["entity_type_hint"] = safe_str(val)

        # Confidence score
        if ent["entity_confidence"] in ("", None):
            val = nested_ent.get("confidence_score") or obj.get("confidence_score")
            if val is not None:
                ent["entity_confidence"] = str(val)

        # Resolution context
        if not ent["entity_resolution_context"]:
            val = nested_ent.get("resolution_context") or obj.get("resolution_context")
            if val is not None:
                ent["entity_resolution_context"] = safe_str(val)

        # Flag
        if not ent["entity_flag"]:
            val = nested_ent.get("flag") or obj.get("flag")
            if val is not None:
                ent["entity_flag"] = safe_str(val)

        # Class info: prefer top-level, then nested _class_*
        if not ent["class_id"]:
            val = obj.get("class_id") or nested_ent.get("_class_id")
            if val is not None:
                ent["class_id"] = safe_str(val)

        if not ent["class_label"]:
            val = obj.get("class_label") or nested_ent.get("_class_label")
            if val is not None:
                ent["class_label"] = safe_str(val)

        if not ent["class_group"]:
            val = obj.get("class_group") or nested_ent.get("_class_group")
            if val is not None:
                ent["class_group"] = safe_str(val)

        # Chunk IDs from the entities file
        chunk_candidates = []
        for key in ("chunk_id",):
            v1 = nested_ent.get(key)
            if v1 is not None:
                if isinstance(v1, list):
                    chunk_candidates.extend(v1)
                else:
                    chunk_candidates.append(v1)
            v2 = obj.get(key)
            if v2 is not None:
                if isinstance(v2, list):
                    chunk_candidates.extend(v2)
                else:
                    chunk_candidates.append(v2)

        for cid in chunk_candidates:
            if cid is not None:
                ent["seen_in_chunks"].add(str(cid))

        # Node properties from entities file
        np = nested_ent.get("node_properties") or obj.get("node_properties") or []
        if isinstance(np, list):
            for item in np:
                ent["node_properties"].append(item)
        else:
            ent["node_properties"].append(np)

print(f"Loaded {len(entities)} entities from classes file.")


# --- 2) Load relation objects (same robustness as before) ---

rows = []
if not relations_jl.exists():
    raise FileNotFoundError(relations_jl)

with relations_jl.open("r", encoding="utf-8") as fh:
    for ln in fh:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            # line might be a JSON array
            try:
                arr = json.loads(ln)
                if isinstance(arr, list):
                    rows.extend(arr)
                continue
            except Exception:
                raise
        # If wrapper has "relations" list
        if isinstance(obj, dict) and "relations" in obj and isinstance(obj["relations"], list):
            rows.extend(obj["relations"])
        elif isinstance(obj, list):
            rows.extend(obj)
        else:
            rows.append(obj)

print(f"Loaded {len(rows)} relation objects.")


def update_entity_from_relation(entities_dict, side_prefix, entity_id, rel_obj, chunk_id):
    """
    Enrich entity records with info available from relations (only filling empty fields),
    and accumulate chunk_ids from relations as well.
    """
    if not entity_id:
        return

    ent = entities_dict.get(entity_id)
    if ent is None:
        ent = new_entity_record(entity_id)
        entities_dict[entity_id] = ent

    if side_prefix == "subject":
        name_keys        = ["subject_entity_name", "subject_name", "subject"]
        desc_keys        = ["subject_entity_description", "subject_description", "subject_desc", "subject_entity_desc"]
        class_label_keys = ["subject_class_label", "subject_entity_class", "subject_entity_type", "subject_label", "subject_cls"]
        class_group_keys = ["subject_class_group", "subject_entity_group", "subject_group", "subject_cls_group"]
    else:
        name_keys        = ["object_entity_name", "object_name", "object"]
        desc_keys        = ["object_entity_description", "object_description", "object_desc", "object_entity_desc"]
        class_label_keys = ["object_class_label", "object_entity_class", "object_entity_type", "object_label", "object_cls"]
        class_group_keys = ["object_class_group", "object_entity_group", "object_group", "object_cls_group"]

    if not ent["entity_name"]:
        name_val = first_non_empty(rel_obj, name_keys)
        if name_val is not None:
            ent["entity_name"] = safe_str(name_val)

    if not ent["entity_description"]:
        desc_val = first_non_empty(rel_obj, desc_keys)
        if desc_val is not None:
            ent["entity_description"] = safe_str(desc_val)

    if not ent["class_label"]:
        cls_val = first_non_empty(rel_obj, class_label_keys)
        if cls_val is not None:
            ent["class_label"] = safe_str(cls_val)

    if not ent["class_group"]:
        grp_val = first_non_empty(rel_obj, class_group_keys)
        if grp_val is not None:
            ent["class_group"] = safe_str(grp_val)

    # accumulate chunk_id from relations as well
    if chunk_id:
        ent["seen_in_chunks"].add(chunk_id)


# --- 3) Build relations CSV + enrich entities from relations ---

rels_rows = []

for r in rows:
    # IDs for subject and object
    subj = (
        r.get("subject_entity_id")
        or r.get("subject_id")
        or r.get("subject_entity")
        or r.get("subject_entity_name")
        or r.get("subject")
    )
    objt = (
        r.get("object_entity_id")
        or r.get("object_id")
        or r.get("object_entity")
        or r.get("object_entity_name")
        or r.get("object")
    )
    subj_id = safe_str(subj)
    objt_id = safe_str(objt)

    # chunk_id (for relation + union into entity.seen_in_chunks)
    chunk_id_val = r.get("chunk_id") or r.get("source_chunk") or r.get("context_chunk") or None
    chunk_id = safe_str(chunk_id_val)

    # qualifiers: sanitize inner quotes -> pretty JSON for nicer Neo4j view
    qualifiers_obj = r.get("qualifiers") or r.get("qualifier") or {}
    if qualifiers_obj is None:
        qualifiers_obj = {}
    qualifiers_sanitized = sanitize_for_json_in_csv(qualifiers_obj)
    qualifiers_json = json.dumps(qualifiers_sanitized, ensure_ascii=False, indent=1)

    # relation row (structure same as before)
    relation_row = {
        "relation_id":         safe_str(r.get("relation_id") or r.get("id") or r.get("rid") or ""),
        "start_id":            subj_id,
        "end_id":              objt_id,
        "raw_relation_name":   safe_str(r.get("relation_name") or r.get("rel_name") or ""),
        "canonical_rel_name":  safe_str(r.get("canonical_rel_name") or r.get("canonical") or ""),
        "canonical_rel_desc":  safe_str(r.get("canonical_rel_desc") or r.get("canonical_desc") or ""),
        "rel_cls":             safe_str(r.get("rel_cls") or r.get("relation_class") or ""),
        "rel_cls_group":       safe_str(r.get("rel_cls_group") or r.get("relation_class_group") or r.get("rel_group") or ""),
        "rel_hint_type":       safe_str(r.get("rel_hint_type") or r.get("hint") or ""),
        "confidence":          safe_str(r.get("confidence") if r.get("confidence") not in (None, "") else ""),
        "resolution_context":  safe_str(r.get("resolution_context") or r.get("resolution") or ""),
        "justification":       safe_str(r.get("justification") or ""),
        "remark":              safe_str(r.get("remark") or r.get("remarks") or ""),
        "evidence_excerpt":    safe_str(r.get("evidence_excerpt") or r.get("evidence") or ""),
        "chunk_id":            chunk_id,
        "qualifiers":          qualifiers_json,
        "rel_desc":            safe_str(r.get("rel_desc") or r.get("relation_description") or ""),
    }
    rels_rows.append(relation_row)

    # Enrich entities from relations (but do not override existing rich info from classes)
    update_entity_from_relation(entities, "subject", subj_id, r, chunk_id)
    update_entity_from_relation(entities, "object", objt_id, r, chunk_id)

print(f"Collected {len(rels_rows)} relation rows.")
print(f"Collected {len(entities)} unique entities (classes + relations).")


# --- 4) Write relations CSV (rels_fixed_no_raw.csv) ---

rel_fieldnames = [
    "relation_id","start_id","end_id","raw_relation_name","canonical_rel_name","canonical_rel_desc",
    "rel_cls","rel_cls_group","rel_hint_type","confidence","resolution_context","justification",
    "remark","evidence_excerpt","chunk_id","qualifiers","rel_desc"
]

rels_out_csv.parent.mkdir(parents=True, exist_ok=True)
with rels_out_csv.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=rel_fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for rrow in rels_rows:
        writer.writerow(rrow)

print("Wrote relations CSV:", rels_out_csv)


# --- 5) Write nodes CSV (nodes.csv) ---
# NOTE: we now expose more intrinsic entity info, and rename "seen_in_chunks" -> "chunk_ids"
# for clearer, more consistent naming with relation.chunk_id.

node_fieldnames = [
    "entity_id",
    "entity_name",
    "entity_description",
    "entity_type_hint",
    "entity_confidence",
    "entity_resolution_context",
    "entity_flag",
    "class_id",
    "class_label",
    "class_group",
    "chunk_ids",
    "node_properties",
]

nodes_out_csv.parent.mkdir(parents=True, exist_ok=True)
with nodes_out_csv.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=node_fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for ent in entities.values():
        # chunk_ids: JSON list string for Neo4j apoc.convert.fromJsonList
        chunk_ids_str = json.dumps(sorted(ent["seen_in_chunks"]), ensure_ascii=False)

        # node_properties: JSON array string, sanitized
        node_props_sanitized = sanitize_for_json_in_csv(ent["node_properties"])
        node_props_str = json.dumps(node_props_sanitized, ensure_ascii=False)

        row = {
            "entity_id":               ent["entity_id"],
            "entity_name":             ent["entity_name"],
            "entity_description":      ent["entity_description"],
            "entity_type_hint":        ent["entity_type_hint"],
            "entity_confidence":       ent["entity_confidence"],
            "entity_resolution_context": ent["entity_resolution_context"],
            "entity_flag":             ent["entity_flag"],
            "class_id":                ent["class_id"],
            "class_label":             ent["class_label"],
            "class_group":             ent["class_group"],
            "chunk_ids":               chunk_ids_str,
            "node_properties":         node_props_str,
        }
        writer.writerow(row)

print("Wrote nodes CSV:", nodes_out_csv)



#endregion#? CSV relations + nodes for Neo4j KG import - V6
#*#########################  End  ##########################






#?######################### Start ##########################
#region:#?   CSV relations + nodes for Neo4j KG import - V7

def export_relations_and_nodes_to_csv():
    import json, csv
    from pathlib import Path

    # Sources
    relations_jl = Path(
        "SGCE-KG/data/Relations/Rel Res_IterativeRuns/overall_summary/relations_resolved.jsonl"
    )
    entities_cls_jl = Path(
        "SGCE-KG/data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl"
    )

    # Outputs
    rels_out_csv  = Path("SGCE-KG/data/KG/rels_fixed_no_raw.csv")
    nodes_out_csv = Path("SGCE-KG/data/KG/nodes.csv")


    def sanitize_string_for_csv_json(s):
        """
        For content that will be embedded inside JSON *inside* CSV (e.g., qualifiers),
        remove problematic double quotes that produce \" sequences which break Neo4j's
        CSV parser. Replace them with single quotes.
        """
        if s is None:
            return ""
        if not isinstance(s, str):
            s = str(s)
        return s.replace('"', "'")


    def sanitize_for_json_in_csv(obj):
        """
        Recursively sanitize dict/list structures so that any string values
        no longer contain raw double quotes. Keys almost never contain quotes,
        but we treat them too for safety.
        """
        if isinstance(obj, dict):
            return {
                sanitize_string_for_csv_json(k) if isinstance(k, str) else k:
                    sanitize_for_json_in_csv(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [sanitize_for_json_in_csv(v) for v in obj]
        elif isinstance(obj, str):
            return sanitize_string_for_csv_json(obj)
        else:
            return obj


    def safe_str(x):
        """
        Safe string conversion for scalar values.
        For dict/list, we JSON-encode after sanitizing for CSV safety.
        """
        if x is None:
            return ""
        if isinstance(x, (dict, list)):
            return json.dumps(sanitize_for_json_in_csv(x), ensure_ascii=False)
        return str(x)


    def first_non_empty(obj, keys):
        """Return the first non-empty value among obj[key] for keys."""
        for k in keys:
            if k in obj and obj[k] not in (None, ""):
                return obj[k]
        return None


    def new_entity_record(entity_id):
        """Initialize a canonical entity record in our in-memory table."""
        return {
            "entity_id":               entity_id,
            "entity_name":             "",
            "entity_description":      "",
            "entity_type_hint":        "",
            "entity_confidence":       "",  # stored as string in CSV
            "entity_resolution_context": "",
            "entity_flag":             "",
            "class_id":                "",
            "class_label":             "",
            "class_group":             "",
            "node_properties":         [],
            "seen_in_chunks":          set(),  # internal; becomes chunk_ids in CSV
        }


    # --- 1) Load entities from entities_with_class.jsonl (primary source of node info) ---

    entities = {}  # entity_id -> entity_record

    if not entities_cls_jl.exists():
        raise FileNotFoundError(entities_cls_jl)

    with entities_cls_jl.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)

            # entity_id is top-level, but we also fall back to nested entity.id
            eid = obj.get("entity_id")
            nested_ent = obj.get("entity") or {}
            if eid is None:
                eid = nested_ent.get("id")
            if not eid:
                continue

            ent = entities.get(eid)
            if ent is None:
                ent = new_entity_record(eid)
                entities[eid] = ent

            # --- Fill from nested "entity" object first ---
            # Name & description
            if not ent["entity_name"]:
                val = nested_ent.get("entity_name") or nested_ent.get("name") or obj.get("entity_name")
                if val is not None:
                    ent["entity_name"] = safe_str(val)

            if not ent["entity_description"]:
                val = nested_ent.get("entity_description") or nested_ent.get("description") or obj.get("entity_description")
                if val is not None:
                    ent["entity_description"] = safe_str(val)

            # Type hint
            if not ent["entity_type_hint"]:
                val = nested_ent.get("entity_type_hint") or obj.get("entity_type_hint")
                if val is not None:
                    ent["entity_type_hint"] = safe_str(val)

            # Confidence score
            if ent["entity_confidence"] in ("", None):
                val = nested_ent.get("confidence_score") or obj.get("confidence_score")
                if val is not None:
                    ent["entity_confidence"] = str(val)

            # Resolution context
            if not ent["entity_resolution_context"]:
                val = nested_ent.get("resolution_context") or obj.get("resolution_context")
                if val is not None:
                    ent["entity_resolution_context"] = safe_str(val)

            # Flag
            if not ent["entity_flag"]:
                val = nested_ent.get("flag") or obj.get("flag")
                if val is not None:
                    ent["entity_flag"] = safe_str(val)

            # Class info: prefer top-level, then nested _class_*
            if not ent["class_id"]:
                val = obj.get("class_id") or nested_ent.get("_class_id")
                if val is not None:
                    ent["class_id"] = safe_str(val)

            if not ent["class_label"]:
                val = obj.get("class_label") or nested_ent.get("_class_label")
                if val is not None:
                    ent["class_label"] = safe_str(val)

            if not ent["class_group"]:
                val = obj.get("class_group") or nested_ent.get("_class_group")
                if val is not None:
                    ent["class_group"] = safe_str(val)

            # Chunk IDs from the entities file
            chunk_candidates = []
            for key in ("chunk_id",):
                v1 = nested_ent.get(key)
                if v1 is not None:
                    if isinstance(v1, list):
                        chunk_candidates.extend(v1)
                    else:
                        chunk_candidates.append(v1)
                v2 = obj.get(key)
                if v2 is not None:
                    if isinstance(v2, list):
                        chunk_candidates.extend(v2)
                    else:
                        chunk_candidates.append(v2)

            for cid in chunk_candidates:
                if cid is not None:
                    ent["seen_in_chunks"].add(str(cid))

            # Node properties from entities file
            np = nested_ent.get("node_properties") or obj.get("node_properties") or []
            if isinstance(np, list):
                for item in np:
                    ent["node_properties"].append(item)
            else:
                ent["node_properties"].append(np)

    print(f"Loaded {len(entities)} entities from classes file.")


    # --- 2) Load relation objects (same robustness as before) ---

    rows = []
    if not relations_jl.exists():
        raise FileNotFoundError(relations_jl)

    with relations_jl.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                # line might be a JSON array
                try:
                    arr = json.loads(ln)
                    if isinstance(arr, list):
                        rows.extend(arr)
                    continue
                except Exception:
                    raise
            # If wrapper has "relations" list
            if isinstance(obj, dict) and "relations" in obj and isinstance(obj["relations"], list):
                rows.extend(obj["relations"])
            elif isinstance(obj, list):
                rows.extend(obj)
            else:
                rows.append(obj)

    print(f"Loaded {len(rows)} relation objects.")


    def update_entity_from_relation(entities_dict, side_prefix, entity_id, rel_obj, chunk_id):
        """
        Enrich entity records with info available from relations (only filling empty fields),
        and accumulate chunk_ids from relations as well.
        """
        if not entity_id:
            return

        ent = entities_dict.get(entity_id)
        if ent is None:
            ent = new_entity_record(entity_id)
            entities_dict[entity_id] = ent

        if side_prefix == "subject":
            name_keys        = ["subject_entity_name", "subject_name", "subject"]
            desc_keys        = ["subject_entity_description", "subject_description", "subject_desc", "subject_entity_desc"]
            class_label_keys = ["subject_class_label", "subject_entity_class", "subject_entity_type", "subject_label", "subject_cls"]
            class_group_keys = ["subject_class_group", "subject_entity_group", "subject_group", "subject_cls_group"]
        else:
            name_keys        = ["object_entity_name", "object_name", "object"]
            desc_keys        = ["object_entity_description", "object_description", "object_desc", "object_entity_desc"]
            class_label_keys = ["object_class_label", "object_entity_class", "object_entity_type", "object_label", "object_cls"]
            class_group_keys = ["object_class_group", "object_entity_group", "object_group", "object_cls_group"]

        if not ent["entity_name"]:
            name_val = first_non_empty(rel_obj, name_keys)
            if name_val is not None:
                ent["entity_name"] = safe_str(name_val)

        if not ent["entity_description"]:
            desc_val = first_non_empty(rel_obj, desc_keys)
            if desc_val is not None:
                ent["entity_description"] = safe_str(desc_val)

        if not ent["class_label"]:
            cls_val = first_non_empty(rel_obj, class_label_keys)
            if cls_val is not None:
                ent["class_label"] = safe_str(cls_val)

        if not ent["class_group"]:
            grp_val = first_non_empty(rel_obj, class_group_keys)
            if grp_val is not None:
                ent["class_group"] = safe_str(grp_val)

        # accumulate chunk_id from relations as well
        if chunk_id:
            ent["seen_in_chunks"].add(chunk_id)


    # --- 3) Build relations CSV + enrich entities from relations ---

    rels_rows = []

    for r in rows:
        # IDs for subject and object
        subj = (
            r.get("subject_entity_id")
            or r.get("subject_id")
            or r.get("subject_entity")
            or r.get("subject_entity_name")
            or r.get("subject")
        )
        objt = (
            r.get("object_entity_id")
            or r.get("object_id")
            or r.get("object_entity")
            or r.get("object_entity_name")
            or r.get("object")
        )
        subj_id = safe_str(subj)
        objt_id = safe_str(objt)

        # chunk_id (for relation + union into entity.seen_in_chunks)
        chunk_id_val = r.get("chunk_id") or r.get("source_chunk") or r.get("context_chunk") or None
        chunk_id = safe_str(chunk_id_val)

        # qualifiers: sanitize inner quotes -> pretty JSON for nicer Neo4j view
        qualifiers_obj = r.get("qualifiers") or r.get("qualifier") or {}
        if qualifiers_obj is None:
            qualifiers_obj = {}
        qualifiers_sanitized = sanitize_for_json_in_csv(qualifiers_obj)
        qualifiers_json = json.dumps(qualifiers_sanitized, ensure_ascii=False, indent=1)

        # relation row (structure same as before)
        relation_row = {
            "relation_id":         safe_str(r.get("relation_id") or r.get("id") or r.get("rid") or ""),
            "start_id":            subj_id,
            "end_id":              objt_id,
            "raw_relation_name":   safe_str(r.get("relation_name") or r.get("rel_name") or ""),
            "canonical_rel_name":  safe_str(r.get("canonical_rel_name") or r.get("canonical") or ""),
            "canonical_rel_desc":  safe_str(r.get("canonical_rel_desc") or r.get("canonical_desc") or ""),
            "rel_cls":             safe_str(r.get("rel_cls") or r.get("relation_class") or ""),
            "rel_cls_group":       safe_str(r.get("rel_cls_group") or r.get("relation_class_group") or r.get("rel_group") or ""),
            "rel_hint_type":       safe_str(r.get("rel_hint_type") or r.get("hint") or ""),
            "confidence":          safe_str(r.get("confidence") if r.get("confidence") not in (None, "") else ""),
            "resolution_context":  safe_str(r.get("resolution_context") or r.get("resolution") or ""),
            "justification":       safe_str(r.get("justification") or ""),
            "remark":              safe_str(r.get("remark") or r.get("remarks") or ""),
            "evidence_excerpt":    safe_str(r.get("evidence_excerpt") or r.get("evidence") or ""),
            "chunk_id":            chunk_id,
            "qualifiers":          qualifiers_json,
            "rel_desc":            safe_str(r.get("rel_desc") or r.get("relation_description") or ""),
        }
        rels_rows.append(relation_row)

        # Enrich entities from relations (but do not override existing rich info from classes)
        update_entity_from_relation(entities, "subject", subj_id, r, chunk_id)
        update_entity_from_relation(entities, "object", objt_id, r, chunk_id)

    print(f"Collected {len(rels_rows)} relation rows.")
    print(f"Collected {len(entities)} unique entities (classes + relations).")


    # --- 4) Write relations CSV (rels_fixed_no_raw.csv) ---

    rel_fieldnames = [
        "relation_id","start_id","end_id","raw_relation_name","canonical_rel_name","canonical_rel_desc",
        "rel_cls","rel_cls_group","rel_hint_type","confidence","resolution_context","justification",
        "remark","evidence_excerpt","chunk_id","qualifiers","rel_desc"
    ]

    rels_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with rels_out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rel_fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for rrow in rels_rows:
            writer.writerow(rrow)

    print("Wrote relations CSV:", rels_out_csv)


    # --- 5) Write nodes CSV (nodes.csv) ---
    # NOTE: we now expose more intrinsic entity info, and rename "seen_in_chunks" -> "chunk_ids"
    # for clearer, more consistent naming with relation.chunk_id.

    node_fieldnames = [
        "entity_id",
        "entity_name",
        "entity_description",
        "entity_type_hint",
        "entity_confidence",
        "entity_resolution_context",
        "entity_flag",
        "class_id",
        "class_label",
        "class_group",
        "chunk_ids",
        "node_properties",
    ]

    nodes_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with nodes_out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=node_fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for ent in entities.values():
            # chunk_ids: JSON list string for Neo4j apoc.convert.fromJsonList
            chunk_ids_str = json.dumps(sorted(ent["seen_in_chunks"]), ensure_ascii=False)

            # node_properties: JSON array string, sanitized
            node_props_sanitized = sanitize_for_json_in_csv(ent["node_properties"])
            node_props_str = json.dumps(node_props_sanitized, ensure_ascii=False)

            row = {
                "entity_id":               ent["entity_id"],
                "entity_name":             ent["entity_name"],
                "entity_description":      ent["entity_description"],
                "entity_type_hint":        ent["entity_type_hint"],
                "entity_confidence":       ent["entity_confidence"],
                "entity_resolution_context": ent["entity_resolution_context"],
                "entity_flag":             ent["entity_flag"],
                "class_id":                ent["class_id"],
                "class_label":             ent["class_label"],
                "class_group":             ent["class_group"],
                "chunk_ids":               chunk_ids_str,
                "node_properties":         node_props_str,
            }
            writer.writerow(row)

    print("Wrote nodes CSV:", nodes_out_csv)



# # -----------------------
# # Export KG to CSVs  - Run statement
# # -----------------------

# export_relations_and_nodes_to_csv()



#endregion#? CSV relations + nodes for Neo4j KG import - V6
#?#########################  End  ##########################






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







#endregion#! KG Assembly
#!#############################################  End Chapter  ##################################################






















#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################
















































#!############################################# Start Chapter ##################################################
#region:#!   

#endregion#! 
#!#############################################  End Chapter  ##################################################






#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################








#?######################### Start ##########################
#region:#?   Run statements


# -----------------------
# Chunking - Run statement
# -----------------------

if __name__ == "__main__":
    sentence_chunks_token_driven(
        "SGCE-KG/data/pdf_to_json/Plain_Text.json",
        "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
        max_tokens_per_chunk=200,   # preferred upper bound (None to disable)
        min_tokens_per_chunk=100,   # expand small chunks to reach this minimum (None to disable)
        sentence_per_line=True,
        keep_ref_text=False,
        strip_leading_headings=True,
        force=True,
        debug=False
    )


# -----------------------
# embed_and_index_chunks  - Run statement
# -----------------------


embed_and_index_chunks(
    "SGCE-KG/data/Chunks/chunks_sentence.jsonl",
    "SGCE-KG/data/Chunks/chunks_emb",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    False,   # use_small_model_for_dev
    32,     # batch_size
    None,   # device -> auto
    True,   # save_index
    True)  # force



# -----------------------
# Entity Recognition  - Run statement
# -----------------------

if __name__ == "__main__":
    run_entity_extraction_on_chunks(
        chunk_ids,
        prev_chunks=5,
        save_debug=False,
        model="gpt-5.1",
        max_tokens=8000
    )





# -----------------------
# Ent Resolution (Multi Run)  - Run statement
# -----------------------

if __name__ == "__main__":
    iterative_resolution()






# -----------------------
# Cls Rec input producer - Run statement
# -----------------------

if __name__ == "__main__":
    produce_clean_jsonl(input_path, out_file)




# -----------------------
# Cls Recognition  - Run statement
# -----------------------



if __name__ == "__main__":
    classrec_iterative_main()



# -----------------------
# Create input for Cls Res  - Run statement
# -----------------------

if __name__ == "__main__":
    main_input_for_cls_res()





# -----------------------
# Cls Res Multi Run - Run statement
# -----------------------
if __name__ == "__main__":
    run_pipeline_iteratively() 









# # -----------------------
# Relation Res Multi Run - Run statement
# -----------------------
if __name__ == "__main__":
    run_relres_iteratively() 



# -----------------------
# Export KG to CSVs  - Run statement
# -----------------------

if __name__ == "__main__":
    export_relations_and_nodes_to_csv()




# -----------------------
# XXXXXXXX  - Run statement
# -----------------------




#endregion#? Run statements
#?#########################  End  ##########################











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











#?######################### Start ##########################
#region:#?   Create KG for each Essay  - V6



"""
TRACE KG multi-essay runner (entity seed fix + entity_final path fix)

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
        [COPY entities_resolved.jsonl -> entities_final.jsonl]
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
# PATHS FOR Cls Rec INPUT  (FIXES input_path / out_file ISSUE)
# --------------------------------------------------------------------

# Directory where class-recognition expects its input entities
CLS_INPUT_DIR = DATA_ROOT / "Classes" / "Cls_Input"
CLS_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# This is the file classrec_iterative_main() reads:
#   INPUT_PATH = "/home/.../data/Classes/Cls_Input/cls_input_entities.jsonl"
out_file = CLS_INPUT_DIR / "cls_input_entities.jsonl"

# This is the path that produce_clean_jsonl() expects as input
# (from your original script / error message).
ENT_RES_ITER_ROOT = ENTITIES_DIR / "Ent_Res_IterativeRuns" / "overall_summary"
input_path = ENT_RES_ITER_ROOT / "entities_final.jsonl"

# This is where iterative_resolution() actually writes its final resolved entities,
# according to your log:
#   /data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl
ENT_1ST_RESOLVED_PATH = ENTITIES_DIR / "Ent_1st" / "Ent_Resolved_1st" / "entities_resolved.jsonl"


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
            # 5) Seed for Ent Resolution
            # --------------------------------------------------------
            prepare_entity_seed_for_iterative_resolution()

            # --------------------------------------------------------
            # 6) Ent Resolution (Multi Run)
            # --------------------------------------------------------
            t0 = time.time()
            iterative_resolution()
            essay_stat["t_entity_resolution"] = time.time() - t0

            # --------------------------------------------------------
            # 6.5) COPY entities_resolved.jsonl -> entities_final.jsonl
            #      so produce_clean_jsonl() sees what it expects.
            # --------------------------------------------------------
            try:
                src = ENT_1ST_RESOLVED_PATH
                if not src.exists():
                    # fallback: search for entities_resolved.jsonl anywhere under Entities
                    candidates = list(ENTITIES_DIR.rglob("entities_resolved.jsonl"))
                    if candidates:
                        # pick the most recent
                        src = max(candidates, key=lambda p: p.stat().st_mtime)
                    else:
                        raise FileNotFoundError(
                            "Could not find entities_resolved.jsonl after iterative_resolution()."
                        )

                ENT_RES_ITER_ROOT.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, input_path)
                print(f"[info] Copied resolved entities from {src} -> {input_path}")
            except Exception as copy_err:
                # Fail fast here so you see a clear error if something is wrong
                raise RuntimeError(
                    f"Failed to prepare entities_final.jsonl for produce_clean_jsonl: {copy_err}"
                )

            # --------------------------------------------------------
            # 7) Cls Rec input producer
            # --------------------------------------------------------
            t0 = time.time()
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


#endregion#? Create KG for each Essay  - V6
#?#########################  End  ##########################



