

#!############################################# Start Chapter ##################################################
#region:#!    Utilities + Config


#?######################### Start ##########################
#region:#?    LLM Model - V4  (DSPy‑centric, reasoning‑aware)

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import re

# DSPy is required for the new flexible model handling.
try:
    import dspy  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "DSPy is required for TRACE KG LLM configuration but is not installed. "
        "Install it with `pip install dspy-ai`."
    ) from e


# ------------------------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------------------------

def coerce_llm_text(raw) -> str:
    """Return a reasonable string for a variety of LLM response shapes."""
    if raw is None:
        return ""
    # already a string
    if isinstance(raw, str):
        return raw

    # dict-like
    if isinstance(raw, dict):
        # common keys returned by various clients
        for k in ("output_text", "text", "content", "message", "output"):
            v = raw.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                # nested message-like: {"content": "..."}
                if "content" in v and isinstance(v["content"], str):
                    return v["content"]
            if isinstance(v, list):
                # join list-of-msgs
                parts = [coerce_llm_text(x) for x in v]
                return " ".join(p for p in parts if p)
        # fallback: try to dump some fields
        for k in ("choices", "outputs", "messages"):
            v = raw.get(k)
            if v:
                try:
                    return json.dumps(v, ensure_ascii=False)
                except Exception:
                    return str(v)
        return str(raw)

    # list-like (e.g., [Message(...)])
    if isinstance(raw, (list, tuple)):
        parts = [coerce_llm_text(x) for x in raw]
        return " ".join(p for p in parts if p)

    # objects: try common attributes used by wrappers
    for attr in ("output_text", "text", "content", "message"):
        if hasattr(raw, attr):
            val = getattr(raw, attr)
            if isinstance(val, str):
                return val
            # if object with nested fields
            try:
                return str(val)
            except Exception:
                pass

    # last fallback
    try:
        return str(raw)
    except Exception:
        return ""




# ------------------------------------------------------------------------------------
# CENTRAL LLM CONFIG FOR TRACE KG
# ------------------------------------------------------------------------------------

@dataclass
class TraceKGLLMConfig:
    """
    Central configuration for all LLM usage in TRACE KG.

    Levels of control:
      - default_model: used if nothing more specific is set.
      - rec_model: shared model for all *recognition* steps
          (entity_rec, class_rec, rel_rec) unless overridden per-step.
      - res_model: shared model for all *resolution* steps
          (entity_res, class_res, rel_res) unless overridden per-step.
      - Per-step overrides: if set, they win over rec_model/res_model.

    Model naming conventions (both are fine; DSPy infers provider):
      - Plain names: "gpt-5.1", "gpt-4.1-mini"
      - Provider-prefixed: "openai/gpt-4o-mini", "google/gemini-1.5-pro"
    """

    # Global default for all steps
    default_model: str = "gpt-5.1"

    # Coarse-grained overrides
    rec_model: Optional[str] = None   # EntityRec, ClassRec, RelRec
    res_model: Optional[str] = None   # EntityRes, ClassRes, RelRes

    # Fine-grained overrides (each step can have its own model)
    entity_rec_model: Optional[str] = None
    entity_res_model: Optional[str] = None
    class_rec_model: Optional[str] = None
    class_res_model: Optional[str] = None
    rel_rec_model: Optional[str] = None
    rel_res_model: Optional[str] = None

    # Common LLM runtime parameters.
    #
    # For GPT‑5 reasoning models (gpt-5.*, gpt-5-nano, etc.) the Responses API
    # often *does not* support `temperature`. Using None lets the provider
    # choose its default.
    temperature: Optional[float] = None
    max_tokens: int = 16000

    # Optional explicit auth / routing (otherwise taken from env + provider defaults)
    api_key: Optional[str] = None          # e.g. OPENAI_API_KEY
    api_base: Optional[str] = None         # e.g. custom proxy; mapped to base_url for DSPy/LiteLLM

    # Whether to disable DSPy cache
    disable_cache: bool = False

    # Optional safety check on our side for GPT‑5 family max_tokens etc.
    enforce_gpt5_constraints: bool = True

    # Responses‑style knobs (for reasoning models etc.)
    reasoning_effort: Optional[str] = None   # e.g. "low", "medium", "high"
    verbosity: Optional[str] = None          # e.g. "low", "medium", "high"

    def validate(self) -> None:
        """
        Optional safety checks on top of DSPy's own constraints.

        Currently we only ensure that if the *default* model looks like a GPT‑5
        reasoning model and enforce_gpt5_constraints is True, then max_tokens
        is at least 16000 (matching DSPy's internal check).
        """
        if not self.enforce_gpt5_constraints:
            return

        family = self.default_model.split("/")[-1].lower()
        # Roughly align with GPT‑5 reasoning family
        if family.startswith("gpt-5") and self.max_tokens < 16000:
            raise ValueError("max_tokens must be >= 16000 for gpt-5 family models")


def get_model_name_for_step(cfg: TraceKGLLMConfig, step: str) -> str:
    """
    Return the model name to use for a given logical step.

    step is one of:
      - "entity_rec"
      - "entity_res"
      - "class_rec"
      - "class_res"
      - "rel_rec"
      - "rel_res"
    """
    # Fine-grained overrides first
    if step == "entity_rec" and cfg.entity_rec_model:
        return cfg.entity_rec_model
    if step == "entity_res" and cfg.entity_res_model:
        return cfg.entity_res_model
    if step == "class_rec" and cfg.class_rec_model:
        return cfg.class_rec_model
    if step == "class_res" and cfg.class_res_model:
        return cfg.class_res_model
    if step == "rel_rec" and cfg.rel_rec_model:
        return cfg.rel_rec_model
    if step == "rel_res" and cfg.rel_res_model:
        return cfg.rel_res_model

    # Coarse-grained overrides
    if step in ("entity_rec", "class_rec", "rel_rec") and cfg.rec_model:
        return cfg.rec_model
    if step in ("entity_res", "class_res", "rel_res") and cfg.res_model:
        return cfg.res_model

    # Fallback to global default
    return cfg.default_model


def parse_model_name(raw: str) -> Tuple[Optional[str], str]:
    """
    Parse a raw model name into (provider, model).

    Examples:
      "gpt-4.1-mini"          -> (None, "gpt-4.1-mini")
      "openai/gpt-4o-mini"    -> ("openai", "gpt-4o-mini")
      "google/gemini-1.5-pro" -> ("google", "gemini-1.5-pro")

    This is mostly advisory. DSPy will also infer provider from the full string.
    """
    if "/" in raw:
        prov, m = raw.split("/", 1)
        prov = prov.strip() or None
        m = m.strip()
        return prov, m
    return None, raw.strip()


def _infer_model_type(model: str) -> str:
    """
    Decide DSPy's model_type ("chat" vs "responses") from the model name.

    - OpenAI reasoning models (o1/o3/o4*), and gpt‑5.* (except gpt‑5‑chat variants)
      -> use the Responses API format: model_type="responses".
    - Everything else -> "chat" (standard chat/completions style).
    """
    family = model.split("/")[-1].lower()

    reasoning_pattern = re.match(
        r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?"
        r"|gpt-5(?!-chat)(?:-.*)?)$",
        family,
    )
    if reasoning_pattern:
        return "responses"

    # Default: treat as normal chat model
    return "chat"


def _supports_temperature(model: str, model_type: str) -> bool:
    """
    Heuristic: when should we send `temperature`?

    - For OpenAI reasoning / GPT‑5 models (Responses API), temperature is often
      not supported (e.g. gpt-5-nano, gpt-5.1, o1/o3). We skip it there.
    - For classic chat/completion models (gpt-4.1, gpt-4o-mini, etc.) it's fine.
    """
    base = model.split("/")[-1].lower()

    if model_type == "responses":
        # treat all gpt-5.* and o* reasoning models as "no temperature"
        if base.startswith("gpt-5"):
            return False
        if base.startswith("o1") or base.startswith("o2") or base.startswith("o3") or base.startswith("o4"):
            return False

    return True


def make_lm_for_step(cfg: TraceKGLLMConfig, step: str) -> "dspy.LM":
    """
    Construct a DSPy LM object for the given step.

    This is the central gateway for all TRACE KG LLM calls.
    Steps should call this (via their llm_config) and then either:

        lm = make_lm_for_step(cfg, "entity_rec")
        outputs = lm(prompt=my_prompt)

    or use `dspy.context(lm=...)` with DSPy modules.
    """
    model_name_raw = get_model_name_for_step(cfg, step)
    cfg.validate()

    model_type = _infer_model_type(model_name_raw)

    lm_kwargs: Dict[str, object] = dict(
        model=model_name_raw,
        max_tokens=cfg.max_tokens,
        cache=not cfg.disable_cache,
        model_type=model_type,
    )

    # Provider / routing options
    if cfg.api_key:
        lm_kwargs["api_key"] = cfg.api_key
    if cfg.api_base:
        # DSPy / LiteLLM expect `base_url` for OpenAI-compatible endpoints
        lm_kwargs["base_url"] = cfg.api_base

    # Temperature: only include when it is supported and explicitly set
    if cfg.temperature is not None and _supports_temperature(model_name_raw, model_type):
        lm_kwargs["temperature"] = cfg.temperature

    # Responses‑style extras (for reasoning / Responses models)
    if model_type == "responses":
        if cfg.reasoning_effort:
            # OpenAI Responses uses: reasoning={"effort": "low"|"medium"|"high"}
            lm_kwargs["reasoning"] = {"effort": cfg.reasoning_effort}
        if cfg.verbosity:
            # Newer models accept verbosity; others will just ignore it.
            lm_kwargs["verbosity"] = cfg.verbosity

    return dspy.LM(**lm_kwargs)


#endregion#?  LLM Model - V4
#?#########################  End  ##########################





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
#         "data/pdf_to_json/Plain_Text.json",
#         "data/Chunks/chunks_sentence.jsonl",
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
    chunks_jsonl_path: str = "data/Chunks/chunks_sentence.jsonl",
    output_prefix: str = "data/Chunks/chunks_emb",
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
#     "data/Chunks/chunks_sentence.jsonl",
#     "data/Chunks/chunks_emb",
#     "BAAI/bge-large-en-v1.5",
#     "BAAI/bge-small-en-v1.5",
#     False,   # use_small_model_for_dev
#     32,     # batch_size
#     None,   # device -> auto
#     True,   # save_index
#     True)  # force



#endregion#? Embedding + FAISS Index
#?#########################  End  ##########################





#endregion#!  Utilities + Config
#!#############################################  End Chapter  ##################################################




#!############################################# Start Chapter ##################################################
#region:#!   Entity Identification





#?######################### Start ##########################
#region:#?   Entity Recognition v10 - DSPy LLM Config

import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# NOTE:
# - This block depends on the DSPy-centric LLM config defined earlier:
#     TraceKGLLMConfig, get_model_name_for_step, make_lm_for_step
#   in the "LLM Model - V3" region of this file.

# ---------- CONFIG: paths ----------
CHUNKS_JSONL =      "data/Chunks/chunks_sentence.jsonl"
ENTITIES_OUT =      "data/Entities/Ent_Raw_0/entities_raw.jsonl"
DEFAULT_DEBUG_DIR = "data/Entities/Ent_Raw_0/entity_raw_debug_prompts_outputs"


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


from TKG_Prompts import ENT_REC_PROMPT_TEMPLATE

def build_entity_prompt_with_context(chunk: Dict, prev_chunks: Optional[List[Dict]] = None) -> str:
    focus_text = chunk.get("text", "") or ""
    context_parts = []
    if prev_chunks:
        for pc in prev_chunks:
            t = (pc.get("text") or "").strip()
            if t:
                context_parts.append(t)
    context_block = "\n\n".join(context_parts) if context_parts else "NO PREVIOUS CONTEXT PROVIDED.\n---"

    return ENT_REC_PROMPT_TEMPLATE.format(
        context_block=context_block,
        focus_chunk_id=chunk.get("id"),
        focus_text=focus_text
    )

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


# ---------- Internal helper: get LM for Entity Rec via DSPy ----------
def _get_lm_for_entity_rec(
    llm_config: Optional["TraceKGLLMConfig"],
    model: str,
    max_tokens: int,
):
    """
    Resolve a DSPy LM for the Entity Recognition step:

      - If llm_config is provided, use it directly (with per-step overrides).
      - Otherwise, build a minimal TraceKGLLMConfig using the function args
        for backward compatibility.
    """
    if llm_config is not None:
        cfg = llm_config
    else:
        # Backward-compatible default: one model everywhere, with given max_tokens.
        cfg = TraceKGLLMConfig(default_model=model, max_tokens=max_tokens)

    try:
        cfg.validate()
    except Exception as e:
        print(f"[EntityRec] WARNING: llm_config.validate() failed: {e}")

    # Centralized LM construction (provider, model_type, cache, api_base, etc.)
    lm = make_lm_for_step(cfg, "entity_rec")
    return lm


# ---------- Main: extract_entities_from_chunk (with optional prev context + debug saving) ----------
def extract_entities_from_chunk(
    chunk_id: str,
    chunks_path: str = CHUNKS_JSONL,
    prev_chunks: int = 2,       # how many previous chunks to include as CONTEXT. Set 0 to disable.
    model: str = "gpt-5.1",
    max_tokens: int = 16000,
    save_debug: bool = False,   # if True, write full prompt+output+parsed to a debug JSON file
    debug_dir: str = DEFAULT_DEBUG_DIR,
    llm_config: Optional["TraceKGLLMConfig"] = None,
) -> List[Dict]:
    """
    Extract entities from the specified focus chunk, optionally including up to `prev_chunks`
    previous chunks from the same section as disambiguating CONTEXT.

    This version uses DSPy exclusively:

      - Resolve a dspy.LM via TraceKGLLMConfig + make_lm_for_step(cfg, "entity_rec").
      - Call lm(prompt) directly; DSPy handles provider, responses vs chat, caching, etc.

    If llm_config is provided, all LLM behavior (model, temperature, tokens, api_base, etc.)
    is governed by that config. If not, a default TraceKGLLMConfig is built from `model`
    and `max_tokens` (backward compatibility).
    """
    # Load and locate the focus chunk
    chunks = load_chunks(chunks_path)
    try:
        chunk = get_chunk_by_id(chunk_id, chunks)
    except ValueError as e:
        print(e)
        return []

    prev_ctx = get_previous_chunks(chunk, chunks, prev_n=prev_chunks)

    prompt = build_entity_prompt_with_context(chunk, prev_ctx)

    # Resolve LM via central DSPy config
    lm = _get_lm_for_entity_rec(llm_config=llm_config, model=model, max_tokens=max_tokens)

    # Call the LM via DSPy
    try:
        lm_outputs = lm(prompt)
    except Exception as e:
        print(f"[EntityRec] LM call error for chunk {chunk_id}: {e}")
        if save_debug:
            dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, "", [], error=str(e))
            print(f"Debug file written to: {dbg_path}")
        return []

    # DSPy returns a list of strings for bare calls: lm("...") -> ['...']
    if isinstance(lm_outputs, list):
        raw = lm_outputs[0] if lm_outputs else ""
    else:
        raw = str(lm_outputs or "")

    if not raw:
        print("Empty LLM response.")
        if save_debug:
            dbg_path = write_debug_file(debug_dir, chunk, prev_ctx, prompt, "", [], error="Empty LLM response")
            print(f"Debug file written to: {dbg_path}")
        return []

    # txt = raw.strip() #INJAM
    # from llm_utils import coerce_llm_text   # add once near top of file (or above this function)
    txt = coerce_llm_text(raw).strip()

    # unwrap markdown fences if present (be liberal)
    if txt.startswith("```") and txt.endswith("```"):
        # Strip surrounding backticks and optional 'json' language tag
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()

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


# ---------- Driver: run on all chunks in the chunks file ----------
def run_entity_extraction_on_chunks(
    chunk_ids: List[str] = None,
    prev_chunks: int = 3,
    save_debug: bool = False,
    debug_dir: str = DEFAULT_DEBUG_DIR,
    model: str = "gpt-5.1",
    max_tokens: int = 16000,
    llm_config: Optional["TraceKGLLMConfig"] = None,
):
    """
    Convenience driver to run entity extraction over a list of chunk_ids.

    If llm_config is provided, a single TraceKGLLMConfig object controls all LLM
    behavior (models, tokens, temperature, api_base, etc.) for this step:

        - default_model
        - rec_model / entity_rec_model
        - temperature, max_tokens, api_key, api_base, disable_cache

    If llm_config is None, `model` and `max_tokens` are used to build a
    minimal TraceKGLLMConfig, preserving backward compatibility with
    older scripts that passed only a model name.
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
            debug_dir=debug_dir,
            llm_config=llm_config,
        )
        if res:
            all_results.extend(res)
    return all_results

#endregion#? Entity Recognition v10 - DSPy LLM Config
#?#########################  End  ##########################



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
    
    
    # labels, probs, clusterer = run_hdbscan(combined,
    #                                       min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    #                                       min_samples=HDBSCAN_MIN_SAMPLES,
    #                                       metric=HDBSCAN_METRIC,
    #                                       use_umap=USE_UMAP if args.use_umap else False)

    # # diagnostics
    # import numpy as np
    # from collections import Counter
    # labels_arr = np.array(labels)
    # n = len(labels_arr)
    # n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    # n_noise = int((labels_arr == -1).sum())
    # print(f"[diagnostic] clusters (excl -1): {n_clusters}  noise: {n_noise} ({n_noise/n*100:.1f}%)")
    # counts = Counter(labels_arr)
    # top = sorted(((lab, sz) for lab, sz in counts.items() if lab != -1), key=lambda x: x[1], reverse=True)[:10]
    # print("[diagnostic] top cluster sizes:", top)

    # save_entities_with_clusters(entities, labels_arr, args.out_jsonl, args.clusters_summary)
    # print("Clustering finished.")
    
    
    labels, clusterer = run_hdbscan(
    combined,
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    metric=HDBSCAN_METRIC,
    use_umap=USE_UMAP if args.use_umap else False,
)

    # diagnostics
    import numpy as np
    from collections import Counter
    labels_arr = np.array(labels)
    n = len(labels_arr)
    n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    n_noise = int((labels_arr == -1).sum())
    print(f"[diagnostic] clusters (excl -1): {n_clusters}  noise: {n_noise} ({n_noise/n*100:.1f}%)")
    counts = Counter(labels_arr)
    top = sorted(
        ((lab, sz) for lab, sz in counts.items() if lab != -1),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    print("[diagnostic] top cluster sizes:", top)

    save_entities_with_clusters(entities, labels_arr, args.out_jsonl, args.clusters_summary)
    print("Clustering finished.")



#endregion#? Embedding and clustering recognized entities    -  V4
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   Final Ent Res v10 - DSPy LLM Config (aligned with EntityRec v10 & LLM Model V3)

"""
Entity resolution orchestrator aligned with Entity Recognition prompt and the updated
embedding pipeline (name/desc/ctx). Performs local sub-clustering, chunk-text inclusion,
token safety guard, and tqdm progress bars.

Original design:
 - stricter input validation (requires _cluster_id + context fields)
 - saves local-subcluster summaries per coarse cluster for debugging
 - fallback to "no local sub-clustering" when fragmentation is excessive (keeps members together)
 - robust min_cluster_size handling and explanatory logging

Updated (Jan 2026, v10):
 - switch LLM backend from OpenAI Responses client (gpt-5.1) to DSPy + TraceKGLLMConfig (LLM Model V3)
 - each LLM call uses a dspy.LM created via make_lm_for_step(cfg, "entity_res")
 - one central config can control all steps; optional per-step model overrides still supported
"""

import json
import uuid
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

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


# ---------------- Paths & config ----------------
CLUSTERED_IN = Path("data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")   # input (from previous clustering)
CHUNKS_JSONL = Path("data/Chunks/chunks_sentence.jsonl")

ENT_OUT = Path("data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")
CANON_OUT = Path("data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
LOG_OUT = Path("data/Entities/Ent_1st/Ent_Resolved_1st/resolution_log.jsonl")

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
LOCAL_USE_UMAP = False   # default OFF for robustness; enable via CLI / config if desired
UMAP_DIMS = 32
UMAP_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0
UMAP_MIN_SAMPLES_TO_RUN = 25  # only run UMAP when cluster size >= this

# LLM defaults (used only when llm_config is None – for backward compatibility)
MODEL = "gpt-5.1"
MAX_TOKENS = 16000

# orchestration thresholds
MAX_CLUSTER_PROMPT = 11        # coarse cluster size threshold to trigger local sub-clustering
MAX_MEMBERS_PER_PROMPT = 10    # <= 10 entities per LLM call
TRUNC_CHUNK_CHARS = 8000
INCLUDE_PREV_CHUNKS = 0        # currently not used (only focus-chunk text included)

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


# ---------------- field builder aligned with EntityRec ----------------
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
        resolution = (
            safe_text(e, "resolution_context")
            or safe_text(e, "text_span")
            or safe_text(e, "context_phrase")
            or safe_text(e, "used_context_excerpt")
            or ""
        )

        # fold type and node_properties into ctx (as text hints)
        etype = safe_text(e, "entity_type_hint") or safe_text(e, "entity_type") or ""
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np_ in node_props:
                if isinstance(np_, dict):
                    pname = np_.get("prop_name") or np_.get("name") or ""
                    pval = np_.get("prop_value") or np_.get("value") or ""
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
def local_subcluster(
    cluster_entities: List[Dict],
    entity_id_to_index: Dict[str, int],
    all_embeddings: np.ndarray,
    min_cluster_size: int = LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = LOCAL_HDBSCAN_MIN_SAMPLES,
    use_umap: bool = LOCAL_USE_UMAP,
    umap_dims: int = UMAP_DIMS,
):
    """
    Returns: dict[label] -> list[entity_dict]
    Notes:
      - min_cluster_size is treated as a true minimum (HDBSCAN may still mark small noise).
      - We protect against trivial inputs and return fallback single cluster if hdbscan fails.
    """
    from collections import defaultdict as _dd
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
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(UMAP_NEIGHBORS, max(2, n - 1)),
                min_dist=UMAP_MIN_DIST,
                metric="cosine",
                random_state=42,
            )
            X_sub = reducer.fit_transform(X)
        except Exception as e:
            print(f"[local_subcluster] UMAP failed for n={n}, n_components={n_components} -> fallback without UMAP. Err: {e}")
            X_sub = X
    else:
        if use_umap and UMAP_AVAILABLE and n < UMAP_MIN_SAMPLES_TO_RUN:
            print(f"[local_subcluster] skipping UMAP for n={n} (threshold {UMAP_MIN_SAMPLES_TO_RUN})")

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=LOCAL_HDBSCAN_METRIC,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(X_sub)
    except Exception as e:
        print(f"[local_subcluster] HDBSCAN failed for n={n} -> fallback single cluster. Err: {e}")
        return {0: list(cluster_entities)}

    groups = _dd(list)
    for ent, lab in zip(cluster_entities, labels):
        groups[int(lab)].append(ent)
    return groups


# ------------------ LLM helpers via DSPy + TraceKGLLMConfig ------------------

def extract_json_array(text: str):
    if not text:
        return None
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None



####################################################
from TKG_Prompts import ENT_RES_PROMPT_TEMPLATE
PROMPT_TEMPLATE = ENT_RES_PROMPT_TEMPLATE
####################################################

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
            ct = ch.get("text", "")
            # (INCLUDE_PREV_CHUNKS is currently 0 / unused)
            chunk_text = " ".join(ct.split())
            if len(chunk_text) > TRUNC_CHUNK_CHARS:
                chunk_text = chunk_text[:TRUNC_CHUNK_CHARS].rsplit(" ", 1)[0] + "..."
    # text_span should show the precise mention proof; prefer context_phrase then resolution_context
    text_span = m.get("context_phrase") or m.get("resolution_context") or m.get("text_span") or ""
    return {
        "id": m.get("id"),
        "name": m.get("entity_name"),
        "type_hint": m.get("entity_type_hint"),
        "confidence": m.get("confidence_score"),
        "desc": m.get("entity_description"),
        "text_span": text_span,
        "chunk_text": chunk_text,
    }


# --------- DSPy integration: LM for Entity Resolution step ---------

def _get_lm_for_entity_res(
    llm_config: Optional["TraceKGLLMConfig"],
    model: str,
    max_tokens: int,
):
    """
    Resolve a DSPy LM for the Entity Resolution step:

      - If llm_config is provided, use it directly (with per-step overrides for 'entity_res').
      - Otherwise, build a minimal TraceKGLLMConfig using the function args
        for backward compatibility (single model everywhere).
    """
    if llm_config is not None:
        cfg = llm_config
    else:
        # Backward-compatible default: one model everywhere, with given max_tokens.
        cfg = TraceKGLLMConfig(default_model=model, max_tokens=max_tokens)

    try:
        cfg.validate()
    except Exception as e:
        print(f"[EntityRes] WARNING: llm_config.validate() failed: {e}")

    # Centralized LM construction (provider, model_type, cache, api_base, etc.)
    lm = make_lm_for_step(cfg, "entity_res")
    return lm


def _call_entity_res_lm(lm, prompt: str) -> str:
    """
    Small helper to call the DSPy LM and normalize the output to a plain string.
    """
    try:
        outputs = lm(prompt)
    except Exception as e:
        print(f"[EntityRes] LM call error: {e}")
        return ""

    # DSPy often returns a list of strings for bare calls: lm("...") -> ['...']
    if isinstance(outputs, list):
        return outputs[0] if outputs else ""
    return str(outputs or "")


# ------------------ apply actions ----------------
def apply_actions(
    members: List[Dict],
    actions: List[Dict],
    entities_by_id: Dict[str, Dict],
    canonical_store: List[Dict],
    log_entries: List[Dict],
):
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
                "source": "LLM_resolution_v10_dspy",
                "rationale": rationale,
                "timestamp": ts,
            }
            canonical_store.append(canonical)
            for eid in ids:
                ent = entities_by_id.get(eid)
                if ent:
                    ent["canonical_id"] = can_id
                    ent["resolved_action"] = "merged"
                    ent["resolution_rationale"] = rationale
                    ent["resolved_time"] = ts
            log_entries.append(
                {
                    "time": ts,
                    "action": "merge",
                    "canonical_id": can_id,
                    "merged_ids": ids,
                    "rationale": rationale,
                }
            )
        elif typ == "ModifyEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            if ent:
                new_name = act.get("new_name")
                new_desc = act.get("new_description")
                new_type = act.get("new_type_hint")
                rationale = act.get("rationale", "")
                if new_name:
                    ent["entity_name"] = new_name
                if new_desc:
                    ent["entity_description"] = new_desc
                if new_type:
                    ent["entity_type_hint"] = new_type
                ent["resolved_action"] = "modified"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append(
                    {
                        "time": ts,
                        "action": "modify",
                        "entity_id": eid,
                        "rationale": rationale,
                    }
                )
        elif typ == "KeepEntity":
            eid = act.get("entity_id")
            ent = entities_by_id.get(eid)
            rationale = act.get("rationale", "")
            if ent:
                ent["resolved_action"] = "kept"
                ent["resolution_rationale"] = rationale
                ent["resolved_time"] = ts
                log_entries.append(
                    {
                        "time": ts,
                        "action": "keep",
                        "entity_id": eid,
                        "rationale": rationale,
                    }
                )
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
        context_present = any(
            k in e and e.get(k) not in (None, "")
            for k in ("resolution_context", "context_phrase", "text_span", "used_context_excerpt")
        )
        if missing or not context_present:
            problems.append(
                {
                    "index": i,
                    "id": e.get("id"),
                    "missing_keys": missing,
                    "has_context": context_present,
                    "sample": {
                        k: e.get(k)
                        for k in [
                            "entity_name",
                            "entity_type_hint",
                            "resolution_context",
                            "context_phrase",
                            "text_span",
                            "_cluster_id",
                        ]
                    },
                }
            )
        # ensure confidence_score exists (may be None but field should be present to avoid KeyError later)
        if "confidence_score" not in e:
            e["confidence_score"] = None

    if problems:
        msg_lines = [
            "Entities schema validation failed — some entries are missing required keys or have no context field:"
        ]
        for p in problems[:20]:
            msg_lines.append(
                f" - idx={p['index']} id={p['id']} missing={p['missing_keys']} "
                f"has_context={p['has_context']} sample={p['sample']}"
            )
        if len(problems) > 20:
            msg_lines.append(f" - ... and {len(problems)-20} more problematic entries")
        raise ValueError("\n".join(msg_lines))


def write_local_subcluster_summary(out_dir: Path, cid: int, subgroups: Dict[int, List[Dict]]):
    """
    Save a compact summary JSON of local subclusters for debugging.
    Format:
      { "cluster_id": cid, "n_entities": N, "n_subclusters": K,
        "clusters": {"0": [ {id,name,type_hint,desc}, ...], "-1": [...] } }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "cluster_id": cid,
        "n_entities": sum(len(v) for v in subgroups.values()),
        "n_subclusters": len(subgroups),
    }
    clusters = {}
    for lab, members in sorted(subgroups.items(), key=lambda x: x[0]):
        clusters[str(lab)] = [
            {
                "id": m.get("id"),
                "entity_name": m.get("entity_name"),
                "entity_type_hint": m.get("entity_type_hint"),
                "entity_description": (m.get("entity_description") or "")[:200],
            }
            for m in members
        ]
    summary["clusters"] = clusters
    path = out_dir / f"cluster_{cid}_subclusters.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return path


def orchestrate(
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    llm_config: Optional["TraceKGLLMConfig"] = None,
):
    """
    Main entry point for Entity Resolution (Final Ent Res).

    LLM behavior:
      - If llm_config is provided, a single TraceKGLLMConfig object controls all LLM
        behavior for this step (models, tokens, temperature, api_base, etc.). The
        'entity_res_model' / 'res_model' / 'default_model' fields are used via
        make_lm_for_step(cfg, "entity_res").
      - If llm_config is None, this function falls back to `model` and `max_tokens`
        by constructing a minimal TraceKGLLMConfig(default_model=model, max_tokens=max_tokens).

    All LLM calls use a single dspy.LM created once and reused.
    """
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

    # Resolve LM for this entire run (reused for all prompts)
    lm = _get_lm_for_entity_res(llm_config=llm_config, model=model, max_tokens=max_tokens)

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
                        chunk = members[s : s + MAX_MEMBERS_PER_PROMPT]
                        payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                        members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                        prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                        est_tokens = max(1, int(len(prompt) / 4))
                        pbar_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                        if est_tokens > PROMPT_TOKEN_LIMIT:
                            # skip LLM call, conservative
                            for m in chunk:
                                log_entries.append(
                                    {
                                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "action": "skip_large_prompt_keep",
                                        "entity_id": m["id"],
                                        "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}",
                                    }
                                )
                                m["resolved_action"] = "kept_skipped_prompt"
                                m["resolution_rationale"] = f"Prompt too large (est_tokens={est_tokens})"
                            continue
                        llm_out = _call_entity_res_lm(lm, prompt)
                        actions = extract_json_array(llm_out)
                        if actions is None:
                            actions = [
                                {
                                    "action": "KeepEntity",
                                    "entity_id": m["id"],
                                    "rationale": "LLM parse failed; conservatively kept",
                                }
                                for m in chunk
                            ]
                        apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)
            else:
                # large cluster -> local sub-cluster
                subgroups = local_subcluster(
                    members,
                    entity_id_to_index,
                    combined_embeddings,
                    min_cluster_size=LOCAL_HDBSCAN_MIN_CLUSTER_SIZE,
                    min_samples=LOCAL_HDBSCAN_MIN_SAMPLES,
                    use_umap=LOCAL_USE_UMAP,
                    umap_dims=UMAP_DIMS,
                )

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
                    print(
                        f"[fallback] cluster {cid} had {num_nonnoise} non-noise subclusters for {len(members)} members; "
                        f"falling back to sequential chunking (preserves grouping)."
                    )
                    log_entries.append(
                        {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "action": "fallback_no_subcluster",
                            "cluster": cid,
                            "n_members": len(members),
                            "n_nonnoise_subclusters": num_nonnoise,
                        }
                    )
                    # chunk original members iteratively
                    n_prompts = math.ceil(len(members) / MAX_MEMBERS_PER_PROMPT)
                    with tqdm(
                        range(n_prompts),
                        desc=f"Cluster {cid} fallback prompts",
                        leave=False,
                        unit="prompt",
                    ) as pbar_prompts:
                        for i in pbar_prompts:
                            s = i * MAX_MEMBERS_PER_PROMPT
                            chunk = members[s : s + MAX_MEMBERS_PER_PROMPT]
                            payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                            members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                            prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                            est_tokens = max(1, int(len(prompt) / 4))
                            pbar_prompts.set_postfix(est_tokens=est_tokens, chunk_size=len(chunk))
                            if est_tokens > PROMPT_TOKEN_LIMIT:
                                for m in chunk:
                                    log_entries.append(
                                        {
                                            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                            "action": "skip_large_prompt_keep",
                                            "entity_id": m["id"],
                                            "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}",
                                        }
                                    )
                                    m["resolved_action"] = "kept_skipped_prompt"
                                    m["resolution_rationale"] = (
                                        f"Prompt too large (est_tokens={est_tokens})"
                                    )
                                continue
                            llm_out = _call_entity_res_lm(lm, prompt)
                            actions = extract_json_array(llm_out)
                            if actions is None:
                                actions = [
                                    {
                                        "action": "KeepEntity",
                                        "entity_id": m["id"],
                                        "rationale": "LLM parse failed; conservatively kept",
                                    }
                                    for m in chunk
                                ]
                            apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)
                else:
                    # proceed with subgroups as planned
                    sub_items = sorted(subgroups.items(), key=lambda x: -len(x[1]))
                    with tqdm(
                        sub_items, desc=f"Cluster {cid} subclusters", leave=False, unit="sub"
                    ) as pbar_subs:
                        for sublab, submembers in pbar_subs:
                            subsize = len(submembers)
                            pbar_subs.set_postfix(sublab=sublab, subsize=subsize)
                            if sublab == -1:
                                for m in submembers:
                                    m["resolved_action"] = "kept_noise_local"
                                    m["resolution_rationale"] = "Local-subcluster noise preserved"
                                    log_entries.append(
                                        {
                                            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                            "action": "keep_noise_local",
                                            "entity_id": m["id"],
                                            "cluster": cid,
                                        }
                                    )
                                continue
                            # prompts for this subcluster
                            n_prompts = math.ceil(subsize / MAX_MEMBERS_PER_PROMPT)
                            with tqdm(
                                range(n_prompts),
                                desc=f"Sub {sublab} prompts",
                                leave=False,
                                unit="prompt",
                            ) as pbar_sub_prompts:
                                for i in pbar_sub_prompts:
                                    s = i * MAX_MEMBERS_PER_PROMPT
                                    chunk = submembers[s : s + MAX_MEMBERS_PER_PROMPT]
                                    payload = [build_member_with_chunk(m, chunks_index) for m in chunk]
                                    members_json = json.dumps(payload, ensure_ascii=False, indent=2)
                                    prompt = PROMPT_TEMPLATE.format(members_json=members_json)
                                    est_tokens = max(1, int(len(prompt) / 4))
                                    pbar_sub_prompts.set_postfix(
                                        est_tokens=est_tokens, chunk_size=len(chunk)
                                    )
                                    if est_tokens > PROMPT_TOKEN_LIMIT:
                                        for m in chunk:
                                            log_entries.append(
                                                {
                                                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                    "action": "skip_large_prompt_keep",
                                                    "entity_id": m["id"],
                                                    "reason": f"est_tokens {est_tokens} > limit {PROMPT_TOKEN_LIMIT}",
                                                }
                                            )
                                            m["resolved_action"] = "kept_skipped_prompt"
                                            m["resolution_rationale"] = (
                                                f"Prompt too large (est_tokens={est_tokens})"
                                            )
                                        continue
                                    llm_out = _call_entity_res_lm(lm, prompt)
                                    actions = extract_json_array(llm_out)
                                    if actions is None:
                                        actions = [
                                            {
                                                "action": "KeepEntity",
                                                "entity_id": m["id"],
                                                "rationale": "LLM parse failed; conservatively kept",
                                            }
                                            for m in chunk
                                        ]
                                    apply_actions(chunk, actions, entities_by_id, canonical_store, log_entries)

    # global noise handling
    for nent in by_cluster.get(-1, []):
        ent = entities_by_id[nent["id"]]
        ent["resolved_action"] = "kept_noise_global"
        ent["resolution_rationale"] = "Global noise preserved for manual review"
        log_entries.append(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "action": "keep_noise_global",
                "entity_id": ent["id"],
            }
        )

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


#endregion#? Final Ent Res v10 - DSPy LLM Config
#?#########################  End  ##########################




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
ENT_RAW_SEED = Path("data/Entities/Ent_Raw_0/entities_raw.jsonl")

CLUSTERED_PATH = Path("data/Entities/Ent_1st/Ent_Clustering_1st/entities_clustered.jsonl")
CANONICAL_PATH = Path("data/Entities/Ent_1st/Ent_Resolved_1st/canonical_entities.jsonl")
RESOLVED_PATH = Path("data/Entities/Ent_1st/Ent_Resolved_1st/entities_resolved.jsonl")

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
    print("DEBUG iterative_resolution running from:", __file__)
    print("DEBUG ENT_RAW_SEED =", ENT_RAW_SEED)
    print("DEBUG ITER_DIR =", ITER_DIR)

    current_input = ENT_RAW_SEED

    # backup seed once
    seed_backup = ITER_DIR / "entities_raw_seed_backup.jsonl"
    if not seed_backup.exists():
        print("DEBUG: creating seed backup:", seed_backup)
        ITER_DIR.mkdir(parents=True, exist_ok=True)
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
RAW_SEED_PATH = Path("data/Entities/Ent_Raw_0/entities_raw.jsonl")

# Default locations for latest iteration entities and class-identification output
DEFAULT_ITER_DIR = Path("data/Entities/iterative_runs/")
DEFAULT_CLS_OUT_DIR = Path("data/Classes/Cls_Input")
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
#region:#?   Cls Rec V10 - DSPy LLM Config (Class hint type included)

#!/usr/bin/env python3
"""
Iterative Class Recognition (ClassRec) with per-iteration class outputs
matching the cluster-file visual/JSON format and including class metadata
(label/desc/confidence/evidence + source cluster id + class_type_hint).

This version:

- Keeps the same clustering + prompt logic as v4.
- Replaces the raw OpenAI client with DSPy + TraceKGLLMConfig (LLM Model V3).
- Uses make_lm_for_step(cfg, "class_rec") to obtain a dspy.LM.
- Allows a single global config (default_model etc.) and optional per-step overrides
  (class_rec_model / rec_model / default_model) via TraceKGLLMConfig.

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


# ----------------------------- CONFIG -----------------------------
INPUT_PATH = Path("data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("data/Classes/Cls_Rec")
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

# prompt and LLM / limits (defaults used if llm_config is None)
CLASSREC_MODEL = "gpt-5.1"
LLM_MAX_TOKENS = 8000
MAX_MEMBERS_PER_PROMPT = 10
PROMPT_CHAR_PER_TOKEN = 4          # crude estimate
MAX_PROMPT_TOKENS_EST = 8000

# iteration control
MAX_RECLUSTER_ROUNDS = 12  # safety cap
VERBOSE = False


# ------------------------- HF Embedder ------------------------------
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
            batch = texts[i : i + batch_size]
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
        resolution = (
            safe_text(e, "resolution_context")
            or safe_text(e, "text_span")
            or safe_text(e, "context_phrase")
            or ""
        )
        et = safe_text(e, "entity_type_hint") or ""
        node_props = e.get("node_properties") or []
        node_props_text = ""
        if isinstance(node_props, list) and node_props:
            pieces = []
            for np_ in node_props:
                if isinstance(np_, dict):
                    pname = np_.get("prop_name") or np_.get("name") or ""
                    pval = np_.get("prop_value") or np_.get("value") or ""
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
    emb_ctx = embedder.encode_batch(ctxs) if any(t.strip() for t in ctxs) else None

    D = None
    for arr in (emb_name, emb_desc, emb_ctx):
        if arr is not None and arr.shape[0] > 0:
            D = arr.shape[1]
            break
    if D is None:
        raise ValueError("No textual field produced embeddings")

    def _ensure(arr):
        if arr is None:
            return np.zeros((len(entities), D))
        if arr.shape[1] != D:
            raise ValueError("embedding dim mismatch")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx = _ensure(emb_ctx)
    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0:
        raise ValueError("invalid weights")
    w_name /= Wsum
    w_desc /= Wsum
    w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)
    return combined


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    metric=HDBSCAN_METRIC,
    use_umap=USE_UMAP,
) -> Tuple[np.ndarray, object]:
    X = embeddings
    if use_umap and UMAP_AVAILABLE and X.shape[0] >= 5:
        reducer = umap.UMAP(
            n_components=min(UMAP_N_COMPONENTS, max(2, X.shape[0] - 1)),
            n_neighbors=min(UMAP_N_NEIGHBORS, max(2, X.shape[0] - 1)),
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=42,
        )
        X = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer


# ------------------ Prompt (REVISED to be MUST) -----------------------
from TKG_Prompts import CLASS_REC_PROMPT_TEMPLATE
CLASS_PROMPT_TEMPLATE = CLASS_REC_PROMPT_TEMPLATE


def build_members_block(members: List[Dict]) -> str:
    rows = []
    for m in members:
        eid = m.get("id", "")
        name = (m.get("entity_name") or "")[:120].replace("\n", " ")
        desc = (m.get("entity_description") or "")[:300].replace("\n", " ")
        res = (
            (m.get("resolution_context") or m.get("context_phrase") or "")
            [:400]
            .replace("\n", " ")
        )
        et = (m.get("entity_type_hint") or "")[:80].replace("\n", " ")
        node_props = m.get("node_properties") or []
        np_txt = json.dumps(node_props, ensure_ascii=False) if node_props else ""
        rows.append(f"{eid} | {name} | {desc} | {res} | {et} | {np_txt}")
    return "\n".join(rows)


def parse_json_array_from_text(txt: str):
    if not txt:
        return None
    # s = txt.strip() #Injam
    s = coerce_llm_text(txt).strip()
    if s.startswith("```"):
        s = s.strip("`")
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        cand = s[start : end + 1]
        try:
            return json.loads(cand)
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        return None


# ------------------- DSPy integration helpers ------------------------
def _get_lm_for_class_rec(
    llm_config: Optional["TraceKGLLMConfig"],
    model: str,
    max_tokens: int,
):
    """
    Resolve a DSPy LM for the Class Recognition step:

      - If llm_config is provided, use it directly (with per-step overrides for 'class_rec').
      - Otherwise, build a minimal TraceKGLLMConfig using the function args
        for backward compatibility (single model everywhere).
    """
    if llm_config is not None:
        cfg = llm_config
    else:
        # Backward-compatible default: one model everywhere, with given max_tokens.
        cfg = TraceKGLLMConfig(default_model=model, max_tokens=max_tokens)

    try:
        cfg.validate()
    except Exception as e:
        print(f"[ClassRec] WARNING: llm_config.validate() failed: {e}")

    lm = make_lm_for_step(cfg, "class_rec")
    return lm


def _call_class_rec_lm(lm, prompt: str) -> str:
    """
    Small helper to call the DSPy LM and normalize the output to a plain string.
    """
    try:
        outputs = lm(prompt)
    except Exception as e:
        print(f"[ClassRec] LM call error: {e}")
        return ""

    if isinstance(outputs, list):
        return outputs[0] if outputs else ""
    return str(outputs or "")


# ------------------- Worker: process a chunk of members --------------------
def process_member_chunk_llm(
    members: List[Dict],
    lm,
    single_entity_mode: bool = False,
) -> List[Dict]:
    members_block = build_members_block(members)
    prompt = CLASS_PROMPT_TEMPLATE.format(members_block=members_block)
    est_tokens = max(1, int(len(prompt) / PROMPT_CHAR_PER_TOKEN))
    if est_tokens > MAX_PROMPT_TOKENS_EST:
        if VERBOSE:
            print(
                f"[warning] prompt too large (est_tokens={est_tokens}) -> skipping chunk of size {len(members)}"
            )
        return []

    llm_out = _call_class_rec_lm(lm, prompt)
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
        # enforce multi-member rule when multiple entities are present
        if not single_entity_mode and len(members) > 1 and len(member_ids) < 2:
            continue
        confidence = (
            float(c.get("confidence", 0.0)) if c.get("confidence") is not None else 0.0
        )
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
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        candidates.append(candidate)
    return candidates


# -------------------- Utility: write cluster files (full entity objects) -----------
def write_cluster_summary(path: Path, cluster_map: Dict[int, List[int]], entities: List[Dict]):
    """
    Write initial cluster summary: full entity objects grouped by cluster label.

    path: output JSON file path, e.g. data/Classes/Cls_Rec/initial_cluster_entities.json
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    n_entities = len(entities)
    clusters = {}
    for k, idxs in sorted(cluster_map.items(), key=lambda x: x[0]):
        arr = []
        for i in idxs:
            ent = entities[i]
            arr.append(ent)
        clusters[str(k)] = arr
    meta = {"n_entities": n_entities, "n_clusters": len(clusters)}

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{\n")
        fh.write(f'  "n_entities": {meta["n_entities"]},\n')
        fh.write(f'  "n_clusters": {meta["n_clusters"]},\n')
        fh.write('  "clusters": {\n')

        cluster_items = list(clusters.items())
        for ci, (k, ents) in enumerate(cluster_items):
            fh.write(f'    "{k}": [\n')
            for ei, ent in enumerate(ents):
                ent_json = json.dumps(ent, ensure_ascii=False, separators=(",", ": "))
                fh.write(f"      {ent_json}")
                if ei < len(ents) - 1:
                    fh.write(",\n")
                else:
                    fh.write("\n")
            fh.write("    ]")
            if ci < len(cluster_items) - 1:
                fh.write(",\n")
            else:
                fh.write("\n")
        fh.write("  }\n")
        fh.write("}\n")


def write_classes_round(
    path: Path, candidates: List[Dict], entities: List[Dict], id_to_index: Dict[str, int]
):
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
            "members": member_objs,
        }
        classes_map[cid] = meta
        total_members += len(member_objs)
    meta = {
        "n_classes": len(classes_map),
        "n_members_total": total_members,
        "classes": classes_map,
    }
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{\n")
        fh.write(f'  "n_classes": {meta["n_classes"]},\n')
        fh.write(f'  "n_members_total": {meta["n_members_total"]},\n')
        fh.write('  "classes": {\n')
        items = list(classes_map.items())
        for ci, (k, cls_meta) in enumerate(items):
            fh.write(f'    "{k}": {{\n')
            fh.write(
                f'      "class_label": {json.dumps(cls_meta["class_label"], ensure_ascii=False)},\n'
            )
            fh.write(
                f'      "class_description": {json.dumps(cls_meta["class_description"], ensure_ascii=False)},\n'
            )
            fh.write(
                f'      "class_type_hint": {json.dumps(cls_meta["class_type_hint"], ensure_ascii=False)},\n'
            )
            fh.write(
                f'      "confidence": {json.dumps(cls_meta["confidence"], ensure_ascii=False)},\n'
            )
            fh.write(
                f'      "evidence_excerpt": {json.dumps(cls_meta["evidence_excerpt"], ensure_ascii=False)},\n'
            )
            fh.write(
                f'      "source_cluster_id": {json.dumps(cls_meta["source_cluster_id"], ensure_ascii=False)},\n'
            )
            fh.write('      "members": [\n')
            for ei, ent in enumerate(cls_meta["members"]):
                ent_json = json.dumps(ent, ensure_ascii=False, separators=(",", ": "))
                fh.write(f"        {ent_json}")
                if ei < len(cls_meta["members"]) - 1:
                    fh.write(",\n")
                else:
                    fh.write("\n")
            fh.write("      ]\n")
            fh.write("    }")
            if ci < len(items) - 1:
                fh.write(",\n")
            else:
                fh.write("\n")
        fh.write("  }\n")
        fh.write("}\n")


# -------------------- Main iterative orchestration -----------------------
def classrec_iterative_main(
    model: str = CLASSREC_MODEL,
    max_tokens: int = LLM_MAX_TOKENS,
    llm_config: Optional["TraceKGLLMConfig"] = None,
):
    """
    Main entry point for Class Recognition (ClassRec).

    LLM behavior:
      - If llm_config is provided, a single TraceKGLLMConfig object controls all LLM
        behavior for this step (models, tokens, temperature, api_base, etc.). The
        'class_rec_model' / 'rec_model' / 'default_model' fields are used via
        make_lm_for_step(cfg, "class_rec").
      - If llm_config is None, this function falls back to `model` and `max_tokens`
        by constructing a minimal TraceKGLLMConfig(default_model=model, max_tokens=max_tokens).

    All LLM calls use a single dspy.LM created once and reused.
    """
    entities = load_entities(INPUT_PATH)
    print(f"[start] loaded {len(entities)} entities from {INPUT_PATH}")

    # Ensure the output directory exists before we try to write initial_cluster_entities.json
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for e in entities:
        if "id" not in e:
            e["id"] = "En_" + uuid.uuid4().hex[:8]

    id_to_index = {e["id"]: i for i, e in enumerate(entities)}

    embedder = HFEmbedder(model_name=EMBED_MODEL, device=DEVICE)
    combined_emb = compute_combined_embeddings(embedder, entities, weights=WEIGHTS)
    print("[info] embeddings computed, shape:", combined_emb.shape)

    labels, _ = run_hdbscan(
        combined_emb,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        use_umap=USE_UMAP,
    )
    print("[info] initial clustering done. unique labels:", len(set(labels)))

    cluster_to_indices: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        cluster_to_indices.setdefault(int(lab), []).append(idx)

    write_cluster_summary(INITIAL_CLUSTER_OUT, cluster_to_indices, entities)
    if VERBOSE:
        print(f"[write] initial cluster file -> {INITIAL_CLUSTER_OUT}")

    seen_by_llm = set()
    assigned_entity_ids = set()
    all_candidates: List[Dict] = []

    # Resolve LM once for this whole run
    lm = _get_lm_for_class_rec(llm_config=llm_config, model=model, max_tokens=max_tokens)

    def call_and_record(
        members_indices: List[int],
        source_cluster: Optional[object] = None,
        single_entity_mode: bool = False,
    ) -> List[Dict]:
        nonlocal seen_by_llm, assigned_entity_ids, all_candidates
        if not members_indices:
            return []
        members = [entities[i] for i in members_indices]
        results: List[Dict] = []
        for i in range(0, len(members), MAX_MEMBERS_PER_PROMPT):
            chunk = members[i : i + MAX_MEMBERS_PER_PROMPT]
            for m in chunk:
                if m.get("id"):
                    seen_by_llm.add(m["id"])
            candidates = process_member_chunk_llm(
                chunk, lm=lm, single_entity_mode=single_entity_mode
            )
            for c in candidates:
                mids = c.get("member_ids", [])
                member_entities = [
                    entities[id_to_index[mid]] for mid in mids if mid in id_to_index
                ]
                if not member_entities:
                    continue
                c["member_entities"] = member_entities
                c["source_cluster_id"] = source_cluster
                all_candidates.append(c)
                for mid in mids:
                    assigned_entity_ids.add(mid)
                results.append(c)
        return results

    # ---------- Round 0: coarse clusters (with optional local subclusters) ----------
    round0_candidates: List[Dict] = []
    if VERBOSE:
        print("[round0] processing coarse non-noise clusters")
    for lab, idxs in sorted(cluster_to_indices.items(), key=lambda x: x[0]):
        if lab == -1:
            continue
        if VERBOSE:
            print(f"[round0] cluster {lab} size={len(idxs)}")
        if len(idxs) > MAX_CLUSTER_SIZE_FOR_LOCAL:
            try:
                sub_emb = combined_emb[idxs]
                local_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(2, LOCAL_HDBSCAN_MIN_CLUSTER_SIZE),
                    min_samples=LOCAL_HDBSCAN_MIN_SAMPLES,
                    metric="euclidean",
                    cluster_selection_method="eom",
                )
                local_labels = local_clusterer.fit_predict(sub_emb)
            except Exception:
                local_labels = np.zeros(len(idxs), dtype=int)
            local_map: Dict[int, List[int]] = {}
            for i_local, lab_local in enumerate(local_labels):
                global_idx = idxs[i_local]
                local_map.setdefault(int(lab_local), []).append(global_idx)
            for sublab, subidxs in local_map.items():
                if sublab == -1:
                    continue
                source_id = {"coarse_cluster": int(lab), "local_subcluster": int(sublab)}
                cand = call_and_record(
                    subidxs,
                    source_cluster=source_id,
                    single_entity_mode=(len(subidxs) == 1),
                )
                round0_candidates.extend(cand)
        else:
            source_id = {"coarse_cluster": int(lab), "local_subcluster": None}
            cand = call_and_record(
                idxs,
                source_cluster=source_id,
                single_entity_mode=(len(idxs) == 1),
            )
            round0_candidates.extend(cand)

    classes_round0_path = Path(f"{CLASSES_PREFIX}0.json")
    write_classes_round(classes_round0_path, round0_candidates, entities, id_to_index)
    if VERBOSE:
        print(f"[write] classes round 0 -> {classes_round0_path}")

    # ---------- Recluster rounds over noise + unassigned ----------
    original_noise_indices = cluster_to_indices.get(-1, [])
    round_num = 0
    while round_num < MAX_RECLUSTER_ROUNDS:
        round_num += 1
        seen_but_unassigned = list(seen_by_llm - assigned_entity_ids)
        pool_ids = {entities[i]["id"] for i in original_noise_indices}
        pool_ids.update(seen_but_unassigned)
        pool_ids = [pid for pid in pool_ids if pid not in assigned_entity_ids]
        if not pool_ids:
            if VERBOSE:
                print(f"[reclust {round_num}] pool empty -> stopping")
            break
        pool_indices = [id_to_index[pid] for pid in pool_ids if pid in id_to_index]
        if not pool_indices:
            if VERBOSE:
                print(f"[reclust {round_num}] no valid pool indices -> stopping")
            break

        if VERBOSE:
            print(f"[reclust {round_num}] reclustering pool size={len(pool_indices)}")
        try:
            sub_emb = combined_emb[pool_indices]
            labels_sub, _ = run_hdbscan(
                sub_emb,
                min_cluster_size=2,
                min_samples=1,
                use_umap=False,
            )
        except Exception:
            labels_sub = np.zeros(len(pool_indices), dtype=int)

        sub_cluster_map: Dict[int, List[int]] = {}
        for local_i, lab_sub in enumerate(labels_sub):
            global_idx = pool_indices[local_i]
            sub_cluster_map.setdefault(int(lab_sub), []).append(global_idx)

        recluster_path = Path(f"{RECLUSTER_PREFIX}{round_num}.json")
        write_cluster_summary(recluster_path, sub_cluster_map, entities)
        if VERBOSE:
            print(f"[write] recluster round {round_num} -> {recluster_path}")

        round_candidates: List[Dict] = []
        new_classes_count = 0
        for lab_sub, gidxs in sorted(
            sub_cluster_map.items(), key=lambda x: (x[0] == -1, x[0])
        ):
            if lab_sub == -1:
                continue
            if VERBOSE:
                print(
                    f"[reclust {round_num}] processing subcluster {lab_sub} size={len(gidxs)}"
                )
            source_id = {"recluster_round": int(round_num), "subcluster": int(lab_sub)}
            cand = call_and_record(
                gidxs,
                source_cluster=source_id,
                single_entity_mode=(len(gidxs) == 1),
            )
            round_candidates.extend(cand)
            new_classes_count += len(cand)

        classes_round_path = Path(f"{CLASSES_PREFIX}{round_num}.json")
        write_classes_round(classes_round_path, round_candidates, entities, id_to_index)
        if VERBOSE:
            print(
                f"[write] classes round {round_num} -> {classes_round_path}  (new_classes={new_classes_count})"
            )

        if new_classes_count == 0:
            if VERBOSE:
                print(
                    f"[reclust {round_num}] no new classes -> stopping recluster loop"
                )
            break

    # ---------- Single-entity pass for leftovers ----------
    remaining_after_reclustering = [
        e for e in entities if e["id"] not in assigned_entity_ids
    ]
    if VERBOSE:
        print(
            f"[single pass] remaining entities (before single-entity pass): {len(remaining_after_reclustering)}"
        )

    single_candidates: List[Dict] = []
    for e in remaining_after_reclustering:
        source_id = {"single_pass": True}
        cand = call_and_record(
            [id_to_index[e["id"]]],
            source_cluster=source_id,
            single_entity_mode=True,
        )
        single_candidates.extend(cand)

    classes_single_path = Path(f"{CLASSES_PREFIX}single.json")
    write_classes_round(classes_single_path, single_candidates, entities, id_to_index)
    if VERBOSE:
        print(f"[write] classes round single -> {classes_single_path}")

    # ---------- Summary outputs ----------
    with open(CLASS_CANDIDATES_OUT, "w", encoding="utf-8") as fh:
        for c in all_candidates:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    if VERBOSE:
        print(
            f"[write] cumulative class_candidates -> {CLASS_CANDIDATES_OUT} (count={len(all_candidates)})"
        )

    final_remaining = [e for e in entities if e["id"] not in assigned_entity_ids]
    with open(REMAINING_OUT, "w", encoding="utf-8") as fh:
        for e in final_remaining:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    if VERBOSE:
        print(
            f"[write] final remaining entities -> {REMAINING_OUT} (count={len(final_remaining)})"
        )

    print("[done] ClassRec iterative v10 (DSPy) finished.")


# -----------------------
# Cls Recognition  - Run statement
# -----------------------

# if __name__ == "__main__":
#     classrec_iterative_main()

#endregion#? Cls Rec V10 - DSPy LLM Config
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?   Create input for Cls Res from per-round classes




#!/usr/bin/env python3
"""
merge_classes_for_cls_res.py

Merge per-round classes files (classes_round_*.json) into a single
JSONL + JSON file suitable as input to the next step (Cls Res).

Output:
 - data/Classes/Cls_Res/Cls_Res_input/classes_for_cls_res.jsonl
 - data/Classes/Cls_Res/Cls_Res_input/classes_for_cls_res.json
"""

import json
from pathlib import Path
from collections import defaultdict

# ROOT = Path("data/Classes/Cls_Rec")
ROOT = Path("data/Classes/Cls_Rec")
PATTERN = "classes_round_*.json"

# OUTPUT_ROOT = Path("data/Classes/Cls_Res/Cls_Res_input")
OUTPUT_ROOT = Path("data/Classes/Cls_Res/Cls_Res_input")

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

    # Ensure output directory exists
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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
#region:#?   Cls Res V10  - Split + Remark + Summary (DSPy LLM Config)


#!/usr/bin/env python3
"""
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
  data/Classes/Cls_Res/Cls_Res_input/classes_for_cls_res.json

Output (written under OUT_DIR):
  - per-cluster decisions: cluster_<N>_decisions.json
  - per-cluster raw llm output: llm_raw/cluster_<N>_llm_raw.txt
  - per-cluster prompts: llm_raw/cluster_<N>_prompt.txt
  - cumulative action log: cls_res_action_log.jsonl
  - final resolved classes: final_classes_resolved.json and .jsonl
  - summary/all_clusters_decisions.json (aggregated decisions)
  - summary/stats_summary.json (aggregate statistics)

V10 (DSPy):
- Removes direct OpenAI client usage.
- All LLM calls go through TraceKGLLMConfig + make_lm_for_step(cfg, "class_res").
- One central config can drive the entire TRACE KG pipeline, with optional
  per-step overrides (class_res_model / res_model / default_model).
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

# ----------------------------- CONFIG -----------------------------
INPUT_CLASSES = Path("data/Classes/Cls_Res/Cls_Res_input/classes_for_cls_res.json")
#INPUT_CLASSES = Path("data/Classes/Cls_Rec/classes_for_cls_res-Wrong.json")
SRC_ENTITIES_PATH = Path("data/Classes/Cls_Input/cls_input_entities.jsonl")
OUT_DIR = Path("data/Classes/Cls_Res")
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
    "members": 0.30,
}

# clustering params
USE_UMAP = True
UMAP_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"

# LLM defaults (only used if llm_config is None)
CLSRES_MODEL = "gpt-5.1"
LLM_MAX_TOKENS = 16000  # mapped into TraceKGLLMConfig.max_tokens

# behavioral flags
VERBOSE = False
WRITE_INTERMEDIATE = True


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
            batch = texts[i : i + batch_size]
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
        "entity_type_hint": safe_str(member.get("entity_type_hint", ""))[:80],
    }


# ---------------------- Build class texts & embeddings ------------------
def build_class_texts(
    classes: List[Dict],
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
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


def compute_class_embeddings(
    embedder: HFEmbedder, classes: List[Dict], weights: Dict[str, float]
) -> np.ndarray:
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
    use_umap: bool = USE_UMAP,
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
                random_state=42,
            )
            X_reduced = reducer.fit_transform(X)
            if X_reduced is not None and X_reduced.shape[0] == N:
                X = X_reduced
            else:
                if VERBOSE:
                    print(
                        f"[warn] UMAP returned invalid shape "
                        f"{None if X_reduced is None else X_reduced.shape}; skipping UMAP"
                    )
        except Exception as e:
            if VERBOSE:
                print(
                    f"[warn] UMAP failed (N={N}, n_comp={safe_n_components}, "
                    f"n_nei={safe_n_neighbors}): {e}. Proceeding without UMAP."
                )
            X = embeddings
    else:
        if use_umap and UMAP_AVAILABLE and VERBOSE:
            print(
                f"[info] Skipping UMAP (N={N} < 6) to avoid unstable spectral computations."
            )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer


# ---------------------- LLM prompt template (extended) ------------------
from TKG_Prompts import CLASS_RES_PROMPT_TEMPLATE
CLSRES_PROMPT_TEMPLATE = CLASS_RES_PROMPT_TEMPLATE


def sanitize_json_like(text: str) -> Optional[Any]:
    # crude sanitizer: extract first [...] region and try loads. Fix common trailing commas and smart quotes.
    if not text or not text.strip():
        return None
    # s = text.strip()  #Injam
    s = coerce_llm_text(text).strip()
    # replace smart quotes
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    # find first [ ... ] block
    start = s.find("[")
    end = s.rfind("]")
    cand = s
    if start != -1 and end != -1 and end > start:
        cand = s[start : end + 1]
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
    new_type: Optional[str],
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
        "_merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
    id_to_entity: Dict[str, Dict],
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
        "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    all_classes[new_cid] = obj
    return new_cid


def execute_reassign_entities(
    all_classes: Dict[str, Dict],
    entity_ids: List[str],
    from_class_id: Optional[str],
    to_class_id: str,
    id_to_entity: Dict[str, Dict],
):
    # remove from source(s)
    for cid, c in list(all_classes.items()):
        if from_class_id and cid != from_class_id:
            continue
        new_members = [
            m for m in c.get("members", []) if m["id"] not in set(entity_ids)
        ]
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
    new_remark: Optional[str],
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
    splits_specs: List[Dict[str, Any]],
) -> List[Tuple[str, Optional[str], List[str]]]:
    """
    Split a source class into several new classes.
    Returns list of (new_class_id, provisional_id, member_ids_used) for each created class.
    """
    if source_class_id not in all_classes:
        raise ValueError(f"split_class: source_class_id {source_class_id} not found")
    src = all_classes[source_class_id]
    src_members = src.get("members", []) or []
    src_member_map = {
        m["id"]: m for m in src_members if isinstance(m, dict) and m.get("id")
    }
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
            "_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
    allow_missing: bool = False,
) -> Optional[str]:
    if raw_id is None:
        return None
    real = provisional_to_real.get(raw_id, raw_id)
    if real not in all_classes and not allow_missing:
        raise ValueError(
            f"resolve_class_id: {raw_id} (resolved to {real}) not found in all_classes"
        )
    return real


# ---------------------- DSPy integration helpers ------------------------
def _get_lm_for_class_res(
    llm_config: Optional["TraceKGLLMConfig"],
    model: str,
    max_tokens: int,
):
    """
    Resolve a DSPy LM for the Class Resolution step:

      - If llm_config is provided, use it directly (with per-step overrides for 'class_res').
      - Otherwise, build a minimal TraceKGLLMConfig using the function args
        for backward compatibility (single model everywhere).
    """
    if llm_config is not None:
        cfg = llm_config
    else:
        cfg = TraceKGLLMConfig(default_model=model, max_tokens=max_tokens)

    try:
        cfg.validate()
    except Exception as e:
        print(f"[ClassRes] WARNING: llm_config.validate() failed: {e}")

    lm = make_lm_for_step(cfg, "class_res")
    return lm


def _call_class_res_lm(lm, prompt: str) -> str:
    """
    Small helper to call the DSPy LM and normalize the output to a plain string.
    """
    try:
        outputs = lm(prompt)
    except Exception as e:
        print(f"[ClassRes] LM call error: {e}")
        return ""

    if isinstance(outputs, list):
        return outputs[0] if outputs else ""
    return str(outputs or "")


# ---------------------- Main orchestration ------------------------------
def classres_main(
    model: str = CLSRES_MODEL,
    max_tokens: int = LLM_MAX_TOKENS,
    llm_config: Optional["TraceKGLLMConfig"] = None,
):
    """
    Main entry point for Class Resolution (Cls Res).

    LLM behavior:
      - If llm_config is provided, a single TraceKGLLMConfig object controls all LLM
        behavior for this step (models, tokens, api_base, etc.). The
        'class_res_model' / 'res_model' / 'default_model' fields are used via
        make_lm_for_step(cfg, "class_res").
      - If llm_config is None, this function falls back to `model` and `max_tokens`
        by constructing a minimal TraceKGLLMConfig(default_model=model, max_tokens=max_tokens).

    All LLM calls use a single dspy.LM created once and reused across clusters.
    """
    # load classes
    if not INPUT_CLASSES.exists():
        raise FileNotFoundError(f"Input classes file not found: {INPUT_CLASSES}")
    classes_list = load_json(INPUT_CLASSES)
    print(
        f"[start] loaded {len(classes_list)} merged candidate classes from {INPUT_CLASSES}"
    )

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
        mids = [
            m["id"]
            for m in members
            if isinstance(m, dict) and m.get("id")
        ]
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
        use_umap=USE_UMAP,
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

    # Resolve LM once for this whole run
    lm = _get_lm_for_class_res(llm_config=llm_config, model=model, max_tokens=max_tokens)

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
            cluster_classes.append(
                {
                    "candidate_id": cid,
                    "class_label": c.get("class_label", ""),
                    "class_description": c.get("class_description", ""),
                    "class_type_hint": c.get("class_type_hint", ""),
                    "class_group": c.get("class_group", "TBD"),
                    "confidence": float(c.get("confidence", 0.0)),
                    "evidence_excerpt": c.get("evidence_excerpt", ""),
                    "member_ids": c.get("member_ids", []),
                    "members": members_compact,
                    "remarks": c.get("remarks", []),
                }
            )

        cluster_block = json.dumps(cluster_classes, ensure_ascii=False, indent=2)
        prompt = CLSRES_PROMPT_TEMPLATE.replace("{cluster_block}", cluster_block)

        # log prompt
        prompt_path = RAW_LLM_DIR / f"cluster_{cluster_label}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # call LLM via DSPy
        raw_out = _call_class_res_lm(lm, prompt)

        # write raw output
        raw_path = RAW_LLM_DIR / f"cluster_{cluster_label}_llm_raw.txt"
        raw_path.write_text(raw_out, encoding="utf-8")

        # try parse/sanitize
        parsed = sanitize_json_like(raw_out)
        if parsed is None:
            print(
                f"[warn] failed to parse LLM output for cluster {cluster_label}; "
                f"skipping automated actions for this cluster."
            )
            dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
            dec_path.write_text(
                json.dumps(
                    {"cluster_label": cluster_label, "raw_llm": raw_out},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
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
                        decisions.append(
                            {
                                "action": "merge_skip_too_few",
                                "requested_class_ids": cids_raw,
                                "valid_class_ids": valid_cids,
                                "justification": justification,
                                "remark": remark_val,
                                "confidence": confidence_val,
                            }
                        )
                        continue

                    new_cid = execute_merge_classes(
                        all_classes, valid_cids, new_name, new_desc, new_type
                    )

                    # map provisional id -> new class id
                    if prov_id:
                        provisional_to_real[prov_id] = new_cid
                    # also map old real ids -> new id so later references can still resolve
                    for old in valid_cids:
                        provisional_to_real.setdefault(old, new_cid)

                    decisions.append(
                        {
                            "action": "merge_classes",
                            "input_class_ids": valid_cids,
                            "result_class_id": new_cid,
                            "provisional_id": prov_id,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val,
                        }
                    )

                elif fn == "create_class":
                    name = args.get("name")
                    desc = args.get("description")
                    t = args.get("class_type_hint")
                    mids = args.get("member_ids", []) or []
                    prov_id = args.get("provisional_id")

                    mids_valid = [m for m in mids if m in id_to_entity]
                    new_cid = execute_create_class(
                        all_classes, name, desc, t, mids_valid, id_to_entity
                    )

                    if prov_id:
                        provisional_to_real[prov_id] = new_cid

                    decisions.append(
                        {
                            "action": "create_class",
                            "result_class_id": new_cid,
                            "provisional_id": prov_id,
                            "member_ids_added": mids_valid,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val,
                        }
                    )

                elif fn == "reassign_entities":
                    eids = args.get("entity_ids", []) or []
                    from_c_raw = args.get("from_class_id")
                    to_c_raw = args.get("to_class_id")

                    eids_valid = [e for e in eids if e in id_to_entity]

                    from_c = resolve_class_id(
                        from_c_raw, all_classes, provisional_to_real, allow_missing=True
                    )
                    to_c = resolve_class_id(
                        to_c_raw, all_classes, provisional_to_real, allow_missing=False
                    )

                    execute_reassign_entities(
                        all_classes, eids_valid, from_c, to_c, id_to_entity
                    )

                    decisions.append(
                        {
                            "action": "reassign_entities",
                            "entity_ids": eids_valid,
                            "from": from_c_raw,
                            "from_resolved": from_c,
                            "to": to_c_raw,
                            "to_resolved": to_c,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val,
                        }
                    )

                elif fn == "modify_class":
                    cid_raw = args.get("class_id")
                    new_name = args.get("new_name")
                    new_desc = args.get("new_description")
                    new_type = args.get("new_class_type_hint")
                    new_group = args.get("new_class_group")
                    new_remark = args.get("remark")

                    cid_real = resolve_class_id(
                        cid_raw, all_classes, provisional_to_real, allow_missing=False
                    )
                    execute_modify_class(
                        all_classes,
                        cid_real,
                        new_name,
                        new_desc,
                        new_type,
                        new_group,
                        new_remark,
                    )

                    decisions.append(
                        {
                            "action": "modify_class",
                            "class_id": cid_raw,
                            "class_id_resolved": cid_real,
                            "new_name": new_name,
                            "new_description": new_desc,
                            "new_class_type_hint": new_type,
                            "new_class_group": new_group,
                            "remark": new_remark,
                            "justification": justification,
                            "confidence": confidence_val,
                        }
                    )

                elif fn == "split_class":
                    source_raw = args.get("source_class_id")
                    splits_specs = args.get("splits", []) or []

                    source_real = resolve_class_id(
                        source_raw, all_classes, provisional_to_real, allow_missing=False
                    )
                    created = execute_split_class(
                        all_classes, source_real, splits_specs
                    )

                    created_summary = []
                    for new_cid, prov_id, mids_used in created:
                        if prov_id:
                            provisional_to_real[prov_id] = new_cid
                        created_summary.append(
                            {
                                "new_class_id": new_cid,
                                "provisional_id": prov_id,
                                "member_ids": mids_used,
                            }
                        )

                    decisions.append(
                        {
                            "action": "split_class",
                            "source_class_id": source_raw,
                            "source_class_id_resolved": source_real,
                            "created_classes": created_summary,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val,
                        }
                    )

                else:
                    decisions.append(
                        {
                            "action": "skip_unknown_function",
                            "raw": step,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val,
                        }
                    )

            except Exception as e:
                decisions.append(
                    {
                        "action": "error_executing",
                        "function": fn,
                        "error": str(e),
                        "input": step,
                        "justification": justification,
                        "remark": remark_val,
                        "confidence": confidence_val,
                    }
                )

        # write decisions file for this cluster
        dec_path = OUT_DIR / f"cluster_{cluster_label}_decisions.json"
        dec_obj = {
            "cluster_label": cluster_label,
            "cluster_classes": cluster_classes,
            "llm_raw": raw_out,
            "parsed_steps": parsed,
            "executed_decisions": decisions,
            "provisional_to_real": provisional_to_real,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        dec_path.write_text(
            json.dumps(dec_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # append to action log
        with open(action_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dec_obj, ensure_ascii=False) + "\n")

    # After all clusters processed: write final classes output
    final_classes = list(all_classes.values())
    out_json = OUT_DIR / "final_classes_resolved.json"
    out_jsonl = OUT_DIR / "final_classes_resolved.jsonl"
    out_json.write_text(
        json.dumps(final_classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for c in final_classes:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(
        f"[done] wrote final resolved classes -> {out_json}  (count={len(final_classes)})"
    )
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
        encoding="utf-8",
    )

    # Compute statistics
    total_clusters = len(cluster_decisions)
    actions_by_type: Dict[str, int] = {}
    total_remarks = 0
    clusters_with_any_decisions = 0
    clusters_with_structural = 0
    clusters_only_classgroup = 0

    structural_actions = {
        "merge_classes",
        "create_class",
        "reassign_entities",
        "split_class",
    }

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
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    stats_path = summary_dir / "stats_summary.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[done] summary decisions -> {all_clusters_decisions_path}")
    print(f"[done] summary stats -> {stats_path}")


# # -----------------------
# # Cls Res  - Run statement
# # -----------------------

# if __name__ == "__main__":
#     classres_main()

#endregion#? Cls Res V10  - Split + Remark + Summary (DSPy LLM Config)
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
BASE_INPUT_CLASSES = Path("data/Classes/Cls_Res/Cls_Res_input/classes_for_cls_res.json")
EXPERIMENT_ROOT = Path("data/Classes/Cls_Res/Cls_Res_IterativeRuns")

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
#region:#?   Rel Rec V10 - DSPy LLM Config (Better Rel Naming)

#!/usr/bin/env python3
"""
Relation Recognition (Rel Rec) — Context-Enriched KG

- Reads:
    - entities_with_class.jsonl
    - chunks_sentence.jsonl
- For each chunk, finds entities that occur in that chunk.
- Calls an LLM (via DSPy + TraceKGLLMConfig) to extract directed relations between those entities.
- Writes:
    - relations_raw.jsonl  (one JSON object per relation instance)

Key design:
- Entities are ALREADY resolved and guaranteed to belong to their chunks.
- This is the LAST time we look at the raw chunk text.
- We aim for HIGH RECALL and rich QUALIFIERS (context-enriched KG).
- Intrinsic node properties were already handled in entity stages; here we treat
  almost everything else as relation-level context.

LLM usage:
- All LLM calls go through a single DSPy LM obtained from TraceKGLLMConfig
  via make_lm_for_step(cfg, "rel_rec").
- If no llm_config is provided, run_rel_rec builds a minimal TraceKGLLMConfig
  using the `model` and `max_tokens` arguments, so existing code that just
  passes a model string still works.
"""

import argparse
import json
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL_NAME = "gpt-5.1"   # default if no llm_config is provided
LLM_MAX_TOKENS_DEFAULT = 16000  # used when we synthesize a TraceKGLLMConfig


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
# LLM Prompt (unchanged semantics)
# -----------------------------------------------------------------------------

from TKG_Prompts import REL_REC_PROMPT_TEMPLATE
REL_REC_INSTRUCTIONS = REL_REC_PROMPT_TEMPLATE


# -----------------------------------------------------------------------------
# DSPy / LLM helpers
# -----------------------------------------------------------------------------

def _get_lm_for_rel_rec(
    llm_config: Optional["TraceKGLLMConfig"],
    model: str,
    max_tokens: int,
):
    """
    Resolve a DSPy LM for the Relation Recognition step.

    - If llm_config is provided, we use it directly (per-step override: 'rel_rec').
    - Otherwise, we synthesize a minimal TraceKGLLMConfig using (model, max_tokens)
      so that existing code can still call run_rel_rec(..., model="...").
    """
    if llm_config is not None:
        cfg = llm_config
    else:
        cfg = TraceKGLLMConfig(default_model=model, max_tokens=max_tokens)

    try:
        cfg.validate()
    except Exception as e:
        logger.warning("[RelRec] llm_config.validate() failed: %s", e)

    lm = make_lm_for_step(cfg, "rel_rec")
    return lm


def _call_rel_rec_lm(lm, prompt: str) -> str:
    """
    Call the DSPy LM and normalize the output into a plain string.
    """
    try:
        outputs = lm(prompt)
    except Exception as e:
        logger.error("[RelRec] LM call error: %s", e)
        return ""

    # dspy.LM may return a string or list[str]; be tolerant.
    if isinstance(outputs, list):
        return outputs[0] if outputs else ""
    return str(outputs or "")


def call_lm_extract_relations_for_chunk(
    lm: Any,
    chunk: Dict[str, Any],
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Call the LLM (via DSPy LM) to extract relations for a single chunk.

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

    # Single prompt string = instructions + JSON payload
    prompt = (
        REL_REC_INSTRUCTIONS
        + "\n\n---\n\n"
        + "Below is a JSON object describing the FOCUS CHUNK and its resolved entities.\n"
        + "Follow ALL instructions above and return ONLY a single JSON object with key 'relations'.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    raw_text = _call_rel_rec_lm(lm, prompt)
    if not raw_text:
        logger.warning("Empty response for chunk %s", chunk["id"])
        return []

    raw_text = coerce_llm_text(raw_text).strip() #Injam
    
    try:
        parsed = json.loads(raw_text)

    except json.JSONDecodeError:
        logger.error(
            "Failed to parse JSON for chunk %s. Raw response (truncated):\n%s",
            chunk["id"],
            raw_text[:2000],
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
    max_tokens: int = LLM_MAX_TOKENS_DEFAULT,
    llm_config: Optional["TraceKGLLMConfig"] = None,
) -> None:
    """
    Full Relation Recognition pipeline:

    - load entities_by_chunk, entities_by_id
    - create a single DSPy LM for rel_rec (shared across all chunks)
    - iterate over chunks
    - for each chunk, call LLM to extract relations
    - enrich relations with relation_id, chunk_id, subject/object class info
    - write to relations_raw.jsonl (streaming, flushed after each chunk)

    LLM behavior:
      * If llm_config is provided, we use it to create the LM via make_lm_for_step(cfg, "rel_rec").
      * If llm_config is None, we build a minimal TraceKGLLMConfig(default_model=model,
        max_tokens=max_tokens) so existing code that passes just `model` continues to work.
    """
    entities_by_chunk, entities_by_id = load_entities_by_chunk(entities_path)

    # Prepare output file
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Single LM instance for the whole run
    lm = _get_lm_for_rel_rec(llm_config=llm_config, model=model, max_tokens=max_tokens)

    n_chunks = 0
    n_chunks_called = 0
    n_relations = 0

    logger.info("Writing relations to %s", output_path)
    with open(out_path, "w", encoding="utf-8") as out_f:
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

            relations = call_lm_extract_relations_for_chunk(
                lm=lm,
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


# make output path directory if it doesn't exist
output_dir = "data/Relations/Rel Rec"
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# -----------------------
# Relation Rec - Run statement
# -----------------------

# Example direct run (without centralized llm_config):
# run_rel_rec(
#     entities_path="data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl",
#     chunks_path="data/Chunks/chunks_sentence.jsonl",
#     output_path="data/Relations/Rel Rec/relations_raw.jsonl",
#     model="gpt-5.1",
# )

# If you want to use a shared TraceKGLLMConfig:
#
# cfg = TraceKGLLMConfig(
#     default_model="gpt-5.1",
#     rec_model=None,
#     res_model=None,
#     rel_rec_model="gpt-5.1",  # optional override
# )
# run_rel_rec(
#     entities_path="...",
#     chunks_path="...",
#     output_path="...",
#     llm_config=cfg,
# )

#endregion#? Rel Rec V10 - DSPy LLM Config (Better Rel Naming)
#?#########################  End  ##########################





#*######################### Start ##########################
#region:#?   Rel Res V10  - DSPy LLM Config + Canonical + RelCls + RelClsGroup + Schema + LocalSubcluster

#!/usr/bin/env python3
"""
relres_iterative_v10_dspy.py

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
  data/Relations/Rel Rec/relations_raw.jsonl

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

LLM usage:
- All LLM calls go through a single DSPy LM obtained from TraceKGLLMConfig
  via make_lm_for_step(cfg, "rel_res").
- If no llm_config is provided, relres_main builds a minimal TraceKGLLMConfig
  using the `model` and `max_tokens` arguments, so existing code that just
  calls relres_main() (or passes a model string) still works.
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

# NOTE: We now use DSPy + TraceKGLLMConfig / make_lm_for_step for all LLM calls.
# These are defined in your LLM Model V3 region:
#   - class TraceKGLLMConfig
#   - def make_lm_for_step(cfg: TraceKGLLMConfig, step: str) -> dspy.LM
# We reference them here (they must be importable or defined in the same module).

# ----------------------------- CONFIG -----------------------------

INPUT_RELATIONS = Path("data/Relations/Rel Rec/relations_raw.jsonl")
OUT_DIR = Path("data/Relations/Rel Res")
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

# LLM (DSPy / central config)
MODEL_NAME = "gpt-5.1"
LLM_MAX_TOKENS_DEFAULT = 16000

VERBOSE = False

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
from TKG_Prompts import REL_RES_PROMPT_TEMPLATE
RELRES_PROMPT_TEMPLATE = REL_RES_PROMPT_TEMPLATE

def sanitize_json_array(text: str) -> Optional[Any]:
    """
    Extract and parse the first JSON array from the text.
    Grab [ ... ] block, fix simple trailing commas, and json.loads.
    """
    if not text or not text.strip():
        return None
    # s = text.strip() #Injam
    s = coerce_llm_text(text).strip()
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

# ---------------------- DSPy / LLM helpers -------------------------

def _get_lm_for_rel_res(
    llm_config: Optional["TraceKGLLMConfig"],
    model: str,
    max_tokens: int,
):
    """
    Resolve a DSPy LM for the Relation Resolution step.

    - If llm_config is provided, we use it directly (per-step override: 'rel_res').
    - Otherwise, we synthesize a minimal TraceKGLLMConfig using (model, max_tokens)
      so existing code can still call relres_main(model="...").
    """
    if llm_config is not None:
        cfg = llm_config
    else:
        cfg = TraceKGLLMConfig(default_model=model, max_tokens=max_tokens)

    try:
        cfg.validate()
    except Exception as e:
        if VERBOSE:
            print(f"[RelRes] llm_config.validate() failed: {e}")

    lm = make_lm_for_step(cfg, "rel_res")
    return lm

def _call_rel_res_lm(lm, prompt: str) -> str:
    """
    Call the DSPy LM and normalize the output into a plain string.
    """
    try:
        outputs = lm(prompt)
    except Exception as e:
        print("[RelRes] LM call error:", e)
        return ""

    if isinstance(outputs, list):
        return outputs[0] if outputs else ""
    return str(outputs or "")

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

def relres_main(
    model: str = MODEL_NAME,
    max_tokens: int = LLM_MAX_TOKENS_DEFAULT,
    llm_config: Optional["TraceKGLLMConfig"] = None,
    input_relations_path: Path = INPUT_RELATIONS,
) -> None:
    """
    Run Relation Resolution end-to-end.

    LLM behavior:
      * If llm_config is provided, it is used via make_lm_for_step(cfg, "rel_res").
      * If llm_config is None, a minimal TraceKGLLMConfig(default_model=model,
        max_tokens=max_tokens) is synthesized so existing code that calls
        relres_main() or relres_main(model="...") still works.
    """
    # load relations
    relations = load_relations(input_relations_path)
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

    # build LM once
    lm = _get_lm_for_rel_res(llm_config=llm_config, model=model, max_tokens=max_tokens)

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

        # call LLM via DSPy
        raw_out = ""
        try:
            raw_out = _call_rel_res_lm(lm, prompt)
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

#endregion#?   Rel Res V10  - DSPy LLM Config + Canonical + RelCls + RelClsGroup + Schema + LocalSubcluster
#*#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Rel Res V10  - DSPy LLM Config + Canonical + RelCls + RelClsGroup + Schema + LocalSubcluster (+ MergeRelations)

#!/usr/bin/env python3
"""
relres_iterative_v10_dspy.py

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
  Exception: exact-duplicate deduplication via merge_relations (LLM-driven) or strict auto-merge (post-pass).
- canonical_rel_name is what will be used as edge label in the KG.
- rel_cls / rel_cls_group give you a 2-layer schema for relations.
- Multi-run friendly: TBD fields can be filled in the first run, refined later.
- Uses global HDBSCAN + optional local subclustering + MAX_MEMBERS_PER_PROMPT
  so LLM chunks stay reasonably small.

Input:
  data/Relations/Rel Rec/relations_raw.jsonl

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

LLM usage:
- All LLM calls go through a single DSPy LM obtained from TraceKGLLMConfig
  via make_lm_for_step(cfg, "rel_res").
- If no llm_config is provided, relres_main builds a minimal TraceKGLLMConfig
  using the `model` and `max_tokens` arguments, so existing code that just
  calls relres_main() (or passes a model string) still works.
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

# ----------------------------- CONFIG -----------------------------

INPUT_RELATIONS = Path("data/Relations/Rel Rec/relations_raw.jsonl")
OUT_DIR = Path("data/Relations/Rel Res")
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

# LLM (DSPy / central config)
MODEL_NAME = "gpt-5.1"
LLM_MAX_TOKENS_DEFAULT = 16000

VERBOSE = False

# NEW: strict deterministic post-pass merge (safe; only merges exact duplicates)
AUTO_MERGE_EXACT_DUPLICATES = True

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

            # ensure qualifiers exists
            if "qualifiers" not in obj or obj.get("qualifiers") is None:
                obj["qualifiers"] = {}

            rels.append(obj)
    if VERBOSE:
        print(f"[start] loaded {len(rels)} relations from {path}")
    return rels

def safe_str(s: Optional[str]) -> str:
    if not s:
        return ""
    return str(s).replace("\n", " ").strip()

def _dedupe_preserve_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _norm_qualifiers(q: Any) -> Dict[str, str]:
    """
    Normalize qualifiers dict into only non-empty string values.
    """
    if not isinstance(q, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in q.items():
        if v is None:
            continue
        sv = str(v).strip()
        if not sv or sv.lower() == "null":
            continue
        out[str(k)] = sv
    return out

def _qualifiers_subset(a: Dict[str, str], b: Dict[str, str]) -> bool:
    """
    True if a is subset of b (all keys in a exist in b with same value).
    """
    for k, v in a.items():
        if k not in b or b[k] != v:
            return False
    return True

def _merge_qualifiers_safe(q_list: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Attempt a safe merge of qualifiers under strict rules:
    - If all normalized qualifiers equal => merge OK (return that)
    - If some are empty and others non-empty => merge OK (return the most informative)
    - If conflicting non-empty values exist => merge NOT OK
    Returns: (merged_qualifiers_or_none, ok_flag)
    """
    if not q_list:
        return {}, True
    normed = [_norm_qualifiers(q) for q in q_list]
    # pick the most informative as base
    base = max(normed, key=lambda d: len(d))
    for d in normed:
        if _qualifiers_subset(d, base):
            continue
        if _qualifiers_subset(base, d):
            base = d
            continue
        # conflict
        return None, False
    return base, True

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
from TKG_Prompts import REL_RES_PROMPT_TEMPLATE
RELRES_PROMPT_TEMPLATE = REL_RES_PROMPT_TEMPLATE

def sanitize_json_array(text: str) -> Optional[Any]:
    """
    Extract and parse the first JSON array from the text.
    Grab [ ... ] block, fix simple trailing commas, and json.loads.
    """
    if not text or not text.strip():
        return None
    s = coerce_llm_text(text).strip()
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

# ---------------------- DSPy / LLM helpers -------------------------

def _get_lm_for_rel_res(
    llm_config: Optional["TraceKGLLMConfig"],
    model: str,
    max_tokens: int,
):
    """
    Resolve a DSPy LM for the Relation Resolution step.

    - If llm_config is provided, we use it directly (per-step override: 'rel_res').
    - Otherwise, we synthesize a minimal TraceKGLLMConfig using (model, max_tokens)
      so existing code can still call relres_main(model="...").
    """
    if llm_config is not None:
        cfg = llm_config
    else:
        cfg = TraceKGLLMConfig(default_model=model, max_tokens=max_tokens)

    try:
        cfg.validate()
    except Exception as e:
        if VERBOSE:
            print(f"[RelRes] llm_config.validate() failed: {e}")

    lm = make_lm_for_step(cfg, "rel_res")
    return lm

def _call_rel_res_lm(lm, prompt: str) -> str:
    """
    Call the DSPy LM and normalize the output into a plain string.
    """
    try:
        outputs = lm(prompt)
    except Exception as e:
        print("[RelRes] LM call error:", e)
        return ""

    if isinstance(outputs, list):
        return outputs[0] if outputs else ""
    return str(outputs or "")

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

def execute_merge_relations(
    rel_by_id: Dict[str, Dict[str, Any]],
    relation_ids: List[str],
    provisional_id: str,
    subject_entity_id: str,
    object_entity_id: str,
    canonical_rel_name: str,
    canonical_rel_desc: Optional[str],
    new_rel_cls: Optional[str],
    new_rel_cls_group: Optional[str],
    relation_name: Optional[str],
    rel_desc: Optional[str],
    rel_hint_type: Optional[str],
    subject_entity_name: Optional[str],
    object_entity_name: Optional[str],
    qualifiers: Optional[Dict[str, Any]],
    remark: Optional[str],
    confidence: Optional[float],
) -> str:
    """
    Merge two or more EXACT duplicate relations into one new relation instance.
    Strict validation:
      - all relation_ids exist
      - all share identical subject_entity_id and object_entity_id
      - canonical_rel_name provided (direction+meaning) and must match (or be set) consistently
    """
    # validate ids exist
    relation_ids = [rid for rid in relation_ids if rid in rel_by_id]
    if len(relation_ids) < 2:
        raise ValueError("merge_relations: need at least 2 valid relation_ids")

    # validate head/tail identical
    for rid in relation_ids:
        r = rel_by_id[rid]
        if str(r.get("subject_entity_id", "")).strip() != str(subject_entity_id).strip():
            raise ValueError("merge_relations: subject_entity_id mismatch across relation_ids")
        if str(r.get("object_entity_id", "")).strip() != str(object_entity_id).strip():
            raise ValueError("merge_relations: object_entity_id mismatch across relation_ids")

    # validate / select canonical
    canon = str(canonical_rel_name or "").strip()
    if not canon:
        raise ValueError("merge_relations: canonical_rel_name is required")

    # qualifiers: use provided if present else safe-merge
    if qualifiers is None:
        q_list = [rel_by_id[rid].get("qualifiers", {}) for rid in relation_ids]
        merged_q, ok = _merge_qualifiers_safe(q_list)
        if not ok:
            raise ValueError("merge_relations: qualifiers conflict; refuse to merge without explicit merged qualifiers")
        qualifiers = merged_q or {}

    # choose base relation (first) for inherited fields
    base = rel_by_id[relation_ids[0]]

    # merge remarks
    merged_remarks: List[str] = []
    for rid in relation_ids:
        rr = rel_by_id[rid]
        rs = rr.get("remarks", [])
        if isinstance(rs, list):
            merged_remarks.extend([str(x) for x in rs if x])
        elif isinstance(rs, str) and rs.strip():
            merged_remarks.append(rs.strip())
    if remark and str(remark).strip():
        merged_remarks.append(str(remark).strip())
    merged_remarks = _dedupe_preserve_order([r for r in merged_remarks if r])

    # merge evidence excerpts (keep both a legacy single field and a list for traceability)
    ev_list: List[str] = []
    for rid in relation_ids:
        rr = rel_by_id[rid]
        ev = rr.get("evidence_excerpt") or rr.get("evidence_excerpts")
        if isinstance(ev, str) and ev.strip():
            ev_list.append(ev.strip())
        elif isinstance(ev, list):
            ev_list.extend([str(x).strip() for x in ev if str(x).strip()])
    ev_list = _dedupe_preserve_order(ev_list)

    # confidence
    if confidence is None:
        confidence = max(float(rel_by_id[rid].get("confidence", 0.0)) for rid in relation_ids)

    new_rid = "RelR_" + uuid.uuid4().hex[:8]

    merged_obj = dict(base)  # start from base to preserve unknown fields safely
    merged_obj["relation_id"] = new_rid
    merged_obj["provisional_id"] = provisional_id
    merged_obj["merged_from"] = list(relation_ids)
    merged_obj["_merged_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # set key schema + identity fields
    merged_obj["subject_entity_id"] = str(subject_entity_id)
    merged_obj["object_entity_id"] = str(object_entity_id)

    if subject_entity_name is not None and str(subject_entity_name).strip():
        merged_obj["subject_entity_name"] = str(subject_entity_name).strip()
    if object_entity_name is not None and str(object_entity_name).strip():
        merged_obj["object_entity_name"] = str(object_entity_name).strip()

    merged_obj["canonical_rel_name"] = canon
    if canonical_rel_desc is not None:
        merged_obj["canonical_rel_desc"] = str(canonical_rel_desc).strip()

    if new_rel_cls is not None and str(new_rel_cls).strip():
        merged_obj["rel_cls"] = str(new_rel_cls).strip()
    if new_rel_cls_group is not None and str(new_rel_cls_group).strip():
        merged_obj["rel_cls_group"] = str(new_rel_cls_group).strip()

    if relation_name is not None and str(relation_name).strip():
        merged_obj["relation_name"] = str(relation_name).strip()
    if rel_desc is not None and str(rel_desc).strip():
        merged_obj["rel_desc"] = str(rel_desc).strip()
    if rel_hint_type is not None and str(rel_hint_type).strip():
        merged_obj["rel_hint_type"] = str(rel_hint_type).strip()

    merged_obj["qualifiers"] = qualifiers or {}
    merged_obj["confidence"] = float(confidence or 0.0)
    merged_obj["remarks"] = merged_remarks

    # evidence fields
    if ev_list:
        merged_obj["evidence_excerpt"] = ev_list[0]
        merged_obj["evidence_excerpts"] = ev_list

    # remove old relations
    for rid in relation_ids:
        rel_by_id.pop(rid, None)

    # insert merged relation
    rel_by_id[new_rid] = merged_obj
    return new_rid

def auto_merge_exact_duplicates(rel_by_id: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Strict deterministic dedup:
    - Merge ONLY when (subject_entity_id, object_entity_id, canonical_rel_name) match
    - Also require rel_cls and rel_cls_group to match (or be both TBD/empty)
    - Qualifiers must be compatible under strict rules (subset/identical), else do not merge
    Returns: (new_rel_by_id, merged_count)
    """
    rels = list(rel_by_id.values())
    buckets: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for r in rels:
        subj = safe_str(r.get("subject_entity_id", ""))
        obj = safe_str(r.get("object_entity_id", ""))
        canon = safe_str(r.get("canonical_rel_name", ""))
        if not subj or not obj or not canon or canon.upper() == "TBD":
            continue
        cls = safe_str(r.get("rel_cls", ""))
        grp = safe_str(r.get("rel_cls_group", ""))
        key = (subj, obj, canon, cls, grp)
        buckets.setdefault(key, []).append(r)

    merged_count = 0
    new_rel_by_id = dict(rel_by_id)

    for key, group in buckets.items():
        if len(group) < 2:
            continue

        # find a compatible merge set around the most informative qualifiers
        # (this merges empties into informative, but rejects conflicts)
        # pick base candidate with max non-empty qualifiers
        group_sorted = sorted(group, key=lambda r: len(_norm_qualifiers(r.get("qualifiers", {}))), reverse=True)
        base = group_sorted[0]
        base_q = _norm_qualifiers(base.get("qualifiers", {}))

        to_merge = [base]
        for r in group_sorted[1:]:
            rq = _norm_qualifiers(r.get("qualifiers", {}))
            if _qualifiers_subset(rq, base_q) or _qualifiers_subset(base_q, rq):
                # update base if r is more informative but compatible
                if _qualifiers_subset(base_q, rq):
                    base_q = rq
                    base = r
                to_merge.append(r)

        if len(to_merge) < 2:
            continue

        # ensure all to_merge still exist (may have been merged already)
        to_merge_ids = [r["relation_id"] for r in to_merge if r.get("relation_id") in new_rel_by_id]
        if len(to_merge_ids) < 2:
            continue

        # compute merged qualifiers safely
        q_list = [new_rel_by_id[rid].get("qualifiers", {}) for rid in to_merge_ids]
        merged_q, ok = _merge_qualifiers_safe(q_list)
        if not ok:
            continue  # strict: skip if conflict

        subj, obj, canon, cls, grp = key
        prov = "MERGE(" + "|".join(to_merge_ids) + ")"

        new_id = execute_merge_relations(
            new_rel_by_id,
            to_merge_ids,
            provisional_id=prov,
            subject_entity_id=subj,
            object_entity_id=obj,
            canonical_rel_name=canon,
            canonical_rel_desc=new_rel_by_id[to_merge_ids[0]].get("canonical_rel_desc", ""),
            new_rel_cls=cls if cls else None,
            new_rel_cls_group=grp if grp else None,
            relation_name=None,
            rel_desc=None,
            rel_hint_type=None,
            subject_entity_name=new_rel_by_id[to_merge_ids[0]].get("subject_entity_name"),
            object_entity_name=new_rel_by_id[to_merge_ids[0]].get("object_entity_name"),
            qualifiers=merged_q or {},
            remark="auto_merge_exact_duplicates: merged exact duplicate edges",
            confidence=None,
        )
        merged_count += (len(to_merge_ids) - 1)

    return new_rel_by_id, merged_count

# ---------------------- Main orchestration -------------------------

def relres_main(
    model: str = MODEL_NAME,
    max_tokens: int = LLM_MAX_TOKENS_DEFAULT,
    llm_config: Optional["TraceKGLLMConfig"] = None,
    input_relations_path: Path = INPUT_RELATIONS,
) -> None:
    """
    Run Relation Resolution end-to-end.

    LLM behavior:
      * If llm_config is provided, it is used via make_lm_for_step(cfg, "rel_res").
      * If llm_config is None, a minimal TraceKGLLMConfig(default_model=model,
        max_tokens=max_tokens) is synthesized so existing code that calls
        relres_main() or relres_main(model="...") still works.
    """
    # load relations
    relations = load_relations(input_relations_path)
    rel_by_id: Dict[str, Dict[str, Any]] = {r["relation_id"]: r for r in relations}

    # NEW: provisional_id mapping + alias mapping (for merges)
    provisional_to_real: Dict[str, str] = {}   # provisional_id -> real relation_id
    alias_map: Dict[str, str] = {}             # old_relation_id -> new_relation_id

    def resolve_relation_ref(rid_or_prov: str) -> str:
        """
        Resolve a reference that might be:
          - a real relation_id
          - a provisional_id from merge_relations
          - an old id that got merged (alias_map)
        """
        if rid_or_prov in provisional_to_real:
            rid_or_prov = provisional_to_real[rid_or_prov]
        # follow alias chain
        seen = set()
        cur = rid_or_prov
        while cur in alias_map and alias_map[cur] != cur and cur not in seen:
            seen.add(cur)
            cur = alias_map[cur]
            if cur in provisional_to_real:
                cur = provisional_to_real[cur]
        return cur

    def resolve_many(ids: List[str]) -> List[str]:
        resolved = [resolve_relation_ref(str(x)) for x in (ids or [])]
        # keep only those that still exist
        resolved = [rid for rid in resolved if rid in rel_by_id]
        return _dedupe_preserve_order(resolved)

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

    # build LM once
    lm = _get_lm_for_rel_res(llm_config=llm_config, model=model, max_tokens=max_tokens)

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
                "subject_entity_id": safe_str(r.get("subject_entity_id", "")),
                "object_entity_id": safe_str(r.get("object_entity_id", "")),
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

        # call LLM via DSPy
        raw_out = ""
        try:
            raw_out = _call_rel_res_lm(lm, prompt)
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
                    rel_ids_valid = resolve_many([str(x) for x in rel_ids_raw])
                    canon_name = args.get("canonical_rel_name")
                    canon_desc = args.get("canonical_rel_desc")

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
                    rel_ids_valid = resolve_many([str(x) for x in rel_ids_raw])
                    rel_cls = args.get("rel_cls")

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
                    rel_ids_valid = resolve_many([str(x) for x in rel_ids_raw])
                    rel_cls_group = args.get("rel_cls_group")

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
                    rel_ids_valid = resolve_many([str(x) for x in rel_ids_raw])

                    canon_name = args.get("canonical_rel_name")
                    canon_desc = args.get("canonical_rel_desc")
                    rel_cls = args.get("rel_cls")
                    rel_cls_group = args.get("rel_cls_group")
                    new_rel_name = args.get("new_relation_name")
                    orig_rel_name = args.get("original_relation_name")

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
                    rel_ids_valid = resolve_many([str(x) for x in rel_ids_raw])
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

                elif fn == "merge_relations":
                    rel_ids_raw = args.get("relation_ids", []) or []
                    rel_ids_valid = resolve_many([str(x) for x in rel_ids_raw])

                    prov = args.get("provisional_id") or ("MERGE(" + "|".join(rel_ids_raw) + ")")
                    subj_id = safe_str(args.get("subject_entity_id", ""))
                    obj_id = safe_str(args.get("object_entity_id", ""))
                    canon_name = safe_str(args.get("canonical_rel_name", ""))
                    canon_desc = args.get("canonical_rel_desc")

                    # accept either new_rel_cls/new_rel_cls_group or rel_cls/rel_cls_group
                    new_rel_cls = args.get("new_rel_cls", None)
                    new_rel_cls_group = args.get("new_rel_cls_group", None)
                    if new_rel_cls is None:
                        new_rel_cls = args.get("rel_cls", None)
                    if new_rel_cls_group is None:
                        new_rel_cls_group = args.get("rel_cls_group", None)

                    rel_name = args.get("relation_name", None)
                    rel_desc = args.get("rel_desc", None)
                    rel_hint = args.get("rel_hint_type", None)
                    subj_name = args.get("subject_entity_name", None)
                    obj_name = args.get("object_entity_name", None)
                    quals = args.get("qualifiers", None)
                    remark = args.get("remark", None)
                    conf = args.get("confidence", None)

                    if len(rel_ids_valid) < 2 or not subj_id or not obj_id or not canon_name:
                        decisions.append({
                            "action": "merge_relations_skip_invalid_args",
                            "requested_relation_ids": rel_ids_raw,
                            "resolved_relation_ids": rel_ids_valid,
                            "provisional_id": prov,
                            "subject_entity_id": subj_id,
                            "object_entity_id": obj_id,
                            "canonical_rel_name": canon_name,
                            "justification": justification,
                            "remark": remark_val,
                            "confidence": confidence_val
                        })
                        continue

                    # strict: ensure canonical_rel_name identical across the to-be-merged items (if already filled)
                    # (We still allow merging TBD into canon_name, but do not allow disagreement.)
                    mismatch = False
                    for rid in rel_ids_valid:
                        existing_c = safe_str(rel_by_id[rid].get("canonical_rel_name", ""))
                        if existing_c and existing_c.upper() != "TBD" and existing_c != canon_name:
                            mismatch = True
                            break
                    if mismatch:
                        decisions.append({
                            "action": "merge_relations_skip_canonical_mismatch",
                            "relation_ids": rel_ids_valid,
                            "canonical_rel_name": canon_name,
                            "justification": justification,
                            "remark": "Refused: canonical_rel_name mismatch across candidates",
                            "confidence": confidence_val
                        })
                        continue

                    new_id = execute_merge_relations(
                        rel_by_id,
                        rel_ids_valid,
                        provisional_id=str(prov),
                        subject_entity_id=subj_id,
                        object_entity_id=obj_id,
                        canonical_rel_name=canon_name,
                        canonical_rel_desc=canon_desc,
                        new_rel_cls=new_rel_cls,
                        new_rel_cls_group=new_rel_cls_group,
                        relation_name=rel_name,
                        rel_desc=rel_desc,
                        rel_hint_type=rel_hint,
                        subject_entity_name=subj_name,
                        object_entity_name=obj_name,
                        qualifiers=quals,
                        remark=remark,
                        confidence=conf,
                    )

                    # record alias + provisional mapping
                    provisional_to_real[str(prov)] = new_id
                    alias_map[str(prov)] = new_id
                    for old in rel_ids_valid:
                        alias_map[old] = new_id

                    decisions.append({
                        "action": "merge_relations",
                        "merged_relation_id": new_id,
                        "provisional_id": prov,
                        "merged_from": rel_ids_valid,
                        "subject_entity_id": subj_id,
                        "object_entity_id": obj_id,
                        "canonical_rel_name": canon_name,
                        "justification": justification,
                        "remark": remark,
                        "confidence": conf
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

    # NEW: strict deterministic post-pass to remove exact duplicates safely
    if AUTO_MERGE_EXACT_DUPLICATES:
        before = len(rel_by_id)
        rel_by_id, merged_n = auto_merge_exact_duplicates(rel_by_id)
        after = len(rel_by_id)
        if merged_n > 0:
            print(f"[dedup] auto_merge_exact_duplicates merged {merged_n} duplicate edges; relations {before} -> {after}")
        else:
            print(f"[dedup] auto_merge_exact_duplicates: no merges; relations {before} -> {after}")

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

#endregion#?   Rel Res V10  - DSPy LLM Config + Canonical + RelCls + RelClsGroup + Schema + LocalSubcluster (+ MergeRelations)
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
BASE_INPUT_RELATIONS = Path("data/Relations/Rel Rec/relations_raw.jsonl")

# Root for iterative runs; each run gets its own subfolder
EXPERIMENT_ROOT = Path("data/Relations/Rel Res_IterativeRuns")

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



#?######################### Start ##########################
#region:#?   CSV relations + nodes for Neo4j KG import - V7


def export_relations_and_nodes_to_csv():
    import json, csv
    from pathlib import Path

    # Sources
    relations_jl = Path(
        "data/Relations/Rel Res_IterativeRuns/overall_summary/relations_resolved.jsonl"
    )

    # Primary source for entities_with_class: ClassRes multi-run
    entities_cls_primary = Path(
        "data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl"
    )
    # Fallback source: RelRes multi-run (your logs show this one exists)
    entities_cls_fallback = Path(
        "data/Relations/Rel Res_IterativeRuns/overall_summary/entities_with_class.jsonl"
    )

    if entities_cls_primary.exists():
        entities_cls_jl = entities_cls_primary
    elif entities_cls_fallback.exists():
        entities_cls_jl = entities_cls_fallback
    else:
        # Neither file exists -> fail with clear message
        raise FileNotFoundError(
            f"entities_with_class.jsonl not found at "
            f"{entities_cls_primary} or {entities_cls_fallback}"
        )

    # Outputs
    rels_out_csv  = Path("data/KG/rels_fixed_no_raw.csv")
    nodes_out_csv = Path("data/KG/nodes.csv")

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

    print(f"Loaded {len(entities)} entities from {entities_cls_jl}.")

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






#endregion#! KG Assembly
#!#############################################  End Chapter  ##################################################









#?######################### Start ##########################
#region:#?   Run statements



# # -----------------------
# # Chunking - Run statement
# # -----------------------

# if __name__ == "__main__":
#     sentence_chunks_token_driven(
#         "data/pdf_to_json/Plain_Text.json",
#         "data/Chunks/chunks_sentence.jsonl",
#         max_tokens_per_chunk=200,   # preferred upper bound (None to disable)
#         min_tokens_per_chunk=100,   # expand small chunks to reach this minimum (None to disable)
#         sentence_per_line=True,
#         keep_ref_text=False,
#         strip_leading_headings=True,
#         force=True,
#         debug=False
#     )


# # -----------------------
# # embed_and_index_chunks  - Run statement
# # -----------------------

# if __name__ == "__main__":
#     embed_and_index_chunks(
#         "data/Chunks/chunks_sentence.jsonl",
#         "data/Chunks/chunks_emb",
#         "BAAI/bge-large-en-v1.5",
#         "BAAI/bge-small-en-v1.5",
#         False,   # use_small_model_for_dev
#         32,     # batch_size
#         None,   # device -> auto
#         True,   # save_index
#         True)  # force



# # -----------------------
# # Entity Recognition  - Run statement
# # -----------------------

# if __name__ == "__main__":
#     run_entity_extraction_on_chunks(
#         chunk_ids,
#         prev_chunks=5,
#         save_debug=False,
#         model="gpt-5.1",
#         max_tokens=8000
#     )





# # -----------------------
# # Ent Resolution (Multi Run)  - Run statement
# # -----------------------

# if __name__ == "__main__":
#     iterative_resolution()






# # -----------------------
# # Cls Rec input producer - Run statement
# # -----------------------

# if __name__ == "__main__":
#     produce_clean_jsonl(input_path, out_file)




# # -----------------------
# # Cls Recognition  - Run statement
# # -----------------------



# if __name__ == "__main__":
#     classrec_iterative_main()



# # -----------------------
# # Create input for Cls Res  - Run statement
# # -----------------------

# if __name__ == "__main__":
#     main_input_for_cls_res()





# # -----------------------
# # Cls Res Multi Run - Run statement
# # -----------------------
# if __name__ == "__main__":
#     run_pipeline_iteratively() 









# # # -----------------------
# # Relation Res Multi Run - Run statement
# # -----------------------
# if __name__ == "__main__":
#     run_relres_iteratively() 



# # -----------------------
# # Export KG to CSVs  - Run statement
# # -----------------------

# if __name__ == "__main__":
#     export_relations_and_nodes_to_csv()




# # -----------------------
# # XXXXXXXX  - Run statement
# # -----------------------




#endregion#? Run statements
#?#########################  End  ##########################














