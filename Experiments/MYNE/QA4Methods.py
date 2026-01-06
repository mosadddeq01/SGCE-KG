


#!############################################# Start Chapter ##################################################
#region:#! Evaluation

  

#endregion#! Evaluation
#!############################################# End Chapter ##################################################












#!############################################# Start Chapter ##################################################
#region:#!   Experiments






#?######################### Start ##########################
#region:#?   QA4Methods - V5   (TRACE names for context, weighted embeddings)







import os
import json
from dataclasses import dataclass
from pathlib import Path
import pwd
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import networkx as nx

from datasets import load_dataset  # not strictly required if you only use JSON dump
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI


# ============================================================
# 0. Global config: weights, models, env
# ============================================================

# Entity weights (must sum to 1 after normalization)
ENT_WEIGHTS = {
    "name": 0.40,   # entity_name
    "desc": 0.25,   # entity_description
    "ctx": 0.35,    # class_label + class_group + node_properties
}

# Relation weights (before normalization)
REL_EMB_WEIGHTS = {
    "name": 0.25,      # relation_name
    "desc+Q": 0.15,    # rel_desc + qualifiers
    "head_tail": 0.20, # subject/object names + class info
    "ctx": 0.40,       # canonical_rel_name + canonical_rel_desc + rel_cls
}

ENT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
REL_EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # same for simplicity
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OpenAI config
OPENAI_MODEL_JUDGE = "gpt-5.1"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


# ============================================================
# 1. Env helper for OpenAI key
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
# 2. HF Embedder (generic)
# ============================================================

def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    masked = token_embeds * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class HFEmbedder:
    def __init__(self, model_name: str, device: str = DEVICE):
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
# 3. SimpleGraph for KG-Gen-style KGs (dataset KGs)
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
# 4. TRACE-KG loaders and weighted embedding builders
# ============================================================

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def build_tracekg_entity_texts(nodes_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return lists: ids, name_texts, desc_texts, ctx_texts."""
    ids, names, descs, ctxs = [], [], [], []
    for _, row in nodes_df.iterrows():
        ent_id = safe_str(row["entity_id"])
        ids.append(ent_id)

        # name ~ entity_name
        name_txt = safe_str(row.get("entity_name", ""))

        # desc ~ entity_description
        desc_txt = safe_str(row.get("entity_description", ""))

        # ctx ~ class_label + class_group + node_properties
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

    print("[TRACE-ENT] encoding name field ...")
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None

    print("[TRACE-ENT] encoding desc field ...")
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None

    print("[TRACE-ENT] encoding ctx field ...")
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
        if arr.shape[1] != D_ref:
            raise ValueError("Entity embedding dimension mismatch.")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx = _ensure(emb_ctx)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0:
        raise ValueError("Sum of ENT_WEIGHTS must be > 0")
    w_name /= Wsum
    w_desc /= Wsum
    w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)

    node_embs: Dict[str, np.ndarray] = {}
    for i, node_id in enumerate(ids):
        node_embs[node_id] = combined[i]
    return node_embs, D_ref


def build_tracekg_relation_texts(
    rels_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, int], Dict[int, Dict[str, str]]]:
    """
    Build relation text fields by bucket:
      - name: relation name
      - desc+Q: rel_desc + qualifiers
      - head_tail: subject/object names + class info
      - ctx: canonical_rel_name + canonical_rel_desc + rel_cls
    """
    # Node helper map
    node_info = {}
    for _, nrow in nodes_df.iterrows():
        nid = safe_str(nrow["entity_id"])
        node_info[nid] = {
            "name": safe_str(nrow.get("entity_name", "")),
            "class_label": safe_str(nrow.get("class_label", "")),
        }

    relation_ids: List[str] = []
    id_to_index: Dict[str, int] = {}
    texts: Dict[int, Dict[str, str]] = {}

    for i, row in rels_df.iterrows():
        rid = safe_str(row.get("relation_id", i))
        relation_ids.append(rid)
        id_to_index[rid] = i

        start_id = safe_str(row.get("start_id", ""))
        end_id = safe_str(row.get("end_id", ""))

        # name bucket
        rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))

        # desc+Q bucket
        rel_desc = safe_str(row.get("rel_desc", "")) or safe_str(row.get("canonical_rel_desc", ""))
        qualifiers = safe_str(row.get("qualifiers", ""))
        desc_plus_q = " ; ".join([p for p in [rel_desc, qualifiers] if p])

        # head_tail bucket
        head_info = node_info.get(start_id, {})
        tail_info = node_info.get(end_id, {})
        head_tail_parts = []
        if head_info.get("name"):
            head_tail_parts.append(f"[H:{head_info['name']}]")
        if head_info.get("class_label"):
            head_tail_parts.append(f"[HCLS:{head_info['class_label']}]")
        if tail_info.get("name"):
            head_tail_parts.append(f"[T:{tail_info['name']}]")
        if tail_info.get("class_label"):
            head_tail_parts.append(f"[TCLS:{tail_info['class_label']}]")
        head_tail = " ".join(head_tail_parts)

        # ctx bucket
        canonical_name = safe_str(row.get("canonical_rel_name", ""))
        canonical_desc = safe_str(row.get("canonical_rel_desc", ""))
        rel_cls = safe_str(row.get("rel_cls", ""))
        ctx_parts = []
        if canonical_name:
            ctx_parts.append(canonical_name)
        if canonical_desc:
            ctx_parts.append(canonical_desc)
        if rel_cls:
            ctx_parts.append(f"[CLS:{rel_cls}]")
        ctx_txt = " ; ".join(ctx_parts)

        texts[i] = {
            "name": rel_name,
            "desc+Q": desc_plus_q,
            "head_tail": head_tail,
            "ctx": ctx_txt,
        }

    return relation_ids, id_to_index, texts


def compute_weighted_relation_embeddings(
    embedder: HFEmbedder,
    rels_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    weights: Dict[str, float] = REL_EMB_WEIGHTS,
) -> Tuple[Dict[str, np.ndarray], int]:
    rel_ids, id_to_index, texts = build_tracekg_relation_texts(rels_df, nodes_df)

    n = len(rel_ids)
    buckets = ["name", "desc+Q", "head_tail", "ctx"]
    bucket_texts = {b: [""] * n for b in buckets}
    for idx in range(n):
        for b in buckets:
            bucket_texts[b][idx] = texts[idx].get(b, "")

    emb_bucket = {}
    D_ref = None
    for b in buckets:
        has_any = any(t.strip() for t in bucket_texts[b])
        if not has_any:
            emb_bucket[b] = None
            continue
        print(f"[TRACE-REL] encoding bucket '{b}' ...")
        eb = embedder.encode_batch(bucket_texts[b])
        emb_bucket[b] = eb
        if D_ref is None:
            D_ref = eb.shape[1]

    if D_ref is None:
        raise ValueError("All relation buckets empty — cannot embed relations.")

    def _ensure(arr):
        if arr is None:
            return np.zeros((n, D_ref))
        if arr.shape[1] != D_ref:
            raise ValueError("Relation embedding dimension mismatch.")
        return arr

    for b in buckets:
        emb_bucket[b] = _ensure(emb_bucket[b])

    w_name = weights.get("name", 0.0)
    w_descq = weights.get("desc+Q", 0.0)
    w_ht = weights.get("head_tail", 0.0)
    w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_descq + w_ht + w_ctx
    if Wsum <= 0:
        raise ValueError("Sum of REL_EMB_WEIGHTS must be > 0")
    w_name /= Wsum
    w_descq /= Wsum
    w_ht /= Wsum
    w_ctx /= Wsum

    combined = (
        w_name * emb_bucket["name"]
        + w_descq * emb_bucket["desc+Q"]
        + w_ht * emb_bucket["head_tail"]
        + w_ctx * emb_bucket["ctx"]
    )
    combined = normalize(combined, axis=1)

    rel_embs: Dict[str, np.ndarray] = {}
    for i, rid in enumerate(rel_ids):
        rel_embs[rid] = combined[i]
    return rel_embs, D_ref


def build_tracekg_nx_and_nodeinfo(
    nodes_df: pd.DataFrame,
    rels_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, Dict[str, Dict[str, str]]]:
    """
    Build TRACE-KG graph and a node_info dict:
      node_id -> {"name": entity_name, "class_label": class_label}
    """
    g = nx.DiGraph()
    node_info: Dict[str, Dict[str, str]] = {}

    for _, row in nodes_df.iterrows():
        nid = safe_str(row["entity_id"])
        name = safe_str(row.get("entity_name", ""))
        cls_label = safe_str(row.get("class_label", ""))

        g.add_node(
            nid,
            entity_name=name,
            entity_description=safe_str(row.get("entity_description", "")),
            class_label=cls_label,
            class_group=safe_str(row.get("class_group", "")),
            node_properties=safe_str(row.get("node_properties", "")),
            chunk_ids=safe_str(row.get("chunk_ids", "")),
        )
        node_info[nid] = {
            "name": name,
            "class_label": cls_label,
        }

    for _, row in rels_df.iterrows():
        sid = safe_str(row.get("start_id", ""))
        eid = safe_str(row.get("end_id", ""))
        rid = safe_str(row.get("relation_id", ""))
        rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))
        qualifiers = safe_str(row.get("qualifiers", ""))
        g.add_edge(
            sid,
            eid,
            relation=rel_name,
            relation_id=rid,
            chunk_id=safe_str(row.get("chunk_id", "")),
            qualifiers=qualifiers,
        )
    return g, node_info


# ============================================================
# 5. Retrieval (weighted node embeddings + graph, with readable context)
# ============================================================

class WeightedGraphRetriever:
    def __init__(
        self,
        node_embeddings: Dict[str, np.ndarray],
        graph: nx.DiGraph,
        node_info: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        node_info (for TRACE-KG) maps:
           node_id -> {"name": ..., "class_label": ...}
        For dataset KGs, this can be None and we fall back to IDs in context.
        """
        self.node_embeddings = node_embeddings
        self.graph = graph
        self.node_info = node_info or {}

    def retrieve_relevant_nodes(
        self,
        query_emb: np.ndarray,
        k: int = 8,
    ) -> List[Tuple[str, float]]:
        sims = []
        for node, emb in self.node_embeddings.items():
            sim = cosine_similarity(query_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
            sims.append((node, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def _format_node_for_context(self, node_id: str) -> str:
        """
        For TRACE-KG, return 'Name (which is of type: ClassLabel)'.
        For others, just return node_id as string.
        """
        info = self.node_info.get(node_id)
        if info is None:
            return str(node_id)

        name = info.get("name") or str(node_id)
        cls = info.get("class_label") or ""
        if cls:
            return f"{name} (which is of type: {cls})"
        return name

    def _format_edge_for_context(self, src: str, dst: str, data: Dict) -> str:
        """
        For TRACE-KG, produce:
          subjectEntName (which is of type: entCleName) has relation
          {rel_canonical_name (with qualifiers:{})} with
          objectEntName (which is of type: entCleName).
        For others, fall back to 'src rel dst.'.
        """
        rel_name = data.get("relation", "")
        qualifiers = data.get("qualifiers", "")

        # Heuristic: if we have node_info, treat this as TRACE-KG style.
        if self.node_info:
            subj = self._format_node_for_context(src)
            obj = self._format_node_for_context(dst)
            if qualifiers:
                return (
                    f"{subj} has relation "
                    f"{{{rel_name} (with qualifiers: {qualifiers})}} "
                    f"with {obj}."
                )
            else:
                return (
                    f"{subj} has relation "
                    f"{{{rel_name}}} "
                    f"with {obj}."
                )
        else:
            # dataset KGs
            return f"{src} {rel_name} {dst}."

    def retrieve_context(
        self,
        node: str,
        depth: int = 2,
    ) -> List[str]:
        context: Set[str] = set()

        def explore(n: str, d: int):
            if d > depth:
                return
            for nbr in self.graph.neighbors(n):
                data = self.graph[n][nbr]
                text = self._format_edge_for_context(n, nbr, data)
                context.add(text)
                explore(nbr, d + 1)
            for nbr in self.graph.predecessors(n):
                data = self.graph[nbr][n]
                text = self._format_edge_for_context(nbr, n, data)
                context.add(text)
                explore(nbr, d + 1)

        explore(node, 1)
        return list(context)

    def retrieve(
        self,
        query_emb: np.ndarray,
        k: int = 8,
    ) -> Tuple[List[Tuple[str, float]], Set[str], str]:
        top_nodes = self.retrieve_relevant_nodes(query_emb, k=k)
        context: Set[str] = set()
        for node, _ in top_nodes:
            ctx = self.retrieve_context(node)
            context.update(ctx)
        context_text = " ".join(context)
        return top_nodes, context, context_text


# ============================================================
# 6. LLM-based evaluator (OpenAI API 5.1)
# ============================================================

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = _load_openai_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY env var or provide it "
            "in .env"
        )
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    """
    Use an OpenAI model as a binary judge.
    Returns 1 if the model believes the context contains the correct answer,
    otherwise 0.
    """
    client = _get_openai_client()

    system_prompt = (
        "You are an evaluation assistant. "
        "You are given a statement that is assumed to be the correct answer, "
        "and a retrieved context. "
        "Return '1' (without quotes) if the context clearly contains enough "
        "information to support that answer. Otherwise return '0'. "
        "Return only a single character: '1' or '0'."
    )

    user_prompt = (
        "Correct answer statement:\n"
        f"{correct_answer}\n\n"
        "Retrieved context from a knowledge graph:\n"
        f"{context}\n\n"
        "Does the retrieved context contain enough information to support "
        "the correctness of the answer statement? "
        "Respond strictly with '1' for yes or '0' for no."
    )

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL_JUDGE,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_output_tokens=64,  # >= 16
        )
        text = resp.output[0].content[0].text.strip()
    except Exception as e:
        print(f"[gpt_evaluate_response] Error calling OpenAI: {e}")
        return 0

    text = text.strip()
    if text == "1":
        return 1
    if text == "0":
        return 0

    # Fallback: heuristic if model output is weird
    ans_tokens = set(t.lower() for t in correct_answer.split() if len(t) > 3)
    if not ans_tokens:
        return 0
    ctx_lower = context.lower()
    for t in ans_tokens:
        if t in ctx_lower:
            return 1
    return 0


# ============================================================
# 7. Evaluation utilities
# ============================================================

def evaluate_accuracy_for_graph(
    query_embedder: HFEmbedder,
    retriever: WeightedGraphRetriever,
    queries: List[str],
    method_name: str,
    essay_idx: int,
    results_dir: str,
    k: int = 8,
    verbose: bool = False,
) -> Dict:
    os.makedirs(results_dir, exist_ok=True)

    print(f"[{method_name}] encoding {len(queries)} queries for essay {essay_idx} ...")
    query_embs = query_embedder.encode_batch(queries)

    correct = 0
    results = []

    for qi, q in enumerate(queries):
        q_emb = query_embs[qi]
        _, _, context_text = retriever.retrieve(q_emb, k=k)
        evaluation = gpt_evaluate_response(q, context_text)
        results.append(
            {
                "correct_answer": q,
                "retrieved_context": context_text,
                "evaluation": int(evaluation),
            }
        )
        correct += evaluation

    accuracy = correct / len(queries) if queries else 0.0
    results.append({"accuracy": f"{accuracy * 100:.2f}%"})

    out_path = os.path.join(results_dir, f"results_{essay_idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(
            f"[{method_name}] Essay {essay_idx}: "
            f"accuracy={accuracy:.4f} ({correct}/{len(queries)})"
        )

    return {
        "accuracy": accuracy,
        "num_queries": len(queries),
        "method": method_name,
        "essay_idx": essay_idx,
    }


def aggregate_method_stats(summaries: List[Dict]) -> Dict[str, float]:
    if not summaries:
        return {"mean_accuracy": 0.0, "num_essays": 0}
    accs = [s["accuracy"] for s in summaries]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "num_essays": len(accs),
    }


def compare_methods(all_summaries: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    return {m: aggregate_method_stats(s) for m, s in all_summaries.items()}


def print_comparison_table(comparison: Dict[str, Dict]):
    print("\n=== Method Comparison (Mean Accuracy) ===")
    print(f"{'Method':<10} | {'Mean Acc':>8} | {'#Essays':>7}")
    print("-" * 32)
    for m, stats in comparison.items():
        print(
            f"{m:<10} | {stats['mean_accuracy']*100:8.2f}% | "
            f"{stats['num_essays']:7d}"
        )


# ============================================================
# 8. Full evaluation over the dataset
# ============================================================

def run_full_evaluation(
    dataset_json_path: str,
    trace_nodes_csv: str,
    trace_rels_csv: str,
    output_root: str,
    methods: List[str],
    k: int = 8,
    max_essays: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    # Load dataset dump you already created
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_list = json.load(f)
    if max_essays is not None:
        dataset_list = dataset_list[:max_essays]

    # TRACE-KG embeddings & graph
    ent_embedder = HFEmbedder(ENT_EMBED_MODEL, DEVICE)
    rel_embedder = HFEmbedder(REL_EMBED_MODEL, DEVICE)  # relation embeddings precomputed but not used yet
    nodes_df = pd.read_csv(trace_nodes_csv)
    rels_df = pd.read_csv(trace_rels_csv)

    trace_node_embs, _ = compute_weighted_entity_embeddings(ent_embedder, nodes_df, ENT_WEIGHTS)
    trace_rel_embs, _ = compute_weighted_relation_embeddings(rel_embedder, rels_df, nodes_df, REL_EMB_WEIGHTS)
    trace_graph, trace_node_info = build_tracekg_nx_and_nodeinfo(nodes_df, rels_df)
    trace_retriever = WeightedGraphRetriever(trace_node_embs, trace_graph, node_info=trace_node_info)

    # Query embedder (same model as entity, for semantic match)
    query_embedder = ent_embedder

    all_summaries: Dict[str, List[Dict]] = {m: [] for m in methods}

    for idx, row in enumerate(dataset_list):
        essay_idx = idx
        queries: List[str] = row.get("generated_queries", [])
        if not queries:
            if verbose:
                print(f"Skipping essay {essay_idx}: no queries.")
            continue

        if verbose:
            print(f"\n=== Essay {essay_idx} | {len(queries)} queries ===")

        # TRACE-KG (one global graph)
        if "tracekg" in methods:
            summaries_dir = os.path.join(output_root, "tracekg")
            s = evaluate_accuracy_for_graph(
                query_embedder=query_embedder,
                retriever=trace_retriever,
                queries=queries,
                method_name="tracekg",
                essay_idx=essay_idx,
                results_dir=summaries_dir,
                k=k,
                verbose=verbose,
            )
            all_summaries["tracekg"].append(s)

        # Other methods: kggen, graphrag, openie
        for method in methods:
            if method == "tracekg":
                continue

            kg_key = None
            if method == "kggen":
                kg_key = "kggen"
            elif method == "graphrag":
                kg_key = "graphrag_kg"
            elif method == "openie":
                kg_key = "openie_kg"
            else:
                continue

            kg_data = row.get(kg_key, None)
            if kg_data is None:
                if verbose:
                    print(f"  [{method}] No KG data for essay {essay_idx}, skipping.")
                continue

            sg = SimpleGraph.from_kggen_dict(kg_data)
            g_nx = sg.to_nx()

            node_ids = list(g_nx.nodes())
            node_texts = [str(n) for n in node_ids]
            node_embs_arr = query_embedder.encode_batch(node_texts)
            node_embs = {nid: node_embs_arr[i] for i, nid in enumerate(node_ids)}
            retriever = WeightedGraphRetriever(node_embs, g_nx, node_info=None)

            summaries_dir = os.path.join(output_root, method)
            s = evaluate_accuracy_for_graph(
                query_embedder=query_embedder,
                retriever=retriever,
                queries=queries,
                method_name=method,
                essay_idx=essay_idx,
                results_dir=summaries_dir,
                k=k,
                verbose=verbose,
            )
            all_summaries[method].append(s)

    return all_summaries


# ============================================================
# 9. Main
# ============================================================

def main():
    dataset_json_path = (
        "Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset-short.json"
    )
    trace_nodes_csv = "Experiments/MYNE/TRACEKG/nodes.csv"
    trace_rels_csv = "Experiments/MYNE/TRACEKG/rels.csv"

    output_root = "./tracekg_mine_results_weighted_openai_v5"

    methods = ["kggen", "graphrag", "openie", "tracekg"]
    #methods = ["tracekg"]

    all_summaries = run_full_evaluation(
        dataset_json_path=dataset_json_path,
        trace_nodes_csv=trace_nodes_csv,
        trace_rels_csv=trace_rels_csv,
        output_root=output_root,
        methods=methods,
        k=8,
        max_essays=1,  # or a small number for debugging
        verbose=True,
    )

    comparison = compare_methods(all_summaries)
    print_comparison_table(comparison)


if __name__ == "__main__":
    main()
    
    



#endregion#? QA4Methods - V5   (TRACE names for context, weighted embeddings)
#?#########################  End  ##########################

# === Method Comparison (Mean Accuracy) ===
# Method | Mean Acc | #Essays
# --------------------------------
# kggen | 66.67% | 1
# graphrag | 40.00% | 1
# openie | 73.33% | 1
# tracekg | 93.33% | 1
# [tracekg] Essay 0: accuracy=0.9333 (14/15)
# [kggen] Essay 0: accuracy=0.6667 (10/15)
# [graphrag] Essay 0: accuracy=0.4000 (6/15)
# [openie] Essay 0: accuracy=0.7333 (11/15)




#?######################### Start ##########################
#region:#?   QA4Methods - V6   (TRACE names for context, weighted embeddings) - Wihout qualifiers in relation text


import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import networkx as nx

from datasets import load_dataset  # not strictly required if you only use JSON dump
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI


# ============================================================
# 0. Global config: weights, models, env
# ============================================================

# Entity weights (must sum to 1 after normalization)
ENT_WEIGHTS = {
    "name": 0.40,   # entity_name
    "desc": 0.25,   # entity_description
    "ctx": 0.35,    # class_label + class_group + node_properties
}

# Relation weights (before normalization)
REL_EMB_WEIGHTS = {
    "name": 0.25,      # relation_name
    "desc+Q": 0.15,    # rel_desc + qualifiers
    "head_tail": 0.20, # subject/object names + class info
    "ctx": 0.40,       # canonical_rel_name + canonical_rel_desc + rel_cls
}

ENT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
REL_EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # same for simplicity
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OpenAI config
OPENAI_MODEL_JUDGE = "gpt-5.1"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


# ============================================================
# 1. Env helper for OpenAI key
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
# 2. HF Embedder (generic)
# ============================================================

def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    masked = token_embeds * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class HFEmbedder:
    def __init__(self, model_name: str, device: str = DEVICE):
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
# 3. SimpleGraph for KG-Gen-style KGs (dataset KGs)
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
# 4. TRACE-KG loaders and weighted embedding builders
# ============================================================

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def build_tracekg_entity_texts(nodes_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return lists: ids, name_texts, desc_texts, ctx_texts."""
    ids, names, descs, ctxs = [], [], [], []
    for _, row in nodes_df.iterrows():
        ent_id = safe_str(row["entity_id"])
        ids.append(ent_id)

        # name ~ entity_name
        name_txt = safe_str(row.get("entity_name", ""))

        # desc ~ entity_description
        desc_txt = safe_str(row.get("entity_description", ""))

        # ctx ~ class_label + class_group + node_properties
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

    print("[TRACE-ENT] encoding name field ...")
    emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None

    print("[TRACE-ENT] encoding desc field ...")
    emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None

    print("[TRACE-ENT] encoding ctx field ...")
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
        if arr.shape[1] != D_ref:
            raise ValueError("Entity embedding dimension mismatch.")
        return arr

    emb_name = _ensure(emb_name)
    emb_desc = _ensure(emb_desc)
    emb_ctx = _ensure(emb_ctx)

    w_name = weights.get("name", 0.0)
    w_desc = weights.get("desc", 0.0)
    w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_desc + w_ctx
    if Wsum <= 0:
        raise ValueError("Sum of ENT_WEIGHTS must be > 0")
    w_name /= Wsum
    w_desc /= Wsum
    w_ctx /= Wsum

    combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
    combined = normalize(combined, axis=1)

    node_embs: Dict[str, np.ndarray] = {}
    for i, node_id in enumerate(ids):
        node_embs[node_id] = combined[i]
    return node_embs, D_ref


def build_tracekg_relation_texts(
    rels_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, int], Dict[int, Dict[str, str]]]:
    """
    Build relation text fields by bucket:
      - name: relation name
      - desc+Q: rel_desc + qualifiers
      - head_tail: subject/object names + class info
      - ctx: canonical_rel_name + canonical_rel_desc + rel_cls
    """
    # Node helper map
    node_info = {}
    for _, nrow in nodes_df.iterrows():
        nid = safe_str(nrow["entity_id"])
        node_info[nid] = {
            "name": safe_str(nrow.get("entity_name", "")),
            "class_label": safe_str(nrow.get("class_label", "")),
        }

    relation_ids: List[str] = []
    id_to_index: Dict[str, int] = {}
    texts: Dict[int, Dict[str, str]] = {}

    for i, row in rels_df.iterrows():
        rid = safe_str(row.get("relation_id", i))
        relation_ids.append(rid)
        id_to_index[rid] = i

        start_id = safe_str(row.get("start_id", ""))
        end_id = safe_str(row.get("end_id", ""))

        # name bucket
        rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))

        # desc+Q bucket
        rel_desc = safe_str(row.get("rel_desc", "")) or safe_str(row.get("canonical_rel_desc", ""))
        qualifiers = safe_str(row.get("qualifiers", ""))
        desc_plus_q = " ; ".join([p for p in [rel_desc, qualifiers] if p])

        # head_tail bucket
        head_info = node_info.get(start_id, {})
        tail_info = node_info.get(end_id, {})
        head_tail_parts = []
        if head_info.get("name"):
            head_tail_parts.append(f"[H:{head_info['name']}]")
        if head_info.get("class_label"):
            head_tail_parts.append(f"[HCLS:{head_info['class_label']}]")
        if tail_info.get("name"):
            head_tail_parts.append(f"[T:{tail_info['name']}]")
        if tail_info.get("class_label"):
            head_tail_parts.append(f"[TCLS:{tail_info['class_label']}]")
        head_tail = " ".join(head_tail_parts)

        # ctx bucket
        canonical_name = safe_str(row.get("canonical_rel_name", ""))
        canonical_desc = safe_str(row.get("canonical_rel_desc", ""))
        rel_cls = safe_str(row.get("rel_cls", ""))
        ctx_parts = []
        if canonical_name:
            ctx_parts.append(canonical_name)
        if canonical_desc:
            ctx_parts.append(canonical_desc)
        if rel_cls:
            ctx_parts.append(f"[CLS:{rel_cls}]")
        ctx_txt = " ; ".join(ctx_parts)

        texts[i] = {
            "name": rel_name,
            "desc+Q": desc_plus_q,
            "head_tail": head_tail,
            "ctx": ctx_txt,
        }

    return relation_ids, id_to_index, texts


def compute_weighted_relation_embeddings(
    embedder: HFEmbedder,
    rels_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    weights: Dict[str, float] = REL_EMB_WEIGHTS,
) -> Tuple[Dict[str, np.ndarray], int]:
    rel_ids, id_to_index, texts = build_tracekg_relation_texts(rels_df, nodes_df)

    n = len(rel_ids)
    buckets = ["name", "desc+Q", "head_tail", "ctx"]
    bucket_texts = {b: [""] * n for b in buckets}
    for idx in range(n):
        for b in buckets:
            bucket_texts[b][idx] = texts[idx].get(b, "")

    emb_bucket = {}
    D_ref = None
    for b in buckets:
        has_any = any(t.strip() for t in bucket_texts[b])
        if not has_any:
            emb_bucket[b] = None
            continue
        print(f"[TRACE-REL] encoding bucket '{b}' ...")
        eb = embedder.encode_batch(bucket_texts[b])
        emb_bucket[b] = eb
        if D_ref is None:
            D_ref = eb.shape[1]

    if D_ref is None:
        raise ValueError("All relation buckets empty — cannot embed relations.")

    def _ensure(arr):
        if arr is None:
            return np.zeros((n, D_ref))
        if arr.shape[1] != D_ref:
            raise ValueError("Relation embedding dimension mismatch.")
        return arr

    for b in buckets:
        emb_bucket[b] = _ensure(emb_bucket[b])

    w_name = weights.get("name", 0.0)
    w_descq = weights.get("desc+Q", 0.0)
    w_ht = weights.get("head_tail", 0.0)
    w_ctx = weights.get("ctx", 0.0)
    Wsum = w_name + w_descq + w_ht + w_ctx
    if Wsum <= 0:
        raise ValueError("Sum of REL_EMB_WEIGHTS must be > 0")
    w_name /= Wsum
    w_descq /= Wsum
    w_ht /= Wsum
    w_ctx /= Wsum

    combined = (
        w_name * emb_bucket["name"]
        + w_descq * emb_bucket["desc+Q"]
        + w_ht * emb_bucket["head_tail"]
        + w_ctx * emb_bucket["ctx"]
    )
    combined = normalize(combined, axis=1)

    rel_embs: Dict[str, np.ndarray] = {}
    for i, rid in enumerate(rel_ids):
        rel_embs[rid] = combined[i]
    return rel_embs, D_ref


def build_tracekg_nx_and_nodeinfo(
    nodes_df: pd.DataFrame,
    rels_df: pd.DataFrame,
) -> Tuple[nx.DiGraph, Dict[str, Dict[str, str]]]:
    """
    Build TRACE-KG graph and a node_info dict:
      node_id -> {"name": entity_name, "class_label": class_label}
    """
    g = nx.DiGraph()
    node_info: Dict[str, Dict[str, str]] = {}

    for _, row in nodes_df.iterrows():
        nid = safe_str(row["entity_id"])
        name = safe_str(row.get("entity_name", ""))
        cls_label = safe_str(row.get("class_label", ""))

        g.add_node(
            nid,
            entity_name=name,
            entity_description=safe_str(row.get("entity_description", "")),
            class_label=cls_label,
            class_group=safe_str(row.get("class_group", "")),
            node_properties=safe_str(row.get("node_properties", "")),
            chunk_ids=safe_str(row.get("chunk_ids", "")),
        )
        node_info[nid] = {
            "name": name,
            "class_label": cls_label,
        }

    for _, row in rels_df.iterrows():
        sid = safe_str(row.get("start_id", ""))
        eid = safe_str(row.get("end_id", ""))
        rid = safe_str(row.get("relation_id", ""))
        rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))
        qualifiers = safe_str(row.get("qualifiers", ""))
        g.add_edge(
            sid,
            eid,
            relation=rel_name,
            relation_id=rid,
            chunk_id=safe_str(row.get("chunk_id", "")),
            qualifiers=qualifiers,
        )
    return g, node_info


# ============================================================
# 5. Retrieval (weighted node embeddings + graph, with readable context)
# ============================================================

class WeightedGraphRetriever:
    def __init__(
        self,
        node_embeddings: Dict[str, np.ndarray],
        graph: nx.DiGraph,
        node_info: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        node_info (for TRACE-KG) maps:
           node_id -> {"name": ..., "class_label": ...}
        For dataset KGs, this can be None and we fall back to IDs in context.
        """
        self.node_embeddings = node_embeddings
        self.graph = graph
        self.node_info = node_info or {}

    def retrieve_relevant_nodes(
        self,
        query_emb: np.ndarray,
        k: int = 8,
    ) -> List[Tuple[str, float]]:
        sims = []
        for node, emb in self.node_embeddings.items():
            sim = cosine_similarity(query_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
            sims.append((node, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def _format_node_for_context(self, node_id: str) -> str:
        """
        For TRACE-KG, return 'Name (which is of type: ClassLabel)'.
        For others, just return node_id as string.
        """
        info = self.node_info.get(node_id)
        if info is None:
            return str(node_id)

        name = info.get("name") or str(node_id)
        cls = info.get("class_label") or ""
        if cls:
            return f"{name} (which is of type: {cls})"
        return name

    def _format_edge_for_context(self, src: str, dst: str, data: Dict) -> str:
        """
        For TRACE-KG, produce:
          subjectEntName (which is of type: entCleName) has relation
          {rel_canonical_name (with qualifiers:{})} with
          objectEntName (which is of type: entCleName).
        For others, fall back to 'src rel dst.'.
        """
        rel_name = data.get("relation", "")
        qualifiers = data.get("qualifiers", "")

        # Heuristic: if we have node_info, treat this as TRACE-KG style.
        if self.node_info:
            subj = self._format_node_for_context(src)
            obj = self._format_node_for_context(dst)
            if qualifiers:
                return (
                    f"{subj} has relation "
                    # f"{{{rel_name} (with qualifiers: {qualifiers})}} "
                    f"{{{rel_name}}} "
                    f"with {obj}."
                )
            else:
                return (
                    f"{subj} has relation "
                    f"{{{rel_name}}} "
                    f"with {obj}."
                )
        else:
            # dataset KGs
            return f"{src} {rel_name} {dst}."

    def retrieve_context(
        self,
        node: str,
        depth: int = 2,
    ) -> List[str]:
        context: Set[str] = set()

        def explore(n: str, d: int):
            if d > depth:
                return
            for nbr in self.graph.neighbors(n):
                data = self.graph[n][nbr]
                text = self._format_edge_for_context(n, nbr, data)
                context.add(text)
                explore(nbr, d + 1)
            for nbr in self.graph.predecessors(n):
                data = self.graph[nbr][n]
                text = self._format_edge_for_context(nbr, n, data)
                context.add(text)
                explore(nbr, d + 1)

        explore(node, 1)
        return list(context)

    def retrieve(
        self,
        query_emb: np.ndarray,
        k: int = 8,
    ) -> Tuple[List[Tuple[str, float]], Set[str], str]:
        top_nodes = self.retrieve_relevant_nodes(query_emb, k=k)
        context: Set[str] = set()
        for node, _ in top_nodes:
            ctx = self.retrieve_context(node)
            context.update(ctx)
        context_text = " ".join(context)
        return top_nodes, context, context_text


# ============================================================
# 6. LLM-based evaluator (OpenAI API 5.1)
# ============================================================

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = _load_openai_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY env var or provide it "
            "in .env"
        )
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    """
    Use an OpenAI model as a binary judge.
    Returns 1 if the model believes the context contains the correct answer,
    otherwise 0.
    """
    client = _get_openai_client()

    system_prompt = (
        "You are an evaluation assistant. "
        "You are given a statement that is assumed to be the correct answer, "
        "and a retrieved context. "
        "Return '1' (without quotes) if the context clearly contains enough "
        "information to support that answer. Otherwise return '0'. "
        "Return only a single character: '1' or '0'."
    )

    user_prompt = (
        "Correct answer statement:\n"
        f"{correct_answer}\n\n"
        "Retrieved context from a knowledge graph:\n"
        f"{context}\n\n"
        "Does the retrieved context contain enough information to support "
        "the correctness of the answer statement? "
        "Respond strictly with '1' for yes or '0' for no."
    )

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL_JUDGE,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_output_tokens=64,  # >= 16
        )
        text = resp.output[0].content[0].text.strip()
    except Exception as e:
        print(f"[gpt_evaluate_response] Error calling OpenAI: {e}")
        return 0

    text = text.strip()
    if text == "1":
        return 1
    if text == "0":
        return 0

    # Fallback: heuristic if model output is weird
    ans_tokens = set(t.lower() for t in correct_answer.split() if len(t) > 3)
    if not ans_tokens:
        return 0
    ctx_lower = context.lower()
    for t in ans_tokens:
        if t in ctx_lower:
            return 1
    return 0


# ============================================================
# 7. Evaluation utilities
# ============================================================

def evaluate_accuracy_for_graph(
    query_embedder: HFEmbedder,
    retriever: WeightedGraphRetriever,
    queries: List[str],
    method_name: str,
    essay_idx: int,
    results_dir: str,
    k: int = 8,
    verbose: bool = False,
) -> Dict:
    os.makedirs(results_dir, exist_ok=True)

    print(f"[{method_name}] encoding {len(queries)} queries for essay {essay_idx} ...")
    query_embs = query_embedder.encode_batch(queries)

    correct = 0
    results = []

    for qi, q in enumerate(queries):
        q_emb = query_embs[qi]
        _, _, context_text = retriever.retrieve(q_emb, k=k)
        evaluation = gpt_evaluate_response(q, context_text)
        results.append(
            {
                "correct_answer": q,
                "retrieved_context": context_text,
                "evaluation": int(evaluation),
            }
        )
        correct += evaluation

    accuracy = correct / len(queries) if queries else 0.0
    results.append({"accuracy": f"{accuracy * 100:.2f}%"})

    out_path = os.path.join(results_dir, f"results_{essay_idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(
            f"[{method_name}] Essay {essay_idx}: "
            f"accuracy={accuracy:.4f} ({correct}/{len(queries)})"
        )

    return {
        "accuracy": accuracy,
        "num_queries": len(queries),
        "method": method_name,
        "essay_idx": essay_idx,
    }


def aggregate_method_stats(summaries: List[Dict]) -> Dict[str, float]:
    if not summaries:
        return {"mean_accuracy": 0.0, "num_essays": 0}
    accs = [s["accuracy"] for s in summaries]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "num_essays": len(accs),
    }


def compare_methods(all_summaries: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    return {m: aggregate_method_stats(s) for m, s in all_summaries.items()}


def print_comparison_table(comparison: Dict[str, Dict]):
    print("\n=== Method Comparison (Mean Accuracy) ===")
    print(f"{'Method':<10} | {'Mean Acc':>8} | {'#Essays':>7}")
    print("-" * 32)
    for m, stats in comparison.items():
        print(
            f"{m:<10} | {stats['mean_accuracy']*100:8.2f}% | "
            f"{stats['num_essays']:7d}"
        )


# ============================================================
# 8. Full evaluation over the dataset
# ============================================================

def run_full_evaluation(
    dataset_json_path: str,
    trace_nodes_csv: str,
    trace_rels_csv: str,
    output_root: str,
    methods: List[str],
    k: int = 8,
    max_essays: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    # Load dataset dump you already created
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_list = json.load(f)
    if max_essays is not None:
        dataset_list = dataset_list[:max_essays]

    # TRACE-KG embeddings & graph
    ent_embedder = HFEmbedder(ENT_EMBED_MODEL, DEVICE)
    rel_embedder = HFEmbedder(REL_EMBED_MODEL, DEVICE)  # relation embeddings precomputed but not used yet
    nodes_df = pd.read_csv(trace_nodes_csv)
    rels_df = pd.read_csv(trace_rels_csv)

    trace_node_embs, _ = compute_weighted_entity_embeddings(ent_embedder, nodes_df, ENT_WEIGHTS)
    trace_rel_embs, _ = compute_weighted_relation_embeddings(rel_embedder, rels_df, nodes_df, REL_EMB_WEIGHTS)
    trace_graph, trace_node_info = build_tracekg_nx_and_nodeinfo(nodes_df, rels_df)
    trace_retriever = WeightedGraphRetriever(trace_node_embs, trace_graph, node_info=trace_node_info)

    # Query embedder (same model as entity, for semantic match)
    query_embedder = ent_embedder

    all_summaries: Dict[str, List[Dict]] = {m: [] for m in methods}

    for idx, row in enumerate(dataset_list):
        essay_idx = idx
        queries: List[str] = row.get("generated_queries", [])
        if not queries:
            if verbose:
                print(f"Skipping essay {essay_idx}: no queries.")
            continue

        if verbose:
            print(f"\n=== Essay {essay_idx} | {len(queries)} queries ===")

        # TRACE-KG (one global graph)
        if "tracekg" in methods:
            summaries_dir = os.path.join(output_root, "tracekg")
            s = evaluate_accuracy_for_graph(
                query_embedder=query_embedder,
                retriever=trace_retriever,
                queries=queries,
                method_name="tracekg",
                essay_idx=essay_idx,
                results_dir=summaries_dir,
                k=k,
                verbose=verbose,
            )
            all_summaries["tracekg"].append(s)

        # Other methods: kggen, graphrag, openie
        for method in methods:
            if method == "tracekg":
                continue

            kg_key = None
            if method == "kggen":
                kg_key = "kggen"
            elif method == "graphrag":
                kg_key = "graphrag_kg"
            elif method == "openie":
                kg_key = "openie_kg"
            else:
                continue

            kg_data = row.get(kg_key, None)
            if kg_data is None:
                if verbose:
                    print(f"  [{method}] No KG data for essay {essay_idx}, skipping.")
                continue

            sg = SimpleGraph.from_kggen_dict(kg_data)
            g_nx = sg.to_nx()

            node_ids = list(g_nx.nodes())
            node_texts = [str(n) for n in node_ids]
            node_embs_arr = query_embedder.encode_batch(node_texts)
            node_embs = {nid: node_embs_arr[i] for i, nid in enumerate(node_ids)}
            retriever = WeightedGraphRetriever(node_embs, g_nx, node_info=None)

            summaries_dir = os.path.join(output_root, method)
            s = evaluate_accuracy_for_graph(
                query_embedder=query_embedder,
                retriever=retriever,
                queries=queries,
                method_name=method,
                essay_idx=essay_idx,
                results_dir=summaries_dir,
                k=k,
                verbose=verbose,
            )
            all_summaries[method].append(s)

    return all_summaries


# ============================================================
# 9. Main
# ============================================================

def main():
    dataset_json_path = (
        "Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset-short.json"
    )
    trace_nodes_csv = "Experiments/MYNE/TRACEKG/nodes.csv"
    trace_rels_csv = "Experiments/MYNE/TRACEKG/rels.csv"

    output_root = "./tracekg_mine_results_weighted_openai_v5"

    methods = ["kggen", "graphrag", "openie", "tracekg"]
    #methods = ["tracekg"]

    all_summaries = run_full_evaluation(
        dataset_json_path=dataset_json_path,
        trace_nodes_csv=trace_nodes_csv,
        trace_rels_csv=trace_rels_csv,
        output_root=output_root,
        methods=methods,
        k=8,
        max_essays=1,  # or a small number for debugging
        verbose=True,
    )

    comparison = compare_methods(all_summaries)
    print_comparison_table(comparison)


if __name__ == "__main__":
    main()

#endregion#? QA4Methods - V6   (TRACE names for context, weighted embeddings) - Wihout qualifiers in relation text
#?#########################  End  ##########################



# === Method Comparison (Mean Accuracy) ===
# Method | Mean Acc | #Essays
# --------------------------------
# kggen | 60.00% | 1
# graphrag | 46.67% | 1
# openie | 73.33% | 1
# tracekg | 80.00% | 1
# [tracekg] encoding 15 queries for essay 0 ...
# [kggen] Essay 0: accuracy=0.6000 (9/15)
# [graphrag] Essay 0: accuracy=0.4667 (7/15)
# [openie] Essay 0: accuracy=0.7333 (11/15)





#endregion#! Experiments
#!#############################################  End Chapter  ##################################################





  

#!############################################# Start Chapter ##################################################
#region:#!   000





#endregion#! 000
#!#############################################  End Chapter  ##################################################




#?######################### Start ##########################
#region:#?   100 KG - V3


#!/usr/bin/env python3
"""
Drive the full TRACE KG pipeline in Trace_KG.py once per essay.

For each essay i in:
    Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json

1) Write essay i into:
       data/pdf_to_json/Plain_Text.json
2) Run, in order (STOP on first failure for that essay):
       sentence_chunks_token_driven(...)
       embed_and_index_chunks(...)
       run_entity_extraction_on_chunks(...)
       iterative_resolution()
       produce_clean_jsonl(...)
       classrec_iterative_main()
       main_input_for_cls_res()
       run_pipeline_iteratively()
       run_relres_iteratively()
       export_relations_and_nodes_to_csv()
3) Snapshot data/ into
       KGs_from_Essays/KG_Essay_{i} or ..._FAILED
4) Clear only:
       data/Chunks, data/Classes, data/Entities, data/KG, data/Relations

We ALWAYS write the global stats file at the end, even if we stop early for some essays.

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
    run_relres_iteratively,
    export_relations_and_nodes_to_csv,
)

# ------------------------------------------------------------------------------------
# CONFIG PATHS
# ------------------------------------------------------------------------------------

REPO_ROOT = Path(".").resolve()

PLAIN_TEXT_JSON = REPO_ROOT / "data/pdf_to_json/Plain_Text.json"
ESSAYS_JSON = REPO_ROOT / "Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json"
DATA_DIR = REPO_ROOT / "data"
KG_OUT_ROOT = REPO_ROOT / "KGs_from_Essays"

DATA_SUBDIRS_TO_CLEAR = [
    DATA_DIR / "Chunks",
    DATA_DIR / "Classes",
    DATA_DIR / "Entities",
    DATA_DIR / "KG",
    DATA_DIR / "Relations",
]

ESSAY_START = 1
ESSAY_END = 1  # inclusive


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

    Returns:
      stats = {
        "steps": {
            step_name: {"ok": bool, "error": str|None, "seconds": float},
            ...
        },
        "ok": bool,             # True only if all steps succeeded
        "error": str|None       # first error message if any
      }
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
        t0 = time.time()
        step_info: Dict[str, Any] = {"ok": True, "error": None, "seconds": None}
        try:
            fn(*args, **kwargs)
        except Exception as e:
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
        # IMPORTANT: match Trace_KG.py's CHUNKS_JSONL path
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

    # 9) Relation Res Multi Run
    if not _run_step("run_relres_iteratively", run_relres_iteratively):
        return stats

    # 10) Export KG to CSVs
    if not _run_step("export_relations_and_nodes_to_csv", export_relations_and_nodes_to_csv):
        return stats

    return stats


# ------------------------------------------------------------------------------------
# ORCHESTRATOR OVER ALL ESSAYS
# ------------------------------------------------------------------------------------

def main():
    ensure_dir(KG_OUT_ROOT)

    try:
        essays = load_essays(ESSAYS_JSON)
    except Exception as e:
        print(f"FATAL: cannot load essays from {ESSAYS_JSON}: {e}")
        # still write an empty stats file to show we tried
        log_path = KG_OUT_ROOT / "Trace_KG_per_essay_stats.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump({"_fatal_error": repr(e)}, f, ensure_ascii=False, indent=2)
        return

    total = len(essays)

    indexed: List[Tuple[int, Dict[str, Any]]] = [
        (i + 1, essays[i]) for i in range(total)
        if ESSAY_START <= i + 1 <= ESSAY_END
    ]

    print(f"Found {total} essays; running from {ESSAY_START} to {ESSAY_END} (inclusive).")
    print(f"Total to process now: {len(indexed)}")

    global_stats: Dict[int, Dict[str, Any]] = {}

    for essay_idx, essay in tqdm(indexed, desc="Essays", unit="essay"):
        print(f"\n================ Essay {essay_idx} / {ESSAY_END} ================\n")
        t0_essay = time.time()

        # 0) Clean pipeline state BEFORE this essay
        clear_pipeline_state()

        # 1) Write this essay to Plain_Text.json
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
            # snapshot current data dir so we know this essay failed at the start
            snap_dir = copy_data_for_essay(essay_idx, ok=False)
            stats["snapshot_dir"] = str(snap_dir)
            global_stats[essay_idx] = stats
            continue

        # 2) Run full pipeline (stops internally on first failed step)
        stats = run_full_pipeline_for_current_plain_text()
        essay_ok = stats.get("ok", False)

        # 3) Snapshot data directory
        snapshot_dir = copy_data_for_essay(essay_idx, ok=essay_ok)

        # 4) Clean pipeline state AFTER snapshot
        clear_pipeline_state()

        stats["seconds_total"] = time.time() - t0_essay
        stats["snapshot_dir"] = str(snapshot_dir)
        global_stats[essay_idx] = stats

        if essay_ok:
            print(f"[Essay {essay_idx}] ✅ Completed in {stats['seconds_total']:.1f}s; snapshot: {snapshot_dir}")
        else:
            print(f"[Essay {essay_idx}] ❌ FAILED (stopped at first failing step). Snapshot: {snapshot_dir}")
            print(f"  Error: {stats.get('error')}")

    # ALWAYS write overall stats, even if some or all essays failed
    log_path = KG_OUT_ROOT / "Trace_KG_per_essay_stats.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)

    print(f"\nAll done. Per‑essay stats written to: {log_path}")


if __name__ == "__main__":
    main()

#endregion#? 100 KG - V3
#?#########################  End  ##########################



