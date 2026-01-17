



  






#!############################################# Start Chapter ##################################################
#region:#!   Experiments 1 - MINE 1 From KG Gen Paper






#?######################### Start ##########################
#region:#?   QA4Methods - V10   (TRACE KG per-snapshot evaluation, id-matched, weighted embeddings)


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

# Paths
DATASET_JSON_PATH = Path("Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset.json")
KG_SNAPSHOTS_ROOT = Path("KGs_from_Essays")
OUTPUT_ROOT = "./tracekg_mine_results_weighted_openai_v10_only4TRACE-KG"

# Limit how many snapshots to run (None = all)
MAX_SNAPSHOTS: Optional[int] = None  # e.g., 3 for just the first 3 discovered


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
                truncation=False,
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
# 3. Baseline SimpleGraph
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

        rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))

        rel_desc = safe_str(row.get("rel_desc", "")) or safe_str(row.get("canonical_rel_desc", ""))
        qualifiers = safe_str(row.get("qualifiers", ""))
        desc_plus_q = " ; ".join([p for p in [rel_desc, qualifiers] if p])

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
# 5. Retrieval
# ============================================================

class WeightedGraphRetriever:
    def __init__(
        self,
        node_embeddings: Dict[str, np.ndarray],
        graph: nx.DiGraph,
        node_info: Optional[Dict[str, Dict[str, str]]] = None,
    ):
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
        info = self.node_info.get(node_id)
        if info is None:
            return str(node_id)

        name = info.get("name") or str(node_id)
        cls = info.get("class_label") or ""
        if cls:
            return f"{name} (which is of type: {cls})"
        return name

    def _format_edge_for_context(self, src: str, dst: str, data: Dict) -> str:
        rel_name = data.get("relation", "")
        qualifiers = data.get("qualifiers", "")

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
# 6. LLM evaluator
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=64,
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

    ans_tokens = set(t.lower() for t in correct_answer.split() if len(t) > 3)
    if not ans_tokens:
        return 0
    ctx_lower = context.lower()
    for t in ans_tokens:
        if t in ctx_lower:
            return 1
    return 0


# ============================================================
# 7. Evaluation helpers
# ============================================================

def evaluate_accuracy_for_graph(
    query_embedder: HFEmbedder,
    retriever: WeightedGraphRetriever,
    queries: List[str],
    method_name: str,
    snapshot_label: str,
    dataset_id: int,
    results_dir: str,
    k: int = 8,
    verbose: bool = False,
) -> Dict:
    os.makedirs(results_dir, exist_ok=True)

    print(f"[{method_name}] encoding {len(queries)} queries "
          f"(snapshot={snapshot_label}, dataset_id={dataset_id}) ...")
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

    out_path = os.path.join(results_dir, f"results_{snapshot_label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(
            f"[{method_name}] Snapshot={snapshot_label} (dataset_id={dataset_id}): "
            f"accuracy={accuracy:.4f} ({correct}/{len(queries)})"
        )

    return {
        "accuracy": accuracy,
        "num_queries": len(queries),
        "method": method_name,
        "snapshot_label": snapshot_label,
        "dataset_id": dataset_id,
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
    print("\n=== Method Comparison (Mean Accuracy across evaluated snapshots) ===")
    print(f"{'Method':<10} | {'Mean Acc':>8} | {'#Snaps':>7}")
    print("-" * 36)
    for m, stats in comparison.items():
        print(
            f"{m:<10} | {stats['mean_accuracy']*100:8.2f}% | "
            f"{stats['num_essays']:7d}"
        )


def print_per_snapshot_table(all_summaries: Dict[str, List[Dict]], methods: List[str]):
    """
    Print a per-snapshot table with one row per snapshot and one accuracy column per method.
    Shows 'xx.xx% (correct/total)' for each available method.
    """
    # Build index: snapshot_label -> {method -> summary}
    by_snap: Dict[str, Dict[str, Dict]] = {}
    for method, summaries in all_summaries.items():
        for s in summaries:
            snap = s.get("snapshot_label", "UNKNOWN")
            d = by_snap.setdefault(snap, {})
            d[method] = s

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
                correct = int(round(acc * n))
                cell = f"{acc*100:5.2f}% ({correct}/{n})"
            line += f" | {cell:>20}"
        print(line)


# ============================================================
# 8. Snapshot discovery and evaluation
# ============================================================

def discover_snapshots(root: Path, max_snapshots: Optional[int]) -> List[Tuple[str, Path]]:
    candidates: List[Tuple[str, Path]] = []
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

    print(f"[info] Discovered {len(candidates)} usable KG snapshots under {root}:")
    for label, path in candidates:
        print(f"  - Snapshot {label}: {path}")
    return candidates


def build_id_to_item_map(dataset: List[Dict]) -> Dict[int, Dict]:
    mapping: Dict[int, Dict] = {}
    for item in dataset:
        if not isinstance(item, dict):
            continue
        if "id" not in item:
            continue
        try:
            key = int(item["id"])
        except Exception:
            continue
        mapping[key] = item
    return mapping


def run_full_evaluation_over_snapshots(
    dataset_json_path: Path,
    snapshots_root: Path,
    output_root: str,
    methods: List[str],
    k: int = 8,
    max_snapshots: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    with dataset_json_path.open("r", encoding="utf-8") as f:
        dataset_list = json.load(f)

    if not isinstance(dataset_list, list):
        raise ValueError(f"Expected top-level list in {dataset_json_path}, got {type(dataset_list)}")

    id_to_item = build_id_to_item_map(dataset_list)
    print(f"[info] Loaded evaluation dataset with {len(dataset_list)} entries from {dataset_json_path}")
    print(f"[info] Built id-to-item map with {len(id_to_item)} IDs.")

    snapshot_dirs = discover_snapshots(snapshots_root, max_snapshots=max_snapshots)
    if not snapshot_dirs:
        return {m: [] for m in methods}

    ent_embedder = HFEmbedder(ENT_EMBED_MODEL, DEVICE)
    rel_embedder = HFEmbedder(REL_EMBED_MODEL, DEVICE)
    query_embedder = ent_embedder

    all_summaries: Dict[str, List[Dict]] = {m: [] for m in methods}

    for snapshot_label, snap_dir in snapshot_dirs:
        try:
            dataset_id = int(snapshot_label)
        except Exception:
            print(f"\n=== Snapshot {snapshot_label} ===")
            print(f"[warn] Cannot parse snapshot label '{snapshot_label}' as int; skipping (no dataset id).")
            continue

        item = id_to_item.get(dataset_id)
        if item is None:
            print(f"\n=== Snapshot {snapshot_label} ===")
            print(f"[warn] No dataset entry with id={dataset_id}; skipping.")
            continue

        queries: List[str] = item.get("generated_queries", [])
        if not queries:
            print(f"\n=== Snapshot {snapshot_label} (dataset_id={dataset_id}) ===")
            print("[info] Skipping: no 'generated_queries' in dataset item.")
            continue

        print(f"\n=== Snapshot {snapshot_label} (dataset_id={dataset_id}) ===")
        print(f"[info] Snapshot dir: {snap_dir}")
        print(f"[info] #queries: {len(queries)}")

        nodes_csv = snap_dir / "KG" / "nodes.csv"
        rels_csv = snap_dir / "KG" / "rels_fixed_no_raw.csv"

        print(f"[info] Using TRACE-KG nodes: {nodes_csv}")
        print(f"[info] Using TRACE-KG rels : {rels_csv}")

        nodes_df = pd.read_csv(nodes_csv)
        rels_df = pd.read_csv(rels_csv)

        print(f"[info] nodes_df rows: {len(nodes_df)}, rels_df rows: {len(rels_df)}")

        trace_node_embs, _ = compute_weighted_entity_embeddings(ent_embedder, nodes_df, ENT_WEIGHTS)
        trace_rel_embs, _ = compute_weighted_relation_embeddings(rel_embedder, rels_df, nodes_df, REL_EMB_WEIGHTS)
        trace_graph, trace_node_info = build_tracekg_nx_and_nodeinfo(nodes_df, rels_df)
        trace_retriever = WeightedGraphRetriever(trace_node_embs, trace_graph, node_info=trace_node_info)

        if "tracekg" in methods:
            summaries_dir = os.path.join(output_root, "tracekg")
            s = evaluate_accuracy_for_graph(
                query_embedder=query_embedder,
                retriever=trace_retriever,
                queries=queries,
                method_name="tracekg",
                snapshot_label=snapshot_label,
                dataset_id=dataset_id,
                results_dir=summaries_dir,
                k=k,
                verbose=verbose,
            )
            all_summaries["tracekg"].append(s)

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
                query_embedder=query_embedder,
                retriever=retriever,
                queries=queries,
                method_name=method,
                snapshot_label=snapshot_label,
                dataset_id=dataset_id,
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
    # methods = ["kggen", "graphrag", "openie", "tracekg"]
    # Or only TRACE-KG:
    methods = ["tracekg"]

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_summaries = run_full_evaluation_over_snapshots(
        dataset_json_path=DATASET_JSON_PATH,
        snapshots_root=KG_SNAPSHOTS_ROOT,
        output_root=OUTPUT_ROOT,
        methods=methods,
        k=8,
        max_snapshots=MAX_SNAPSHOTS,
        verbose=True,
    )

    comparison = compare_methods(all_summaries)
    print_per_snapshot_table(all_summaries, methods)
    print_comparison_table(comparison)


if __name__ == "__main__":
    main()
    
    

#endregion#?   QA4Methods - V10   (TRACE KG per-snapshot evaluation, id-matched, weighted embeddings)
#?#########################  End  ##########################



#?######################### Start ##########################
#region:#?   Results for 52 randomly selected essays


# #first 12 results
    
# === Per-Snapshot Accuracy (all methods) ===
#   Snapshot | DatasetID |       tracekg        |  Reponses mentioned in baselines by mistake     |     KgGen     |                        GraphRAG                   |     OpenIE 
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
#        001 |         1 |       93.33% (14/15) |  Response accuracy for essay ID 1:              {'kggen_accuracy': 0.5333,             'graphrag_accuracy': 0.4667, 'openie_accuracy': 0.2667}       
#        002 |         2 |      100.00% (15/15) |  Response accuracy for essay ID 2:              {'kggen_accuracy': 0.6,                'graphrag_accuracy': 0.3333, 'openie_accuracy': 0.0} 
#        010 |        10 |      100.00% (15/15) |  Response accuracy for essay ID 10:             {'kggen_accuracy': 0.6667000000000001, 'graphrag_accuracy': 0.6667, 'openie_accuracy': 0.3333}    
#        015 |        15 |      100.00% (15/15) |  Response accuracy for essay ID 15:             {'kggen_accuracy': 0.6667000000000001, 'graphrag_accuracy': 0.4667, 'openie_accuracy': 0.3333}    
#        024 |        24 |      100.00% (15/15) |  Response accuracy for essay ID 25:             {'kggen_accuracy': 0.8667,             'graphrag_accuracy': 0.6,    'openie_accuracy': 0.2667}    
#        047 |        47 |      100.00% (15/15) |  Response accuracy for essay ID 48:             {'kggen_accuracy': 0.8667,             'graphrag_accuracy': 0.9333, 'openie_accuracy': 0.7333}    
#        052 |        52 |      100.00% (15/15) |  Response accuracy for essay ID 53:             {'kggen_accuracy': 0.6667000000000001, 'graphrag_accuracy': 0.3333, 'openie_accuracy': 0.6}    
#        053 |        53 |      100.00% (15/15) |  Response accuracy for essay ID 54:             {'kggen_accuracy': 0.8,                'graphrag_accuracy': 0.7333, 'openie_accuracy': 0.4}    
#        064 |        64 |       86.67% (13/15) |  NULL                
#        067 |        67 |      100.00% (15/15) |  Response accuracy for essay ID 68:             {'kggen_accuracy': 0.6,                'graphrag_accuracy': 0.4,    'openie_accuracy': 0.0667} 
#        088 |        88 |      100.00% (15/15) |  Response accuracy for essay ID 89:             {'kggen_accuracy': 0.9333,             'graphrag_accuracy': 0.3333, 'openie_accuracy': 0.4667}    
#        091 |        91 |      100.00% (15/15) |  Response accuracy for essay ID 92:             {'kggen_accuracy': 0.9333,             'graphrag_accuracy': 0.5333, 'openie_accuracy': 0.5333}



# === Method Comparison (Mean Accuracy across evaluated snapshots) ===
# Method     | Mean Acc |   Essays
# ------------------------------------
# tracekg    |    98.33% |      52
# KGGen      |    66.67% |      52
# GraphRAG   |    53.31% |      52
# OpenIE     |    36.36% |      52



#! For TRACE KG the above code "" has been run the results
#! For other 3 baseline methods, the results has been mentioned in wrong ID. I found the true one. And following code reads the accuracy for the true ID. 
# (HERE IS THE LINK: https://huggingface.co/datasets/josancamon/kg-gen-MINE-evaluation-dataset/viewer/default/train?row=22)


#*######################### Start ##########################
#region:#?   reads the accuracy for the true ID.



# import json
# from pathlib import Path

# DATASET_JSON_PATH = Path("Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset.json")

# def get_response_accuracy(essay_id: int, dataset_path: Path):
#     """
#     Get response accuracy for kggen, graphrag, and openie methods for a given essay ID.
    
#     :param essay_id: The ID of the essay (e.g., 23 for id=23).
#     :param dataset_path: Path to the evaluation dataset JSON file.
#     :return: Dictionary of accuracies for the 3 methods or None if ID not found.
#     """
#     with dataset_path.open("r", encoding="utf-8") as f:
#         dataset = json.load(f)

#     # Find the item with the matching ID
#     essay_item = next((item for item in dataset if item.get("id") == essay_id), None)
#     if essay_item is None:
#         return None  # ID not found

#     # Get response accuracy for each method
#     return {
#         "kggen_accuracy": essay_item.get("kggen_accuracy", None),
#         "graphrag_accuracy": essay_item.get("graphrag_accuracy", None),
#         "openie_accuracy": essay_item.get("openie_accuracy", None),
#     }

# # Example usage
# essay_id = 92
# response_accuracy = get_response_accuracy(essay_id, DATASET_JSON_PATH)
# if response_accuracy:
#     print(f"Response accuracy for essay ID {essay_id}: {response_accuracy}")
# else:
#     print(f"Essay ID {essay_id} not found in dataset.")
    

#endregion#? reads the accuracy for the true ID.
#*#########################  End  ##########################



#endregion#? Results for 12 randomly selected essays
#?#########################  End  ##########################


#endregion#! Experiments 1 - MINE 1 From KG Gen Paper
#!#############################################  End Chapter  ##################################################

  
