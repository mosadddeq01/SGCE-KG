
  



#!############################################# Start Chapter ##################################################
#region:#!   Experiments 1 - MINE 1 From KG Gen Paper






#?######################### Start ##########################
#region:#?   QA4Methods - V10   (TRACE KG per-snapshot evaluation, id-matched, weighted embeddings)


# import os
# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Set

# import numpy as np
# import pandas as pd
# import networkx as nx

# from datasets import load_dataset  # not strictly required if you only use JSON dump
# import torch
# from transformers import AutoTokenizer, AutoModel
# from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity

# from openai import OpenAI

# # ============================================================
# # 0. Global config: weights, models, env
# # ============================================================

# ENT_WEIGHTS = {
#     "name": 0.40,
#     "desc": 0.25,
#     "ctx": 0.35,
# }

# REL_EMB_WEIGHTS = {
#     "name": 0.25,
#     "desc+Q": 0.15,
#     "head_tail": 0.20,
#     "ctx": 0.40,
# }

# ENT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# REL_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# BATCH_SIZE = 32
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OPENAI_MODEL_JUDGE =  "gpt-5.1" #"gpt-4.1-nano"
# OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# # Paths
# DATASET_JSON_PATH = Path("Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset.json")
# KG_SNAPSHOTS_ROOT = Path("Experiments/MYNE/Ex1/KGs_from_Essays_KFE_test")
# OUTPUT_ROOT = "./tracekg_mine_results_weighted_openai_v10_only4TRACE-KG"

# # Limit how many snapshots to run (None = all)
# MAX_SNAPSHOTS: Optional[int] = None  # e.g., 3 for just the first 3 discovered


# # ============================================================
# # 1. Env helper
# # ============================================================

# def _load_openai_key(
#     envvar: str = OPENAI_API_KEY_ENV,
#     fallback_path: str = ".env",
# ):
#     key = os.getenv(envvar, None)
#     if key:
#         return key
#     if Path(fallback_path).exists():
#         txt = Path(fallback_path).read_text(encoding="utf-8").strip()
#         if txt:
#             return txt
#     return None


# # ============================================================
# # 2. HF Embedder
# # ============================================================

# def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
#     masked = token_embeds * mask
#     summed = masked.sum(dim=1)
#     counts = mask.sum(dim=1).clamp(min=1e-9)
#     return summed / counts


# class HFEmbedder:
#     def __init__(self, model_name: str, device: str = DEVICE):
#         # print(f"[Embedder] loading model {model_name} on {device} ...")
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#         self.model = AutoModel.from_pretrained(model_name)
#         self.model.to(device)
#         self.model.eval()
#         for p in self.model.parameters():
#             p.requires_grad = False

#     @torch.no_grad()
#     def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
#         embs = []
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]
#             enc = self.tokenizer(
#                 batch,
#                 padding=True,
#                 truncation=False,
#                 return_tensors="pt",
#                 max_length=1024,
#             )
#             input_ids = enc["input_ids"].to(self.device)
#             attention_mask = enc["attention_mask"].to(self.device)
#             out = self.model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 return_dict=True,
#             )
#             token_embeds = out.last_hidden_state
#             pooled = mean_pool(token_embeds, attention_mask)
#             pooled = pooled.cpu().numpy()
#             embs.append(pooled)
#         embs = np.vstack(embs)
#         embs = normalize(embs, axis=1)
#         return embs


# # ============================================================
# # 3. Baseline SimpleGraph
# # ============================================================

# @dataclass
# class SimpleGraph:
#     entities: Set[str]
#     relations: Set[Tuple[str, str, str]]

#     @staticmethod
#     def from_kggen_dict(d: Dict) -> "SimpleGraph":
#         entities = set(d.get("entities", []))
#         rels_raw = d.get("relations", [])
#         relations = set()
#         for r in rels_raw:
#             if isinstance(r, (list, tuple)) and len(r) == 3:
#                 s, rel, t = r
#                 relations.add((str(s), str(rel), str(t)))
#         return SimpleGraph(entities=entities, relations=relations)

#     def to_nx(self) -> nx.DiGraph:
#         g = nx.DiGraph()
#         for e in self.entities:
#             g.add_node(e, text=str(e))
#         for s, rel, t in self.relations:
#             g.add_edge(s, t, relation=str(rel))
#         return g


# # ============================================================
# # 4. TRACE-KG utilities
# # ============================================================

# def safe_str(x) -> str:
#     if x is None:
#         return ""
#     if isinstance(x, float) and np.isnan(x):
#         return ""
#     return str(x)


# def build_tracekg_entity_texts(nodes_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
#     ids, names, descs, ctxs = [], [], [], []
#     for _, row in nodes_df.iterrows():
#         ent_id = safe_str(row["entity_id"])
#         ids.append(ent_id)

#         name_txt = safe_str(row.get("entity_name", ""))
#         desc_txt = safe_str(row.get("entity_description", ""))

#         cls_label = safe_str(row.get("class_label", ""))
#         cls_group = safe_str(row.get("class_group", ""))
#         node_props = safe_str(row.get("node_properties", ""))

#         ctx_parts = []
#         if cls_label:
#             ctx_parts.append(f"[CLASS:{cls_label}]")
#         if cls_group:
#             ctx_parts.append(f"[GROUP:{cls_group}]")
#         if node_props:
#             ctx_parts.append(f"[PROPS:{node_props}]")
#         ctx_txt = " ; ".join(ctx_parts)

#         names.append(name_txt)
#         descs.append(desc_txt)
#         ctxs.append(ctx_txt)
#     return ids, names, descs, ctxs


# def compute_weighted_entity_embeddings(
#     embedder: HFEmbedder,
#     nodes_df: pd.DataFrame,
#     weights: Dict[str, float] = ENT_WEIGHTS,
# ) -> Tuple[Dict[str, np.ndarray], int]:
#     ids, names, descs, ctxs = build_tracekg_entity_texts(nodes_df)

#     # print("[TRACE-ENT] encoding name field ...")
#     emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None

#     # print("[TRACE-ENT] encoding desc field ...")
#     emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None

#     # print("[TRACE-ENT] encoding ctx field ...")
#     emb_ctx = embedder.encode_batch(ctxs) if any(t.strip() for t in ctxs) else None

#     D_ref = None
#     for arr in [emb_name, emb_desc, emb_ctx]:
#         if arr is not None:
#             D_ref = arr.shape[1]
#             break
#     if D_ref is None:
#         raise ValueError("All entity fields empty — cannot embed.")

#     def _ensure(arr):
#         if arr is None:
#             return np.zeros((len(ids), D_ref))
#         if arr.shape[1] != D_ref:
#             raise ValueError("Entity embedding dimension mismatch.")
#         return arr

#     emb_name = _ensure(emb_name)
#     emb_desc = _ensure(emb_desc)
#     emb_ctx = _ensure(emb_ctx)

#     w_name = weights.get("name", 0.0)
#     w_desc = weights.get("desc", 0.0)
#     w_ctx = weights.get("ctx", 0.0)
#     Wsum = w_name + w_desc + w_ctx
#     if Wsum <= 0:
#         raise ValueError("Sum of ENT_WEIGHTS must be > 0")
#     w_name /= Wsum
#     w_desc /= Wsum
#     w_ctx /= Wsum

#     combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
#     combined = normalize(combined, axis=1)

#     node_embs: Dict[str, np.ndarray] = {}
#     for i, node_id in enumerate(ids):
#         node_embs[node_id] = combined[i]
#     return node_embs, D_ref


# def build_tracekg_relation_texts(
#     rels_df: pd.DataFrame,
#     nodes_df: pd.DataFrame,
# ) -> Tuple[List[str], Dict[str, int], Dict[int, Dict[str, str]]]:
#     node_info = {}
#     for _, nrow in nodes_df.iterrows():
#         nid = safe_str(nrow["entity_id"])
#         node_info[nid] = {
#             "name": safe_str(nrow.get("entity_name", "")),
#             "class_label": safe_str(nrow.get("class_label", "")),
#         }

#     relation_ids: List[str] = []
#     id_to_index: Dict[str, int] = {}
#     texts: Dict[int, Dict[str, str]] = {}

#     for i, row in rels_df.iterrows():
#         rid = safe_str(row.get("relation_id", i))
#         relation_ids.append(rid)
#         id_to_index[rid] = i

#         start_id = safe_str(row.get("start_id", ""))
#         end_id = safe_str(row.get("end_id", ""))

#         rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))

#         rel_desc = safe_str(row.get("rel_desc", "")) or safe_str(row.get("canonical_rel_desc", ""))
#         qualifiers = safe_str(row.get("qualifiers", ""))
#         desc_plus_q = " ; ".join([p for p in [rel_desc, qualifiers] if p])

#         head_info = node_info.get(start_id, {})
#         tail_info = node_info.get(end_id, {})
#         head_tail_parts = []
#         if head_info.get("name"):
#             head_tail_parts.append(f"[H:{head_info['name']}]")
#         if head_info.get("class_label"):
#             head_tail_parts.append(f"[HCLS:{head_info['class_label']}]")
#         if tail_info.get("name"):
#             head_tail_parts.append(f"[T:{tail_info['name']}]")
#         if tail_info.get("class_label"):
#             head_tail_parts.append(f"[TCLS:{tail_info['class_label']}]")
#         head_tail = " ".join(head_tail_parts)

#         canonical_name = safe_str(row.get("canonical_rel_name", ""))
#         canonical_desc = safe_str(row.get("canonical_rel_desc", ""))
#         rel_cls = safe_str(row.get("rel_cls", ""))
#         ctx_parts = []
#         if canonical_name:
#             ctx_parts.append(canonical_name)
#         if canonical_desc:
#             ctx_parts.append(canonical_desc)
#         if rel_cls:
#             ctx_parts.append(f"[CLS:{rel_cls}]")
#         ctx_txt = " ; ".join(ctx_parts)

#         texts[i] = {
#             "name": rel_name,
#             "desc+Q": desc_plus_q,
#             "head_tail": head_tail,
#             "ctx": ctx_txt,
#         }

#     return relation_ids, id_to_index, texts


# def compute_weighted_relation_embeddings(
#     embedder: HFEmbedder,
#     rels_df: pd.DataFrame,
#     nodes_df: pd.DataFrame,
#     weights: Dict[str, float] = REL_EMB_WEIGHTS,
# ) -> Tuple[Dict[str, np.ndarray], int]:
#     rel_ids, id_to_index, texts = build_tracekg_relation_texts(rels_df, nodes_df)

#     n = len(rel_ids)
#     buckets = ["name", "desc+Q", "head_tail", "ctx"]
#     bucket_texts = {b: [""] * n for b in buckets}
#     for idx in range(n):
#         for b in buckets:
#             bucket_texts[b][idx] = texts[idx].get(b, "")

#     emb_bucket = {}
#     D_ref = None
#     for b in buckets:
#         has_any = any(t.strip() for t in bucket_texts[b])
#         if not has_any:
#             emb_bucket[b] = None
#             continue
#         # print(f"[TRACE-REL] encoding bucket '{b}' ...")
#         eb = embedder.encode_batch(bucket_texts[b])
#         emb_bucket[b] = eb
#         if D_ref is None:
#             D_ref = eb.shape[1]

#     if D_ref is None:
#         raise ValueError("All relation buckets empty — cannot embed relations.")

#     def _ensure(arr):
#         if arr is None:
#             return np.zeros((n, D_ref))
#         if arr.shape[1] != D_ref:
#             raise ValueError("Relation embedding dimension mismatch.")
#         return arr

#     for b in buckets:
#         emb_bucket[b] = _ensure(emb_bucket[b])

#     w_name = weights.get("name", 0.0)
#     w_descq = weights.get("desc+Q", 0.0)
#     w_ht = weights.get("head_tail", 0.0)
#     w_ctx = weights.get("ctx", 0.0)
#     Wsum = w_name + w_descq + w_ht + w_ctx
#     if Wsum <= 0:
#         raise ValueError("Sum of REL_EMB_WEIGHTS must be > 0")
#     w_name /= Wsum
#     w_descq /= Wsum
#     w_ht /= Wsum
#     w_ctx /= Wsum

#     combined = (
#         w_name * emb_bucket["name"]
#         + w_descq * emb_bucket["desc+Q"]
#         + w_ht * emb_bucket["head_tail"]
#         + w_ctx * emb_bucket["ctx"]
#     )
#     combined = normalize(combined, axis=1)

#     rel_embs: Dict[str, np.ndarray] = {}
#     for i, rid in enumerate(rel_ids):
#         rel_embs[rid] = combined[i]
#     return rel_embs, D_ref


# def build_tracekg_nx_and_nodeinfo(
#     nodes_df: pd.DataFrame,
#     rels_df: pd.DataFrame,
# ) -> Tuple[nx.DiGraph, Dict[str, Dict[str, str]]]:
#     g = nx.DiGraph()
#     node_info: Dict[str, Dict[str, str]] = {}

#     for _, row in nodes_df.iterrows():
#         nid = safe_str(row["entity_id"])
#         name = safe_str(row.get("entity_name", ""))
#         cls_label = safe_str(row.get("class_label", ""))

#         g.add_node(
#             nid,
#             entity_name=name,
#             entity_description=safe_str(row.get("entity_description", "")),
#             class_label=cls_label,
#             class_group=safe_str(row.get("class_group", "")),
#             node_properties=safe_str(row.get("node_properties", "")),
#             chunk_ids=safe_str(row.get("chunk_ids", "")),
#         )
#         node_info[nid] = {
#             "name": name,
#             "class_label": cls_label,
#         }

#     for _, row in rels_df.iterrows():
#         sid = safe_str(row.get("start_id", ""))
#         eid = safe_str(row.get("end_id", ""))
#         rid = safe_str(row.get("relation_id", ""))
#         rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))
#         qualifiers = safe_str(row.get("qualifiers", ""))
#         g.add_edge(
#             sid,
#             eid,
#             relation=rel_name,
#             relation_id=rid,
#             chunk_id=safe_str(row.get("chunk_id", "")),
#             qualifiers=qualifiers,
#         )
#     return g, node_info


# # ============================================================
# # 5. Retrieval
# # ============================================================

# class WeightedGraphRetriever:
#     def __init__(
#         self,
#         node_embeddings: Dict[str, np.ndarray],
#         graph: nx.DiGraph,
#         node_info: Optional[Dict[str, Dict[str, str]]] = None,
#     ):
#         self.node_embeddings = node_embeddings
#         self.graph = graph
#         self.node_info = node_info or {}

#     def retrieve_relevant_nodes(
#         self,
#         query_emb: np.ndarray,
#         k: int = 8,
#     ) -> List[Tuple[str, float]]:
#         sims = []
#         for node, emb in self.node_embeddings.items():
#             sim = cosine_similarity(query_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
#             sims.append((node, sim))
#         sims.sort(key=lambda x: x[1], reverse=True)
#         return sims[:k]

#     def _format_node_for_context(self, node_id: str) -> str:
#         info = self.node_info.get(node_id)
#         if info is None:
#             return str(node_id)

#         name = info.get("name") or str(node_id)
#         cls = info.get("class_label") or ""
#         if cls:
#             return f"{name} (which is of type: {cls})"
#         return name

#     def _format_edge_for_context(self, src: str, dst: str, data: Dict) -> str:
#         rel_name = data.get("relation", "")
#         qualifiers = data.get("qualifiers", "")

#         if self.node_info:
#             subj = self._format_node_for_context(src)
#             obj = self._format_node_for_context(dst)
#             if qualifiers:
#                 return (
#                     f"{subj} has relation "
#                     f"{{{rel_name} (with qualifiers: {qualifiers})}} "
#                     f"with {obj}."
#                 )
#             else:
#                 return (
#                     f"{subj} has relation "
#                     f"{{{rel_name}}} "
#                     f"with {obj}."
#                 )
#         else:
#             return f"{src} {rel_name} {dst}."

#     def retrieve_context(
#         self,
#         node: str,
#         depth: int = 2,
#     ) -> List[str]:
#         context: Set[str] = set()

#         def explore(n: str, d: int):
#             if d > depth:
#                 return
#             for nbr in self.graph.neighbors(n):
#                 data = self.graph[n][nbr]
#                 text = self._format_edge_for_context(n, nbr, data)
#                 context.add(text)
#                 explore(nbr, d + 1)
#             for nbr in self.graph.predecessors(n):
#                 data = self.graph[nbr][n]
#                 text = self._format_edge_for_context(nbr, n, data)
#                 context.add(text)
#                 explore(nbr, d + 1)

#         explore(node, 1)
#         return list(context)

#     def retrieve(
#         self,
#         query_emb: np.ndarray,
#         k: int = 8,
#     ) -> Tuple[List[Tuple[str, float]], Set[str], str]:
#         top_nodes = self.retrieve_relevant_nodes(query_emb, k=k)
#         context: Set[str] = set()
#         for node, _ in top_nodes:
#             ctx = self.retrieve_context(node)
#             context.update(ctx)
#         context_text = " ".join(context)
#         return top_nodes, context, context_text


# # ============================================================
# # 6. LLM evaluator
# # ============================================================

# _openai_client: Optional[OpenAI] = None

# def _get_openai_client() -> OpenAI:
#     global _openai_client
#     if _openai_client is not None:
#         return _openai_client
#     api_key = _load_openai_key()
#     if not api_key:
#         raise RuntimeError(
#             "OpenAI API key not found. Set OPENAI_API_KEY env var or provide it "
#             "in .env"
#         )
#     _openai_client = OpenAI(api_key=api_key)
#     return _openai_client


# def gpt_evaluate_response(correct_answer: str, context: str) -> int:
#     client = _get_openai_client()

#     system_prompt = (
#         "You are an evaluation assistant. "
#         "You are given a statement that is assumed to be the correct answer, "
#         "and a retrieved context. "
#         "Return '1' (without quotes) if the context clearly contains enough "
#         "information to support that answer. Otherwise return '0'. "
#         "Return only a single character: '1' or '0'."
#     )

#     user_prompt = (
#         "Correct answer statement:\n"
#         f"{correct_answer}\n\n"
#         "Retrieved context from a knowledge graph:\n"
#         f"{context}\n\n"
#         "Does the retrieved context contain enough information to support "
#         "the correctness of the answer statement? "
#         "Respond strictly with '1' for yes or '0' for no."
#     )

#     try:
#         resp = client.responses.create(
#             model=OPENAI_MODEL_JUDGE,
#             input=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             max_output_tokens=64,
#         )
#         text = resp.output[0].content[0].text.strip()
#     except Exception as e:
#         print(f"[gpt_evaluate_response] Error calling OpenAI: {e}")
#         return 0

#     text = text.strip()
#     if text == "1":
#         return 1
#     if text == "0":
#         return 0

#     ans_tokens = set(t.lower() for t in correct_answer.split() if len(t) > 3)
#     if not ans_tokens:
#         return 0
#     ctx_lower = context.lower()
#     for t in ans_tokens:
#         if t in ctx_lower:
#             return 1
#     return 0


# # ============================================================
# # 7. Evaluation helpers
# # ============================================================

# def evaluate_accuracy_for_graph(
#     query_embedder: HFEmbedder,
#     retriever: WeightedGraphRetriever,
#     queries: List[str],
#     method_name: str,
#     snapshot_label: str,
#     dataset_id: int,
#     results_dir: str,
#     k: int = 8,
#     verbose: bool = False,
# ) -> Dict:
#     os.makedirs(results_dir, exist_ok=True)

#     # print(f"[{method_name}] encoding {len(queries)} queries "
#           # f"(snapshot={snapshot_label}, dataset_id={dataset_id}) ...")
#     query_embs = query_embedder.encode_batch(queries)

#     correct = 0
#     results = []

#     for qi, q in enumerate(queries):
#         q_emb = query_embs[qi]
#         _, _, context_text = retriever.retrieve(q_emb, k=k)
#         evaluation = gpt_evaluate_response(q, context_text)
#         results.append(
#             {
#                 "correct_answer": q,
#                 "retrieved_context": context_text,
#                 "evaluation": int(evaluation),
#             }
#         )
#         correct += evaluation

#     accuracy = correct / len(queries) if queries else 0.0
#     results.append({"accuracy": f"{accuracy * 100:.2f}%"})

#     out_path = os.path.join(results_dir, f"results_{snapshot_label}.json")
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)

#     if verbose:
#         print(
#             f"[{method_name}] Snapshot={snapshot_label} (dataset_id={dataset_id}): "
#             f"accuracy={accuracy:.4f} ({correct}/{len(queries)})"
#         )

#     return {
#         "accuracy": accuracy,
#         "num_queries": len(queries),
#         "method": method_name,
#         "snapshot_label": snapshot_label,
#         "dataset_id": dataset_id,
#     }


# def aggregate_method_stats(summaries: List[Dict]) -> Dict[str, float]:
#     if not summaries:
#         return {"mean_accuracy": 0.0, "num_essays": 0}
#     accs = [s["accuracy"] for s in summaries]
#     return {
#         "mean_accuracy": float(np.mean(accs)),
#         "num_essays": len(accs),
#     }


# def compare_methods(all_summaries: Dict[str, List[Dict]]) -> Dict[str, Dict]:
#     return {m: aggregate_method_stats(s) for m, s in all_summaries.items()}


# def print_comparison_table(comparison: Dict[str, Dict]):
#     print("\n=== Method Comparison (Mean Accuracy across evaluated snapshots) ===")
#     print(f"{'Method':<10} | {'Mean Acc':>8} | {'#Snaps':>7}")
#     print("-" * 36)
#     for m, stats in comparison.items():
#         print(
#             f"{m:<10} | {stats['mean_accuracy']*100:8.2f}% | "
#             f"{stats['num_essays']:7d}"
#         )


# def print_per_snapshot_table(all_summaries: Dict[str, List[Dict]], methods: List[str]):
#     """
#     Print a per-snapshot table with one row per snapshot and one accuracy column per method.
#     Shows 'xx.xx% (correct/total)' for each available method.
#     """
#     # Build index: snapshot_label -> {method -> summary}
#     by_snap: Dict[str, Dict[str, Dict]] = {}
#     for method, summaries in all_summaries.items():
#         for s in summaries:
#             snap = s.get("snapshot_label", "UNKNOWN")
#             d = by_snap.setdefault(snap, {})
#             d[method] = s

#     if not by_snap:
#         print("\n[INFO] No per-snapshot summaries to report.")
#         return

#     header = f"{'Snapshot':>10} | {'DatasetID':>9}"
#     for m in methods:
#         header += f" | {m:^20}"
#     print("\n=== Per-Snapshot Accuracy (all methods) ===")
#     print(header)
#     print("-" * len(header))

#     for snap in sorted(by_snap.keys()):
#         row_methods = by_snap[snap]
#         any_summary = next(iter(row_methods.values()))
#         dataset_id = any_summary.get("dataset_id", -1)

#         line = f"{snap:>10} | {dataset_id:9d}"
#         for m in methods:
#             s = row_methods.get(m)
#             if s is None:
#                 cell = "N/A"
#             else:
#                 acc = s["accuracy"]
#                 n = s["num_queries"]
#                 correct = int(round(acc * n))
#                 cell = f"{acc*100:5.2f}% ({correct}/{n})"
#             line += f" | {cell:>20}"
#         print(line)


# # ============================================================
# # 8. Snapshot discovery and evaluation
# # ============================================================

# def discover_snapshots(root: Path, max_snapshots: Optional[int]) -> List[Tuple[str, Path]]:
#     candidates: List[Tuple[str, Path]] = []
#     for p in sorted(root.glob("KG_Essay_*")):
#         if not p.is_dir():
#             continue
#         snapshot_label = p.name.replace("KG_Essay_", "")

#         nodes_csv = p / "KG" / "nodes.csv"
#         rels_csv = p / "KG" / "rels_fixed_no_raw.csv"
#         if not nodes_csv.exists() or not rels_csv.exists():
#             print(f"[warn] Missing KG CSVs in {p}, skipping.")
#             continue

#         candidates.append((snapshot_label, p))

#     candidates.sort(key=lambda x: x[0])

#     if max_snapshots is not None:
#         candidates = candidates[:max_snapshots]

#     print(f"[info] Discovered {len(candidates)} usable KG snapshots under {root}:")
#     for label, path in candidates:
#         print(f"  - Snapshot {label}: {path}")
#     return candidates


# def build_id_to_item_map(dataset: List[Dict]) -> Dict[int, Dict]:
#     mapping: Dict[int, Dict] = {}
#     for item in dataset:
#         if not isinstance(item, dict):
#             continue
#         if "id" not in item:
#             continue
#         try:
#             key = int(item["id"])
#         except Exception:
#             continue
#         mapping[key] = item
#     return mapping


# def run_full_evaluation_over_snapshots(
#     dataset_json_path: Path,
#     snapshots_root: Path,
#     output_root: str,
#     methods: List[str],
#     k: int = 8,
#     max_snapshots: Optional[int] = None,
#     verbose: bool = True,
# ) -> Dict[str, List[Dict]]:
#     with dataset_json_path.open("r", encoding="utf-8") as f:
#         dataset_list = json.load(f)

#     if not isinstance(dataset_list, list):
#         raise ValueError(f"Expected top-level list in {dataset_json_path}, got {type(dataset_list)}")

#     id_to_item = build_id_to_item_map(dataset_list)
#     print(f"[info] Loaded evaluation dataset with {len(dataset_list)} entries from {dataset_json_path}")
#     print(f"[info] Built id-to-item map with {len(id_to_item)} IDs.")

#     snapshot_dirs = discover_snapshots(snapshots_root, max_snapshots=max_snapshots)
#     if not snapshot_dirs:
#         return {m: [] for m in methods}

#     ent_embedder = HFEmbedder(ENT_EMBED_MODEL, DEVICE)
#     rel_embedder = HFEmbedder(REL_EMBED_MODEL, DEVICE)
#     query_embedder = ent_embedder

#     all_summaries: Dict[str, List[Dict]] = {m: [] for m in methods}

#     for snapshot_label, snap_dir in snapshot_dirs:
#         try:
#             dataset_id = int(snapshot_label)
#         except Exception:
#             print(f"\n=== Snapshot {snapshot_label} ===")
#             print(f"[warn] Cannot parse snapshot label '{snapshot_label}' as int; skipping (no dataset id).")
#             continue

#         item = id_to_item.get(dataset_id)
#         if item is None:
#             print(f"\n=== Snapshot {snapshot_label} ===")
#             print(f"[warn] No dataset entry with id={dataset_id}; skipping.")
#             continue

#         queries: List[str] = item.get("generated_queries", [])
#         if not queries:
#             print(f"\n=== Snapshot {snapshot_label} (dataset_id={dataset_id}) ===")
#             print("[info] Skipping: no 'generated_queries' in dataset item.")
#             continue

#         print(f"\n=== Snapshot {snapshot_label} (dataset_id={dataset_id}) ===")
#         print(f"[info] Snapshot dir: {snap_dir}")
#         print(f"[info] #queries: {len(queries)}")

#         nodes_csv = snap_dir / "KG" / "nodes.csv"
#         rels_csv = snap_dir / "KG" / "rels_fixed_no_raw.csv"

#         print(f"[info] Using TRACE-KG nodes: {nodes_csv}")
#         print(f"[info] Using TRACE-KG rels : {rels_csv}")

#         nodes_df = pd.read_csv(nodes_csv)
#         rels_df = pd.read_csv(rels_csv)

#         print(f"[info] nodes_df rows: {len(nodes_df)}, rels_df rows: {len(rels_df)}")

#         trace_node_embs, _ = compute_weighted_entity_embeddings(ent_embedder, nodes_df, ENT_WEIGHTS)
#         trace_rel_embs, _ = compute_weighted_relation_embeddings(rel_embedder, rels_df, nodes_df, REL_EMB_WEIGHTS)
#         trace_graph, trace_node_info = build_tracekg_nx_and_nodeinfo(nodes_df, rels_df)
#         trace_retriever = WeightedGraphRetriever(trace_node_embs, trace_graph, node_info=trace_node_info)

#         if "tracekg" in methods:
#             summaries_dir = os.path.join(output_root, "tracekg")
#             s = evaluate_accuracy_for_graph(
#                 query_embedder=query_embedder,
#                 retriever=trace_retriever,
#                 queries=queries,
#                 method_name="tracekg",
#                 snapshot_label=snapshot_label,
#                 dataset_id=dataset_id,
#                 results_dir=summaries_dir,
#                 k=k,
#                 verbose=verbose,
#             )
#             all_summaries["tracekg"].append(s)

#         for method in methods:
#             if method == "tracekg":
#                 continue

#             kg_key = None
#             if method == "kggen":
#                 kg_key = "kggen"
#             elif method == "graphrag":
#                 kg_key = "graphrag_kg"
#             elif method == "openie":
#                 kg_key = "openie_kg"
#             else:
#                 continue

#             kg_data = item.get(kg_key, None)
#             if kg_data is None:
#                 if verbose:
#                     print(f"  [{method}] No KG data under key '{kg_key}' for dataset_id={dataset_id}, skipping.")
#                 continue

#             sg = SimpleGraph.from_kggen_dict(kg_data)
#             g_nx = sg.to_nx()

#             node_ids = list(g_nx.nodes())
#             if not node_ids:
#                 if verbose:
#                     print(f"  [{method}] Empty KG for dataset_id={dataset_id}, skipping.")
#                 continue

#             node_texts = [str(n) for n in node_ids]
#             node_embs_arr = query_embedder.encode_batch(node_texts)
#             node_embs = {nid: node_embs_arr[i] for i, nid in enumerate(node_ids)}
#             retriever = WeightedGraphRetriever(node_embs, g_nx, node_info=None)

#             summaries_dir = os.path.join(output_root, method)
#             s = evaluate_accuracy_for_graph(
#                 query_embedder=query_embedder,
#                 retriever=retriever,
#                 queries=queries,
#                 method_name=method,
#                 snapshot_label=snapshot_label,
#                 dataset_id=dataset_id,
#                 results_dir=summaries_dir,
#                 k=k,
#                 verbose=verbose,
#             )
#             all_summaries[method].append(s)

#     return all_summaries


# # ============================================================
# # 9. Main
# # ============================================================

# def main():
#     # methods = ["kggen", "graphrag", "openie", "tracekg"]
#     # Or only TRACE-KG:
#     methods = ["tracekg"]

#     os.makedirs(OUTPUT_ROOT, exist_ok=True)

#     all_summaries = run_full_evaluation_over_snapshots(
#         dataset_json_path=DATASET_JSON_PATH,
#         snapshots_root=KG_SNAPSHOTS_ROOT,
#         output_root=OUTPUT_ROOT,
#         methods=methods,
#         k=8,
#         max_snapshots=MAX_SNAPSHOTS,
#         verbose=False,
#     )

#     comparison = compare_methods(all_summaries)
#     print_per_snapshot_table(all_summaries, methods)
#     print_comparison_table(comparison)


# if __name__ == "__main__":
#     main()
    
    

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





#?######################### Start ##########################
#region:#?   QA4Methods - V11 (MINE-1 closer: induced subgraph retrieval + strict judge, no heuristics)




# import os
# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Set, Any

# import numpy as np
# import pandas as pd
# import networkx as nx

# from datasets import load_dataset  # not strictly required if you only use JSON dump
# import torch
# from transformers import AutoTokenizer, AutoModel
# from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity

# from openai import OpenAI

# # ============================================================
# # 0. Global config: weights, models, env
# # ============================================================

# ENT_WEIGHTS = {
#     "name": 0.40,
#     "desc": 0.25,
#     "ctx": 0.35,
# }

# REL_EMB_WEIGHTS = {
#     "name": 0.25,
#     "desc+Q": 0.15,
#     "head_tail": 0.20,
#     "ctx": 0.40,
# }

# ENT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# REL_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# BATCH_SIZE = 32
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OPENAI_MODEL_JUDGE = "gpt-5.1"
# OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# # Paths
# DATASET_JSON_PATH = Path("Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset.json")
# KG_SNAPSHOTS_ROOT = Path("Experiments/MYNE/Ex1/KGs_from_Essays_KFE_test")
# OUTPUT_ROOT = "Experiments/MYNE/Ex1/RES/tracekg_mine_results_weighted_openai_v11_inducedsubgraph_strictjudge"

# # Limit how many snapshots to run (None = all)
# MAX_SNAPSHOTS: Optional[int] = None  # e.g., 3 for just the first 3 discovered

# # Retrieval params (MINE-1-ish)
# TOP_K_NODES = 8 #todo make sure what is a good k here!
# HOPS = 2

# # Context caps (to avoid huge prompts)
# MAX_CONTEXT_NODES = 250
# MAX_CONTEXT_EDGES = 300

# # Optional: log potential verbatim leakage
# LOG_VERBATIM_FACT_IN_CONTEXT = True


# # ============================================================
# # 1. Env helper
# # ============================================================

# def _load_openai_key(
#     envvar: str = OPENAI_API_KEY_ENV,
#     fallback_path: str = ".env",
# ):
#     key = os.getenv(envvar, None)
#     if key:
#         return key
#     if Path(fallback_path).exists():
#         txt = Path(fallback_path).read_text(encoding="utf-8").strip()
#         if txt:
#             return txt
#     return None


# # ============================================================
# # 2. HF Embedder
# # ============================================================

# def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
#     masked = token_embeds * mask
#     summed = masked.sum(dim=1)
#     counts = mask.sum(dim=1).clamp(min=1e-9)
#     return summed / counts


# class HFEmbedder:
#     def __init__(self, model_name: str, device: str = DEVICE):
#         # print(f"[Embedder] loading model {model_name} on {device} ...")
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#         self.model = AutoModel.from_pretrained(model_name)
#         self.model.to(device)
#         self.model.eval()
#         for p in self.model.parameters():
#             p.requires_grad = False

#     @torch.no_grad()
#     def encode_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
#         embs = []
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]
#             enc = self.tokenizer(
#                 batch,
#                 padding=True,
#                 truncation=False,
#                 return_tensors="pt",
#                 max_length=1024,
#             )
#             input_ids = enc["input_ids"].to(self.device)
#             attention_mask = enc["attention_mask"].to(self.device)
#             out = self.model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 return_dict=True,
#             )
#             token_embeds = out.last_hidden_state
#             pooled = mean_pool(token_embeds, attention_mask)
#             pooled = pooled.cpu().numpy()
#             embs.append(pooled)
#         embs = np.vstack(embs)
#         embs = normalize(embs, axis=1)
#         return embs


# # ============================================================
# # 3. Baseline SimpleGraph
# # ============================================================

# @dataclass
# class SimpleGraph:
#     entities: Set[str]
#     relations: Set[Tuple[str, str, str]]

#     @staticmethod
#     def from_kggen_dict(d: Dict) -> "SimpleGraph":
#         entities = set(d.get("entities", []))
#         rels_raw = d.get("relations", [])
#         relations = set()
#         for r in rels_raw:
#             if isinstance(r, (list, tuple)) and len(r) == 3:
#                 s, rel, t = r
#                 relations.add((str(s), str(rel), str(t)))
#         return SimpleGraph(entities=entities, relations=relations)

#     def to_nx(self) -> nx.DiGraph:
#         g = nx.DiGraph()
#         for e in self.entities:
#             g.add_node(e, text=str(e))
#         for s, rel, t in self.relations:
#             g.add_edge(s, t, relation=str(rel))
#         return g


# # ============================================================
# # 4. TRACE-KG utilities
# # ============================================================

# def safe_str(x) -> str:
#     if x is None:
#         return ""
#     if isinstance(x, float) and np.isnan(x):
#         return ""
#     return str(x)


# def build_tracekg_entity_texts(nodes_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
#     ids, names, descs, ctxs = [], [], [], []
#     for _, row in nodes_df.iterrows():
#         ent_id = safe_str(row["entity_id"])
#         ids.append(ent_id)

#         name_txt = safe_str(row.get("entity_name", ""))
#         desc_txt = safe_str(row.get("entity_description", ""))

#         cls_label = safe_str(row.get("class_label", ""))
#         cls_group = safe_str(row.get("class_group", ""))
#         node_props = safe_str(row.get("node_properties", ""))

#         ctx_parts = []
#         if cls_label:
#             ctx_parts.append(f"[CLASS:{cls_label}]")
#         if cls_group:
#             ctx_parts.append(f"[GROUP:{cls_group}]")
#         if node_props:
#             ctx_parts.append(f"[PROPS:{node_props}]")
#         ctx_txt = " ; ".join(ctx_parts)

#         names.append(name_txt)
#         descs.append(desc_txt)
#         ctxs.append(ctx_txt)
#     return ids, names, descs, ctxs


# def compute_weighted_entity_embeddings(
#     embedder: HFEmbedder,
#     nodes_df: pd.DataFrame,
#     weights: Dict[str, float] = ENT_WEIGHTS,
# ) -> Tuple[Dict[str, np.ndarray], int]:
#     ids, names, descs, ctxs = build_tracekg_entity_texts(nodes_df)

#     # print("[TRACE-ENT] encoding name field ...")
#     emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None

#     # print("[TRACE-ENT] encoding desc field ...")
#     emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None

#     # print("[TRACE-ENT] encoding ctx field ...")
#     emb_ctx = embedder.encode_batch(ctxs) if any(t.strip() for t in ctxs) else None

#     D_ref = None
#     for arr in [emb_name, emb_desc, emb_ctx]:
#         if arr is not None:
#             D_ref = arr.shape[1]
#             break
#     if D_ref is None:
#         raise ValueError("All entity fields empty — cannot embed.")

#     def _ensure(arr):
#         if arr is None:
#             return np.zeros((len(ids), D_ref))
#         if arr.shape[1] != D_ref:
#             raise ValueError("Entity embedding dimension mismatch.")
#         return arr

#     emb_name = _ensure(emb_name)
#     emb_desc = _ensure(emb_desc)
#     emb_ctx = _ensure(emb_ctx)

#     w_name = weights.get("name", 0.0)
#     w_desc = weights.get("desc", 0.0)
#     w_ctx = weights.get("ctx", 0.0)
#     Wsum = w_name + w_desc + w_ctx
#     if Wsum <= 0:
#         raise ValueError("Sum of ENT_WEIGHTS must be > 0")
#     w_name /= Wsum
#     w_desc /= Wsum
#     w_ctx /= Wsum

#     combined = (w_name * emb_name) + (w_desc * emb_desc) + (w_ctx * emb_ctx)
#     combined = normalize(combined, axis=1)

#     node_embs: Dict[str, np.ndarray] = {}
#     for i, node_id in enumerate(ids):
#         node_embs[node_id] = combined[i]
#     return node_embs, D_ref


# def build_tracekg_relation_texts(
#     rels_df: pd.DataFrame,
#     nodes_df: pd.DataFrame,
# ) -> Tuple[List[str], Dict[str, int], Dict[int, Dict[str, str]]]:
#     node_info = {}
#     for _, nrow in nodes_df.iterrows():
#         nid = safe_str(nrow["entity_id"])
#         node_info[nid] = {
#             "name": safe_str(nrow.get("entity_name", "")),
#             "class_label": safe_str(nrow.get("class_label", "")),
#         }

#     relation_ids: List[str] = []
#     id_to_index: Dict[str, int] = {}
#     texts: Dict[int, Dict[str, str]] = {}

#     for i, row in rels_df.iterrows():
#         rid = safe_str(row.get("relation_id", i))
#         relation_ids.append(rid)
#         id_to_index[rid] = i

#         start_id = safe_str(row.get("start_id", ""))
#         end_id = safe_str(row.get("end_id", ""))

#         rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))

#         rel_desc = safe_str(row.get("rel_desc", "")) or safe_str(row.get("canonical_rel_desc", ""))
#         qualifiers = safe_str(row.get("qualifiers", ""))
#         desc_plus_q = " ; ".join([p for p in [rel_desc, qualifiers] if p])

#         head_info = node_info.get(start_id, {})
#         tail_info = node_info.get(end_id, {})
#         head_tail_parts = []
#         if head_info.get("name"):
#             head_tail_parts.append(f"[H:{head_info['name']}]")
#         if head_info.get("class_label"):
#             head_tail_parts.append(f"[HCLS:{head_info['class_label']}]")
#         if tail_info.get("name"):
#             head_tail_parts.append(f"[T:{tail_info['name']}]")
#         if tail_info.get("class_label"):
#             head_tail_parts.append(f"[TCLS:{tail_info['class_label']}]")
#         head_tail = " ".join(head_tail_parts)

#         canonical_name = safe_str(row.get("canonical_rel_name", ""))
#         canonical_desc = safe_str(row.get("canonical_rel_desc", ""))
#         rel_cls = safe_str(row.get("rel_cls", ""))
#         ctx_parts = []
#         if canonical_name:
#             ctx_parts.append(canonical_name)
#         if canonical_desc:
#             ctx_parts.append(canonical_desc)
#         if rel_cls:
#             ctx_parts.append(f"[CLS:{rel_cls}]")
#         ctx_txt = " ; ".join(ctx_parts)

#         texts[i] = {
#             "name": rel_name,
#             "desc+Q": desc_plus_q,
#             "head_tail": head_tail,
#             "ctx": ctx_txt,
#         }

#     return relation_ids, id_to_index, texts


# def compute_weighted_relation_embeddings(
#     embedder: HFEmbedder,
#     rels_df: pd.DataFrame,
#     nodes_df: pd.DataFrame,
#     weights: Dict[str, float] = REL_EMB_WEIGHTS,
# ) -> Tuple[Dict[str, np.ndarray], int]:
#     rel_ids, id_to_index, texts = build_tracekg_relation_texts(rels_df, nodes_df)

#     n = len(rel_ids)
#     buckets = ["name", "desc+Q", "head_tail", "ctx"]
#     bucket_texts = {b: [""] * n for b in buckets}
#     for idx in range(n):
#         for b in buckets:
#             bucket_texts[b][idx] = texts[idx].get(b, "")

#     emb_bucket = {}
#     D_ref = None
#     for b in buckets:
#         has_any = any(t.strip() for t in bucket_texts[b])
#         if not has_any:
#             emb_bucket[b] = None
#             continue
#         # print(f"[TRACE-REL] encoding bucket '{b}' ...")
#         eb = embedder.encode_batch(bucket_texts[b])
#         emb_bucket[b] = eb
#         if D_ref is None:
#             D_ref = eb.shape[1]

#     if D_ref is None:
#         raise ValueError("All relation buckets empty — cannot embed relations.")

#     def _ensure(arr):
#         if arr is None:
#             return np.zeros((n, D_ref))
#         if arr.shape[1] != D_ref:
#             raise ValueError("Relation embedding dimension mismatch.")
#         return arr

#     for b in buckets:
#         emb_bucket[b] = _ensure(emb_bucket[b])

#     w_name = weights.get("name", 0.0)
#     w_descq = weights.get("desc+Q", 0.0)
#     w_ht = weights.get("head_tail", 0.0)
#     w_ctx = weights.get("ctx", 0.0)
#     Wsum = w_name + w_descq + w_ht + w_ctx
#     if Wsum <= 0:
#         raise ValueError("Sum of REL_EMB_WEIGHTS must be > 0")
#     w_name /= Wsum
#     w_descq /= Wsum
#     w_ht /= Wsum
#     w_ctx /= Wsum

#     combined = (
#         w_name * emb_bucket["name"]
#         + w_descq * emb_bucket["desc+Q"]
#         + w_ht * emb_bucket["head_tail"]
#         + w_ctx * emb_bucket["ctx"]
#     )
#     combined = normalize(combined, axis=1)

#     rel_embs: Dict[str, np.ndarray] = {}
#     for i, rid in enumerate(rel_ids):
#         rel_embs[rid] = combined[i]
#     return rel_embs, D_ref


# def build_tracekg_nx_and_nodeinfo(
#     nodes_df: pd.DataFrame,
#     rels_df: pd.DataFrame,
# ) -> Tuple[nx.DiGraph, Dict[str, Dict[str, str]]]:
#     g = nx.DiGraph()
#     node_info: Dict[str, Dict[str, str]] = {}

#     for _, row in nodes_df.iterrows():
#         nid = safe_str(row["entity_id"])
#         name = safe_str(row.get("entity_name", ""))
#         cls_label = safe_str(row.get("class_label", ""))

#         g.add_node(
#             nid,
#             entity_name=name,
#             entity_description=safe_str(row.get("entity_description", "")),
#             class_label=cls_label,
#             class_group=safe_str(row.get("class_group", "")),
#             node_properties=safe_str(row.get("node_properties", "")),
#             chunk_ids=safe_str(row.get("chunk_ids", "")),
#         )
#         node_info[nid] = {
#             "name": name,
#             "class_label": cls_label,
#         }

#     for _, row in rels_df.iterrows():
#         sid = safe_str(row.get("start_id", ""))
#         eid = safe_str(row.get("end_id", ""))
#         rid = safe_str(row.get("relation_id", ""))
#         rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))
#         qualifiers = safe_str(row.get("qualifiers", ""))
#         g.add_edge(
#             sid,
#             eid,
#             relation=rel_name,
#             relation_id=rid,
#             chunk_id=safe_str(row.get("chunk_id", "")),
#             qualifiers=qualifiers,
#         )
#     return g, node_info


# # ============================================================
# # 5. Retrieval (V11: induced subgraph formatting)
# # ============================================================

# class WeightedGraphRetriever:
#     def __init__(
#         self,
#         node_embeddings: Dict[str, np.ndarray],
#         graph: nx.DiGraph,
#         node_info: Optional[Dict[str, Dict[str, str]]] = None,
#     ):
#         self.node_embeddings = node_embeddings
#         self.graph = graph
#         self.node_info = node_info or {}

#     def retrieve_relevant_nodes(
#         self,
#         query_emb: np.ndarray,
#         k: int = TOP_K_NODES,
#     ) -> List[Tuple[str, float]]:
#         sims = []
#         for node, emb in self.node_embeddings.items():
#             sim = cosine_similarity(query_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
#             sims.append((node, sim))
#         sims.sort(key=lambda x: x[1], reverse=True)
#         return sims[:k]

#     def expand_nodes_within_hops(self, seed_nodes: List[str], hops: int = HOPS) -> Set[str]:
#         """
#         Expand using undirected hop distance (matches 'within two relations' more closely).
#         """
#         und = self.graph.to_undirected(as_view=True)
#         expanded: Set[str] = set(seed_nodes)

#         for s in seed_nodes:
#             if s not in und:
#                 continue
#             lengths = nx.single_source_shortest_path_length(und, s, cutoff=hops)
#             expanded.update(lengths.keys())

#         return expanded

#     def induced_subgraph_edges(self, nodes: Set[str]) -> List[Tuple[str, str, Dict[str, Any]]]:
#         edges: List[Tuple[str, str, Dict[str, Any]]] = []
#         for u, v, data in self.graph.edges(data=True):
#             if u in nodes and v in nodes:
#                 edges.append((u, v, dict(data)))
#         return edges

#     def format_induced_subgraph(self, nodes: Set[str], edges: List[Tuple[str, str, Dict[str, Any]]]) -> str:
#         node_list = list(nodes)[:MAX_CONTEXT_NODES]
#         edge_list = edges[:MAX_CONTEXT_EDGES]

#         lines: List[str] = []
#         lines.append("SUBGRAPH NODES:")
#         for nid in node_list:
#             info = self.node_info.get(nid, {})
#             name = (info.get("name") or str(nid)).strip()
#             cls = (info.get("class_label") or "").strip()
#             if cls:
#                 # lines.append(f"- {nid} | {name} | class={cls}")
#                 lines.append(f"{name}")
#             else:
#                 # lines.append(f"- {nid} | {name}")
#                 lines.append(f"{name}")

#         lines.append("")
#         lines.append("SUBGRAPH EDGES (directed):")
#         # for u, v, data in edge_list:
#         #     rel = str(data.get("relation", "")).strip()
#         #     rid = str(data.get("relation_id", "")).strip()
#         #     q = str(data.get("qualifiers", "")).strip()
#         #     if q and q.lower() != "nan":
#         #         # lines.append(f"- ({u}) -[{rel} | id={rid} | qualifiers={q}]-> ({v})")
#         #         lines.append(f"- ({u}) -[{rel} | id={rid} | qualifiers={q}]-> ({v})")
#         #     else:
#         #         # lines.append(f"- ({u}) -[{rel} | id={rid}]-> ({v})")
#         #         lines.append(f"- ({u}) -[{rel} | id={rid}]-> ({v})")
        
#         for u, v, data in edge_list:    
#             rel = str(data.get("relation", "")).strip()
#             rid = str(data.get("relation_id", "")).strip()
#             q = str(data.get("qualifiers", "")).strip()

#             u_label = self._node_label(u)
#             v_label = self._node_label(v)

#             if q and q.lower() != "nan":
#                 lines.append(f"- ({u_label}) -[{rel} | qualifiers={q}]-> ({v_label})")
#             else:
#                 lines.append(f"- ({u_label}) -[{rel}]-> ({v_label})")

#         return "\n".join(lines)

#     def retrieve_induced_subgraph_context(
#         self,
#         query_emb: np.ndarray,
#         k: int = TOP_K_NODES,
#         hops: int = HOPS,
#     ) -> Tuple[List[Tuple[str, float]], Set[str], List[Tuple[str, str, Dict[str, Any]]], str]:
#         top_nodes = self.retrieve_relevant_nodes(query_emb, k=k)
#         seed = [nid for nid, _ in top_nodes]
#         expanded_nodes = self.expand_nodes_within_hops(seed, hops=hops)
#         edges = self.induced_subgraph_edges(expanded_nodes)
#         context_text = self.format_induced_subgraph(expanded_nodes, edges)
#         return top_nodes, expanded_nodes, edges, context_text
    
#     def _node_label(self, nid: str) -> str:
#         info = self.node_info.get(nid, {})
#         name = (info.get("name") or str(nid)).strip()
#         cls = (info.get("class_label") or "").strip()
#         # names only:
#         return name


# # ============================================================
# # 6. LLM evaluator (V11: strict, temp=0, no heuristics)
# # ============================================================

# _openai_client: Optional[OpenAI] = None

# def _get_openai_client() -> OpenAI:
#     global _openai_client
#     if _openai_client is not None:
#         return _openai_client
#     api_key = _load_openai_key()
#     if not api_key:
#         raise RuntimeError(
#             "OpenAI API key not found. Set OPENAI_API_KEY env var or provide it in .env"
#         )
#     _openai_client = OpenAI(api_key=api_key)
#     return _openai_client


# def _normalize_ws(s: str) -> str:
#     return " ".join((s or "").split()).strip()


# def contains_full_fact_verbatim(fact: str, context: str) -> bool:
#     f = _normalize_ws(fact).lower()
#     c = _normalize_ws(context).lower()
#     if not f:
#         return False
#     return f in c


# def gpt_evaluate_response_strict(correct_answer: str, context: str) -> int:
#     """
#     Strict binary judge:
#     - temperature=0
#     - no fallback heuristics
#     - "no world knowledge" constraint
#     """
#     client = _get_openai_client()

#     system_prompt = (
#         "You are a strict evaluator for a knowledge-graph retention benchmark.\n"
#         "You will be given:\n"
#         "1) A FACT statement.\n"
#         "2) A SUBGRAPH (nodes + directed edges) retrieved from a KG.\n\n"
#         "Decide whether the FACT can be supported or inferred using ONLY the provided SUBGRAPH.\n"
#         "- Do NOT use external or world knowledge.\n"
#         "- Do NOT assume missing edges.\n"
#         "- If the subgraph is insufficient or ambiguous, answer 0.\n\n"
#         "Output format: return exactly one character: 1 or 0."
#     )

#     user_prompt = (
#         "FACT:\n"
#         f"{correct_answer}\n\n"
#         "SUBGRAPH:\n"
#         f"{context}\n\n"
#         "Can the FACT be supported or inferred from the SUBGRAPH alone?\n"
#         "Answer with exactly: 1 or 0."
#     )

#     try:
#         resp = client.responses.create(
#             model=OPENAI_MODEL_JUDGE,
#             input=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             temperature=0,
#             max_output_tokens=64,
#         )
#         text = resp.output[0].content[0].text.strip()
#     except Exception as e:
#         print(f"[judge] Error calling OpenAI: {e}")
#         return 0

#     if text == "1":
#         return 1
#     if text == "0":
#         return 0

#     # Strict: anything else is a fail (0)
#     return 0


# # ============================================================
# # 7. Evaluation helpers
# # ============================================================

# def evaluate_accuracy_for_graph(
#     query_embedder: HFEmbedder,
#     retriever: WeightedGraphRetriever,
#     queries: List[str],
#     method_name: str,
#     snapshot_label: str,
#     dataset_id: int,
#     results_dir: str,
#     k: int = TOP_K_NODES,
#     verbose: bool = False,
# ) -> Dict:
#     os.makedirs(results_dir, exist_ok=True)

#     # print(f"[{method_name}] encoding {len(queries)} facts "
#           # f"(snapshot={snapshot_label}, dataset_id={dataset_id}) ...")
#     query_embs = query_embedder.encode_batch(queries)

#     correct = 0
#     results = []
#     verbatim_hits = 0

#     for qi, fact in enumerate(queries):
#         q_emb = query_embs[qi]

#         top_nodes, expanded_nodes, edges, context_text = retriever.retrieve_induced_subgraph_context(
#             q_emb, k=k, hops=HOPS
#         )

#         if LOG_VERBATIM_FACT_IN_CONTEXT and contains_full_fact_verbatim(fact, context_text):
#             verbatim_hits += 1

#         evaluation = gpt_evaluate_response_strict(fact, context_text)

#         results.append(
#             {
#                 "correct_answer": fact,
#                 "retrieved_context": context_text,
#                 "evaluation": int(evaluation),
#                 "top_nodes": [{"id": nid, "sim": float(sim)} for nid, sim in top_nodes],
#                 "num_expanded_nodes": int(len(expanded_nodes)),
#                 "num_induced_edges": int(len(edges)),
#                 "verbatim_fact_in_context": bool(contains_full_fact_verbatim(fact, context_text)),
#             }
#         )
#         correct += evaluation

#     accuracy = correct / len(queries) if queries else 0.0
#     results.append(
#         {
#             "accuracy": f"{accuracy * 100:.2f}%",
#             "verbatim_fact_in_context_count": int(verbatim_hits),
#             "verbatim_fact_in_context_rate": float(verbatim_hits / len(queries)) if queries else 0.0,
#             "k": int(k),
#             "hops": int(HOPS),
#             "max_context_nodes": int(MAX_CONTEXT_NODES),
#             "max_context_edges": int(MAX_CONTEXT_EDGES),
#             "judge_model": OPENAI_MODEL_JUDGE,
#             "judge_temperature": 0,
#         }
#     )

#     out_path = os.path.join(results_dir, f"results_{snapshot_label}.json")
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)

#     if verbose:
#         print(
#             f"[{method_name}] Snapshot={snapshot_label} (dataset_id={dataset_id}): "
#             f"accuracy={accuracy:.4f} ({correct}/{len(queries)}) | "
#             f"verbatim_hits={verbatim_hits}/{len(queries)}"
#         )

#     return {
#         "accuracy": accuracy,
#         "num_queries": len(queries),
#         "method": method_name,
#         "snapshot_label": snapshot_label,
#         "dataset_id": dataset_id,
#         "verbatim_hits": verbatim_hits,
#     }


# def aggregate_method_stats(summaries: List[Dict]) -> Dict[str, float]:
#     if not summaries:
#         return {"mean_accuracy": 0.0, "num_essays": 0}
#     accs = [s["accuracy"] for s in summaries]
#     return {
#         "mean_accuracy": float(np.mean(accs)),
#         "num_essays": len(accs),
#     }


# def compare_methods(all_summaries: Dict[str, List[Dict]]) -> Dict[str, Dict]:
#     return {m: aggregate_method_stats(s) for m, s in all_summaries.items()}


# def print_comparison_table(comparison: Dict[str, Dict]):
#     print("\n=== Method Comparison (Mean Accuracy across evaluated snapshots) ===")
#     print(f"{'Method':<10} | {'Mean Acc':>8} | {'#Snaps':>7}")
#     print("-" * 36)
#     for m, stats in comparison.items():
#         print(
#             f"{m:<10} | {stats['mean_accuracy']*100:8.2f}% | "
#             f"{stats['num_essays']:7d}"
#         )


# def print_per_snapshot_table(all_summaries: Dict[str, List[Dict]], methods: List[str]):
#     """
#     Print a per-snapshot table with one row per snapshot and one accuracy column per method.
#     Shows 'xx.xx% (correct/total)' for each available method.
#     """
#     by_snap: Dict[str, Dict[str, Dict]] = {}
#     for method, summaries in all_summaries.items():
#         for s in summaries:
#             snap = s.get("snapshot_label", "UNKNOWN")
#             d = by_snap.setdefault(snap, {})
#             d[method] = s

#     if not by_snap:
#         print("\n[INFO] No per-snapshot summaries to report.")
#         return

#     header = f"{'Snapshot':>10} | {'DatasetID':>9}"
#     for m in methods:
#         header += f" | {m:^20}"
#     print("\n=== Per-Snapshot Accuracy (all methods) ===")
#     print(header)
#     print("-" * len(header))

#     for snap in sorted(by_snap.keys()):
#         row_methods = by_snap[snap]
#         any_summary = next(iter(row_methods.values()))
#         dataset_id = any_summary.get("dataset_id", -1)

#         line = f"{snap:>10} | {dataset_id:9d}"
#         for m in methods:
#             s = row_methods.get(m)
#             if s is None:
#                 cell = "N/A"
#             else:
#                 acc = s["accuracy"]
#                 n = s["num_queries"]
#                 correct = int(round(acc * n))
#                 cell = f"{acc*100:5.2f}% ({correct}/{n})"
#             line += f" | {cell:>20}"
#         print(line)


# # ============================================================
# # 8. Snapshot discovery and evaluation
# # ============================================================

# def discover_snapshots(root: Path, max_snapshots: Optional[int]) -> List[Tuple[str, Path]]:
#     candidates: List[Tuple[str, Path]] = []
#     for p in sorted(root.glob("KG_Essay_*")):
#         if not p.is_dir():
#             continue
#         snapshot_label = p.name.replace("KG_Essay_", "")

#         nodes_csv = p / "KG" / "nodes.csv"
#         rels_csv = p / "KG" / "rels_fixed_no_raw.csv"
#         if not nodes_csv.exists() or not rels_csv.exists():
#             print(f"[warn] Missing KG CSVs in {p}, skipping.")
#             continue

#         candidates.append((snapshot_label, p))

#     candidates.sort(key=lambda x: x[0])

#     if max_snapshots is not None:
#         candidates = candidates[:max_snapshots]

#     print(f"[info] Discovered {len(candidates)} usable KG snapshots under {root}:")
#     for label, path in candidates:
#         print(f"  - Snapshot {label}: {path}")
#     return candidates


# def build_id_to_item_map(dataset: List[Dict]) -> Dict[int, Dict]:
#     mapping: Dict[int, Dict] = {}
#     for item in dataset:
#         if not isinstance(item, dict):
#             continue
#         if "id" not in item:
#             continue
#         try:
#             key = int(item["id"])
#         except Exception:
#             continue
#         mapping[key] = item
#     return mapping

# def validate_id_alignment(
#     dataset: List[Dict],
#     snapshots: List[Tuple[str, Path]],
#     methods: List[str],
# ) -> None:
#     """
#     Pre-flight check: ensure that for every snapshot we're about to evaluate,
#     the essay ID derived from the snapshot folder name exists in the dataset,
#     and every requested baseline method has KG data under that same ID.

#     Raises ValueError with a detailed message if any mismatch is found.
#     """
#     id_map = build_id_to_item_map(dataset)

#     METHOD_TO_KEY = {
#         "kggen":        "kggen",
#         "graphrag":     "graphrag_kg",
#         "openie":       "openie_kg",
#         "autoschemakg": "autoschemakg",
#     }

#     errors = []

#     for snapshot_label, snapshot_path in snapshots:
#         # Parse dataset_id the same way the evaluation loop does
#         try:
#             dataset_id = int(snapshot_label.lstrip("0") or "0")
#         except ValueError:
#             errors.append(
#                 f"  Snapshot '{snapshot_label}': cannot parse integer ID from folder name"
#             )
#             continue

#         # Check 1: does this ID exist in the dataset at all?
#         item = id_map.get(dataset_id)
#         if item is None:
#             errors.append(
#                 f"  Snapshot '{snapshot_label}' → dataset_id={dataset_id}: "
#                 f"NOT FOUND in mine_evaluation_dataset.json"
#             )
#             continue

#         # Check 2: does the item have generated_queries?
#         if not item.get("generated_queries"):
#             errors.append(
#                 f"  Snapshot '{snapshot_label}' → dataset_id={dataset_id}: "
#                 f"has no 'generated_queries' — nothing to evaluate"
#             )

#         # Check 3: for each baseline method requested, does the dataset item
#         #           have KG data under the expected key?
#         for method in methods:
#             if method == "tracekg":
#                 # TRACE KG comes from the snapshot folder itself — already validated
#                 # by discover_snapshots (checks nodes.csv + rels_fixed_no_raw.csv exist)
#                 continue

#             kg_key = METHOD_TO_KEY.get(method)
#             if kg_key is None:
#                 continue

#             kg_data = item.get(kg_key)
#             if kg_data is None:
#                 errors.append(
#                     f"  Snapshot '{snapshot_label}' → dataset_id={dataset_id}: "
#                     f"method '{method}' expects key '{kg_key}' but it is MISSING"
#                 )
#             elif not kg_data.get("relations"):
#                 errors.append(
#                     f"  Snapshot '{snapshot_label}' → dataset_id={dataset_id}: "
#                     f"method '{method}' key '{kg_key}' exists but has NO relations"
#                 )

#     if errors:
#         msg = (
#             "\n╔══════════════════════════════════════════════════════════╗\n"
#             "║  ID ALIGNMENT VALIDATION FAILED                        ║\n"
#             "╚══════════════════════════════════════════════════════════╝\n"
#             "The following mismatches were detected BEFORE running evaluation.\n"
#             "This means some methods would be evaluated on the wrong essay.\n\n"
#             + "\n".join(errors)
#             + "\n\nFix the data sources so all methods align on the same essay IDs."
#         )
#         raise ValueError(msg)

#     print(f"[✓] ID alignment validated: {len(snapshots)} snapshots × {len(methods)} methods — all consistent.")
    

# def validate_id_alignment(
#     dataset: List[Dict],
#     snapshots: List[Tuple[str, Path]],
#     methods: List[str],
# ) -> None:
#     """
#     Pre-flight check: for every essay ID we intend to evaluate, verify that:
#       1. The ID exists in the evaluation dataset
#       2. The dataset item has generated_queries
#       3. Every baseline method has KG data under the correct key
#       4. TRACE KG snapshot has the required CSV files
#     Raises ValueError with a detailed message on any mismatch.
#     """
#     id_map = build_id_to_item_map(dataset)

#     METHOD_TO_KEY = {
#         "kggen":        "kggen",
#         "graphrag":     "graphrag_kg",
#         "openie":       "openie_kg",
#         "autoschemakg": "autoschemakg",
#     }

#     errors = []

#     for snapshot_label, snapshot_path in snapshots:
#         try:
#             dataset_id = int(snapshot_label.lstrip("0") or "0")
#         except ValueError:
#             errors.append(f"  Snapshot '{snapshot_label}': cannot parse integer ID from folder name")
#             continue

#         # 1. ID exists in dataset?
#         item = id_map.get(dataset_id)
#         if item is None:
#             errors.append(
#                 f"  essay_id={dataset_id}: NOT FOUND in mine_evaluation_dataset.json"
#             )
#             continue

#         # 2. Has queries?
#         if not item.get("generated_queries"):
#             errors.append(
#                 f"  essay_id={dataset_id}: has no 'generated_queries' — nothing to evaluate"
#             )

#         # 3. Each method has data?
#         for method in methods:
#             if method == "tracekg":
#                 # Check snapshot CSVs exist
#                 nodes_csv = snapshot_path / "KG" / "nodes.csv"
#                 rels_csv = snapshot_path / "KG" / "rels_fixed_no_raw.csv"
#                 if not nodes_csv.exists():
#                     errors.append(f"  essay_id={dataset_id}: TRACE KG missing {nodes_csv}")
#                 if not rels_csv.exists():
#                     errors.append(f"  essay_id={dataset_id}: TRACE KG missing {rels_csv}")
#                 continue

#             kg_key = METHOD_TO_KEY.get(method)
#             if kg_key is None:
#                 continue

#             kg_data = item.get(kg_key)
#             if kg_data is None:
#                 errors.append(
#                     f"  essay_id={dataset_id}: method '{method}' expects key '{kg_key}' but it is MISSING"
#                 )
#             elif not kg_data.get("relations"):
#                 errors.append(
#                     f"  essay_id={dataset_id}: method '{method}' key '{kg_key}' exists but has NO relations"
#                 )

#     if errors:
#         msg = (
#             "\n╔══════════════════════════════════════════════════════════╗\n"
#             "║  ID ALIGNMENT VALIDATION FAILED                        ║\n"
#             "╚══════════════════════════════════════════════════════════╝\n"
#             "The following mismatches were detected BEFORE running evaluation.\n"
#             "This means some methods would be evaluated on the wrong essay.\n\n"
#             + "\n".join(errors)
#             + "\n\nFix the data sources so all methods align on the same essay IDs."
#         )
#         raise ValueError(msg)

#     print(f"[✓] ID alignment validated: {len(snapshots)} snapshots × {len(methods)} methods — all consistent.")
    

# def run_full_evaluation_over_snapshots(
#     dataset_json_path: Path,
#     snapshots_root: Path,
#     output_root: str,
#     methods: List[str],
#     k: int = TOP_K_NODES,
#     max_snapshots: Optional[int] = None,
#     verbose: bool = True,
# ) -> Dict[str, List[Dict]]:
#     with dataset_json_path.open("r", encoding="utf-8") as f:
#         dataset_list = json.load(f)

#     if not isinstance(dataset_list, list):
#         raise ValueError(f"Expected top-level list in {dataset_json_path}, got {type(dataset_list)}")

#     id_to_item = build_id_to_item_map(dataset_list)
#     print(f"[info] Loaded evaluation dataset with {len(dataset_list)} entries from {dataset_json_path}")
#     print(f"[info] Built id-to-item map with {len(id_to_item)} IDs.")

#     snapshot_dirs = discover_snapshots(snapshots_root, max_snapshots=max_snapshots)
#     if not snapshot_dirs:
#         return {m: [] for m in methods}

#     ent_embedder = HFEmbedder(ENT_EMBED_MODEL, DEVICE)
#     rel_embedder = HFEmbedder(REL_EMBED_MODEL, DEVICE)
#     query_embedder = ent_embedder

#     all_summaries: Dict[str, List[Dict]] = {m: [] for m in methods}

#     for snapshot_label, snap_dir in snapshot_dirs:
#         try:
#             dataset_id = int(snapshot_label)
#         except Exception:
#             print(f"\n=== Snapshot {snapshot_label} ===")
#             print(f"[warn] Cannot parse snapshot label '{snapshot_label}' as int; skipping (no dataset id).")
#             continue

#         item = id_to_item.get(dataset_id)
#         if item is None:
#             print(f"\n=== Snapshot {snapshot_label} ===")
#             print(f"[warn] No dataset entry with id={dataset_id}; skipping.")
#             continue

#         queries: List[str] = item.get("generated_queries", [])
#         if not queries:
#             print(f"\n=== Snapshot {snapshot_label} (dataset_id={dataset_id}) ===")
#             print("[info] Skipping: no 'generated_queries' in dataset item.")
#             continue

#         print(f"\n=== Snapshot {snapshot_label} (dataset_id={dataset_id}) ===")
#         print(f"[info] Snapshot dir: {snap_dir}")
#         print(f"[info] #facts: {len(queries)}")

#         nodes_csv = snap_dir / "KG" / "nodes.csv"
#         rels_csv = snap_dir / "KG" / "rels_fixed_no_raw.csv"

#         print(f"[info] Using TRACE-KG nodes: {nodes_csv}")
#         print(f"[info] Using TRACE-KG rels : {rels_csv}")

#         nodes_df = pd.read_csv(nodes_csv)
#         rels_df = pd.read_csv(rels_csv)

#         print(f"[info] nodes_df rows: {len(nodes_df)}, rels_df rows: {len(rels_df)}")

#         trace_node_embs, _ = compute_weighted_entity_embeddings(ent_embedder, nodes_df, ENT_WEIGHTS)
#         _trace_rel_embs, _ = compute_weighted_relation_embeddings(rel_embedder, rels_df, nodes_df, REL_EMB_WEIGHTS)  # kept for parity/logging
#         trace_graph, trace_node_info = build_tracekg_nx_and_nodeinfo(nodes_df, rels_df)
#         trace_retriever = WeightedGraphRetriever(trace_node_embs, trace_graph, node_info=trace_node_info)

#         if "tracekg" in methods:
#             summaries_dir = os.path.join(output_root, "tracekg")
#             s = evaluate_accuracy_for_graph(
#                 query_embedder=query_embedder,
#                 retriever=trace_retriever,
#                 queries=queries,
#                 method_name="tracekg",
#                 snapshot_label=snapshot_label,
#                 dataset_id=dataset_id,
#                 results_dir=summaries_dir,
#                 k=k,
#                 verbose=verbose,
#             )
#             all_summaries["tracekg"].append(s)

#         # Baselines (kept as-is per your request; still use induced-subgraph retrieval + strict judge)
#         for method in methods:
#             if method == "tracekg":
#                 continue

#             kg_key = None
#             if method == "kggen":
#                 kg_key = "kggen"
#             elif method == "graphrag":
#                 kg_key = "graphrag_kg"
#             elif method == "openie":
#                 kg_key = "openie_kg"
#             else:
#                 continue

#             kg_data = item.get(kg_key, None)
#             if kg_data is None:
#                 if verbose:
#                     print(f"  [{method}] No KG data under key '{kg_key}' for dataset_id={dataset_id}, skipping.")
#                 continue

#             sg = SimpleGraph.from_kggen_dict(kg_data)
#             g_nx = sg.to_nx()

#             node_ids = list(g_nx.nodes())
#             if not node_ids:
#                 if verbose:
#                     print(f"  [{method}] Empty KG for dataset_id={dataset_id}, skipping.")
#                 continue

#             node_texts = [str(n) for n in node_ids]
#             node_embs_arr = query_embedder.encode_batch(node_texts)
#             node_embs = {nid: node_embs_arr[i] for i, nid in enumerate(node_ids)}
#             retriever = WeightedGraphRetriever(node_embs, g_nx, node_info=None)

#             summaries_dir = os.path.join(output_root, method)
#             s = evaluate_accuracy_for_graph(
#                 query_embedder=query_embedder,
#                 retriever=retriever,
#                 queries=queries,
#                 method_name=method,
#                 snapshot_label=snapshot_label,
#                 dataset_id=dataset_id,
#                 results_dir=summaries_dir,
#                 k=k,
#                 verbose=verbose,
#             )
#             all_summaries[method].append(s)

#     return all_summaries


# # ============================================================
# # 9. Main
# # ============================================================

# def main():
#     methods = ["kggen", "graphrag", "openie", "tracekg"]
#     # Or only TRACE-KG:
#     # methods = ["tracekg"]

#     os.makedirs(OUTPUT_ROOT, exist_ok=True)

#     all_summaries = run_full_evaluation_over_snapshots(
#         dataset_json_path=DATASET_JSON_PATH,
#         snapshots_root=KG_SNAPSHOTS_ROOT,
#         output_root=OUTPUT_ROOT,
#         methods=methods,
#         k=TOP_K_NODES,
#         max_snapshots=MAX_SNAPSHOTS,
#         verbose=True,
#     )

#     comparison = compare_methods(all_summaries)
#     print_per_snapshot_table(all_summaries, methods)
#     print_comparison_table(comparison)


# if __name__ == "__main__":
#     main()

#endregion#?   QA4Methods - V11 (MINE-1 closer: induced subgraph retrieval + strict judge, no heuristics)
#?#########################  End  ##########################


#?######################### Start ##########################
#region:#?   Results from v11

# === Per-Snapshot Accuracy (all methods) ===
#   Snapshot | DatasetID |        kggen         |       graphrag       |        openie        |       tracekg       
# ------------------------------------------------------------------------------------------------------------------
#        001 |         1 |        33.33% (5/15) |        33.33% (5/15) |        53.33% (8/15) |       93.33% (14/15)
#        002 |         2 |        60.00% (9/15) |        26.67% (4/15) |        53.33% (8/15) |      100.00% (15/15)


# === Method Comparison (Mean Accuracy across evaluated snapshots) ===
# Method     | Mean Acc |  #Snaps
# ------------------------------------
# kggen      |    46.67% |       2
# graphrag   |    30.00% |       2
# openie     |    53.33% |       2
# tracekg    |    96.67% |       2





#endregion#? Results from v11
#?#########################  End  ##########################


#endregion#! Experiments 1 - MINE 1 From KG Gen Paper
#!#############################################  End Chapter  ##################################################

  




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
ESSAY_IDS: Optional[List[int]] = None  # e.g., [1, 2] for testing


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




    # # 1. Load essays
    # print(f"\n[1] Loading essays from: {ESSAYS_JSON_PATH}")
    # with open(ESSAYS_JSON_PATH, "r", encoding="utf-8") as f:
    #     essays = json.load(f)
    # print(f"    Loaded {len(essays)} essays.")

    # # 2. Load evaluation dataset
    # print(f"\n[2] Loading evaluation dataset from: {DATASET_JSON_PATH}")
    # with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
    #     dataset = json.load(f)
    # print(f"    Loaded {len(dataset)} evaluation items.")

    # # Build map: dataset_id → index in dataset list
    # id_to_idx = {}
    # for idx, item in enumerate(dataset):
    #     did = item.get("id")
    #     if did is not None:
    #         id_to_idx[int(did)] = idx

    # # Determine which IDs to process
    # target_ids = ESSAY_IDS if ESSAY_IDS else sorted(id_to_idx.keys())
    # print(f"    Will process {len(target_ids)} essay(s): {target_ids[:10]}{'...' if len(target_ids) > 10 else ''}")
        

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


# if __name__ == "__main__":
#     main()











# # Find Essays ids

# import re
# from pathlib import Path

# KFE_ROOT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/Ex1/KGs_from_Essays_KFE")

# def discover_essay_ids_from_kfe(root: Path) -> list[int]:
#     """Scan KGs_from_Essays_KFE for KG_Essay_XXX folders and return sorted list of integer IDs."""
#     ids = []
#     for p in sorted(root.glob("KG_Essay_*")):
#         if not p.is_dir():
#             continue
#         label = p.name.replace("KG_Essay_", "")
#         try:
#             ids.append(int(label))
#         except ValueError:
#             continue
#     return sorted(ids)

# # Use this instead of a hardcoded list:
# ESSAY_IDS = discover_essay_ids_from_kfe(KFE_ROOT)
# print(f"Will generate AutoSchemaKG for essay IDs: {ESSAY_IDS}")





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

ESSAY_IDS: Optional[List[int]] = [1, 2] #, 10, 15, 24, 47, 52, 53, 64, 67, 88, 91] #None      # e.g., [1, 2, 3] for a quick test

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
EVAL_ESSAY_IDS: List[int] = [1, 2  ] #, 10, 15, 24, 47, 52, 53, 64, 67, 88, 91]

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
        verbose=False,
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
#region:#?       Post-hoc Compression & Information-Leakage Analysis for MINE-1 Evaluation


"""
Post-hoc Compression & Information-Leakage Analysis for MINE-1 Evaluation
==========================================================================

Run this AFTER you have already computed the 5-method comparison.
It reads the same data sources and adds compression/leakage metrics.

Usage:
    python TKG_Experiment_1_CompressionAnalysis.py
"""

import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict


# ============================================================
# 1. CONFIG — same paths as your main experiment
# ============================================================

REPO_ROOT = Path(".").resolve()

DATASET_JSON_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/QA_and_OthersAnswers/mine_evaluation_dataset.json"
KG_SNAPSHOTS_ROOT = REPO_ROOT / "Experiments/MYNE/Ex1/KGs_from_Essays_KFE_test"

# The results directory where your 5-method comparison wrote per-method JSON files
RESULTS_ROOT = REPO_ROOT / "Experiments/MYNE/Ex1/RES/tracekg_mine_results_weighted_openai_v11_inducedsubgraph_strictjudge"

# Output
OUTPUT_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis.json"
OUTPUT_TABLE_PATH = REPO_ROOT / "Experiments/MYNE/Ex1/RES/compression_analysis_table.txt"

# Which essay IDs were evaluated (must match your EVAL_ESSAY_IDS)
EVAL_ESSAY_IDS: List[int] = [1, 2] #, 10, 15, 24, 47, 52, 53, 64, 67, 88, 91]

# Method name → key in mine_evaluation_dataset.json
METHOD_TO_KEY = {
    "kggen":        "kggen",
    "graphrag":     "graphrag_kg",
    "openie":       "openie_kg",
    "autoschemakg": "autoschemakg",
}


# ============================================================
# 2. Core analysis functions
# ============================================================

def clean_essay_text(text: str) -> str:
    """Strip backtick wrappers and normalize whitespace."""
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def word_count(text: str) -> int:
    """Simple whitespace-based word count."""
    return len(text.split()) if text and text.strip() else 0


def compute_kg_compression_stats(
    kg_data: Dict,
    essay_text: str,
) -> Dict[str, Any]:
    """
    Compute compression and granularity metrics for one KG on one essay.

    Metrics:
        - num_entities: total entity count
        - num_edges: unique edge/relation type count
        - num_relations: total relation triple count
        - avg_entity_words: mean word count per entity string
        - median_entity_words: median word count per entity
        - max_entity_words: longest entity (in words)
        - pct_entities_gt5: % of entities with >5 words (sentence-like)
        - pct_entities_gt10: % of entities with >10 words (almost certainly a sentence)
        - total_entity_words: sum of words across all entities
        - total_triple_words: sum of words across all (head + rel + tail)
        - essay_words: word count of the source essay
        - entity_compression_ratio: total_entity_words / essay_words
        - triple_compression_ratio: total_triple_words / essay_words
        - avg_head_words: mean word count of relation heads
        - avg_tail_words: mean word count of relation tails
        - avg_rel_words: mean word count of relation predicates (edge labels)
        - verbatim_overlap_score: fraction of essay 4-grams found in entity text
    """
    entities = [str(e) for e in kg_data.get("entities", [])]
    edges = kg_data.get("edges", [])
    relations = kg_data.get("relations", [])

    essay_clean = clean_essay_text(essay_text)
    essay_words = word_count(essay_clean)

    # --- Entity-level stats ---
    ent_word_counts = [word_count(e) for e in entities]
    total_entity_words = sum(ent_word_counts)
    avg_entity_words = np.mean(ent_word_counts) if ent_word_counts else 0.0
    median_entity_words = float(np.median(ent_word_counts)) if ent_word_counts else 0.0
    max_entity_words = max(ent_word_counts) if ent_word_counts else 0
    pct_gt5 = (sum(1 for w in ent_word_counts if w > 5) / len(ent_word_counts) * 100) if ent_word_counts else 0.0
    pct_gt10 = (sum(1 for w in ent_word_counts if w > 10) / len(ent_word_counts) * 100) if ent_word_counts else 0.0

    # --- Relation triple stats ---
    head_words_list = []
    tail_words_list = []
    rel_words_list = []
    total_triple_words = 0

    for r in relations:
        if isinstance(r, (list, tuple)) and len(r) == 3:
            hw = word_count(str(r[0]))
            rw = word_count(str(r[1]))
            tw = word_count(str(r[2]))
            head_words_list.append(hw)
            rel_words_list.append(rw)
            tail_words_list.append(tw)
            total_triple_words += hw + rw + tw

    avg_head_words = np.mean(head_words_list) if head_words_list else 0.0
    avg_tail_words = np.mean(tail_words_list) if tail_words_list else 0.0
    avg_rel_words = np.mean(rel_words_list) if rel_words_list else 0.0

    # --- Compression ratios ---
    entity_compression_ratio = total_entity_words / max(essay_words, 1)
    triple_compression_ratio = total_triple_words / max(essay_words, 1)

    # --- Verbatim overlap: what fraction of essay 4-grams appear in entity text ---
    verbatim_score = _compute_ngram_overlap(essay_clean, entities, n=4)

    return {
        "num_entities": len(entities),
        "num_edges": len(edges) if isinstance(edges, list) else 0,
        "num_relations": len(relations),
        "avg_entity_words": round(float(avg_entity_words), 2),
        "median_entity_words": round(median_entity_words, 2),
        "max_entity_words": max_entity_words,
        "pct_entities_gt5_words": round(pct_gt5, 1),
        "pct_entities_gt10_words": round(pct_gt10, 1),
        "total_entity_words": total_entity_words,
        "total_triple_words": total_triple_words,
        "essay_words": essay_words,
        "entity_compression_ratio": round(entity_compression_ratio, 4),
        "triple_compression_ratio": round(triple_compression_ratio, 4),
        "avg_head_words": round(float(avg_head_words), 2),
        "avg_tail_words": round(float(avg_tail_words), 2),
        "avg_rel_words": round(float(avg_rel_words), 2),
        "verbatim_4gram_overlap": round(verbatim_score, 4),
    }


def _compute_ngram_overlap(essay_text: str, entities: List[str], n: int = 4) -> float:
    """
    What fraction of the essay's n-grams appear verbatim in the entity strings?
    High score = the KG entities are just copying the source text.
    """
    def get_ngrams(text: str, n: int) -> Set[Tuple[str, ...]]:
        words = text.lower().split()
        if len(words) < n:
            return set()
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

    essay_ngrams = get_ngrams(essay_text, n)
    if not essay_ngrams:
        return 0.0

    # Combine all entity text into one big string
    all_entity_text = " ".join(entities)
    entity_ngrams = get_ngrams(all_entity_text, n)

    overlap = essay_ngrams & entity_ngrams
    return len(overlap) / len(essay_ngrams)


def compute_tracekg_compression_stats(
    snapshot_path: Path,
    essay_text: str,
) -> Dict[str, Any]:
    """
    Compute compression stats for TRACE KG from its CSV files.
    Converts to the same format as baseline KGs for uniform analysis.
    """
    nodes_csv = snapshot_path / "KG" / "nodes.csv"
    rels_csv = snapshot_path / "KG" / "rels_fixed_no_raw.csv"

    if not nodes_csv.exists() or not rels_csv.exists():
        return {"error": f"Missing CSV files in {snapshot_path}"}

    nodes_df = pd.read_csv(nodes_csv)
    rels_df = pd.read_csv(rels_csv)

    # Build a pseudo kg_data dict matching baseline format
    entities = []
    for _, row in nodes_df.iterrows():
        name = str(row.get("name", "")) if pd.notna(row.get("name")) else ""
        if name:
            entities.append(name)

    relations = []
    edges_set = set()
    for _, row in rels_df.iterrows():
        src = str(row.get("source_name", row.get("source", ""))) if pd.notna(row.get("source_name", row.get("source"))) else ""
        rel = str(row.get("relation", "")) if pd.notna(row.get("relation")) else ""
        tgt = str(row.get("target_name", row.get("target", ""))) if pd.notna(row.get("target_name", row.get("target"))) else ""
        if src and rel and tgt:
            relations.append([src, rel, tgt])
            edges_set.add(rel)

    kg_data = {
        "entities": entities,
        "edges": sorted(edges_set),
        "relations": relations,
    }

    return compute_kg_compression_stats(kg_data, essay_text)


# ============================================================
# 3. Load accuracy results from already-computed evaluation
# ============================================================

def load_accuracy_from_results(results_root: Path, methods: List[str], essay_ids: List[int]) -> Dict[str, Dict[int, float]]:
    """
    Read the per-snapshot JSON result files to get accuracy per method per essay.
    Returns: {method: {essay_id: accuracy}}
    """
    acc_map: Dict[str, Dict[int, float]] = {m: {} for m in methods}

    for method in methods:
        method_dir = results_root / method
        if not method_dir.exists():
            continue
        for f in sorted(method_dir.glob("results_*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                summary = data.get("summary", {})
                did = summary.get("dataset_id")
                acc = summary.get("accuracy")
                if did is not None and acc is not None:
                    acc_map[method][int(did)] = float(acc)
            except Exception:
                continue

    return acc_map


# ============================================================
# 4. Main analysis
# ============================================================

def run_compression_analysis():
    print("=" * 80)
    print("MINE-1 Compression & Information-Leakage Analysis")
    print("=" * 80)

    # Load evaluation dataset
    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_item = {int(item["id"]): item for item in dataset if "id" in item}
    print(f"Loaded {len(id_to_item)} items from evaluation dataset")

    # All methods including tracekg
    all_methods = ["kggen", "graphrag", "openie", "autoschemakg", "tracekg"]

    # Load pre-computed accuracy
    acc_map = load_accuracy_from_results(RESULTS_ROOT, all_methods, EVAL_ESSAY_IDS)

    # Compute compression stats per method per essay
    all_stats: Dict[str, List[Dict[str, Any]]] = {m: [] for m in all_methods}

    for eid in EVAL_ESSAY_IDS:
        item = id_to_item.get(eid)
        if item is None:
            print(f"  [warn] essay_id={eid} not in dataset, skipping")
            continue

        essay_text = item.get("essay_content", "")

        # Baseline methods (from dataset JSON)
        for method in ["kggen", "graphrag", "openie", "autoschemakg"]:
            kg_key = METHOD_TO_KEY[method]
            kg_data = item.get(kg_key)
            if kg_data is None:
                continue
            stats = compute_kg_compression_stats(kg_data, essay_text)
            stats["essay_id"] = eid
            stats["method"] = method
            stats["accuracy"] = acc_map.get(method, {}).get(eid, None)
            all_stats[method].append(stats)

        # TRACE KG (from snapshot CSV)
        label = f"{eid:03d}"
        snap_path = KG_SNAPSHOTS_ROOT / f"KG_Essay_{label}"
        if snap_path.is_dir():
            stats = compute_tracekg_compression_stats(snap_path, essay_text)
            if "error" not in stats:
                stats["essay_id"] = eid
                stats["method"] = "tracekg"
                stats["accuracy"] = acc_map.get("tracekg", {}).get(eid, None)
                all_stats["tracekg"].append(stats)

    # ============================================================
    # 5. Aggregate and print
    # ============================================================

    print("\n" + "=" * 80)
    print("PER-METHOD AGGREGATE STATISTICS")
    print("=" * 80)

    agg_results = {}

    for method in all_methods:
        entries = all_stats[method]
        if not entries:
            print(f"\n  {method}: NO DATA")
            continue

        n = len(entries)
        agg = {
            "method": method,
            "n_essays": n,
            "mean_accuracy": _safe_mean([e["accuracy"] for e in entries if e.get("accuracy") is not None]),
            "mean_num_entities": _safe_mean([e["num_entities"] for e in entries]),
            "mean_num_relations": _safe_mean([e["num_relations"] for e in entries]),
            "mean_avg_entity_words": _safe_mean([e["avg_entity_words"] for e in entries]),
            "mean_median_entity_words": _safe_mean([e["median_entity_words"] for e in entries]),
            "mean_max_entity_words": _safe_mean([e["max_entity_words"] for e in entries]),
            "mean_pct_entities_gt5": _safe_mean([e["pct_entities_gt5_words"] for e in entries]),
            "mean_pct_entities_gt10": _safe_mean([e["pct_entities_gt10_words"] for e in entries]),
            "mean_entity_compression_ratio": _safe_mean([e["entity_compression_ratio"] for e in entries]),
            "mean_triple_compression_ratio": _safe_mean([e["triple_compression_ratio"] for e in entries]),
            "mean_avg_head_words": _safe_mean([e["avg_head_words"] for e in entries]),
            "mean_avg_tail_words": _safe_mean([e["avg_tail_words"] for e in entries]),
            "mean_avg_rel_words": _safe_mean([e["avg_rel_words"] for e in entries]),
            "mean_verbatim_4gram_overlap": _safe_mean([e["verbatim_4gram_overlap"] for e in entries]),
        }
        agg_results[method] = agg

    # --- Print compact comparison table ---
    print("\n" + "=" * 120)
    print("COMPARISON TABLE")
    print("=" * 120)

    header = (
        f"{'Method':<16} | {'Acc%':>6} | {'#Ent':>5} | {'#Rel':>5} | "
        f"{'AvgEW':>6} | {'MedEW':>6} | {'MaxEW':>6} | "
        f"{'%>5w':>6} | {'%>10w':>6} | "
        f"{'EntCR':>7} | {'TriCR':>7} | {'4gram%':>7}"
    )
    print(header)
    print("-" * len(header))

    for method in all_methods:
        a = agg_results.get(method)
        if a is None:
            print(f"{method:<16} | {'N/A':>6} |")
            continue
        print(
            f"{method:<16} | "
            f"{a['mean_accuracy']*100:5.1f}% | "
            f"{a['mean_num_entities']:5.0f} | "
            f"{a['mean_num_relations']:5.0f} | "
            f"{a['mean_avg_entity_words']:6.1f} | "
            f"{a['mean_median_entity_words']:6.1f} | "
            f"{a['mean_max_entity_words']:6.0f} | "
            f"{a['mean_pct_entities_gt5']:5.1f}% | "
            f"{a['mean_pct_entities_gt10']:5.1f}% | "
            f"{a['mean_entity_compression_ratio']:7.3f} | "
            f"{a['mean_triple_compression_ratio']:7.3f} | "
            f"{a['mean_verbatim_4gram_overlap']*100:6.1f}%"
        )

    # --- Print interpretation guide ---
    print("\n" + "-" * 80)
    print("COLUMN LEGEND:")
    print("  Acc%     = Mean accuracy from MINE-1 evaluation (LLM judge)")
    print("  #Ent     = Mean number of entities in the KG")
    print("  #Rel     = Mean number of relation triples")
    print("  AvgEW    = Mean words per entity (lower = more atomic)")
    print("  MedEW    = Median words per entity")
    print("  MaxEW    = Max words in any single entity")
    print("  %>5w     = % of entities with >5 words (sentence fragments)")
    print("  %>10w    = % of entities with >10 words (full sentences)")
    print("  EntCR    = Entity compression ratio (total entity words / essay words)")
    print("             <1.0 = KG compresses; >1.0 = KG is BIGGER than source")
    print("  TriCR    = Triple compression ratio (total triple words / essay words)")
    print("  4gram%   = % of essay 4-grams found verbatim in entity text")
    print("             High = information leakage (entities copy source text)")
    print("-" * 80)

    # Save detailed results
    output = {
        "config": {
            "eval_essay_ids": EVAL_ESSAY_IDS,
            "dataset_path": str(DATASET_JSON_PATH),
            "snapshots_root": str(KG_SNAPSHOTS_ROOT),
            "results_root": str(RESULTS_ROOT),
        },
        "aggregate": agg_results,
        "per_essay": {m: entries for m, entries in all_stats.items()},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")

    # Save table as text
    with open(OUTPUT_TABLE_PATH, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for method in all_methods:
            a = agg_results.get(method)
            if a is None:
                continue
            f.write(
                f"{method:<16} | "
                f"{a['mean_accuracy']*100:5.1f}% | "
                f"{a['mean_num_entities']:5.0f} | "
                f"{a['mean_num_relations']:5.0f} | "
                f"{a['mean_avg_entity_words']:6.1f} | "
                f"{a['mean_median_entity_words']:6.1f} | "
                f"{a['mean_max_entity_words']:6.0f} | "
                f"{a['mean_pct_entities_gt5']:5.1f}% | "
                f"{a['mean_pct_entities_gt10']:5.1f}% | "
                f"{a['mean_entity_compression_ratio']:7.3f} | "
                f"{a['mean_triple_compression_ratio']:7.3f} | "
                f"{a['mean_verbatim_4gram_overlap']*100:6.1f}%\n"
            )
    print(f"Table saved to: {OUTPUT_TABLE_PATH}")


def _safe_mean(values: list) -> float:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else 0.0


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_compression_analysis()
    
#endregion#?     Post-hoc Compression & Information-Leakage Analysis for MINE-1 Evaluation
#?#########################  End  ##########################



#endregion#! Addomg AutoSchemaKG to the comparison
#!#############################################  End Chapter  ##################################################




#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################


































#?######################### Start ##########################
#region:#?     Jetstream





#!pip install -U open-webui


import requests

def chat_with_model(token):
    url = 'http://localhost:3000/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": "granite3.1-dense:8b",
      "messages": [
        {
          "role": "user",
          "content": "Why is the sky blue?"
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

x = chat_with_model("sk-fbc98f5dd11044b99597b66a489b9a91")
print(x)




!curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:3000/api/models
!curl -H "sk-Oybl2tBTLJM1nhY79IlpQA" http://localhost:3000/api/models

!curl -H "sk-fbc98f5dd11044b99597b66a489b9a91" http://localhost:3000/api/models


curl -H "Authorization: bearer sk-fbc98f5dd11044b99597b66a489b9a91"













curl https://llm.jetstream-cloud.org/api/chat/completions \
  -H "Authorization: bearer your-token-here" \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "llama-4-scout",
        "messages": [
          {
            "role": "user",
            "content": "What is the difference between SSH and SSL"
          }
        ],
        "max_tokens": 64
      }'
      
      
      
      curl -H "Authorization: Bearer sk-fbc98f5dd11044b99597b66a489b9a91" http://localhost:3000/api/models
      
      
      
      
      
      
      
import requests

def chat_with_model(token):
    url = "https://llm.jetstream-cloud.org/api/"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": "",
      "messages": [
        {
          "role": "user",
          "content": "Why is the sky blue?"
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()


x = chat_with_model("XXXX")
print(x)



import os, time, requests

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

start_time = int(time.time()) - 7 * 24 * 3600  # last 7 days
url = "https://api.openai.com/v1/organization/usage/completions"
params = {"start_time": start_time, "bucket_width": "1d"}

r = requests.get(
    url,
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
    params=params,
    timeout=30,
)

print("status:", r.status_code)
print(r.text[:1000])



def log_usage(resp):
    u = resp.usage
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "total_tokens": u.total_tokens,
    }







import os
from openai import OpenAI

# make sure your key is set:
# export OPENAI_API_KEY="sk-..."

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def log_usage(resp):
    u = resp.usage
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "total_tokens": u.total_tokens,
    }

# ---- simple task ----
response = client.responses.create(
    model="gpt-4o-mini",
    input="Give me one sentence explaining what a knowledge graph is."
)

# print model output
print("MODEL OUTPUT:")
print(response.output_text)

# print usage info
usage = log_usage(response)
print("\nUSAGE:")
print(usage)

#endregion#?   Jetstream
#?#########################  End  ##########################