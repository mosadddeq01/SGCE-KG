





#!############################################# Start Chapter ##################################################
#region:#!    Compare TRACE KG with KG-Gen

  
  
  

#endregion#!  Compare TRACE KG with KG-Gen
#!############################################# End Chapter ##################################################
  
  
  
  
  
  
  
  
  

#!############################################# Start Chapter ##################################################
#region:#!   Comparing Schema of TRACE KG with X

  
  
  

#endregion#! Comparing Schema of TRACE KG with X
#!############################################# End Chapter ##################################################
  
  
  
  






#!############################################# Start Chapter ##################################################
#region:#!   Experiments 1 - MINE 1 From KG Gen Paper









#*######################### Start ##########################
#region:#?   QA4Methods - V5   (TRACE names for context, weighted embeddings)





# import os
# import json
# from dataclasses import dataclass
# from pathlib import Path
# import pwd
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

# # Entity weights (must sum to 1 after normalization)
# ENT_WEIGHTS = {
#     "name": 0.40,   # entity_name
#     "desc": 0.25,   # entity_description
#     "ctx": 0.35,    # class_label + class_group + node_properties
# }

# # Relation weights (before normalization)
# REL_EMB_WEIGHTS = {
#     "name": 0.25,      # relation_name
#     "desc+Q": 0.15,    # rel_desc + qualifiers
#     "head_tail": 0.20, # subject/object names + class info
#     "ctx": 0.40,       # canonical_rel_name + canonical_rel_desc + rel_cls
# }

# ENT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# REL_EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # same for simplicity
# BATCH_SIZE = 32
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # OpenAI config
# OPENAI_MODEL_JUDGE = "gpt-5.1"
# OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


# # ============================================================
# # 1. Env helper for OpenAI key
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
# # 2. HF Embedder (generic)
# # ============================================================

# def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
#     masked = token_embeds * mask
#     summed = masked.sum(dim=1)
#     counts = mask.sum(dim=1).clamp(min=1e-9)
#     return summed / counts


# class HFEmbedder:
#     def __init__(self, model_name: str, device: str = DEVICE):
#         print(f"[Embedder] loading model {model_name} on {device} ...")
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
# # 3. SimpleGraph for KG-Gen-style KGs (dataset KGs)
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
# # 4. TRACE-KG loaders and weighted embedding builders
# # ============================================================

# def safe_str(x) -> str:
#     if x is None:
#         return ""
#     if isinstance(x, float) and np.isnan(x):
#         return ""
#     return str(x)


# def build_tracekg_entity_texts(nodes_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
#     """Return lists: ids, name_texts, desc_texts, ctx_texts."""
#     ids, names, descs, ctxs = [], [], [], []
#     for _, row in nodes_df.iterrows():
#         ent_id = safe_str(row["entity_id"])
#         ids.append(ent_id)

#         # name ~ entity_name
#         name_txt = safe_str(row.get("entity_name", ""))

#         # desc ~ entity_description
#         desc_txt = safe_str(row.get("entity_description", ""))

#         # ctx ~ class_label + class_group + node_properties
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

#     print("[TRACE-ENT] encoding name field ...")
#     emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None

#     print("[TRACE-ENT] encoding desc field ...")
#     emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None

#     print("[TRACE-ENT] encoding ctx field ...")
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
#     """
#     Build relation text fields by bucket:
#       - name: relation name
#       - desc+Q: rel_desc + qualifiers
#       - head_tail: subject/object names + class info
#       - ctx: canonical_rel_name + canonical_rel_desc + rel_cls
#     """
#     # Node helper map
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

#         # name bucket
#         rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))

#         # desc+Q bucket
#         rel_desc = safe_str(row.get("rel_desc", "")) or safe_str(row.get("canonical_rel_desc", ""))
#         qualifiers = safe_str(row.get("qualifiers", ""))
#         desc_plus_q = " ; ".join([p for p in [rel_desc, qualifiers] if p])

#         # head_tail bucket
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

#         # ctx bucket
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
#         print(f"[TRACE-REL] encoding bucket '{b}' ...")
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
#     """
#     Build TRACE-KG graph and a node_info dict:
#       node_id -> {"name": entity_name, "class_label": class_label}
#     """
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
# # 5. Retrieval (weighted node embeddings + graph, with readable context)
# # ============================================================

# class WeightedGraphRetriever:
#     def __init__(
#         self,
#         node_embeddings: Dict[str, np.ndarray],
#         graph: nx.DiGraph,
#         node_info: Optional[Dict[str, Dict[str, str]]] = None,
#     ):
#         """
#         node_info (for TRACE-KG) maps:
#            node_id -> {"name": ..., "class_label": ...}
#         For dataset KGs, this can be None and we fall back to IDs in context.
#         """
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
#         """
#         For TRACE-KG, return 'Name (which is of type: ClassLabel)'.
#         For others, just return node_id as string.
#         """
#         info = self.node_info.get(node_id)
#         if info is None:
#             return str(node_id)

#         name = info.get("name") or str(node_id)
#         cls = info.get("class_label") or ""
#         if cls:
#             return f"{name} (which is of type: {cls})"
#         return name

#     def _format_edge_for_context(self, src: str, dst: str, data: Dict) -> str:
#         """
#         For TRACE-KG, produce:
#           subjectEntName (which is of type: entCleName) has relation
#           {rel_canonical_name (with qualifiers:{})} with
#           objectEntName (which is of type: entCleName).
#         For others, fall back to 'src rel dst.'.
#         """
#         rel_name = data.get("relation", "")
#         qualifiers = data.get("qualifiers", "")

#         # Heuristic: if we have node_info, treat this as TRACE-KG style.
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
#             # dataset KGs
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
# # 6. LLM-based evaluator (OpenAI API 5.1)
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
#     """
#     Use an OpenAI model as a binary judge.
#     Returns 1 if the model believes the context contains the correct answer,
#     otherwise 0.
#     """
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
#                 {
#                     "role": "system",
#                     "content": system_prompt,
#                 },
#                 {
#                     "role": "user",
#                     "content": user_prompt,
#                 },
#             ],
#             max_output_tokens=64,  # >= 16
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

#     # Fallback: heuristic if model output is weird
#     ans_tokens = set(t.lower() for t in correct_answer.split() if len(t) > 3)
#     if not ans_tokens:
#         return 0
#     ctx_lower = context.lower()
#     for t in ans_tokens:
#         if t in ctx_lower:
#             return 1
#     return 0


# # ============================================================
# # 7. Evaluation utilities
# # ============================================================

# def evaluate_accuracy_for_graph(
#     query_embedder: HFEmbedder,
#     retriever: WeightedGraphRetriever,
#     queries: List[str],
#     method_name: str,
#     essay_idx: int,
#     results_dir: str,
#     k: int = 8,
#     verbose: bool = False,
# ) -> Dict:
#     os.makedirs(results_dir, exist_ok=True)

#     print(f"[{method_name}] encoding {len(queries)} queries for essay {essay_idx} ...")
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

#     out_path = os.path.join(results_dir, f"results_{essay_idx}.json")
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)

#     if verbose:
#         print(
#             f"[{method_name}] Essay {essay_idx}: "
#             f"accuracy={accuracy:.4f} ({correct}/{len(queries)})"
#         )

#     return {
#         "accuracy": accuracy,
#         "num_queries": len(queries),
#         "method": method_name,
#         "essay_idx": essay_idx,
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
#     print("\n=== Method Comparison (Mean Accuracy) ===")
#     print(f"{'Method':<10} | {'Mean Acc':>8} | {'#Essays':>7}")
#     print("-" * 32)
#     for m, stats in comparison.items():
#         print(
#             f"{m:<10} | {stats['mean_accuracy']*100:8.2f}% | "
#             f"{stats['num_essays']:7d}"
#         )


# # ============================================================
# # 8. Full evaluation over the dataset
# # ============================================================

# def run_full_evaluation(
#     dataset_json_path: str,
#     trace_nodes_csv: str,
#     trace_rels_csv: str,
#     output_root: str,
#     methods: List[str],
#     k: int = 8,
#     max_essays: Optional[int] = None,
#     verbose: bool = True,
# ) -> Dict[str, List[Dict]]:
#     # Load dataset dump you already created
#     with open(dataset_json_path, "r", encoding="utf-8") as f:
#         dataset_list = json.load(f)
#     if max_essays is not None:
#         dataset_list = dataset_list[:max_essays]

#     # TRACE-KG embeddings & graph
#     ent_embedder = HFEmbedder(ENT_EMBED_MODEL, DEVICE)
#     rel_embedder = HFEmbedder(REL_EMBED_MODEL, DEVICE)  # relation embeddings precomputed but not used yet
#     nodes_df = pd.read_csv(trace_nodes_csv)
#     rels_df = pd.read_csv(trace_rels_csv)

#     trace_node_embs, _ = compute_weighted_entity_embeddings(ent_embedder, nodes_df, ENT_WEIGHTS)
#     trace_rel_embs, _ = compute_weighted_relation_embeddings(rel_embedder, rels_df, nodes_df, REL_EMB_WEIGHTS)
#     trace_graph, trace_node_info = build_tracekg_nx_and_nodeinfo(nodes_df, rels_df)
#     trace_retriever = WeightedGraphRetriever(trace_node_embs, trace_graph, node_info=trace_node_info)

#     # Query embedder (same model as entity, for semantic match)
#     query_embedder = ent_embedder

#     all_summaries: Dict[str, List[Dict]] = {m: [] for m in methods}

#     for idx, row in enumerate(dataset_list):
#         essay_idx = idx
#         queries: List[str] = row.get("generated_queries", [])
#         if not queries:
#             if verbose:
#                 print(f"Skipping essay {essay_idx}: no queries.")
#             continue

#         if verbose:
#             print(f"\n=== Essay {essay_idx} | {len(queries)} queries ===")

#         # TRACE-KG (one global graph)
#         if "tracekg" in methods:
#             summaries_dir = os.path.join(output_root, "tracekg")
#             s = evaluate_accuracy_for_graph(
#                 query_embedder=query_embedder,
#                 retriever=trace_retriever,
#                 queries=queries,
#                 method_name="tracekg",
#                 essay_idx=essay_idx,
#                 results_dir=summaries_dir,
#                 k=k,
#                 verbose=verbose,
#             )
#             all_summaries["tracekg"].append(s)

#         # Other methods: kggen, graphrag, openie
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

#             kg_data = row.get(kg_key, None)
#             if kg_data is None:
#                 if verbose:
#                     print(f"  [{method}] No KG data for essay {essay_idx}, skipping.")
#                 continue

#             sg = SimpleGraph.from_kggen_dict(kg_data)
#             g_nx = sg.to_nx()

#             node_ids = list(g_nx.nodes())
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
#                 essay_idx=essay_idx,
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
#     dataset_json_path = (
#         "Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset-short.json"
#     )
#     trace_nodes_csv = "Experiments/MYNE/TRACEKG/EssaysKG_CSVs/002/nodes.csv"
#     trace_rels_csv = "Experiments/MYNE/TRACEKG/EssaysKG_CSVs/002/rels.csv"

#     output_root = "./tracekg_mine_results_weighted_openai_v5"

#     methods = ["kggen", "graphrag", "openie", "tracekg"]
#     #methods = ["tracekg"]

#     all_summaries = run_full_evaluation(
#         dataset_json_path=dataset_json_path,
#         trace_nodes_csv=trace_nodes_csv,
#         trace_rels_csv=trace_rels_csv,
#         output_root=output_root,
#         methods=methods,
#         k=8,
#         max_essays=None,  # or a small number for debugging
#         verbose=True,
#     )

#     comparison = compare_methods(all_summaries)
#     print_comparison_table(comparison)


# if __name__ == "__main__":
#     main()
    
    



#endregion#? QA4Methods - V5   (TRACE names for context, weighted embeddings)
#*#########################  End  ##########################




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

# OPENAI_MODEL_JUDGE = "gpt-5.1"
# OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# # Paths
# DATASET_JSON_PATH = Path("Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset.json")
# KG_SNAPSHOTS_ROOT = Path("KGs_from_Essays")
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
#         print(f"[Embedder] loading model {model_name} on {device} ...")
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

#     print("[TRACE-ENT] encoding name field ...")
#     emb_name = embedder.encode_batch(names) if any(t.strip() for t in names) else None

#     print("[TRACE-ENT] encoding desc field ...")
#     emb_desc = embedder.encode_batch(descs) if any(t.strip() for t in descs) else None

#     print("[TRACE-ENT] encoding ctx field ...")
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
#         print(f"[TRACE-REL] encoding bucket '{b}' ...")
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

#     print(f"[{method_name}] encoding {len(queries)} queries "
#           f"(snapshot={snapshot_label}, dataset_id={dataset_id}) ...")
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
#         verbose=True,
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

#first 12 results
    
=== Per-Snapshot Accuracy (all methods) ===
  Snapshot | DatasetID |       tracekg        |  Reponses mentioned in baselines by mistake     |     KgGen     |                        GraphRAG                   |     OpenIE 
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
       001 |         1 |       93.33% (14/15) |  Response accuracy for essay ID 1:              {'kggen_accuracy': 0.5333,             'graphrag_accuracy': 0.4667, 'openie_accuracy': 0.2667}       
       002 |         2 |      100.00% (15/15) |  Response accuracy for essay ID 2:              {'kggen_accuracy': 0.6,                'graphrag_accuracy': 0.3333, 'openie_accuracy': 0.0} 
       010 |        10 |      100.00% (15/15) |  Response accuracy for essay ID 10:             {'kggen_accuracy': 0.6667000000000001, 'graphrag_accuracy': 0.6667, 'openie_accuracy': 0.3333}    
       015 |        15 |      100.00% (15/15) |  Response accuracy for essay ID 15:             {'kggen_accuracy': 0.6667000000000001, 'graphrag_accuracy': 0.4667, 'openie_accuracy': 0.3333}    
       024 |        24 |      100.00% (15/15) |  Response accuracy for essay ID 25:             {'kggen_accuracy': 0.8667,             'graphrag_accuracy': 0.6,    'openie_accuracy': 0.2667}    
       047 |        47 |      100.00% (15/15) |  Response accuracy for essay ID 48:             {'kggen_accuracy': 0.8667,             'graphrag_accuracy': 0.9333, 'openie_accuracy': 0.7333}    
       052 |        52 |      100.00% (15/15) |  Response accuracy for essay ID 53:             {'kggen_accuracy': 0.6667000000000001, 'graphrag_accuracy': 0.3333, 'openie_accuracy': 0.6}    
       053 |        53 |      100.00% (15/15) |  Response accuracy for essay ID 54:             {'kggen_accuracy': 0.8,                'graphrag_accuracy': 0.7333, 'openie_accuracy': 0.4}    
       064 |        64 |       86.67% (13/15) |  NULL                
       067 |        67 |      100.00% (15/15) |  Response accuracy for essay ID 68:             {'kggen_accuracy': 0.6,                'graphrag_accuracy': 0.4,    'openie_accuracy': 0.0667} 
       088 |        88 |      100.00% (15/15) |  Response accuracy for essay ID 89:             {'kggen_accuracy': 0.9333,             'graphrag_accuracy': 0.3333, 'openie_accuracy': 0.4667}    
       091 |        91 |      100.00% (15/15) |  Response accuracy for essay ID 92:             {'kggen_accuracy': 0.9333,             'graphrag_accuracy': 0.5333, 'openie_accuracy': 0.5333}



=== Method Comparison (Mean Accuracy across evaluated snapshots) ===
Method     | Mean Acc |   Essays
------------------------------------
tracekg    |    98.33% |      52
KGGen      |    66.67% |      52
GraphRAG   |    53.31% |      52
OpenIE     |    36.36% |      52



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

  




#!############################################# Start Chapter ##################################################
#region:#!   Experiments 2 - MINE 2 From KG Gen Paper






#?######################### Start ##########################
#region:#?   V1

"""
Run MINE-2 (KG-assisted RAG) baseline vs TRACE KG on the WikiQA validation subset.

Usage (from repo root):
  python3 experiments/run_mine2_vs_trace.py \
    --validation_csv Experiments/MYNE/Ex2/wiki_qa/wiki_qa_Validation/validation_clean.csv \
    --articles_dir Experiments/MYNE/Ex2/wiki_qa/wiki_qa_Validation/articles \
    --out_dir experiments/results_mine2_vs_trace

Notes / Defaults:
  - If you already produced a KG triples CSV/JSONL from KG-Gen or TRACE KG, point
    --triples_path to it; otherwise the script will create a fallback pseudo-KG
    from article sentences so the retrieval+RAG steps can run.
  - You must set OPENAI_API_KEY in your environment for the example LLM calls,
    or modify generate_answer_with_llm and judge_answer_with_llm to use your LLM.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import logging
import time
import csv

# Optional OpenAI import for LLM calls. Replace or adapt to your LLM of choice.
try:
    import openai
except Exception:
    openai = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------
# Utilities to load data
# -------------------------
def load_validation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    # Expect columns: question_id, question, document_title, answer, label
    return df


def load_articles_map(articles_dir: Path) -> Dict[str, str]:
    """
    Map document_title -> article text (string).
    """
    articles = {}
    for p in sorted(articles_dir.iterdir()):
        if p.suffix.lower() in (".txt",):
            key = p.stem  # matches your CSV document_title
            text = p.read_text(encoding="utf-8")
            articles[key] = text
    return articles


# -------------------------
# KG loading / fallback
# -------------------------
def load_triples_from_csv(path: Path) -> List[Dict[str, Any]]:
    """
    Try to load a triples CSV with common columns:
      subject, predicate, object, triple_text, source_chunk, source_doc
    If some columns are missing we will synthesize triple_text by concatenation.
    """
    triples = []
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, keep_default_na=False)
        for _, row in df.iterrows():
            subj = row.get("subject") or row.get("head") or row.get("s") or ""
            pred = row.get("predicate") or row.get("rel") or row.get("p") or ""
            obj = row.get("object") or row.get("tail") or row.get("o") or ""
            triple_text = row.get("triple_text") or f"{subj} {pred} {obj}".strip()
            source_chunk = row.get("source_chunk") or row.get("text") or ""
            source_doc = row.get("source_doc") or row.get("source") or row.get("file") or ""
            triples.append({
                "subject": str(subj),
                "predicate": str(pred),
                "object": str(obj),
                "triple_text": str(triple_text),
                "source_chunk": str(source_chunk),
                "source_doc": str(source_doc)
            })
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                subj = obj.get("subject", "")
                pred = obj.get("predicate", "")
                objt = obj.get("object", "")
                triple_text = obj.get("triple_text") or f"{subj} {pred} {objt}"
                triples.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": objt,
                    "triple_text": triple_text,
                    "source_chunk": obj.get("source_chunk", ""),
                    "source_doc": obj.get("source_doc", ""),
                })
    else:
        raise ValueError(f"Unsupported triples file extension: {path}")
    return triples


def build_pseudo_kg_from_articles(articles_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Build simple triple set from article sentences. This is a fallback used if no
    KG exports are present. Each sentence becomes a triple:
      (article_title, "has_sentence", sentence)
    """
    from nltk import sent_tokenize
    triples = []
    tid = 0
    for doc_title, text in articles_map.items():
        # naive sentence split
        sents = sent_tokenize(text)
        for s in sents:
            triples.append({
                "subject": doc_title,
                "predicate": "has_sentence",
                "object": s,
                "triple_text": f"{doc_title} has_sentence {s}",
                "source_chunk": s,
                "source_doc": doc_title,
                "id": f"T{tid}"
            })
            tid += 1
    return triples


# -------------------------
# Embeddings & BM25
# -------------------------
class Retriever:
    def __init__(self, triples: List[Dict[str, Any]], embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.triples = triples
        self.triple_texts = [t.get("triple_text", "") for t in triples]
        self.tokenized_for_bm25 = [tt.lower().split() for tt in self.triple_texts]
        self.bm25 = BM25Okapi(self.tokenized_for_bm25)
        self.embedder = SentenceTransformer(embed_model_name, device=device)
        logger.info("Encoding triple texts for embeddings...")
        self.embeddings = self.embedder.encode(self.triple_texts, show_progress_bar=True, convert_to_numpy=True)

    def retrieve_top_k(self, query: str, k: int = 10) -> List[Tuple[int, float, float]]:
        # BM25
        q_tok = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(q_tok), dtype=float)

        # Embedding similarity
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        cos_sim = cosine_similarity([q_emb], self.embeddings)[0]  # shape (N,)

        # Normalize each score vector to [0,1]
        bm25_norm = minmax_scale(bm25_scores) if bm25_scores.max() != bm25_scores.min() else np.zeros_like(bm25_scores)
        cos_norm = minmax_scale(cos_sim) if cos_sim.max() != cos_sim.min() else np.zeros_like(cos_sim)

        # Combine equally
        combined = 0.5 * bm25_norm + 0.5 * cos_norm

        topk_idx = np.argsort(combined)[-k:][::-1]
        return [(int(i), float(combined[i]), float(bm25_scores[i])) for i in topk_idx]

    def find_two_hop_expansion(self, topk_indices: List[int], hops: int = 2, max_extra: int = 10) -> List[int]:
        """
        Build graph between subjects/objects present in triples; nodes are subject/object strings.
        Return indices of triples within `hops` of topk triples' nodes.
        """
        G = nx.Graph()
        # map triple idx to nodes
        for i, t in enumerate(self.triples):
            subj = t.get("subject", "")
            obj = t.get("object", "")
            G.add_node(f"S:{subj}")
            G.add_node(f"O:{obj}")
            # add edge between subj and obj via triple id
            G.add_edge(f"S:{subj}", f"O:{obj}", triple_idx=i)

        # seeds: nodes that appear in topk triples
        seed_nodes = set()
        for idx in topk_indices:
            t = self.triples[idx]
            seed_nodes.add(f"S:{t.get('subject','')}")
            seed_nodes.add(f"O:{t.get('object','')}")

        # perform BFS to collect nodes within hops
        nodes_within = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(hops):
            new_frontier = set()
            for node in frontier:
                for nbr in G.neighbors(node):
                    if nbr not in nodes_within:
                        new_frontier.add(nbr)
                        nodes_within.add(nbr)
            frontier = new_frontier
            if not frontier:
                break

        # collect triples that touch any node in nodes_within
        triple_idxs = set()
        for a, b, data in G.edges(data=True):
            if a in nodes_within or b in nodes_within:
                triple_idxs.add(int(data.get("triple_idx")))
        # exclude already included topk
        extra = [i for i in sorted(triple_idxs - set(topk_indices), reverse=True)]
        return extra[:max_extra]


# -------------------------
# LLM / RAG interface
# -------------------------
def generate_rag_prompt(question: str, triples_context: List[Dict[str, Any]]) -> str:
    header = "You are given a question and a set of retrieved knowledge triples along with their source text chunks. Use ONLY the provided evidence to answer the question as precisely as possible.\n\n"
    ctx_lines = []
    for i, t in enumerate(triples_context, start=1):
        src = t.get("source_doc", "") or ""
        chunk = t.get("source_chunk", "") or t.get("triple_text", "")
        ctx_lines.append(f"[{i}] triple: {t.get('triple_text','')}\nsource_doc: {src}\nsource_chunk: {chunk}\n")
    ctx_text = "\n".join(ctx_lines)
    prompt = f"{header}Question:\n{question}\n\nRetrieved evidence (top to bottom):\n{ctx_text}\n\nAnswer:"
    return prompt


def generate_answer_with_llm(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 256, temperature: float = 0.0) -> str:
    """
    Example using OpenAI chat completion. Replace/adapt to your LLM.
    Make sure OPENAI_API_KEY is in your environment if using openai.
    """
    if openai is None:
        raise RuntimeError("openai library not installed or import failed. Replace generate_answer_with_llm with your LLM client.")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp["choices"][0]["message"]["content"]
    return text.strip()


def judge_answer_with_llm(question: str, gold: str, pred: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """
    LLM-as-judge: returns a judgement dict. This prompt is a lightweight example;
    adapt if you already have the judge prompt from KGGen Appendix B.
    """
    judge_prompt = (
        "You are a strict evaluator. Given a question, a gold answer, and a generated answer, "
        "return JSON: {correct: true/false, verdict: <one-sentence explanation>}.\n\n"
        f"Question: {question}\n\nGold: {gold}\n\nGenerated: {pred}\n\nIs the generated answer correct and supported by the gold answer?"
    )
    if openai is None:
        # fallback: simple string match for reproducibility
        corr = 1 if gold.strip().lower() in pred.strip().lower() else 0
        return {"correct": bool(corr), "verdict": "simple substring match fallback"}
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":judge_prompt}],
        max_tokens=128,
        temperature=0.0,
    )
    verdict_text = resp["choices"][0]["message"]["content"].strip()
    # try to parse simple yes/no from verdict_text
    correct = False
    txt_low = verdict_text.lower()
    if "true" in txt_low or "correct" in txt_low or "yes" in txt_low:
        correct = True
    return {"correct": correct, "verdict": verdict_text}


# -------------------------
# Main experiment flow
# -------------------------
def run_experiment(validation_csv: Path, articles_dir: Path, out_dir: Path, triples_path: Optional[Path] = None, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2", llm_model: str = "gpt-3.5-turbo"):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_validation_csv(validation_csv)
    articles_map = load_articles_map(articles_dir)

    # Load or build KG triples
    triples: List[Dict[str, Any]]
    if triples_path and triples_path.exists():
        logger.info(f"Loading triples from {triples_path}")
        triples = load_triples_from_csv(triples_path)
    else:
        logger.warning("No triples file provided or found. Building fallback pseudo-KG from article sentences.")
        triples = build_pseudo_kg_from_articles(articles_map)

    # Build retriever (embeddings + BM25)
    retriever = Retriever(triples, embed_model_name=embed_model)

    results = []
    start_time = time.time()
    for idx, row in df.iterrows():
        qid = row["question_id"]
        question = str(row["question"])
        gold = str(row["answer"])
        # 1) retrieve top 10
        topk = retriever.retrieve_top_k(question, k=10)
        topk_idx = [i for i, _, _ in topk]
        # 2) two-hop expand and append up to 10 more (total 20)
        extra_idx = retriever.find_two_hop_expansion(topk_idx, hops=2, max_extra=10)
        selected_indices = topk_idx + extra_idx
        # build context
        ctx = [triples[i] for i in selected_indices]
        prompt = generate_rag_prompt(question, ctx)
        try:
            pred = generate_answer_with_llm(prompt, model=llm_model)
        except Exception as e:
            logger.exception("LLM call failed; saving failure. Set up LLM client or adapt generate_answer_with_llm.")
            pred = f"[LLM_CALL_FAILED: {e}]"
        # judge
        try:
            judgement = judge_answer_with_llm(question, gold, pred, model=llm_model)
        except Exception:
            judgement = {"correct": False, "verdict": "judge-run-failed"}
        res = {
            "question_id": qid,
            "question": question,
            "gold": gold,
            "prediction": pred,
            "judge": judgement,
            "num_retrieved": len(selected_indices),
            "retrieved_ids": selected_indices,
        }
        results.append(res)
        logger.info(f"Processed {qid}: judge.correct={judgement.get('correct')}")
    elapsed = time.time() - start_time
    out_path = out_dir / "mine2_trace_results.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Finished. Wrote {len(results)} results to {out_path} in {elapsed:.1f}s.")
    # Summarize
    correct = sum(1 for r in results if r["judge"].get("correct"))
    logger.info(f"Accuracy (LLM-as-judge): {correct}/{len(results)} = {correct/len(results):.4f}")
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_csv",
        type=Path,
        required=True,
        help="Path to validation CSV (question_id,question,document_title,answer,label)"
    )
    parser.add_argument(
        "--articles_dir",
        type=Path,
        required=True,
        help="Path to articles directory"
    )
    parser.add_argument("--triples_path", type=Path, default=None,
                        help="Optional triples CSV/JSONL path to use (KGGen or TRACE KG exported triples). If not provided, script builds pseudo-KG from article sentences.")
    parser.add_argument("--out_dir", type=Path, default=Path("experiments/results_mine2_vs_trace"))
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm_model", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()

    # If using OpenAI example LLM
    if openai is not None and os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    run_experiment(args.validation_csv, args.articles_dir, args.out_dir, args.triples_path,
                   embed_model=args.embed_model, llm_model=args.llm_model)


if __name__ == "__main__":
    main()


#endregion#? V1
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   

#endregion#? 
#?#########################  End  ##########################



#endregion#! Experiments 2 - MINE 2 From KG Gen Paper
#!#############################################  End Chapter  ##################################################





  
  

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




import Trace_KG


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

from Trace_KG import (  # core pipeline functions
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

from Trace_KG import (  # core pipeline functions
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



