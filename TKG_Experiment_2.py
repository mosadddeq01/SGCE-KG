
  




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





  