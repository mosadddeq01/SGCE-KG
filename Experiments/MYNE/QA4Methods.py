


#!############################################# Start Chapter ##################################################
#region:#! Evaluation

  

#endregion#! Evaluation
#!############################################# End Chapter ##################################################












#!############################################# Start Chapter ##################################################
#region:#!   Experiments



#*######################### Start ##########################
#region:#?   QA4Methods  - V1  (Wants TRACE KG API)



import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Literal

from datasets import load_dataset
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# If you want real visualizations, install graphviz and uncomment.
# import graphviz


# ============================================================
# 1. Data Model for Knowledge Graphs (generic, not KG-Gen specific)
# ============================================================

@dataclass
class SimpleGraph:
    """
    A minimal KG representation compatible with what KG-Gen exports.

    - entities: set of node identifiers (usually strings).
    - relations: set of triples (source, relation, target).
    """
    entities: Set[str]
    relations: Set[Tuple[str, str, str]]

    @staticmethod
    def from_kggen_dict(d: Dict) -> "SimpleGraph":
        """
        Convert a KG-Gen style dict into SimpleGraph.

        Expected keys (from kg-gen exports / dataset):
        - entities: list[str]
        - relations: list[list or tuple of 3 items]
        """
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
            g.add_node(e)
        for s, rel, t in self.relations:
            g.add_edge(s, t, relation=rel)
        return g


# ============================================================
# 2. TRACE-KG integration stubs
#    You MUST implement/replace the TODOs for your own model.
# ============================================================

def generate_tracekg(text: str) -> SimpleGraph:
    """
    Generate a KG from essay text using your TRACE-KG generator.

    TODO: Replace this stub with calls to your own model.

    It should return a SimpleGraph where:
    - entities are the node strings you want to index.
    - relations are (source, relation, target) triples.
    """
    # --------------------------------------------------------
    # >>>> TODO: Implement this with your real TRACE-KG <<<<
    # --------------------------------------------------------
    # Example: dummy KG using each sentence as a node, no edges.
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    entities = set(sentences)
    relations = set()
    return SimpleGraph(entities=entities, relations=relations)


def tracekg_to_nx(graph: SimpleGraph) -> nx.DiGraph:
    """
    Convert your TRACE-KG graph to a NetworkX DiGraph.

    If your internal representation is not SimpleGraph, adapt accordingly.
    """
    return graph.to_nx()


# ============================================================
# 3. Embedding-based graph retrieval (KG-Gen style)
# ============================================================

class GraphRetriever:
    """
    Embedding + graph-neighborhood query engine,
    roughly mirroring KGGen.retrieve + helpers.
    """

    def __init__(self, retrieval_model_name: str = "all-MiniLM-L6-v2"):
        """
        retrieval_model_name: any SentenceTransformers model string.
        """
        self.model_name = retrieval_model_name
        self.model = SentenceTransformer(retrieval_model_name)

    def generate_embeddings(
        self, graph: nx.DiGraph
    ) -> Dict[str, np.ndarray]:
        """
        Generate node embeddings for the graph.

        In KG-Gen, node text == node id string.
        """
        node_embeddings: Dict[str, np.ndarray] = {}
        for node in graph.nodes:
            emb = self.model.encode(str(node))
            node_embeddings[node] = emb
        return node_embeddings

    def retrieve_relevant_nodes(
        self,
        query: str,
        node_embeddings: Dict[str, np.ndarray],
        k: int = 8,
    ) -> List[Tuple[str, float]]:
        """
        Top-k node retrieval by cosine similarity.
        """
        q_emb = self.model.encode(query).reshape(1, -1)
        sims: List[Tuple[str, float]] = []
        for node, emb in node_embeddings.items():
            t_emb = emb.reshape(1, -1)
            sim = cosine_similarity(q_emb, t_emb)[0][0]
            sims.append((node, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def retrieve_context(
        self,
        node: str,
        graph: nx.DiGraph,
        depth: int = 2,
    ) -> List[str]:
        """
        Get local text context around a node up to given depth.
        Mirrors KG-Gen's BFS over outgoing + incoming edges.
        """
        context: Set[str] = set()

        def explore(n: str, d: int):
            if d > depth:
                return
            # outgoing
            for nbr in graph.neighbors(n):
                rel = graph[nbr].get("relation", None)
                if "relation" in graph[n][nbr]:
                    rel = graph[n][nbr]["relation"]
                text = f"{n} {rel} {nbr}."
                context.add(text)
                explore(nbr, d + 1)
            # incoming
            for nbr in graph.predecessors(n):
                if "relation" in graph[nbr][n]:
                    rel = graph[nbr][n]["relation"]
                else:
                    rel = None
                text = f"{nbr} {rel} {n}."
                context.add(text)
                explore(nbr, d + 1)

        explore(node, 1)
        return list(context)

    def retrieve(
        self,
        query: str,
        node_embeddings: Dict[str, np.ndarray],
        graph: nx.DiGraph,
        k: int = 8,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[str, float]], Set[str], str]:
        """
        Approximate KG-Gen's KGGen.retrieve signature:

        Returns:
        - top_nodes: [(node, similarity), ...]
        - context_set: set of local-triple strings
        - context_text: joined context string
        """
        top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, k=k)
        context: Set[str] = set()
        for node, _ in top_nodes:
            node_ctx = self.retrieve_context(node, graph)
            if verbose:
                print(f"Context for node {node}: {node_ctx}")
            context.update(node_ctx)
        context_text = " ".join(context)
        if verbose:
            print("Combined context:", context_text)
        return top_nodes, context, context_text


# ============================================================
# 4. LLM-based evaluator (ResponseEvaluator / gpt_evaluate_response)
# ============================================================

def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    """
    Binary evaluator: returns 1 if context is judged to contain the answer, else 0.

    In KG-Gen, this is implemented via DSPy + an LLM (GPT-5 etc.).
    Here we provide a placeholder you MUST replace with your LLM.

    TODO: Replace this with a real LLM call (OpenAI / other).
    """
    # --------------------------------------------------------
    # >>>> TODO: Implement actual LLM scoring here <<<<
    # --------------------------------------------------------
    # For now: extremely naive heuristic (DO NOT use for real experiments):
    # 1 if any non-trivial overlap between answer tokens and context.
    ans_tokens = set(t.lower() for t in correct_answer.split() if len(t) > 3)
    if not ans_tokens:
        return 0
    ctx_lower = context.lower()
    for t in ans_tokens:
        if t in ctx_lower:
            return 1
    return 0


# ============================================================
# 5. Evaluation Loop (similar to experiments/MINE/_1_evaluation.py)
# ============================================================

def evaluate_accuracy_for_graph(
    retriever: GraphRetriever,
    queries: List[str],
    graph: nx.DiGraph,
    method_name: str,
    essay_idx: int,
    results_dir: str,
    k: int = 8,
    verbose: bool = False,
) -> Dict:
    """
    Run query-answer evaluation against a single KG (one essay).

    Saves a JSON results file similar to KG-Gen.
    """
    os.makedirs(results_dir, exist_ok=True)
    node_embeddings = retriever.generate_embeddings(graph)

    correct = 0
    results: List[Dict] = []

    for q in queries:
        _, _, context_text = retriever.retrieve(q, node_embeddings, graph, k=k)
        evaluation = gpt_evaluate_response(q, context_text)
        result = {
            "correct_answer": q,
            "retrieved_context": context_text,
            "evaluation": int(evaluation),
        }
        results.append(result)
        correct += evaluation

    accuracy = correct / len(queries) if queries else 0.0
    summary = {
        "accuracy": accuracy,
        "num_queries": len(queries),
        "method": method_name,
        "essay_idx": essay_idx,
    }
    results.append({"accuracy": f"{accuracy * 100:.2f}%"})

    out_path = os.path.join(results_dir, f"results_{essay_idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"[{method_name}] Essay {essay_idx}: "
              f"accuracy={accuracy:.4f}, saved to {out_path}")
    return summary


def run_full_evaluation(
    dataset_path: str,
    output_root: str,
    methods: List[str],
    k: int = 8,
    max_essays: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Run evaluation for all essays and all methods.

    methods must be a subset of:
    - "kggen"   (uses dataset[i]["kggen"])
    - "graphrag" (uses dataset[i]["graphrag_kg"])
    - "openie"  (uses dataset[i]["openie_kg"])
    - "tracekg" (your TRACE-KG from essay_content)

    Returns a dict mapping method -> list of per-essay summaries.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_list = json.load(f)

    if max_essays is not None:
        dataset_list = dataset_list[:max_essays]

    retriever = GraphRetriever(retrieval_model_name="all-MiniLM-L6-v2")

    all_summaries: Dict[str, List[Dict]] = {m: [] for m in methods}

    for idx, row in enumerate(dataset_list):
        essay_idx = idx  # zero-based index for file naming

        generated_queries: List[str] = row.get("generated_queries", [])
        essay_content: str = row.get("essay_content", "")

        if not generated_queries:
            if verbose:
                print(f"Skipping essay {essay_idx} (no queries)")
            continue

        if verbose:
            print(f"\n=== Essay {essay_idx} | {len(generated_queries)} queries ===")

        for method in methods:
            # Build or load KG
            graph_nx: Optional[nx.DiGraph] = None

            if method == "kggen":
                kg_data = row.get("kggen", None)
                if kg_data is None:
                    if verbose:
                        print(f"  [kggen] No KG data for essay {essay_idx}, skipping.")
                    continue
                sg = SimpleGraph.from_kggen_dict(kg_data)
                graph_nx = sg.to_nx()

            elif method == "graphrag":
                kg_data = row.get("graphrag_kg", None)
                if kg_data is None:
                    if verbose:
                        print(f"  [graphrag] No KG data for essay {essay_idx}, skipping.")
                    continue
                sg = SimpleGraph.from_kggen_dict(kg_data)
                graph_nx = sg.to_nx()

            elif method == "openie":
                kg_data = row.get("openie_kg", None)
                if kg_data is None:
                    if verbose:
                        print(f"  [openie] No KG data for essay {essay_idx}, skipping.")
                    continue
                sg = SimpleGraph.from_kggen_dict(kg_data)
                graph_nx = sg.to_nx()

            elif method == "tracekg":
                if not essay_content:
                    if verbose:
                        print(f"  [tracekg] No essay content for {essay_idx}, skipping.")
                    continue
                trace_graph = generate_tracekg(essay_content)
                graph_nx = tracekg_to_nx(trace_graph)

            else:
                raise ValueError(f"Unknown method: {method}")

            method_dir = os.path.join(output_root, method)
            summary = evaluate_accuracy_for_graph(
                retriever=retriever,
                queries=generated_queries,
                graph=graph_nx,
                method_name=method,
                essay_idx=essay_idx,
                results_dir=method_dir,
                k=k,
                verbose=verbose,
            )
            all_summaries[method].append(summary)

    return all_summaries


# ============================================================
# 6. Comparison & Analysis (similar to _2_compare_results.py & _4_analysis.py)
# ============================================================

def aggregate_method_stats(
    summaries: List[Dict],
) -> Dict[str, float]:
    """
    Compute aggregate metrics for one method.
    """
    if not summaries:
        return {
            "mean_accuracy": 0.0,
            "num_essays": 0,
        }

    accuracies = [s["accuracy"] for s in summaries]
    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "num_essays": len(accuracies),
    }


def compare_methods(all_summaries: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Compare all methods by mean accuracy across essays.
    """
    comparison: Dict[str, Dict] = {}
    for method, summaries in all_summaries.items():
        stats = aggregate_method_stats(summaries)
        comparison[method] = stats
    return comparison


def print_comparison_table(comparison: Dict[str, Dict]):
    """
    Pretty-print comparison results.
    """
    print("\n=== Method Comparison (Mean Accuracy) ===")
    print(f"{'Method':<10} | {'Mean Acc':>8} | {'#Essays':>7}")
    print("-" * 32)
    for method, stats in comparison.items():
        print(
            f"{method:<10} | {stats['mean_accuracy']*100:8.2f}% | "
            f"{stats['num_essays']:7d}"
        )


# ============================================================
# 7. Simple Visualization (similar in spirit to _3_visualize.py)
# ============================================================

def save_graph_as_dot(
    graph: nx.DiGraph,
    path: str,
):
    """
    Save a small graph to Graphviz .dot format.
    You can then render with `dot -Tpng path.dot -o out.png`.

    This is a minimal substitute for KG-Gen's HTML visualization.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("digraph G {\n")
        for n in graph.nodes:
            f.write(f'  "{n}" ;\n')
        for u, v, data in graph.edges(data=True):
            rel = data.get("relation", "")
            f.write(f'  "{u}" -> "{v}" [label="{rel}"] ;\n')
        f.write("}\n")


def export_example_visualizations(
    dataset_path: str,
    output_root: str,
    methods: List[str],
    essay_indices: List[int],
):
    """
    For a few essays and methods, save their KGs to .dot files for quick inspection.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_list = json.load(f)

    retriever = GraphRetriever(retrieval_model_name="all-MiniLM-L6-v2")

    for essay_idx in essay_indices:
        if essay_idx >= len(dataset_list):
            print(f"Essay index {essay_idx} out of range, skipping.")
            continue
        row = dataset_list[essay_idx]
        essay_content = row.get("essay_content", "")

        for method in methods:
            if method == "kggen":
                d = row.get("kggen", None)
                if not d:
                    continue
                g = SimpleGraph.from_kggen_dict(d).to_nx()
            elif method == "graphrag":
                d = row.get("graphrag_kg", None)
                if not d:
                    continue
                g = SimpleGraph.from_kggen_dict(d).to_nx()
            elif method == "openie":
                d = row.get("openie_kg", None)
                if not d:
                    continue
                g = SimpleGraph.from_kggen_dict(d).to_nx()
            elif method == "tracekg":
                if not essay_content:
                    continue
                trace_graph = generate_tracekg(essay_content)
                g = tracekg_to_nx(trace_graph)
            else:
                continue

            out_dir = os.path.join(output_root, "visualizations", method)
            out_path = os.path.join(out_dir, f"essay_{essay_idx}.dot")
            save_graph_as_dot(g, out_path)
            print(f"Saved {method} KG for essay {essay_idx} to {out_path}")


# ============================================================
# 8. Main entry point
# ============================================================

def main():
    """
    Example usage:

    1. Ensure you have downloaded the MINE evaluation dataset
       with your code and saved it to:

       /home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/KG-Gen/kg-gen/experiments/MINE/dataset_dumps/mine_evaluation_dataset.json

       (Adjust the path below if different.)

    2. Implement:
       - generate_tracekg(...)
       - gpt_evaluate_response(...)

    3. Run:
       python tracekg_mine_evaluation.py
    """
    dataset_path = "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/KG-Gen/kg-gen/experiments/MINE/dataset_dumps/mine_evaluation_dataset.json"

    output_root = "./tracekg_mine_results"  # where all outputs will go

    # Methods to evaluate. Include any subset of these:
    methods: List[str] = ["kggen", "graphrag", "openie", "tracekg"]

    all_summaries = run_full_evaluation(
        dataset_path=dataset_path,
        output_root=output_root,
        methods=methods,
        k=8,
        max_essays=None,  # or set a small number for quick tests
        verbose=True,
    )

    comparison = compare_methods(all_summaries)
    print_comparison_table(comparison)

    # Optional: export a few example KGs for visualization
    export_example_visualizations(
        dataset_path=dataset_path,
        output_root=output_root,
        methods=methods,
        essay_indices=[0, 1, 2],
    )


if __name__ == "__main__":
    main()

#endregion#? QA4Methods  - V1
#*#########################  End  ##########################

#*######################### Start ##########################
#region:#?   QA4Methods - V2   (load TRACE from CSV)

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import networkx as nx
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. Data Model for Generic Knowledge Graphs
# ============================================================

@dataclass
class SimpleGraph:
    """
    Minimal KG representation used for KG-Gen-like graphs.

    - entities: set of node identifiers (strings).
    - relations: set of triples (source, relation, target).
    """
    entities: Set[str]
    relations: Set[Tuple[str, str, str]]

    @staticmethod
    def from_kggen_dict(d: Dict) -> "SimpleGraph":
        """
        Convert a KG-Gen style dict into SimpleGraph.

        Expected keys:
        - entities: list[str]
        - relations: list[list or tuple of 3 items]
        """
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
            g.add_node(e, label=str(e))
        for s, rel, t in self.relations:
            g.add_edge(s, t, relation=rel)
        return g


# ============================================================
# 2. TRACE-KG Loader from CSV
# ============================================================

"""
We assume a single TRACE-KG over the whole corpus is stored as:

- nodes CSV:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/nodes.csv
  Columns:
    "entity_id","entity_name","entity_description","entity_type_hint",
    "entity_confidence","entity_resolution_context","entity_flag",
    "class_id","class_label","class_group","chunk_ids","node_properties"

- rels CSV:
  /home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/rels.csv
  Columns:
    "relation_id","start_id","end_id","raw_relation_name","canonical_rel_name",
    "canonical_rel_desc","rel_cls","rel_cls_group","rel_hint_type","confidence",
    "resolution_context","justification","remark","evidence_excerpt",
    "chunk_id","qualifiers","rel_desc"

We build a single NetworkX DiGraph from these two files.
Later you can:
- Filter by essay / chunk_ids.
- Change the node text construction and weighting.
"""


def build_tracekg_graph_from_csv(
    nodes_csv_path: str,
    rels_csv_path: str,
) -> nx.DiGraph:
    # Load nodes
    nodes_df = pd.read_csv(nodes_csv_path)

    # Load relationships
    rels_df = pd.read_csv(rels_csv_path)

    g = nx.DiGraph()

    # Add nodes with rich attributes
    for _, row in nodes_df.iterrows():
        node_id = str(row["entity_id"])

        # Build a descriptive label/text combining multiple fields.
        # You can later change the fields or add explicit weighting.
        parts = []
        if not pd.isna(row.get("entity_name", None)):
            parts.append(f"Name: {row['entity_name']}")
        if not pd.isna(row.get("entity_description", None)):
            parts.append(f"Description: {row['entity_description']}")
        if not pd.isna(row.get("entity_type_hint", None)):
            parts.append(f"Type: {row['entity_type_hint']}")
        if not pd.isna(row.get("class_label", None)):
            parts.append(f"Class: {row['class_label']}")
        if not pd.isna(row.get("class_group", None)):
            parts.append(f"ClassGroup: {row['class_group']}")
        if not pd.isna(row.get("entity_resolution_context", None)):
            parts.append(f"Context: {row['entity_resolution_context']}")
        if not pd.isna(row.get("node_properties", None)):
            parts.append(f"Properties: {row['node_properties']}")

        # This is the textual representation used for embeddings.
        # Later you can implement real weighting here.
        node_text = " | ".join(parts) if parts else str(row["entity_name"])

        g.add_node(
            node_id,
            text=node_text,
            entity_name=row.get("entity_name", None),
            entity_description=row.get("entity_description", None),
            entity_type_hint=row.get("entity_type_hint", None),
            entity_confidence=row.get("entity_confidence", None),
            entity_resolution_context=row.get("entity_resolution_context", None),
            entity_flag=row.get("entity_flag", None),
            class_id=row.get("class_id", None),
            class_label=row.get("class_label", None),
            class_group=row.get("class_group", None),
            chunk_ids=row.get("chunk_ids", None),
            node_properties=row.get("node_properties", None),
        )

    # Add edges with attributes
    for _, row in rels_df.iterrows():
        start_id = str(row["start_id"])
        end_id = str(row["end_id"])

        # Fallback if canonical name is missing
        rel_label = row.get("canonical_rel_name", None)
        if isinstance(rel_label, float) and np.isnan(rel_label):
            rel_label = row.get("raw_relation_name", "")

        g.add_edge(
            start_id,
            end_id,
            relation=str(rel_label),
            relation_id=row.get("relation_id", None),
            raw_relation_name=row.get("raw_relation_name", None),
            canonical_rel_name=row.get("canonical_rel_name", None),
            canonical_rel_desc=row.get("canonical_rel_desc", None),
            rel_cls=row.get("rel_cls", None),
            rel_cls_group=row.get("rel_cls_group", None),
            rel_hint_type=row.get("rel_hint_type", None),
            confidence=row.get("confidence", None),
            resolution_context=row.get("resolution_context", None),
            justification=row.get("justification", None),
            remark=row.get("remark", None),
            evidence_excerpt=row.get("evidence_excerpt", None),
            chunk_id=row.get("chunk_id", None),
            qualifiers=row.get("qualifiers", None),
            rel_desc=row.get("rel_desc", None),
        )

    return g


# ============================================================
# 3. Embedding-based Graph Retriever (KG-Gen style)
# ============================================================

class GraphRetriever:
    """
    Embedding + graph-neighborhood query engine, roughly
    mirroring KGGen.retrieve + helpers.
    """

    def __init__(self, retrieval_model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(retrieval_model_name)

    def _node_text(self, graph: nx.DiGraph, node: str) -> str:
        """
        Return the text used to embed this node.
        - For TRACE-KG graph: use node["text"] (rich concatenation).
        - For KG-Gen graphs: use node id as a string.
        """
        data = graph.nodes[node]
        if "text" in data and data["text"]:
            return str(data["text"])
        # fallback for KG-Gen style: node itself
        return str(node)

    def generate_embeddings(
        self, graph: nx.DiGraph
    ) -> Dict[str, np.ndarray]:
        node_embeddings: Dict[str, np.ndarray] = {}
        for n in graph.nodes:
            txt = self._node_text(graph, n)
            emb = self.model.encode(txt)
            node_embeddings[n] = emb
        return node_embeddings

    def retrieve_relevant_nodes(
        self,
        query: str,
        node_embeddings: Dict[str, np.ndarray],
        k: int = 8,
    ) -> List[Tuple[str, float]]:
        q_emb = self.model.encode(query).reshape(1, -1)
        sims: List[Tuple[str, float]] = []
        for node, emb in node_embeddings.items():
            t_emb = emb.reshape(1, -1)
            sim = cosine_similarity(q_emb, t_emb)[0][0]
            sims.append((node, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def retrieve_context(
        self,
        node: str,
        graph: nx.DiGraph,
        depth: int = 2,
    ) -> List[str]:
        """
        BFS over outgoing + incoming edges, up to 'depth',
        collecting text triples like "A rel B.".
        """
        context: Set[str] = set()

        def explore(n: str, d: int):
            if d > depth:
                return
            # outgoing
            for nbr in graph.neighbors(n):
                data = graph[n][nbr]
                rel = data.get("relation", "")
                context.add(f"{n} {rel} {nbr}.")
                explore(nbr, d + 1)
            # incoming
            for nbr in graph.predecessors(n):
                data = graph[nbr][n]
                rel = data.get("relation", "")
                context.add(f"{nbr} {rel} {n}.")
                explore(nbr, d + 1)

        explore(node, 1)
        return list(context)

    def retrieve(
        self,
        query: str,
        node_embeddings: Dict[str, np.ndarray],
        graph: nx.DiGraph,
        k: int = 8,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[str, float]], Set[str], str]:
        top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, k=k)
        context: Set[str] = set()
        for node, _ in top_nodes:
            node_ctx = self.retrieve_context(node, graph)
            if verbose:
                print(f"Context for node {node}: {node_ctx}")
            context.update(node_ctx)
        context_text = " ".join(context)
        if verbose:
            print("Combined context:", context_text)
        return top_nodes, context, context_text


# ============================================================
# 4. LLM-based Evaluator (you must implement properly)
# ============================================================

def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    """
    Binary evaluator: 1 if context is judged to contain the answer, 0 otherwise.

    TODO: Replace this naive heuristic with your LLM-based judge.
    """
    # --------------------------------------------------------
    # >>>> TODO: Implement actual LLM scoring here <<<<
    # --------------------------------------------------------
    ans_tokens = set(t.lower() for t in correct_answer.split() if len(t) > 3)
    if not ans_tokens:
        return 0
    ctx_lower = context.lower()
    for t in ans_tokens:
        if t in ctx_lower:
            return 1
    return 0


# ============================================================
# 5. Evaluation Loop (per graph)
# ============================================================

def evaluate_accuracy_for_graph(
    retriever: GraphRetriever,
    queries: List[str],
    graph: nx.DiGraph,
    method_name: str,
    essay_idx: int,
    results_dir: str,
    k: int = 8,
    verbose: bool = False,
) -> Dict:
    os.makedirs(results_dir, exist_ok=True)
    node_embeddings = retriever.generate_embeddings(graph)

    correct = 0
    results: List[Dict] = []

    for q in queries:
        _, _, context_text = retriever.retrieve(q, node_embeddings, graph, k=k)
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


# ============================================================
# 6. Full Evaluation: all essays, 4 methods
# ============================================================

def run_full_evaluation(
    dataset_json_path: str,
    tracekg_nodes_csv: str,
    tracekg_rels_csv: str,
    output_root: str,
    methods: List[str],
    k: int = 8,
    max_essays: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """
    methods ⊆ {"kggen", "graphrag", "openie", "tracekg"}.

    For "tracekg" we use the **same global TRACE-KG** for every essay.
    For the others we use per-essay KGs from the dataset.
    """
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_list = json.load(f)
    if max_essays is not None:
        dataset_list = dataset_list[:max_essays]

    retriever = GraphRetriever("all-MiniLM-L6-v2")

    # Build TRACE-KG once (global KG)
    tracekg_graph: Optional[nx.DiGraph] = None
    if "tracekg" in methods:
        if verbose:
            print("Building TRACE-KG graph from CSVs...")
        tracekg_graph = build_tracekg_graph_from_csv(
            tracekg_nodes_csv,
            tracekg_rels_csv,
        )
        if verbose:
            print(
                f"TRACE-KG: {tracekg_graph.number_of_nodes()} nodes, "
                f"{tracekg_graph.number_of_edges()} edges."
            )

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

        for method in methods:
            graph_nx: Optional[nx.DiGraph] = None

            if method == "kggen":
                kg_data = row.get("kggen", None)
                if kg_data is None:
                    if verbose:
                        print(f"  [kggen] No KG data for essay {essay_idx}, skipping.")
                    continue
                sg = SimpleGraph.from_kggen_dict(kg_data)
                graph_nx = sg.to_nx()

            elif method == "graphrag":
                kg_data = row.get("graphrag_kg", None)
                if kg_data is None:
                    if verbose:
                        print(f"  [graphrag] No KG data for essay {essay_idx}, skipping.")
                    continue
                sg = SimpleGraph.from_kggen_dict(kg_data)
                graph_nx = sg.to_nx()

            elif method == "openie":
                kg_data = row.get("openie_kg", None)
                if kg_data is None:
                    if verbose:
                        print(f"  [openie] No KG data for essay {essay_idx}, skipping.")
                    continue
                sg = SimpleGraph.from_kggen_dict(kg_data)
                graph_nx = sg.to_nx()

            elif method == "tracekg":
                if tracekg_graph is None:
                    raise RuntimeError("TRACE-KG graph not built.")
                graph_nx = tracekg_graph

            else:
                raise ValueError(f"Unknown method: {method}")

            method_dir = os.path.join(output_root, method)
            summary = evaluate_accuracy_for_graph(
                retriever=retriever,
                queries=queries,
                graph=graph_nx,
                method_name=method,
                essay_idx=essay_idx,
                results_dir=method_dir,
                k=k,
                verbose=verbose,
            )
            all_summaries[method].append(summary)

    return all_summaries


# ============================================================
# 7. Comparison & Simple Analysis
# ============================================================

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
# 8. Simple Visualization: export small DOT graphs
# ============================================================

def save_graph_as_dot(graph: nx.DiGraph, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("digraph G {\n")
        for n, data in graph.nodes(data=True):
            label = data.get("text", str(n))
            f.write(f'  "{n}" [label="{label}"] ;\n')
        for u, v, edata in graph.edges(data=True):
            rel = edata.get("relation", "")
            f.write(f'  "{u}" -> "{v}" [label="{rel}"] ;\n')
        f.write("}\n")


def export_example_visualizations(
    dataset_json_path: str,
    tracekg_nodes_csv: str,
    tracekg_rels_csv: str,
    output_root: str,
    methods: List[str],
    essay_indices: List[int],
):
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_list = json.load(f)

    tracekg_graph = None
    if "tracekg" in methods:
        tracekg_graph = build_tracekg_graph_from_csv(
            tracekg_nodes_csv,
            tracekg_rels_csv,
        )

    for essay_idx in essay_indices:
        if essay_idx >= len(dataset_list):
            print(f"Essay index {essay_idx} out of range; skipping.")
            continue

        row = dataset_list[essay_idx]
        for method in methods:
            if method == "kggen":
                d = row.get("kggen", None)
                if not d:
                    continue
                g = SimpleGraph.from_kggen_dict(d).to_nx()
            elif method == "graphrag":
                d = row.get("graphrag_kg", None)
                if not d:
                    continue
                g = SimpleGraph.from_kggen_dict(d).to_nx()
            elif method == "openie":
                d = row.get("openie_kg", None)
                if not d:
                    continue
                g = SimpleGraph.from_kggen_dict(d).to_nx()
            elif method == "tracekg":
                if tracekg_graph is None:
                    continue
                g = tracekg_graph
            else:
                continue

            out_dir = os.path.join(output_root, "visualizations", method)
            out_path = os.path.join(out_dir, f"essay_{essay_idx}.dot")
            save_graph_as_dot(g, out_path)
            print(f"Saved {method} KG for essay {essay_idx} to {out_path}")


# ============================================================
# 9. Main
# ============================================================

def main():
    # Path where you dumped the HF dataset to JSON
    dataset_json_path = (
        "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/KG-Gen/kg-gen/"
        "experiments/MINE/dataset_dumps/mine_evaluation_dataset.json"
    )

    # Paths to your TRACE-KG CSVs
    tracekg_nodes_csv = (
        "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/nodes.csv"
    )
    tracekg_rels_csv = (
        "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/rels.csv"
    )

    output_root = "./tracekg_mine_results_csv"

    methods = ["kggen", "graphrag", "openie", "tracekg"]

    all_summaries = run_full_evaluation(
        dataset_json_path=dataset_json_path,
        tracekg_nodes_csv=tracekg_nodes_csv,
        tracekg_rels_csv=tracekg_rels_csv,
        output_root=output_root,
        methods=methods,
        k=8,
        max_essays=None,  # or small number for quick debug
        verbose=True,
    )

    comparison = compare_methods(all_summaries)
    print_comparison_table(comparison)

    # Optional: export some visualizations
    export_example_visualizations(
        dataset_json_path=dataset_json_path,
        tracekg_nodes_csv=tracekg_nodes_csv,
        tracekg_rels_csv=tracekg_rels_csv,
        output_root=output_root,
        methods=methods,
        essay_indices=[0, 1, 2],
    )


if __name__ == "__main__":
    main()


#endregion#? QA4Methods - V2   (load TRACE from CSV)
#*#########################  End  ##########################

#*######################### Start ##########################
#region:#?   QA4Methods - V3   (Weighted Retrieval)




import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import networkx as nx

from datasets import load_dataset  # only needed if you prefer reloading HF; for now we read your JSON dump
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 0. Global config: weights & models
# ============================================================

# Entity weights (must sum to 1 after normalization)
ENT_WEIGHTS = {
    "name": 0.40,   # entity_name
    "desc": 0.25,   # entity_description
    "ctx": 0.35,    # class_label + class_group + node_properties (bucketed into "ctx")
}

# Relation weights (before normalization)
REL_EMB_WEIGHTS = {
    "name": 0.25,      # relation_name (we'll use canonical_rel_name or raw_relation_name)
    "desc+Q": 0.15,    # rel_desc + qualifiers
    "head_tail": 0.20, # subject/object + class info
    "ctx": 0.40,       # canonical_rel_name + canonical_rel_desc + rel_cls
}

ENT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
REL_EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # same model; can be different if you want
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. HF Embedder (generic)
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
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_embeds = out.last_hidden_state
            pooled = mean_pool(token_embeds, attention_mask)
            pooled = pooled.cpu().numpy()
            embs.append(pooled)
        embs = np.vstack(embs)
        embs = normalize(embs, axis=1)
        return embs


# ============================================================
# 2. SimpleGraph for KG-Gen-style KGs (dataset KGs)
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
# 3. TRACE-KG loaders and weighted embedding builders
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

    Returns:
      - relation_ids: list of rel_id strings
      - id_to_index: maps relation_id -> row index
      - texts: index -> {bucket: text}
    """
    # Build helper maps from node_id -> (name, class_label)
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

    # Per-bucket text lists by index
    n = len(rel_ids)
    buckets = ["name", "desc+Q", "head_tail", "ctx"]
    bucket_texts = {b: [""] * n for b in buckets}
    for idx in range(n):
        for b in buckets:
            bucket_texts[b][idx] = texts[idx].get(b, "")

    # Encode each bucket
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

    # Normalize weights
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


def build_tracekg_nx(
    nodes_df: pd.DataFrame,
    rels_df: pd.DataFrame,
) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        nid = safe_str(row["entity_id"])
        g.add_node(
            nid,
            entity_name=safe_str(row.get("entity_name", "")),
            entity_description=safe_str(row.get("entity_description", "")),
            class_label=safe_str(row.get("class_label", "")),
            class_group=safe_str(row.get("class_group", "")),
            node_properties=safe_str(row.get("node_properties", "")),
            chunk_ids=safe_str(row.get("chunk_ids", "")),
        )
    for _, row in rels_df.iterrows():
        sid = safe_str(row.get("start_id", ""))
        eid = safe_str(row.get("end_id", ""))
        rid = safe_str(row.get("relation_id", ""))
        rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))
        g.add_edge(
            sid,
            eid,
            relation=rel_name,
            relation_id=rid,
            chunk_id=safe_str(row.get("chunk_id", "")),
        )
    return g


# ============================================================
# 4. Retrieval (uses entity embeddings + graph)
# ============================================================

class WeightedGraphRetriever:
    def __init__(
        self,
        node_embeddings: Dict[str, np.ndarray],
        graph: nx.DiGraph,
    ):
        self.node_embeddings = node_embeddings
        self.graph = graph

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
                rel = data.get("relation", "")
                context.add(f"{n} {rel} {nbr}.")
                explore(nbr, d + 1)
            for nbr in self.graph.predecessors(n):
                data = self.graph[nbr][n]
                rel = data.get("relation", "")
                context.add(f"{nbr} {rel} {n}.")
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
# 5. LLM-based evaluator (placeholder)
# ============================================================

def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    """
    TODO: Replace with your real LLM judge.
    Must return 1 if context contains answer, else 0.
    """
    ans_tokens = set(t.lower() for t in correct_answer.split() if len(t) > 3)
    if not ans_tokens:
        return 0
    ctx_lower = context.lower()
    for t in ans_tokens:
        if t in ctx_lower:
            return 1
    return 0


# ============================================================
# 6. Evaluation utilities
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
# 7. Full evaluation over the dataset
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
    rel_embedder = HFEmbedder(REL_EMBED_MODEL, DEVICE)  # not used in retrieval yet, but ready
    nodes_df = pd.read_csv(trace_nodes_csv)
    rels_df = pd.read_csv(trace_rels_csv)

    trace_node_embs, _ = compute_weighted_entity_embeddings(ent_embedder, nodes_df, ENT_WEIGHTS)
    trace_rel_embs, _ = compute_weighted_relation_embeddings(rel_embedder, rels_df, nodes_df, REL_EMB_WEIGHTS)
    trace_graph = build_tracekg_nx(nodes_df, rels_df)
    trace_retriever = WeightedGraphRetriever(trace_node_embs, trace_graph)

    # Query embedder (can reuse entity model for simplicity)
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

        # 1) TRACE-KG evaluation (one global graph)
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

        # 2) Other methods: kggen, graphrag, openie
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

            # Build crude (unweighted) node embeddings using same HF model (name only = node id)
            node_ids = list(g_nx.nodes())
            node_texts = [str(n) for n in node_ids]
            node_embs_arr = query_embedder.encode_batch(node_texts)
            node_embs = {nid: node_embs_arr[i] for i, nid in enumerate(node_ids)}
            retriever = WeightedGraphRetriever(node_embs, g_nx)

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
# 8. Main
# ============================================================

def main():
    dataset_json_path = (
        "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/KG-Gen/kg-gen/"
        "experiments/MINE/dataset_dumps/mine_evaluation_dataset.json"
    )
    trace_nodes_csv = "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/nodes.csv"
    trace_rels_csv = "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/rels.csv"

    output_root = "./tracekg_mine_results_weighted"

    methods = ["kggen", "graphrag", "openie", "tracekg"]

    all_summaries = run_full_evaluation(
        dataset_json_path=dataset_json_path,
        trace_nodes_csv=trace_nodes_csv,
        trace_rels_csv=trace_rels_csv,
        output_root=output_root,
        methods=methods,
        k=8,
        max_essays=None,  # or a small number for debugging
        verbose=True,
    )

    comparison = compare_methods(all_summaries)
    print_comparison_table(comparison)


if __name__ == "__main__":
    main()


#endregion#? QA4Methods - V3   (Weighted Retrieval)
#*#########################  End  ##########################





#*######################### Start ##########################
#region:#?   QA4Methods - V4   (gpt 5.1)


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
    fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env",
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


def build_tracekg_nx(
    nodes_df: pd.DataFrame,
    rels_df: pd.DataFrame,
) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        nid = safe_str(row["entity_id"])
        g.add_node(
            nid,
            entity_name=safe_str(row.get("entity_name", "")),
            entity_description=safe_str(row.get("entity_description", "")),
            class_label=safe_str(row.get("class_label", "")),
            class_group=safe_str(row.get("class_group", "")),
            node_properties=safe_str(row.get("node_properties", "")),
            chunk_ids=safe_str(row.get("chunk_ids", "")),
        )
    for _, row in rels_df.iterrows():
        sid = safe_str(row.get("start_id", ""))
        eid = safe_str(row.get("end_id", ""))
        rid = safe_str(row.get("relation_id", ""))
        rel_name = safe_str(row.get("canonical_rel_name", "")) or safe_str(row.get("raw_relation_name", ""))
        g.add_edge(
            sid,
            eid,
            relation=rel_name,
            relation_id=rid,
            chunk_id=safe_str(row.get("chunk_id", "")),
        )
    return g


# ============================================================
# 5. Retrieval (weighted node embeddings + graph)
# ============================================================

class WeightedGraphRetriever:
    def __init__(
        self,
        node_embeddings: Dict[str, np.ndarray],
        graph: nx.DiGraph,
    ):
        self.node_embeddings = node_embeddings
        self.graph = graph

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
                rel = data.get("relation", "")
                context.add(f"{n} {rel} {nbr}.")
                explore(nbr, d + 1)
            for nbr in self.graph.predecessors(n):
                data = self.graph[nbr][n]
                rel = data.get("relation", "")
                context.add(f"{nbr} {rel} {n}.")
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
            "in /home/mabolhas/MyReposOnSOL/SGCE-KG/.env"
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
        "You are given a question (or statement of a correct answer) and a retrieved context. "
        "Return '1' (without quotes) if the context clearly contains enough information "
        "to answer/justify the correct answer. Otherwise return '0'. "
        "Return only a single character: '1' or '0'."
    )

    user_prompt = (
        "Question / Correct answer:\n"
        f"{correct_answer}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        "Does the retrieved context contain enough information to support the correct answer? "
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
            # must be >= 16 for Responses API
            max_output_tokens=800,
        )
        text = resp.output[0].content[0].text.strip()
    except Exception as e:
        print(f"[gpt_evaluate_response] Error calling OpenAI: {e}")
        return 0

    # Normalize and check strictly for '1' or '0'
    text = text.strip()
    if text == "1":
        return 1
    if text == "0":
        return 0

    # Fallback: heuristic string match if model gives weird output
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
    trace_graph = build_tracekg_nx(nodes_df, rels_df)
    trace_retriever = WeightedGraphRetriever(trace_node_embs, trace_graph)

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
            retriever = WeightedGraphRetriever(node_embs, g_nx)

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
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset-short.json"
    )
    trace_nodes_csv = "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/nodes.csv"
    trace_rels_csv = "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/rels.csv"

    output_root = "./tracekg_mine_results_weighted_openai"

    methods = ["kggen", "graphrag", "openie", "tracekg"]

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

#endregion#? QA4Methods - V4   (gpt 5.1)
#*#########################  End  ##########################







#?######################### Start ##########################
#region:#?   QA4Methods - V5   (TRACE names for context, weighted embeddings)







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
    fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env",
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
            "in /home/mabolhas/MyReposOnSOL/SGCE-KG/.env"
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
        "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset-short.json"
    )
    trace_nodes_csv = "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/nodes.csv"
    trace_rels_csv = "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/rels.csv"

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
    fallback_path: str = "/home/mabolhas/MyReposOnSOL/SGCE-KG/.env",
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
            "in /home/mabolhas/MyReposOnSOL/SGCE-KG/.env"
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
        "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset-short.json"
    )
    trace_nodes_csv = "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/nodes.csv"
    trace_rels_csv = "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACEKG/rels.csv"

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












#?######################### Start ##########################
#region:#?   Run statements


# -----------------------
# Chunking - Run statement
# -----------------------

if __name__ == "__main__":
    sentence_chunks_token_driven(
        "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/Plain_Text.json",
        "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
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
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_emb",
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
#region:#?   Create JSON with 100 essays from the mine_evaluation_dataset.json





import json

with open("/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset.json", "r") as f:
    data = json.load(f)

print(type(data))
if isinstance(data, dict):
    print(data.keys())
elif isinstance(data, list) and data:
    print(data[0].keys())







import json

with open("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/Plain_Text.json", "r") as f:
    data = json.load(f)

print(type(data))
if isinstance(data, dict):
    print(data.keys())
elif isinstance(data, list) and data:
    print(data[0].keys())






import json
from pathlib import Path

# -------- paths --------
SRC = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/mine_evaluation_dataset.json")
OUT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/Plain_Text_100_Essays.json")

# -------- load source --------
with SRC.open("r", encoding="utf-8") as f:
    data = json.load(f)   # list of dicts

# -------- transform --------
out = []
for row in data[:100]:
    out.append({
        "title": row["essay_topic"],
        "start_page": row["id"],
        "end_page": row["id"],
        "text": row["essay_content"],
        "kind": "Essay",
    })

# -------- write output --------
with OUT.open("w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"Saved {len(out)} essays to {OUT}")







#endregion#? Create JSON with 100 essays from the mine_evaluation_dataset.json
#?#########################  End  ##########################




#*######################### Start ##########################
#region:#?     Create KG for each Essay V1




import json
import shutil
import time
import traceback
from pathlib import Path

from tqdm import tqdm  # pip install tqdm if needed


# =========================
# CONFIG
# =========================

BASE_DIR = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG")

DATA_DIR      = BASE_DIR / "data"
CHUNKS_DIR    = DATA_DIR / "Chunks"
CLASSES_DIR   = DATA_DIR / "Classes"
ENTITIES_DIR  = DATA_DIR / "Entities"
KG_DIR        = DATA_DIR / "KG"
REL_DIR       = DATA_DIR / "Relations"
PDF_JSON_DIR  = DATA_DIR / "pdf_to_json"

# File that sentence_chunks_token_driven reads from
SINGLE_PLAIN_TEXT_PATH = PDF_JSON_DIR / "Plain_Text.json"

# The ONLY external input file you told me to use
ALL_ESSAYS_PATH = BASE_DIR / "Experiments" / "MYNE" / "QA_and_OthersAnswers" / "Plain_Text_100_Essays.json"

# Where to store per‑essay snapshots of /data
# (siblings of /data): KG_Essay_000, KG_Essay_001, ...
PER_ESSAY_SNAPSHOTS_ROOT = BASE_DIR


# =========================
# HELPERS
# =========================

def load_all_essays():
    """
    Load Plain_Text_100_Essays.json and return:
      - root: original JSON root
      - essays: list of essay objects
    """
    with ALL_ESSAYS_PATH.open("r", encoding="utf-8") as f:
        root = json.load(f)

    if isinstance(root, list):
        essays = root
    elif isinstance(root, dict):
        essays = None
        for key in ("essays", "data", "items"):
            v = root.get(key)
            if isinstance(v, list):
                essays = v
                break
        if essays is None:
            raise ValueError(
                f"Could not find a list of essays in {ALL_ESSAYS_PATH}; "
                f"expected a list or a dict with 'essays'/'data'/'items'."
            )
    else:
        raise ValueError(
            f"Unexpected JSON root type in {ALL_ESSAYS_PATH}: {type(root)}"
        )

    return root, essays


def write_single_essay_to_plain_text(original_root, essay_obj):
    """
    Overwrite Plain_Text.json so that it contains ONLY this essay,
    without changing the overall container structure more than necessary.

    - If original_root is a list: [essay_obj]
    - If original_root is a dict with a list field (essays/data/items):
        same dict but that list replaced by [essay_obj]
    - Otherwise: just essay_obj as root.
    """
    SINGLE_PLAIN_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(original_root, list):
        new_root = [essay_obj]
    elif isinstance(original_root, dict):
        new_root = dict(original_root)
        replaced = False
        for key in ("essays", "data", "items"):
            if isinstance(original_root.get(key), list):
                new_root[key] = [essay_obj]
                replaced = True
                break
        if not replaced:
            new_root = {"essay": essay_obj}
    else:
        new_root = essay_obj

    with SINGLE_PLAIN_TEXT_PATH.open("w", encoding="utf-8") as f:
        json.dump(new_root, f, ensure_ascii=False, indent=2)


def clean_data_subdirs():
    """
    Remove everything inside (but not the dirs themselves):

      /data/Chunks
      /data/Classes
      /data/Entities
      /data/KG
      /data/Relations
    """
    for subdir in [CHUNKS_DIR, CLASSES_DIR, ENTITIES_DIR, KG_DIR, REL_DIR]:
        if not subdir.exists():
            continue
        for item in subdir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                print(f"[WARN] Failed to delete {item}: {e}")


def build_chunk_ids_from_file(chunks_jsonl_path: Path):
    """
    Build chunk_ids list for run_entity_extraction_on_chunks
    from /data/Chunks/chunks_sentence.jsonl
    """
    chunk_ids = []
    if not chunks_jsonl_path.exists():
        return chunk_ids

    with chunks_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = (
                obj.get("id")
                or obj.get("chunk_id")
                or obj.get("uid")
                or obj.get("chunkId")
            )
            if cid is not None:
                chunk_ids.append(cid)
    return chunk_ids


def snapshot_data_dir_for_essay(idx: int):
    """
    Copy /data to KG_Essay_{idx:03d} under BASE_DIR.
    """
    snapshot_dir = PER_ESSAY_SNAPSHOTS_ROOT / f"KG_Essay_{idx:03d}"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    print(f"[INFO] Copying {DATA_DIR} -> {snapshot_dir}")
    shutil.copytree(DATA_DIR, snapshot_dir)


# =========================
# MAIN DRIVER
# =========================

def run_trace_kg_for_all_essays():
    """
    For each essay in Plain_Text_100_Essays.json:

      1) Write it into /data/pdf_to_json/Plain_Text.json
      2) Run your run-statements IN THE ORDER YOU SPECIFIED:
         - sentence_chunks_token_driven(...)
         - embed_and_index_chunks(...)
         - run_entity_extraction_on_chunks(...)
         - iterative_resolution()
         - produce_clean_jsonl(input_path, out_file)
         - classrec_iterative_main()
         - main_input_for_cls_res()
         - run_pipeline_iteratively()
         - run_rel_rec(...  # from your Rel Rec run-statement)
         - run_relres_iteratively()
         - export_relations_and_nodes_to_csv()
      3) Copy /data -> KG_Essay_{i}
      4) Empty Chunks / Classes / Entities / KG / Relations
    """
    original_root, essays = load_all_essays()
    n_essays = len(essays)
    print(f"[INFO] Loaded {n_essays} essays from {ALL_ESSAYS_PATH}")

    # Optional: start from a clean state
    clean_data_subdirs()

    run_stats = []

    for idx, essay in enumerate(tqdm(essays, desc="TRACE KG per essay"), start=0):
        essay_label = (
            essay.get("id")
            if isinstance(essay, dict) and essay.get("id") is not None
            else essay.get("essay_id")
            if isinstance(essay, dict) and essay.get("essay_id") is not None
            else essay.get("title")
            if isinstance(essay, dict) and essay.get("title") is not None
            else f"essay_{idx}"
        )

        print("\n" + "=" * 80)
        print(f"=== Essay {idx:03d} :: {essay_label} ===")
        print("=" * 80)

        essay_stat = {
            "index": idx,
            "essay_id": essay_label,
        }

        t_run_start = time.time()
        success = False
        error_msg = None

        try:
            # --------------------------------
            # 0) Prepare input for chunking
            # --------------------------------
            write_single_essay_to_plain_text(original_root, essay)

            # --------------------------------
            # 1) Chunking - EXACT run statement
            # --------------------------------
            t0 = time.time()
            sentence_chunks_token_driven(
                "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/Plain_Text.json",
                "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                max_tokens_per_chunk=200,
                min_tokens_per_chunk=100,
                sentence_per_line=True,
                keep_ref_text=False,
                strip_leading_headings=True,
                force=True,
                debug=False,
            )
            essay_stat["t_chunking"] = time.time() - t0

            # --------------------------------
            # 2) embed_and_index_chunks - EXACT run statement
            # --------------------------------
            t0 = time.time()
            embed_and_index_chunks(
                "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_emb",
                "BAAI/bge-large-en-v1.5",
                "BAAI/bge-small-en-v1.5",
                False,   # use_small_model_for_dev
                32,      # batch_size
                None,    # device -> auto
                True,    # save_index
                True,    # force
            )
            essay_stat["t_embed_index"] = time.time() - t0

            # --------------------------------
            # 3) Build chunk_ids for Entity Recognition
            # --------------------------------
            t0 = time.time()
            chunks_path = Path(
                "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl"
            )
            chunk_ids = build_chunk_ids_from_file(chunks_path)
            essay_stat["n_chunks"] = len(chunk_ids)
            essay_stat["t_build_chunk_ids"] = time.time() - t0

            # --------------------------------
            # 4) Entity Recognition - EXACT run statement
            # --------------------------------
            t0 = time.time()
            run_entity_extraction_on_chunks(
                chunk_ids,
                prev_chunks=5,
                save_debug=False,
                model="gpt-5.1",
                max_tokens=8000,
            )
            essay_stat["t_entity_recognition"] = time.time() - t0

            # --------------------------------
            # 5) Ent Resolution (Multi Run) - EXACT run statement
            # --------------------------------
            t0 = time.time()
            iterative_resolution()
            essay_stat["t_entity_resolution"] = time.time() - t0

            # --------------------------------
            # 6) Cls Rec input producer - EXACT run statement
            # --------------------------------
            # you had: produce_clean_jsonl(input_path, out_file)
            # we just define input_path / out_file (paths are from your code)
            t0 = time.time()
            input_path = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/Ent_Res/Ent_Res_IterativeRuns/overall_summary/entities_with_class.jsonl"
            out_file   = "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Input/cls_input_entities.jsonl"
            produce_clean_jsonl(input_path, out_file)
            essay_stat["t_cls_input"] = time.time() - t0

            # --------------------------------
            # 7) Cls Recognition - EXACT run statement
            # --------------------------------
            t0 = time.time()
            classrec_iterative_main()
            essay_stat["t_cls_recognition"] = time.time() - t0

            # --------------------------------
            # 8) Create input for Cls Res - EXACT run statement
            # --------------------------------
            t0 = time.time()
            main_input_for_cls_res()
            essay_stat["t_cls_res_input"] = time.time() - t0

            # --------------------------------
            # 9) Cls Res Multi Run - EXACT run statement
            # --------------------------------
            t0 = time.time()
            run_pipeline_iteratively()
            essay_stat["t_cls_resolution"] = time.time() - t0

            # --------------------------------
            # 10) Relation Rec (Rel Rec) - from your Part Three code
            #     (you didn't list it in the last block, but it's needed
            #      to create relations_raw.jsonl for Rel Res)
            # --------------------------------
            t0 = time.time()
            run_rel_rec(
                entities_path="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl",
                chunks_path="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                output_path="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Relations/Rel Rec/relations_raw.jsonl",
                model="gpt-5.1",
            )
            essay_stat["t_rel_recognition"] = time.time() - t0

            # --------------------------------
            # 11) Relation Res Multi Run - EXACT run statement
            # --------------------------------
            t0 = time.time()
            run_relres_iteratively()
            essay_stat["t_rel_resolution"] = time.time() - t0

            # --------------------------------
            # 12) Export KG to CSVs - EXACT run statement
            # --------------------------------
            t0 = time.time()
            export_relations_and_nodes_to_csv()
            essay_stat["t_export_csv"] = time.time() - t0

            success = True

        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Essay {idx:03d} failed: {error_msg}")
            traceback.print_exc()

        finally:
            essay_stat["success"] = success
            essay_stat["error"] = error_msg
            essay_stat["t_total"] = time.time() - t_run_start

            # 13) Copy /data to KG_Essay_{idx}
            try:
                snapshot_data_dir_for_essay(idx)
            except Exception as e_snap:
                print(f"[WARN] Failed to snapshot data for essay {idx:03d}: {e_snap}")

            # 14) Clean data subdirs for next essay
            try:
                clean_data_subdirs()
            except Exception as e_clean:
                print(f"[WARN] Failed to clean data dirs after essay {idx:03d}: {e_clean}")

            run_stats.append(essay_stat)

    # Save per‑essay stats
    stats_path = BASE_DIR / "trace_kg_per_essay_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Per‑essay stats written to {stats_path}")


# =========================
# KICK IT OFF
# =========================

if __name__ == "__main__":
    run_trace_kg_for_all_essays()



#endregion#?   Create KG for each Essay
#*#########################  End  ##########################










#?######################### Start ##########################
#region:#?     Create KG for each Essay - V2


#!/usr/bin/env python3
"""
TRACE-KG batch driver (complete revised orchestration)

What this script does (drop-in driver; assumes all pipeline functions are already
defined in the Python environment you loaded earlier):

- Reads the essays file:
    /home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json

  The file is expected to contain either:
    - a JSON array of objects where each object contains at least a text field,
      and optionally an 'essay_id' or 'title' field; or
    - a dict with a top-level key like "essays" mapping to such an array.

  The code is defensive: it tries several common shapes.

- For each essay (i from 0..N-1):
    1. Writes a one-essay JSON file to the exact path your chunker expects:
       /home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/Plain_Text.json
       (we write a JSON array with a single object {"id": essay_id, "text": essay_text})
       This is the ONLY input path we modify/prepare for the chunker step.
    2. Runs the pipeline run statements in EXACT the order you provided:
       sentence_chunks_token_driven(...)  (chunking)
       embed_and_index_chunks(...)
       run_entity_extraction_on_chunks(...)
       iterative_resolution()
       produce_clean_jsonl(...)
       classrec_iterative_main()
       main_input_for_cls_res()
       run_pipeline_iteratively()
       run_relres_iteratively()
       export_relations_and_nodes_to_csv()
       (calls are made exactly as you specified; they assume functions exist)
    3. Measures timings for each step and captures success/failure + error message.
    4. After a successful (or partially successful) run for essay i:
       - Copies the entire data folder:
         /home/mabolhas/MyReposOnSOL/SGCE-KG/data
         ->  /home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/KG_Essay_{i:03d}
       - Then completely empties the following directories (so next essay starts fresh):
         /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks
         /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes
         /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities
         /home/mabolhas/MyReposOnSOL/SGCE-KG/data/KG
         /home/mabolhas/MyReposOnSOL/SGCE-KG/data/Relations
       - Recreates those directories as empty folders.
    5. If anything fails during the pipeline for essay i, the driver:
       - records the exception & stack trace into the per-essay stat,
       - still attempts to copy whatever was produced to KG_Essay_{i:03d} (best-effort),
       - then empties the data folders and continues with the next essay.
    6. Writes an overall JSON report at the end:
       /home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/TRACE_KG_batch_report.json

Notes / assumptions:
- This driver does NOT redefine or import any of your pipeline functions; it
  expects them to be available already in the Python process (as you requested).
- The only file it writes to the pipeline input paths is the single-essay JSON at
  /home/.../data/pdf_to_json/Plain_Text.json which your chunker will ingest.
- The helper ensure_entities_seed_exists() (below) will try to copy a sensible
  "entities" file into the expected seed path before iterative_resolution() runs.
  This addresses the FileNotFoundError you reported.
- The driver uses tqdm for progress and robust try/except to ensure one failing
  essay does not stop the whole batch.

Paste this file next to your other code and run it after you've defined all
functions. It will call them in order.
"""

import json
import os
import shutil
import time
import traceback
import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

# -----------------------
# Configuration (edit paths only if you *really* must)
# -----------------------

ESSAYS_PATH = Path(
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json"
)

# The single-essay JSON the chunker expects (we will overwrite this for each essay)
PIPELINE_SINGLE_ESSAY_PATH = Path(
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/Plain_Text.json"
)

DATA_ROOT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/data")
EXPERIMENTS_ROOT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE")
KG_OUTPUT_ROOT = EXPERIMENTS_ROOT / "KG_Essays"  # will contain KG_Essay_001, KG_Essay_002, ...
REPORT_PATH = EXPERIMENTS_ROOT / "TRACE_KG_batch_report.json"

# folders to clear between essays
DATA_SUBFOLDERS_TO_RESET = [
    DATA_ROOT / "Chunks",
    DATA_ROOT / "Classes",
    DATA_ROOT / "Entities",
    DATA_ROOT / "KG",
    DATA_ROOT / "Relations",
]

# iterative_resolution expects a seed file here (fixes missing-file issue)
ITERATIVE_SEED_EXPECTED = Path(
    "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Entities/iterative_runs/entities_raw_seed_backup.jsonl"
)


# -----------------------
# Helper: ensure the iterative_resolution seed file exists (robust)
# -----------------------
def ensure_entities_seed_exists(target_seed_path: str) -> str:
    """
    Ensure that the iterative_resolution seed file exists at target_seed_path.
    If it doesn't, search for common entity output files and copy the first one found.
    Returns the path of the file used as seed (target_seed_path).
    Raises FileNotFoundError if nothing suitable is found.
    """
    import shutil
    import glob

    target = Path(target_seed_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        return str(target)

    # Candidate explicit files (ordered)
    candidates = [
        str(DATA_ROOT / "Entities" / "Ent_Raw_0" / "entities_raw.jsonl"),
        str(DATA_ROOT / "Entities" / "Ent_Raw_0" / "entities_raw_seed.jsonl"),
        str(DATA_ROOT / "Entities" / "Ent_Raw_0" / "entities_raw_seed_backup.jsonl"),
        str(DATA_ROOT / "Entities" / "Ent_Res" / "entities_raw.jsonl"),
        str(DATA_ROOT / "Entities" / "Ent_Res" / "entities_with_class.jsonl"),
        str(DATA_ROOT / "Entities" / "entities_raw.jsonl"),
        str(DATA_ROOT / "Entities" / "entities_with_class.jsonl"),
        # fallback: overall_summary locations used elsewhere
        str(DATA_ROOT / "Entities" / "Ent_Res_IterativeRuns" / "overall_summary" / "entities_with_class.jsonl"),
        str(DATA_ROOT / "Entities" / "Ent_Res_IterativeRuns" / "overall_summary" / "entities_raw.jsonl"),
    ]

    # glob patterns to search
    glob_patterns = [
        str(DATA_ROOT / "Entities" / "**" / "*entities*.jsonl"),
        str(DATA_ROOT / "Entities" / "**" / "*ent*.jsonl"),
        str(DATA_ROOT / "Entities" / "**" / "*.jsonl"),
    ]

    # try explicit candidates
    for c in candidates:
        p = Path(c)
        if p.exists() and p.is_file():
            shutil.copy2(p, target)
            return str(target)

    # try globs and prefer files that include 'entity' or 'entities' in the name
    for pat in glob_patterns:
        for pstr in glob.glob(pat, recursive=True):
            p = Path(pstr)
            if not p.is_file():
                continue
            name = p.name.lower()
            # prefer containing 'entity' or 'entities' or 'raw'
            if ("entity" in name) or ("entities" in name) or ("raw" in name):
                shutil.copy2(p, target)
                return str(target)

    # last resort: try ANY jsonl inside Entities
    all_jsonl = list(Path(DATA_ROOT / "Entities").rglob("*.jsonl"))
    for p in all_jsonl:
        if p.is_file():
            shutil.copy2(p, target)
            return str(target)

    # if nothing found, raise helpful error
    raise FileNotFoundError(
        f"iterative_resolution seed missing and no candidate entity outputs found.\n"
        f"Expected: {target}\n"
        f"Checked explicit candidates and glob patterns under {DATA_ROOT}/Entities."
    )


# -----------------------
# Helper: write the single-essay JSON the chunker expects
# -----------------------
def write_single_essay_for_chunker(essay_id: str, essay_text: str) -> None:
    """
    Writes a single-essay JSON file to PIPELINE_SINGLE_ESSAY_PATH.
    We write a JSON array with one object:
      [{"id": essay_id, "text": essay_text}]
    This format is commonly accepted by many simple chunkers; it's also what the
    user requested: single-essay content at that exact path.
    """
    PIPELINE_SINGLE_ESSAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    obj = [{"id": essay_id, "text": essay_text}]
    with PIPELINE_SINGLE_ESSAY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)


# -----------------------
# Helper: copy data folder to KG_Essay_{i} (timestamped)
# -----------------------
def copy_data_to_kg_folder(essay_index: int, essay_id: str) -> Path:
    """
    Copies the entire DATA_ROOT folder to an experiment folder:
      EXPERIMENTS_ROOT/KG_Essay_{i:03d}__{safe_essay_id}__{timestamp}
    Returns path to the destination folder.
    """
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in (essay_id or f"essay_{essay_index}"))[:100]
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    dest_dir = KG_OUTPUT_ROOT / f"KG_Essay_{essay_index:03d}__{safe_id}__{ts}"
    dest_dir_parent = dest_dir.parent
    dest_dir_parent.mkdir(parents=True, exist_ok=True)
    # copytree requires dest not exist
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(DATA_ROOT, dest_dir)
    return dest_dir


# -----------------------
# Helper: empty/reset data subfolders
# -----------------------
def reset_data_subfolders() -> None:
    """
    Remove all files and folders inside the configured DATA_SUBFOLDERS_TO_RESET.
    Recreate them as empty directories.
    """
    for folder in DATA_SUBFOLDERS_TO_RESET:
        try:
            if folder.exists():
                shutil.rmtree(folder)
        except Exception:
            # don't fail the whole run if one remove fails; try to continue
            print(f"[warn] failed to remove {folder}: {traceback.format_exc()}")
        finally:
            # recreate
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"[warn] failed to recreate {folder}: {e}")


# -----------------------
# Helper: robust call wrapper for pipeline steps
# -----------------------
def call_step(func, *args, step_name: str = "", **kwargs):
    """
    Calls func(*args, **kwargs) and measures time. Returns (success, elapsed_seconds, exception_info_or_None).
    """
    t0 = time.time()
    try:
        func(*args, **kwargs)
        dt = time.time() - t0
        return True, dt, None
    except Exception as e:
        dt = time.time() - t0
        tb = traceback.format_exc()
        return False, dt, {"error": str(e), "traceback": tb, "step": step_name}


# -----------------------
# Load essays defensively
# -----------------------
def load_essays(path: Path) -> List[Dict[str, Any]]:
    """
    Try to load essays from path. Accepts common shapes:
      - JSON array of objects -> returns that array
      - JSON object with 'essays' or 'items' key -> returns that value if it's a list
      - object with numeric keys -> treat values as essays
    Each essay object is normalized to contain 'essay_id' and 'text'.
    """
    if not path.exists():
        raise FileNotFoundError(f"Essays file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    candidates = []

    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, dict):
        # try common keys
        for k in ("essays", "items", "data", "documents"):
            if k in raw and isinstance(raw[k], list):
                candidates = raw[k]
                break
        # fallback: values that are lists or dicts
        if not candidates:
            # if values look like essays
            vals = [v for v in raw.values() if isinstance(v, (list, dict, str))]
            if vals and isinstance(vals[0], list):
                candidates = vals[0]
            else:
                # try to convert dict entries to essay objects
                # e.g., {"0": {"text": "...", "title": ...}, ...}
                maybe = []
                for key, val in raw.items():
                    if isinstance(val, dict) and ("text" in val or "essay" in val or "body" in val):
                        maybe.append(val)
                if maybe:
                    candidates = maybe
    # last resort: if none discovered, but the file itself contains a text field
    if not candidates:
        # if raw looks like a single essay dict
        keys_lower = set(k.lower() for k in (raw.keys() if isinstance(raw, dict) else []))
        if isinstance(raw, dict) and ("text" in keys_lower or "essay" in keys_lower or "body" in keys_lower):
            candidates = [raw]
    # normalize
    normalized = []
    for i, item in enumerate(candidates):
        if isinstance(item, str):
            normalized.append({"essay_id": f"essay_{i}", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        essay_id = item.get("essay_id") or item.get("id") or item.get("title") or item.get("essay_title") or item.get("name")
        # prefer explicit text fields in common names
        text = item.get("text") or item.get("body") or item.get("essay") or item.get("content")
        if text is None:
            # maybe the item itself is plain string in some nested key
            # try to find the first string value
            for v in item.values():
                if isinstance(v, str) and len(v) > 20:
                    text = v
                    break
        if text is None:
            continue
        normalized.append({"essay_id": essay_id or f"essay_{i}", "text": text})
    return normalized


# -----------------------
# Main batch loop
# -----------------------
def run_batch():
    essays = load_essays(ESSAYS_PATH)
    n = len(essays)
    print(f"[INFO] Loaded {n} essays from {ESSAYS_PATH}")

    # create KG output root
    KG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    overall_stats = {
        "total_essays": n,
        "essays": [],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Step-by-step function names expected to be present in the environment
    # The user insisted these exact calls be used.
    # We'll call them via globals() lookups to avoid importing/redefining.
    def get_callable(name: str):
        obj = globals().get(name)
        if obj is None:
            raise RuntimeError(f"Required pipeline function not found in environment: {name}")
        return obj

    # Lookup functions (these must have been loaded previously by the user)
    # We do not redefine them here; we simply call them.
    required_functions = {
        "sentence_chunks_token_driven": "sentence_chunks_token_driven",
        "embed_and_index_chunks": "embed_and_index_chunks",
        "run_entity_extraction_on_chunks": "run_entity_extraction_on_chunks",
        "iterative_resolution": "iterative_resolution",
        "produce_clean_jsonl": "produce_clean_jsonl",
        "classrec_iterative_main": "classrec_iterative_main",
        "main_input_for_cls_res": "main_input_for_cls_res",
        "run_pipeline_iteratively": "run_pipeline_iteratively",
        "run_relres_iteratively": "run_relres_iteratively",
        "export_relations_and_nodes_to_csv": "export_relations_and_nodes_to_csv",
    }

    # Validate all required functions exist (fail fast)
    for fname in required_functions.values():
        if fname not in globals():
            raise RuntimeError(f"Pipeline function not defined in current environment: {fname}")

    # Prepare the chunk_ids variable required by run_entity_extraction_on_chunks
    # Many of your run statements expected that `chunk_ids` or `input_path` variables exist.
    # We will try to set chunk_ids to None and rely on the function to derive from chunks file.
    # If your function expects a specific chunk_ids list, adapt accordingly in your environment.
    chunk_ids = None
    input_path = None
    out_file = None

    # For each essay, run the pipeline steps in order
    for idx, essay in enumerate(tqdm(essays, desc="Essays")):
        essay_stat: Dict[str, Any] = {
            "index": idx,
            "essay_id": essay.get("essay_id") or f"essay_{idx}",
            "t_chunking": None,
            "t_embed_index": None,
            "n_chunks": None,
            "t_build_chunk_ids": None,
            "t_entity_recognition": None,
            "t_entity_resolution": None,
            "t_cls_rec_input": None,
            "t_cls_rec": None,
            "t_cls_res_input": None,
            "t_cls_res": None,
            "t_rel_res": None,
            "t_export_csv": None,
            "success": False,
            "error": None,
            "t_total": None,
        }
        start_total = time.time()
        essay_id = essay_stat["essay_id"]
        essay_text = essay.get("text", "")

        print(f"\n[INFO] Starting essay {idx} id={essay_id}")

        try:
            # 0) Prepare single-essay JSON the chunker reads
            write_single_essay_for_chunker(essay_id=essay_id, essay_text=essay_text)

            # -----------------------
            # 1) Chunking - Run statement (must use the exact call)
            # -----------------------
            t0 = time.time()
            # call the function as the user wrote it
            sentence_chunks_token_driven(
                str(PIPELINE_SINGLE_ESSAY_PATH),
                str(DATA_ROOT / "Chunks" / "chunks_sentence.jsonl"),
                max_tokens_per_chunk=200,
                min_tokens_per_chunk=100,
                sentence_per_line=True,
                keep_ref_text=False,
                strip_leading_headings=True,
                force=True,
                debug=False
            )
            essay_stat["t_chunking"] = time.time() - t0

            # set n_chunks if the chunker wrote the chunks file
            chunks_file = DATA_ROOT / "Chunks" / "chunks_sentence.jsonl"
            if chunks_file.exists():
                try:
                    with chunks_file.open("r", encoding="utf-8") as fh:
                        n_chunks = sum(1 for _ in fh)
                    essay_stat["n_chunks"] = n_chunks
                except Exception:
                    essay_stat["n_chunks"] = None

            # -----------------------
            # 2) embed_and_index_chunks  - Run statement
            # -----------------------
            t0 = time.time()
            embed_and_index_chunks(
                str(DATA_ROOT / "Chunks" / "chunks_sentence.jsonl"),
                str(DATA_ROOT / "Chunks" / "chunks_emb"),
                "BAAI/bge-large-en-v1.5",
                "BAAI/bge-small-en-v1.5",
                False,   # use_small_model_for_dev
                32,     # batch_size
                None,   # device -> auto
                True,   # save_index
                True    # force
            )
            essay_stat["t_embed_index"] = time.time() - t0

            # -----------------------
            # 3) Entity Recognition  - Run statement
            # -----------------------
            t0 = time.time()
            # Note: run_entity_extraction_on_chunks originally used variable chunk_ids.
            # We'll pass chunk_ids as-is (None) and let the function read chunks file if implemented that way.
            run_entity_extraction_on_chunks(
                chunk_ids,
                prev_chunks=5,
                save_debug=False,
                model="gpt-5.1",
                max_tokens=8000
            )
            essay_stat["t_entity_recognition"] = time.time() - t0

            # 4) Ensure seed exists for iterative_resolution (avoid FileNotFoundError)
            try:
                ensure_entities_seed_exists(str(ITERATIVE_SEED_EXPECTED))
            except Exception as e_seed:
                # If we cannot prepare seed, raise to be caught by outer try
                raise RuntimeError(f"Failed to prepare iterative_resolution seed: {e_seed}")

            # -----------------------
            # 5) Ent Resolution (Multi Run)  - Run statement
            # -----------------------
            t0 = time.time()
            iterative_resolution()
            essay_stat["t_entity_resolution"] = time.time() - t0

            # -----------------------
            # 6) Cls Rec input producer - Run statement
            # -----------------------
            t0 = time.time()
            # produce_clean_jsonl(input_path, out_file)
            # We don't know the exact var names used by your function; attempt to call with placeholders.
            # If your function is signature-sensitive, please ensure the names input_path/out_file are defined in your environment.
            produce_clean_jsonl(input_path, out_file)
            essay_stat["t_cls_rec_input"] = time.time() - t0

            # -----------------------
            # 7) Cls Recognition  - Run statement
            # -----------------------
            t0 = time.time()
            classrec_iterative_main()
            essay_stat["t_cls_rec"] = time.time() - t0

            # -----------------------
            # 8) Create input for Cls Res  - Run statement
            # -----------------------
            t0 = time.time()
            main_input_for_cls_res()
            essay_stat["t_cls_res_input"] = time.time() - t0

            # -----------------------
            # 9) Cls Res Multi Run - Run statement
            # -----------------------
            t0 = time.time()
            run_pipeline_iteratively()
            essay_stat["t_cls_res"] = time.time() - t0

            # -----------------------
            # 10) Relation Res Multi Run - Run statement
            # -----------------------
            t0 = time.time()
            run_relres_iteratively()
            essay_stat["t_rel_res"] = time.time() - t0

            # -----------------------
            # 11) Export KG to CSVs  - Run statement
            # -----------------------
            t0 = time.time()
            export_relations_and_nodes_to_csv()
            essay_stat["t_export_csv"] = time.time() - t0

            # Mark success
            essay_stat["success"] = True

        except Exception as e:
            # record error and continue to copy/reset
            tb = traceback.format_exc()
            essay_stat["error"] = {"error": str(e), "traceback": tb}
            essay_stat["success"] = False
            print(f"[error] essay {idx} failed: {e}\n{tb}")

        finally:
            # measure total time
            essay_stat["t_total"] = time.time() - start_total

            # copy DATA_ROOT to KG_Essay folder (best-effort)
            try:
                dest = copy_data_to_kg_folder(idx, essay_id)
                essay_stat["copied_to"] = str(dest)
                print(f"[info] copied data to {dest}")
            except Exception as e_copy:
                essay_stat.setdefault("copy_error", str(e_copy))
                print(f"[warn] failed to copy data for essay {idx}: {e_copy}\n{traceback.format_exc()}")

            # reset the data subfolders so next essay runs fresh
            try:
                reset_data_subfolders()
            except Exception as e_reset:
                print(f"[warn] failed to reset data subfolders after essay {idx}: {e_reset}\n{traceback.format_exc()}")

            # append essay stat to overall
            overall_stats["essays"].append(essay_stat)

            # write incremental report to disk (so progress is saved even if driver crashes)
            try:
                REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
                with REPORT_PATH.open("w", encoding="utf-8") as fh:
                    json.dump(overall_stats, fh, ensure_ascii=False, indent=2)
            except Exception as e_report:
                print(f"[warn] failed to write report: {e_report}")

    # final report write (ensure last snapshot)
    overall_stats["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(overall_stats, fh, ensure_ascii=False, indent=2)

    print(f"\n[done] Batch finished. Report: {REPORT_PATH}")


# -----------------------
# Run the batch when executed directly
# -----------------------
if __name__ == "__main__":
    run_batch()



#endregion#?   Create KG for each Essay - V2
#?#########################  End  ##########################


  


#?######################### Start ##########################
#region:#?  Create KG for each Essay  - V3


"""
TRACE KG multi-essay runner

Drop this near the BOTTOM of your main .py file (AFTER all function definitions),
or put it in a separate script that imports those functions.

It will:

  - Read 100 essays from:
        /home/mabolhas/MyReposOnSOL/SGCE-KG/Experiments/MYNE/QA_and_OthersAnswers/Plain_Text_100_Essays.json

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
              /home/mabolhas/MyReposOnSOL/SGCE-KG/KGs_from_Essays/KG_Essay_<i>

        * Clear the pipeline data folders again so the next essay is independent

  - Use tqdm for progress
  - Record per-essay timings, success/fail, and basic counts to:
        /home/mabolhas/MyReposOnSOL/SGCE-KG/KGs_from_Essays/trace_kg_essays_run_stats.json
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

BASE_ROOT = Path("/home/mabolhas/MyReposOnSOL/SGCE-KG")
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
                "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/pdf_to_json/Plain_Text.json",
                "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
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
                "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                "/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_emb",
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
                entities_path="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Classes/Cls_Res/Cls_Res_IterativeRuns/overall_summary/entities_with_class.jsonl",
                chunks_path="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Chunks/chunks_sentence.jsonl",
                output_path="/home/mabolhas/MyReposOnSOL/SGCE-KG/data/Relations/Rel Rec/relations_raw.jsonl",
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

