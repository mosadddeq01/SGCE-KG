
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




