from typing import Dict
import numpy as np
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from atlas_rag.retriever.base import BaseEdgeRetriever, BasePassageRetriever
from atlas_rag.retriever.inference_config import InferenceConfig
import json_repair
from networkx import DiGraph

class SimpleGraphRetriever(BaseEdgeRetriever):

    def __init__(self, llm_generator:LLMGenerator, sentence_encoder:BaseEmbeddingModel, 
                 data:dict):
        
        self.KG = data["KG"]
        self.node_list = data["node_list"]
        self.edge_list = data["edge_list"]
        
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder

        self.node_faiss_index = data["node_faiss_index"]
        self.edge_faiss_index = data["edge_faiss_index"]


    def retrieve(self, query, topN=5, **kwargs):
        # retrieve the top k edges
        topk_edges = []
        query_embedding = self.sentence_encoder.encode([query], query_type='edge')
        D, I = self.edge_faiss_index.search(query_embedding, topN)

        topk_edges += [self.edge_list[i] for i in I[0]]

        topk_edges_with_data = [(edge[0], self.KG.edges[edge]["relation"], edge[1]) for edge in topk_edges]
        string_edge_edges = [f"{self.KG.nodes[edge[0]]['id']}  {edge[1]}  {self.KG.nodes[edge[2]]['id']}" for edge in topk_edges_with_data]

        return string_edge_edges, ["N/A" for _ in range(len(string_edge_edges))]

class SimpleTextRetriever(BasePassageRetriever):
    def __init__(self, passage_dict:Dict[str,str], sentence_encoder:BaseEmbeddingModel, data:dict, inference_config:InferenceConfig=None):  
        self.sentence_encoder = sentence_encoder
        self.passage_dict = passage_dict
        self.passage_list = list(passage_dict.values())
        self.passage_keys = list(passage_dict.keys())
        self.text_embeddings = data["text_embeddings"]
        self.KG = data["KG"]
        self.inference_config = inference_config if inference_config is not None else InferenceConfig()
        node_id_to_file_id = {}
        text_id_to_node_name = {}
        for node_id in list(self.KG.nodes):
            if self.inference_config.keyword == "musique" and self.KG.nodes[node_id]['type']=="passage":
                text_id_to_node_name[node_id] = self.KG.nodes[node_id]["id"]
            elif self.KG.nodes[node_id]['type']=="passage":
                text_id_to_node_name[node_id] = self.KG.nodes[node_id]["id"]
            else:
                node_id_to_file_id[node_id] = self.KG.nodes[node_id]["file_id"]
        self.node_id_to_file_id = node_id_to_file_id
        self.text_id_to_node_name = text_id_to_node_name
        
    def retrieve(self, query, topN=5, **kwargs):
        query_emb = self.sentence_encoder.encode([query], query_type="passage")
        sim_scores = self.text_embeddings @ query_emb[0].T
        topk_indices = np.argsort(sim_scores)[-topN:][::-1]  # Get indices of top-k scores

        # Retrieve top-k passages
        topk_passages = [self.passage_list[i] for i in topk_indices]
        topk_passages_ids = [self.passage_keys[i] for i in topk_indices]
        topk_passages_ids = [self.text_id_to_node_name[pid] for pid in topk_passages_ids]
        return topk_passages, topk_passages_ids

class SubgraphRetriever(BaseEdgeRetriever):
    def __init__(self, llm_generator: LLMGenerator, sentence_encoder: BaseEmbeddingModel, data: dict, config: InferenceConfig = None):
        self.config = config if config is not None else InferenceConfig()
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.KG = data["KG"]
        self.node_list = data["node_list"]
        self.edge_list = data["edge_list"]
        self.node_faiss_index = data["node_faiss_index"]
        self.edge_faiss_index = data["edge_faiss_index"]
        self.node_embeddings = data["node_embeddings"]
        self.triple_embeddings = data["edge_embeddings"]
        self.num_hop = self.config.num_hop

        self.node_id_to_attr_id = {self.KG.nodes[n]['id']: n for n in self.KG.nodes}
        self.KG = self.KG.subgraph(self.node_list)  # Ensure KG only contains nodes in node_list

    def ner(self, text):
        """Extract topic entities from the query using LLM."""
        messages = [
            {
                "role": "system",
                "content": "Extract the named entities from the provided question and output them as a JSON object in the format: {\"entities\": [\"entity1\", \"entity2\", ...]}"
            },
            {
                "role": "user",
                "content": f"Extract all the named entities from: {text}"
            }
        ]
        response = self.llm_generator.generate_response(messages)
        entities_json = json_repair.loads(response)
        if "entities" not in entities_json or not isinstance(entities_json["entities"], list):
            return {}
        return entities_json

    def retrieve_topk_nodes(self, query, topN=5):
        """Retrieve top-k nodes relevant to the query using FAISS index."""
        entities_json = self.ner(query)
        entities = entities_json.get("entities", [])
        if not entities:
            entities = [query]
        topk_nodes = []
        entities_not_in_kg = []
        entities = list(set(str(e) for e in entities))  # Remove duplicates

        for entity in entities:
            entity = self.node_id_to_attr_id.get(entity, entity)
            if entity in self.node_list:
                topk_nodes.append(entity)
            else:
                entities_not_in_kg.append(entity)
        if entities_not_in_kg:
            query_embeddings = self.sentence_encoder.encode(entities_not_in_kg, query_type='node')
            D, I = self.node_faiss_index.search(query_embeddings, 1)  # Get top-1 node per entity
            for i in range(I.shape[0]):
                top_node = self.node_list[I[i][0]]
                topk_nodes.append(top_node)
        return list(set(topk_nodes))  # Remove duplicates

    def construct_subgraph(self, initial_nodes):
        """Construct a multi-hop subgraph around initial nodes up to self.num_hop."""
        subgraph = DiGraph()
        visited = set()
        queue = [(node, 0) for node in initial_nodes if node in self.node_list]

        # Add initial nodes
        for node, _ in queue:
            subgraph.add_node(node)
            visited.add(node)

        # Breadth-first search to collect neighbors
        while queue:
            current_node, hop_count = queue.pop(0)
            if hop_count >= self.num_hop:
                continue
            # Add successors (outgoing edges)
            for neighbor in self.KG.successors(current_node):
                neighbor_id = self.KG.nodes[neighbor].get('id', None)
                if neighbor_id.isdigit():
                    # Do not further explore this neighbor
                    relation = self.KG.edges[(current_node, neighbor)]["relation"]
                    subgraph.add_edge(current_node, neighbor, relation=relation)
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                relation = self.KG.edges[(current_node, neighbor)]["relation"]
                subgraph.add_edge(current_node, neighbor, relation=relation)

            # Add predecessors (incoming edges)
            for neighbor in self.KG.predecessors(current_node):
                neighbor_id = self.KG.nodes[neighbor].get('id', None)
                if neighbor_id.isdigit():
                    # Do not further explore this neighbor
                    relation = self.KG.edges[(neighbor, current_node)]["relation"]
                    subgraph.add_edge(neighbor, current_node, relation=relation)
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                relation = self.KG.edges[(neighbor, current_node)]["relation"]
                subgraph.add_edge(neighbor, current_node, relation=relation)

        return subgraph

    def retrieve(self, question, **kwargs) -> str:
        """Retrieve a subgraph (or full KG) and generate an answer."""
        self.sub_queries = kwargs.get("sub_queries", [])

        initial_nodes = self.retrieve_topk_nodes(question)
        subgraph = self.construct_subgraph(initial_nodes)
        subgraph_edges = len(subgraph.edges)
        triples = [(self.KG.nodes[u]['id'], d["relation"], self.KG.nodes[v]['id']) for u, v, d in subgraph.edges(data=True)]
        return [f"({s}, {r}, {o})" for s, r, o in triples], ["N/A" for _ in range(subgraph_edges)]
