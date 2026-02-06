import os
import json
import numpy as np
from logging import Logger
from atlas_rag.retriever.base import BaseRetriever, BaseEdgeRetriever, BasePassageRetriever
from typing import List
from datetime import datetime
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from atlas_rag.vectorstore.embedding_model import NvEmbed, SentenceEmbedding
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from atlas_rag.evaluation.evaluation import QAJudger
from dataclasses import dataclass
from atlas_rag.llm_generator.prompt.react import ReAct
from dataclasses import asdict

def normalize_embeddings(embeddings):
    """Normalize the embeddings to unit length (L2 norm)."""
    if isinstance(embeddings, torch.Tensor):
        # Handle PyTorch tensors
        norm_emb = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
    elif isinstance(embeddings, np.ndarray):
        # Handle numpy arrays
        norm_emb = F.normalize(torch.tensor(embeddings), p=2, dim=1).detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported input type: {type(embeddings)}. Must be torch.Tensor or np.ndarray")
    
    return norm_emb

@dataclass
class BenchMarkConfig:
    """
    Configuration class for benchmarking.

    Attributes:
        dataset_name (str): Name of the dataset. Default is "hotpotqa".
        question_file (str): Path to the question file. Default is "hotpotqa".
        graph_file (str): Path to the graph file. Default is "hotpotqa_concept.graphml".
        include_events (bool): Whether to include events. Default is False.
        include_concept (bool): Whether to include concepts. Default is False.
        reader_model_name (str): Name of the reader model. Default is "meta-llama/Llama-2-7b-chat-hf".
        encoder_model_name (str): Name of the encoder model. Default is "nvidia/NV-Embed-v2".
        number_of_samples (int): Number of samples to use from the dataset. Default is -1 (use all samples).
        react_max_iterations (int): Maximum iterations for ReAct. Default is 5.
        result_dir (str): Directory to store results. Default is "./result".
        upper_bound_mode (bool): Whether to use upper bound mode. Default is False.
        topN (int): Number of top passages to retrieve. Default is 5.
    """
    dataset_name: str = "hotpotqa"
    question_file: str = "hotpotqa"
    include_events: bool = False
    include_concept: bool = False
    reader_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    encoder_model_name: str = "nvidia/NV-Embed-v2"
    number_of_samples: int = -1  # Default to -1 to use all samples
    react_max_iterations: int = 5
    result_dir: str = "./result"
    upper_bound_mode: bool = False
    topN: int = 5  # Number of top passages to retrieve

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in asdict(self).items())
    

class RAGBenchmark:
    def __init__(self, config:BenchMarkConfig, logger:Logger = None):
        self.config = config
        self.logger = logger
        self.logging : bool = self.logger is not None

    def load_encoder_model(self, encoder_model_name, **kwargs):
        if encoder_model_name == "nvidia/NV-Embed-v2":
            sentence_encoder = AutoModel.from_pretrained("nvidia/NV-Embed-v2", **kwargs)
            return NvEmbed(sentence_encoder)
        else:
            sentence_encoder = SentenceTransformer(encoder_model_name, **kwargs)
            return SentenceEmbedding(sentence_encoder)

    def run(self, retrievers:List[BaseRetriever], 
                  llm_generator:LLMGenerator,
                  use_react: bool = False):
        qa_judge = QAJudger()
        if use_react:
            react_agent = ReAct(llm_generator=llm_generator)
        result_list = []
        with open(self.config.question_file, "r") as f:
            data = json.load(f)
            print(f"Data loaded from {self.config.question_file}")
        if self.config.number_of_samples > 0:
            data = data[:self.config.number_of_samples]
            print(f"Using only the first {self.config.number_of_samples} samples from the dataset")
        for sample in tqdm(data):
            question = sample.get("question","")
            answer = sample.get("answer", "")

            gold_file_ids = []
            gold_paragraphs = []
            full_list_passages_contents = set()
            full_list_passages_ids = set()
            if self.config.dataset_name in ("hotpotqa", "2wikimultihopqa","popqa","nq"):
                for fact in sample["supporting_facts"]:
                    gold_file_ids.append(fact[0])
                    if self.config.dataset_name in ["2wikimultihopqa", "popqa", "nq"]:
                        for text in sample["context"]:
                            if text[0] == fact[0]:
                                # text[1] in the form of [ "Teutberga( died 11 November 875) was a queen of Lotharingia by marriage to Lothair II.","She was a daughter of Bosonid Boso the Elder and sister of Hucbert, the lay- abbot of St. Maurice's Abbey."]
                                # join them as one string
                                gold_para = " ".join(text[1])
                                gold_paragraphs.append(f"{fact[0]}: {gold_para}")
                            else:
                                full_list_passages_contents.add(f"{text[0]}: {' '.join(text[1])}")
                                full_list_passages_ids.add(text[0])
                    elif self.config.dataset_name == "hotpotqa":
                        for text in sample["context"]:
                            if text[0] == fact[0]:
                                gold_paragraphs.append(f"{fact[0]}: {' '.join(text[1])}")
                            else:
                                full_list_passages_contents.add(f"{text[0]}: {' '.join(text[1])}")
                                full_list_passages_ids.add(text[0])
            elif self.config.dataset_name == "musique":
                answer_list = []
                answer_list.append(answer)
                for answer in sample.get("answer_aliases", []):
                    answer_list.append(answer)
                answer = answer_list
                for paragraph in sample["paragraphs"]:
                    if paragraph["is_supporting"]:
                        gold_file_ids.append(paragraph["paragraph_text"])
                        gold_paragraphs.append(f'{paragraph["title"]}: {paragraph["paragraph_text"]}')
                    else:
                        full_list_passages_contents.add(f'{paragraph["title"]}: {paragraph["paragraph_text"]}')
                        full_list_passages_ids.add(paragraph["paragraph_text"])
            elif self.config.dataset_name == "hipporag_nq":
                answer = sample.get("reference") # it is a list here
                for context in sample["contexts"]:
                    if context['is_supporting']:
                        gold_file_ids.append(context["title"])
                        gold_paragraphs.append(f"{context['title']}: {context['text']}")
                    else:
                        full_list_passages_contents.add(f"{context['title']}: {context['text']}")
                        full_list_passages_ids.add(context["title"])
            elif self.config.dataset_name == "hipporag_popqa":
                gold_ans = set([sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
                gold_ans = list(gold_ans)
                answer = gold_ans
                for context in sample["paragraphs"]:
                    gold_file_ids.append(context["title"])
                    gold_paragraphs.append(f"{context['title']}: {context['text']}")
            else:
                print("Dataset not supported")
                continue
            
            result = {
                "question": question,
                "answer": answer,
                "gold_file_ids": gold_file_ids,
            } 
            
            if self.logging:
                self.logger.info(f"Question: {question}")
            for retriever in retrievers:
                if use_react:
                    # Use RAG with ReAct
                    llm_generated_answer, search_history = react_agent.generate_with_rag_react(
                        question=question,
                        retriever=retriever,
                        max_iterations=self.config.react_max_iterations,
                        max_new_tokens=2048, 
                        logger=self.logger
                    )
                    self.logger.info(f"Search history: {search_history}")
                    self.logger.info(f"Final answer: {llm_generated_answer}")
                    # Store search history in results
                    result[f"{retriever.__class__.__name__}_search_history"] = search_history
                    
                    # Extract all retrieved contexts from search history
                    all_contexts = []
                    for _, action, observation in search_history:
                        if "search" in action.lower() or "look up" in action.lower():
                            all_contexts.append(observation)
                    
                    sorted_context = "\n".join(all_contexts)
                    sorted_context_ids = []  # We don't track IDs in ReAct mode
                elif self.config.upper_bound_mode:
                    # use the golden doc for rag
                    sorted_context, sorted_context_ids = retriever.retrieve(question, topN=self.config.topN, 
                                                                          sorted_passages_contents=gold_paragraphs, 
                                                                          sorted_passage_ids=gold_file_ids,
                                                                          full_list_passages_contents=full_list_passages_contents,
                                                                          full_list_passages_ids=full_list_passages_ids)
                    llm_generated_answer = llm_generator.generate_with_context(question, sorted_context, max_new_tokens=2048, temperature=0.5)
                else:
                    # Original RAG implementation
                    sorted_context, sorted_context_ids = retriever.retrieve(question, topN=self.config.topN)
                    
                    if isinstance(retriever, BaseEdgeRetriever):
                        retrieved_context = "\n".join(sorted_context)
                        llm_generated_answer = llm_generator.generate_with_context_kg(question, retrieved_context, max_new_tokens=2048, temperature=0.0)
                    elif isinstance(retriever, BasePassageRetriever):
                        retrieved_context = "\n".join(sorted_context)
                        llm_generated_answer = llm_generator.generate_with_context(question, retrieved_context, max_new_tokens=2048, temperature=0.0)
                
                if self.logging:
                    self.logger.info(f"{retriever.__class__.__name__} retrieved passages: {sorted_context}")
                    self.logger.info(f"{retriever.__class__.__name__} generated answer: {llm_generated_answer}")
                    self.logger.info(f"Gold answer: {answer}")
                    self.logger.info(f'Gold paragraphs: {gold_paragraphs}')
                short_answer = qa_judge.split_answer(llm_generated_answer)
                em, f1 = qa_judge.judge(short_answer, answer)
                
                result[f"{retriever.__class__.__name__ }_em"] = em
                result[f"{retriever.__class__.__name__ }_f1"] = f1
                result[f"{retriever.__class__.__name__ }_passages"] = sorted_context
                if not use_react:
                    result[f"{retriever.__class__.__name__ }_id"] = sorted_context_ids
                result[f"{retriever.__class__.__name__ }_generated_answer"] = llm_generated_answer
                result[f"{retriever.__class__.__name__ }short_answer"] = short_answer
                
                # Calculate recall
                if not use_react:  # Only calculate recall for non-ReAct mode
                    recall_2, recall_5 = qa_judge.recall(sorted_context, gold_paragraphs)
                    
                    result[f"{retriever.__class__.__name__ }_recall@2"] = recall_2
                    result[f"{retriever.__class__.__name__ }_recall@5"] = recall_5
                
            result_list.append(result)

        self.save_results(result_list, [retriever.__class__.__name__ for retriever in retrievers])
    

    def save_results(self, result_list, retriever_names:List[str]):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        
        dataset_name = self.config.dataset_name
        include_events = self.config.include_events
        include_concept = self.config.include_concept
        encoder_model_name = self.config.encoder_model_name
        reader_model_name = self.config.reader_model_name
        
        # use last part of model name as identifier
        if "/" in encoder_model_name:
            encoder_model_name = encoder_model_name.split("/")[-1]
        if "/" in reader_model_name:
            reader_model_name = reader_model_name.split("/")[-1]

        summary_file = f"{self.config.result_dir}/{dataset_name}/summary_{formatted_time}_event{include_events}_concept{include_concept}_{encoder_model_name}_{reader_model_name}.json"
        if not os.path.exists(os.path.dirname(summary_file)):
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)

        result_dir = f"{self.config.result_dir}/{dataset_name}/result_{formatted_time}_event{include_events}_concept{include_concept}_{encoder_model_name}_{reader_model_name}.json"
        if not os.path.exists(os.path.dirname(result_dir)):
            os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        
        summary_dict = self.calculate_summary(result_list, retriever_names)
        
        with open(summary_file, "w") as f_summary:
            json.dump(summary_dict, f_summary)
            f_summary.write("\n")

        with open(result_dir, "w") as f:
            for result in result_list:
                json.dump(result, f)
                f.write("\n")
    
    def calculate_summary(self, result_list, method):
        summary_dict = {}
        for retriever_name in method:
            if not all(f"{retriever_name}_em" in result for result in result_list):
                raise ValueError(f"Missing {retriever_name}_em in results")
            if not all(f"{retriever_name}_f1" in result for result in result_list):
                raise ValueError(f"Missing {retriever_name}_f1 in results")
            
            average_em = sum([result[f"{retriever_name}_em"] for result in result_list]) / len(result_list)
            average_f1 = sum([result[f"{retriever_name}_f1"] for result in result_list]) / len(result_list)
            
            # Only calculate recall metrics if they exist in the results
            if all(f"{retriever_name}_recall@2" in result for result in result_list):
                average_recall_2 = sum([result[f"{retriever_name}_recall@2"] for result in result_list]) / len(result_list)
                average_recall_5 = sum([result[f"{retriever_name}_recall@5"] for result in result_list]) / len(result_list)
                summary_dict.update({
                    f"{retriever_name}_average_f1": average_f1,
                    f"{retriever_name}_average_em": average_em,
                    f"{retriever_name}_average_recall@2": average_recall_2,
                    f"{retriever_name}_average_recall@5": average_recall_5,
                })
            else:
                # For ReAct mode where recall metrics don't exist
                summary_dict.update({
                    f"{retriever_name}_average_f1": average_f1,
                    f"{retriever_name}_average_em": average_em,
                })
                
        return summary_dict
