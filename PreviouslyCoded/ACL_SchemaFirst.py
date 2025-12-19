
#!############################################# Start Chapter ##################################################
#region:#!   Schema Guided KG Generation  |   v0.1.0




#?######################### Start ##########################
#region:#?   Chunking + Embedding + VDB Insertion   |   v0.1.0

# ------------------------------
# CHUNKING + EMBEDDING + FAISS (dual-mode support w/ spaCy)
# ------------------------------

import re
import json
import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Literal
import spacy

# ------------------------------
# CONFIGURATION
# ------------------------------
EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # or "BAAI/bge-small-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
model.eval()

# ------------------------------
# LOAD SPACY SENTENCE TOKENIZER
# ------------------------------
nlp = spacy.load("en_core_web_sm")

def split_sentences_spacy(text: str) -> List[str]:
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# ------------------------------
# CHUNKING FUNCTION (dual mode, spaCy-powered)
# ------------------------------
def chunk_text(
    text: str,
    mode: Literal["narrative", "disjoint"] = "narrative",
    max_chunk_len: int = 2,
    sentence_per_line: bool = True,
    max_tokens_per_chunk: int = 60  # optional cap
) -> List[Dict]:
    chunks = []

    if mode == "narrative":
        paragraphs = text.strip().split("\n\n")  # split on double-newline (optional)
        sentences = []
        for p in paragraphs:
            sentences.extend(split_sentences_spacy(p.strip()))

        for i in range(0, len(sentences), max_chunk_len):
            chunk_sentences = sentences[i:i + max_chunk_len]
            chunk_text = ' '.join(chunk_sentences).strip()
            if chunk_text:
                chunks.append({
                    "id": f"Ch_{len(chunks)+1}",
                    "text": chunk_text,
                    "index": len(chunks),
                    "mode": mode
                })

    elif mode == "disjoint":
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if sentence_per_line:
                sentences = [line]
            else:
                sentences = split_sentences_spacy(line)

            for s in sentences:
                if not s:
                    continue
                words = s.split()
                if len(words) > max_tokens_per_chunk:
                    for j in range(0, len(words), max_tokens_per_chunk):
                        partial = ' '.join(words[j:j+max_tokens_per_chunk])
                        chunks.append({
                            "id": f"Ch_{len(chunks)+1}",
                            "text": partial,
                            "index": len(chunks),
                            "mode": mode
                        })
                else:
                    chunks.append({
                        "id": f"Ch_{len(chunks)+1}",
                        "text": s,
                        "index": len(chunks),
                        "mode": mode
                    })

    else:
        raise ValueError("Mode must be 'narrative' or 'disjoint'")

    return chunks


 

# text = """
# Transparency often conflicts with accountability in AI decision-making. When conflicts arise, regulators usually favor accountability over transparency.\n
# The company’s new logo uses 80% transparency to create a layered visual effect. This transparency improves the readability of text on colored backgrounds.\n
# During filming, an actor and the director had several conflicts about the script. Later, the conflict was resolved after Actor J. Stone agreed to rewrite one scene.\n
# """

# text = """
# The new library GraphGlow implements the Graph Neural Flow idea and outperforms GNNChef on small social networks.
# GNNChef (v2) is a framework; its motifX module detects recurring motifs and reports motifX-score per node.
# When using shabakeh-augment the training stability improves, but only if dropout < 0.3.
# GraphGlow also released GraphGlowLib, a set of utilities wrapping GraphGlow’s core API.
# motifX in the paper is described both as an algorithm and as a benchmark dataset (MotifX-DS).
# Despite similar names, MotifX-DS and motifX (algorithm) should not be merged; one is a dataset, the other a method.
# """

# now test chunking
#chunks = chunk_text(text, mode="disjoint", sentence_per_line=False)






# ------------------------------
# EMBEDDING FUNCTION
# ------------------------------
@torch.no_grad()
def embed_chunks(
    chunks: List[Dict],
    batch_size: int = 16
) -> (List[Dict], np.ndarray):
    all_metadata = []
    all_vectors = []

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Chunks"):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]

        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        for j, emb in enumerate(embeddings):
            all_metadata.append({
                "id": ids[j],
                "flag": "chunk",
                "chunk_index": batch[j]["index"],
                "mode": batch[j]["mode"],
                "original_text": batch[j]["text"]
            })
            all_vectors.append(emb.cpu().numpy())

    return all_metadata, np.stack(all_vectors)

# ------------------------------
# SAVE + LOAD + FAISS
# ------------------------------
def save_chunks(metadata: List[Dict], embeddings: np.ndarray, output_prefix: str = "chunk"):
    with open(f"{output_prefix}_metadata.jsonl", "w") as f:
        for item in metadata:
            json.dump(item, f)
            f.write("\n")
    np.save(f"{output_prefix}_embeddings.npy", embeddings)

def load_chunks(prefix: str = "chunk") -> (List[Dict], np.ndarray):
    with open(f"{prefix}_metadata.jsonl", "r") as f:
        metadata = [json.loads(line) for line in f]
    vectors = np.load(f"{prefix}_embeddings.npy")
    return metadata, vectors

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index




# text = """
# Transparency often conflicts with accountability in AI decision-making. When conflicts arise, regulators usually favor accountability over transparency.\n
# The company’s new logo uses 80% transparency to create a layered visual effect. This transparency improves the readability of text on colored backgrounds.\n
# During filming, an actor and the director had several conflicts about the script. Later, the conflict was resolved after Actor J. Stone agreed to rewrite one scene.\n
# """

# text = """
# The blockbuster Avatar was directed by James Cameron. Cameron is known for high-budget cinematic hits. Inception, a psychological thriller, was written and directed by Christopher Nolan. Nolan’s films explore memory and identity. The hit movie Titanic was also directed by Cameron.
# The chart-topping album Divide was created by Ed Sheeran. The British singer-songwriter is known for his heartfelt lyrics and acoustic sound. 1989, a pop-driven reinvention, was written and performed by Taylor Swift. Her work often explores themes of love, growth, and public identity. The Grammy-winning hit Everything Has Changed was also a collaboration between Swift and Sheeran.
# """

# text = """
# The blockbuster film Avatar was directed by James Cameron. 
# Cameron, a pioneer in cinematic technology, is known for creating immersive worlds. 
# The genre-bending masterpiece Inception, a psychological thriller about dreams and reality, was written and directed by Christopher Nolan. 
# Nolan’s storytelling philosophy has influenced a generation of filmmakers. 
# The tragic romance Titanic, both a movie and a cultural phenomenon, was also directed by Cameron.

# In music, the chart-topping album Divide was produced by Ed Sheeran, whose songwriting often blends pop and folk elements. 
# The record 1989 marked Taylor Swift’s transformation from country artist to pop icon. 
# Her collaboration with Sheeran on Everything Has Changed blurred the line between friendship and artistry. 
# Music critics often compare Swift’s lyrical authenticity to the narrative depth found in Nolan’s films, suggesting an underlying creative parallel between cinema and music.
# """


mode = "disjoint" #"narrative"  # or "disjoint"

chunks = chunk_text(text, mode=mode , sentence_per_line=False)
metadata, vectors = embed_chunks(chunks)



# Save
save_chunks(metadata, vectors, output_prefix="chunks_disjoint")


# Load later (if needed)
# loaded_meta, loaded_vecs = load_chunks("chunks_narrative")

# Build FAISS index
faiss_index = build_faiss_index(vectors) # 


# Combine metadata and vectors for easy access

# combined = [
#     {**metadata[i], "embedding": vectors[i]}
#     for i in range(len(metadata))
# ]





#endregion#? Chunking + Embedding + VDB Insertion  |   v0.1.0
#?#########################  End  ##########################





#?######################### Start ##########################
#region:#?  Chunk Retrieval + Context (IMPROVED)   |   v0.1.0


# ------------------------------
# IMPORTS
# ------------------------------

import json
import uuid
import numpy as np
import faiss
from typing import List, Dict, Literal
from openai import OpenAI
import os


os.environ["OPENAI_API_KEY"] = "sk-svcacct-nPW5ufSOF8XAU00GTHlafXNnn6TQOOI0DMbTFnl94sIFILngTS2d0b8mEwz-p8r1xT3BlbkFJ8I7u_wrq2TyB6GPDmTGCaSLgXRTTv1avc7TY_gX7kMC3z8R6QkUEuEX9XIv5mw2AA"
# Set your key (or load from environment securely)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))






# ------------------------------
# CHUNK RETRIEVAL + CONTEXT (IMPROVED)
# ------------------------------

def get_chunk_by_id(chunk_id: str, metadata: List[Dict]) -> (int, Dict):
    for i, chunk in enumerate(metadata):
        if chunk["id"] == chunk_id:
            return i, chunk
    raise ValueError(f"Chunk ID {chunk_id} not found.")

# # test on chunk 7
# chunk_7 = get_chunk_by_id("Ch_7", metadata)

def get_context_chunks(chunk: Dict, metadata: List[Dict], mode: str) -> List[Dict]:
    idx = chunk["chunk_index"]
    if mode == "narrative":
        prev_chunk = next((c for c in metadata if c["chunk_index"] == idx - 1), None)
        next_chunk = next((c for c in metadata if c["chunk_index"] == idx + 1), None)
        return [c for c in [prev_chunk, chunk, next_chunk] if c is not None]
    elif mode == "disjoint":
        return [chunk]
    else:
        raise ValueError("Mode must be 'narrative' or 'disjoint'")


def get_similar_chunks(
    vector_index: int,
    metadata: List[Dict],
    vectors: np.ndarray,
    faiss_index: faiss.IndexFlatIP,
    top_k: int = 4
) -> List[Dict]:
    """
    Returns top_k most similar chunks (excluding the query chunk itself),
    including their FAISS similarity scores and ranks.
    """
    query_vector = vectors[vector_index].reshape(1, -1)
    D, I = faiss_index.search(query_vector, top_k + 1)  # +1 to include self

    results = []
    rank = 1
    for idx, score in zip(I[0], D[0]):
        if idx == vector_index:
            continue  # skip self
        chunk = metadata[idx]
        results.append({
            "id": chunk["id"],
            "chunk_index": chunk["chunk_index"],
            "original_text": chunk["original_text"],
            "similarity_score": float(score),  # convert to float for JSON compatibility
            "rank": rank
        })
        rank += 1
        if len(results) == top_k:
            break

    return results


#endregion#?  Chunk Retrieval + Context (IMPROVED)  |   v0.1.0
#?#########################  End  ##########################






#?######################### Start ##########################
#region:#?   Entity Recognition phase   |   v0.1.1 (debugged)


import json
import uuid
from typing import List, Dict, Literal
from openai import OpenAI
import os

# ------------------------------
# OPENAI CLIENT (load once)
# ------------------------------
os.environ["OPENAI_API_KEY"] = "sk-svcacct-nPW5ufSOF8XAU00GTHlafXNnn6TQOOI0DMbTFnl94sIFILngTS2d0b8mEwz-p8r1xT3BlbkFJ8I7u_wrq2TyB6GPDmTGCaSLgXRTTv1avc7TY_gX7kMC3z8R6QkUEuEX9XIv5mw2AA"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------
# PROMPT GENERATOR — Entity Extraction
# ------------------------------
def build_entity_prompt(
    chunk_id: str,
    metadata: List[Dict],
    vectors: np.ndarray,
    faiss_index,
    mode: Literal["narrative", "disjoint"] = "narrative",
    top_k: int = 5
) -> str:
    """Prepare LLM prompt to extract entity mentions from one chunk."""
    vector_index, chunk = get_chunk_by_id(chunk_id, metadata)
    vicinity_context = get_context_chunks(chunk, metadata, mode)
    similar_chunks = get_similar_chunks(vector_index, metadata, vectors, faiss_index, top_k)

    prompt = f"""
# ENTITY EXTRACTION TASK

You are an entity recognition agent.
Your goal is to identify **specific entities** (named or descriptive) from natural language text.
Each entity is a distinct instance that could later be linked to an ontology class.

---

## FOCUS CHUNK
Chunk ID: {chunk['id']}
{chunk['original_text']}

"""

    if mode == "narrative":
        prompt += "## VICINITY CONTEXT (Previous / Next Chunks)\n"
        for ctx in vicinity_context:
            if ctx["id"] != chunk["id"]:
                prompt += f"[{ctx['id']}] {ctx['original_text']}\n"
        prompt += "\n"

    prompt += "## SEMANTICALLY SIMILAR CHUNKS (for disambiguation aid)\n"
    for s in similar_chunks:
        prompt += f"[{s['id']}] {s['original_text']}\n"
    prompt += "\n"

    prompt += """
---

## INSTRUCTIONS

From the **focus chunk only**, extract all entities that represent **individual objects, people, places, works, or items**.
Include both proper names (e.g., “Avatar”, “James Cameron”) and descriptive entities (e.g., “the chart-topping album”).

For each entity:
- `entity_name`: exact surface form as it appears
- `entity_type_hint`: coarse category such as "person", "organization", "work", "object", "location", "event", or "other".
- `context_phrase`: short phrase (5–10 words) from surrounding text
- `confidence_score`: float between 0 and 1 for how sure you are

---

## OUTPUT FORMAT (strict JSON)

[
  {{
    "entity_name": "Avatar",
    "entity_type_hint": "work",
    "context_phrase": "The blockbuster Avatar was directed by James Cameron",
    "confidence_score": 0.96
  }},
  ...
]
Return **only** the JSON list — no commentary, markdown, or text.
"""
    return prompt.strip()

# ------------------------------
# CALL OPENAI
# ------------------------------
def call_openai(prompt: str, model: str = "gpt-4o", max_tokens: int = 600) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ OpenAI API Error:", e)
        return ""

# ------------------------------
# MAIN WRAPPER
# ------------------------------
def extract_entities_from_chunk(
    chunk_id: str,
    metadata: List[Dict],
    vectors: np.ndarray,
    faiss_index,
    mode: Literal["narrative", "disjoint"] = "narrative",
    top_k: int = 4
) -> List[Dict]:
    """LLM-based entity extraction with context and FAISS hints."""
    try:
        _, chunk = get_chunk_by_id(chunk_id, metadata)
    except ValueError as e:
        print(f"❌ {e}")
        print("Available chunk IDs:", [c["id"] for c in metadata])
        return []

    prompt = build_entity_prompt(chunk_id, metadata, vectors, faiss_index, mode, top_k)
    print(f"\n🟦 ENTITY PROMPT for {chunk_id}:\n", prompt, "\n")

    raw_response = call_openai(prompt)
    if not raw_response:
        print("⚠️ Empty response from API.")
        return []

    print(f"🟩 Raw Entity Response ({chunk_id}):\n", raw_response, "\n")

    try:
        txt = raw_response.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").replace("json", "").strip()
        entity_list = json.loads(txt)
    except Exception as e:
        print(f"❌ JSON parse error for {chunk_id}: {e}")
        return []

    results = []
    for e in entity_list:
        results.append({
            "id": f"En_{uuid.uuid4().hex[:8]}",
            "flag": "entity",
            "chunk_id": chunk_id,
            "Chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["original_text"],
            "entity_name": e.get("entity_name"),
            "entity_type_hint": e.get("entity_type_hint"),
            "context_phrase": e.get("context_phrase"),
            "confidence_score": e.get("confidence_score"),
        })
    return results




# ------------------------------
# Test Safely
# ------------------------------
# print("Available chunks:", [c["id"] for c in metadata])
# entities = extract_entities_from_chunk(
#     chunk_id=metadata[3]["id"],  # always pick an existing chunk
#     metadata=metadata,
#     vectors=vectors,
#     faiss_index=faiss_index,
#     mode="disjoint"
# )
# print(json.dumps(entities, indent=2))

#endregion#? Entity Recognition phase  |   v0.1.1
#?#########################  End  ##########################



# ########################## Start ##########################
 #region:?  --------------------------- Entity Resolution phase   |   v0.2.0 (with examples + rename + LLM guidance)

# """
# Entity Resolution Phase
# -----------------------
# Goal:
# Merge duplicate or co-referent entity mentions, rename ambiguous ones, and keep distinct ones separate.

# Differences from v0.1.0:
# - Adds LLM examples and clearer reasoning instructions.
# - Introduces actions: merge_entities, rename_entity, keep_entity.
# - Lets the LLM modify entity names and descriptions.
# - Clarifies that these resolved entities will later be classified under schema-level classes,
#   helping the model understand the context and avoid confusing entities with classes.

# Output:
#     - entities_resolved.jsonl
# """

# import os
# import json
# import uuid
# import numpy as np
# import torch
# import faiss
# from tqdm import tqdm
# from typing import List, Dict, Any, Literal
# from transformers import AutoTokenizer, AutoModel
# from openai import OpenAI

# # ------------------------------
# # CONFIG
# # ------------------------------
# EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TOP_K = 6
# MODEL = "gpt-4o"

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "sk-xxx"
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # ------------------------------
# # LOAD EMBEDDING MODEL
# # ------------------------------
# print("⚙️ Loading embedding model for entity resolution...")
# tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
# model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
# model.eval()
# print("✅ Model loaded on", DEVICE)

# # ------------------------------
# # EMBEDDING FUNCTION
# # ------------------------------
# @torch.no_grad()
# def embed_entities(entities: List[Dict], batch_size: int = 16) -> np.ndarray:
#     """Embed each entity using name + context_phrase"""
#     all_vectors = []
#     texts = [
#         f"{e['entity_name']}. Context: {e.get('context_phrase','')}"
#         for e in entities
#     ]
#     for i in tqdm(range(0, len(texts), batch_size), desc="Embedding entities"):
#         batch = texts[i:i+batch_size]
#         inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
#         outputs = model(**inputs)
#         emb = outputs.last_hidden_state[:, 0, :]
#         emb = torch.nn.functional.normalize(emb, dim=1)
#         all_vectors.append(emb.cpu().numpy())
#     return np.vstack(all_vectors)

# # ------------------------------
# # BUILD FAISS INDEX
# # ------------------------------
# def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
#     dim = vectors.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     faiss.normalize_L2(vectors)
#     index.add(vectors)
#     return index

# # ------------------------------
# # PROMPT EXAMPLES
# # ------------------------------
# PROMPT_EXAMPLES = r"""
# ===============================
# ### EXAMPLES FOR ENTITY RESOLUTION
# ===============================

# 1️⃣ **Merge Example**
# Input group:
# [
#   {"entity_name": "GraphGlow", "context_phrase": "The new library GraphGlow implements Graph Neural Flow."},
#   {"entity_name": "GraphGlow v2", "context_phrase": "GraphGlow v2 outperforms GNNChef."},
#   {"entity_name": "the library", "context_phrase": "The library was tested on social networks."}
# ]

# Output:
# [
#   {
#     "action": "merge_entities",
#     "canonical_id": "En_merge_01",
#     "canonical_name": "GraphGlow",
#     "merged_ids": ["En_001", "En_002", "En_003"],
#     "new_description": "Unified mention of the GraphGlow library and its versions."
#   }
# ]

# Explanation:
# All refer to the same software system (GraphGlow).

# ---

# 2️⃣ **Rename Example**
# Input group:
# [
#   {"entity_name": "motifX", "context_phrase": "motifX detects recurring motifs in graphs."},
#   {"entity_name": "MotifX", "context_phrase": "MotifX-DS is a benchmark dataset for motif detection."}
# ]

# Output:
# [
#   {"action": "rename_entity", "entity_id": "En_002", "new_name": "MotifX_Dataset", "new_description": "A benchmark dataset named MotifX-DS for motif detection."},
#   {"action": "rename_entity", "entity_id": "En_001", "new_name": "MotifX_Algorithm", "new_description": "An algorithm for detecting recurring motifs in graphs."}
# ]

# Explanation:
# Names are similar but represent distinct concepts; renaming avoids confusion.

# ---

# 3️⃣ **Keep Example**
# Input group:
# [
#   {"entity_name": "GNNChef", "context_phrase": "GNNChef is a framework for GNN evaluation."}
# ]

# Output:
# [
#   {"action": "keep_entity", "entity_id": "En_010"}
# ]

# Explanation:
# No merge or rename needed; keep as-is.
# """

# # ------------------------------
# # PROMPT BUILDER
# # ------------------------------
# def build_entity_resolution_prompt(focus_entity: Dict, candidate_entities: List[Dict], similar_chunks: List[Dict]) -> str:
#     prompt = f"""
# # ENTITY RESOLUTION TASK

# You are an **Entity Resolution Agent** working in a Knowledge Graph construction pipeline.

# Your job: decide which entity mentions refer to the **same real-world entity**, which should be **merged** or **renamed** for clarity, 
# and which should be **kept separate**.

# ⚠️ Important Context:
# - These resolved entities will later be **classified under ontology classes extracted from the same corpus**.
# - Therefore, you are NOT creating classes — only deciding which *mentions* refer to the same or distinct *entities*.
# - Be careful to distinguish between *entities* (individual instances) and *classes* (types/categories).

# ---

# ## RULES
# - **Merge** mentions that are aliases, abbreviations, or coreferences of the same real-world object. Do not be overly stringent; minor name variations are acceptable as we care semantics much more than the wording.
# - **Rename** entities that share surface form but refer to different items. Create clear new names and concise updated descriptions.
# - **Keep** distinct entities unchanged.
# - Preserve all semantics from context; if uncertain, prefer keeping them separate.
# - Always include a clear, one-sentence `new_description` for any merge or rename.
# - Return **only JSON**, following the exact format.

# ---

# ## OUTPUT FORMAT
# [
#   {{
#     "action": "merge_entities",
#     "canonical_id": "En_merge_01",
#     "canonical_name": "GraphGlow",
#     "merged_ids": ["En_001","En_005"],
#     "new_description": "Unified reference to the GraphGlow library."
#   }},
#   {{
#     "action": "rename_entity",
#     "entity_id": "En_002",
#     "new_name": "MotifX_Algorithm",
#     "new_description": "Algorithm for detecting motifs in graphs."
#   }},
#   {{
#     "action": "keep_entity",
#     "entity_id": "En_003"
#   }}
# ]

# ---

# {PROMPT_EXAMPLES}

# ---

# ## INPUT GROUP

# ### FOCUS ENTITY
# {json.dumps(focus_entity, ensure_ascii=False, indent=2)}

# ### CANDIDATE ENTITIES
# {json.dumps(candidate_entities, ensure_ascii=False, indent=2)}

# ### CONTEXTUALLY SIMILAR CHUNKS
# {json.dumps(similar_chunks, ensure_ascii=False, indent=2)}

# Return only the JSON array, no commentary.
# """
#     return prompt.strip()

# # ------------------------------
# # LLM CALLER
# # ------------------------------
# def call_llm(prompt: str, max_tokens: int = 1000) -> str:
#     try:
#         response = client.chat.completions.create(
#             model=MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             max_tokens=max_tokens,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print("❌ OpenAI API Error:", e)
#         return ""

# # ------------------------------
# # ACTION EXECUTOR
# # ------------------------------
# def execute_entity_actions(actions: List[Dict], active_entities: Dict[str, Dict], final_entities: Dict[str, Dict]):
#     for act in actions:
#         action = act.get("action")
#         if action == "keep_entity":
#             eid = act["entity_id"]
#             if eid not in active_entities:
#                 continue
#             e = active_entities.pop(eid)
#             new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
#             final_entities[new_id] = {
#                 "id_final": new_id,
#                 "label": e["entity_name"],
#                 "aliases": [],
#                 "description": e.get("context_phrase"),
#                 "source_chunks": [e.get("chunk_id")],
#                 "embedding": e.get("embedding"),
#                 "flag": "resolved_entity"
#             }
#             print(f"   ✅ keep_entity: {eid} -> {new_id}")

#         elif action == "rename_entity":
#             eid = act["entity_id"]
#             if eid not in active_entities:
#                 continue
#             e = active_entities.pop(eid)
#             new_label = act.get("new_name", e["entity_name"])
#             new_desc = act.get("new_description", e.get("context_phrase"))
#             new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
#             final_entities[new_id] = {
#                 "id_final": new_id,
#                 "label": new_label,
#                 "aliases": [e["entity_name"]] if e["entity_name"] != new_label else [],
#                 "description": new_desc,
#                 "source_chunks": [e.get("chunk_id")],
#                 "embedding": e.get("embedding"),
#                 "flag": "resolved_entity",
#                 "revision_log": [f"renamed_from:{e['entity_name']}"]
#             }
#             print(f"   🔤 rename_entity: {eid} ({e['entity_name']} -> {new_label})")

#         elif action == "merge_entities":
#             merge_ids = act.get("merged_ids", [])
#             canonical_name = act.get("canonical_name")
#             new_desc = act.get("new_description", "")
#             members = [active_entities.pop(mid) for mid in merge_ids if mid in active_entities]
#             if not members:
#                 continue
#             embeddings = [np.array(m["embedding"]) for m in members if m.get("embedding") is not None]
#             mean_emb = np.mean(np.vstack(embeddings), axis=0) if embeddings else None
#             new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
#             final_entities[new_id] = {
#                 "id_final": new_id,
#                 "label": canonical_name,
#                 "aliases": [m["entity_name"] for m in members if m["entity_name"] != canonical_name],
#                 "description": new_desc,
#                 "source_chunks": sorted(list(set([m["chunk_id"] for m in members]))),
#                 "embedding": mean_emb.tolist() if mean_emb is not None else None,
#                 "flag": "resolved_entity",
#                 "revision_log": [f"merged:{merge_ids}"]
#             }
#             print(f"   🔁 merge_entities: {merge_ids} -> {new_id} ({canonical_name})")

#         else:
#             print("⚠️ Unknown action:", action)

# # ------------------------------
# # SAFE PARSE
# # ------------------------------
# def safe_json_parse(text: str):
#     txt = text.strip()
#     if txt.startswith("```"):
#         txt = txt.strip("`").replace("json", "").strip()
#     try:
#         return json.loads(txt)
#     except Exception as e:
#         print("⚠️ JSON parse failed:", e)
#         print("Raw output:\n", text)
#         return []

# # ------------------------------
# # MAIN RUNNER
# # ------------------------------
# def run_entity_resolution(entities: List[Dict], metadata: List[Dict], vectors: np.ndarray, faiss_index) -> List[Dict]:
#     print(f"Starting entity resolution on {len(entities)} entities...")

#     entity_vectors = embed_entities(entities)
#     faiss.normalize_L2(entity_vectors)
#     index = build_faiss_index(entity_vectors)
#     for i, e in enumerate(entities):
#         e["embedding"] = entity_vectors[i].tolist()

#     active_entities = {e["id"]: e for e in entities}
#     final_entities = {}
#     processed = set()

#     for idx, e in enumerate(entities):
#         if e["id"] in processed or e["id"] not in active_entities:
#             continue
#         print(f"\n🟢 Processing entity {idx+1}/{len(entities)}: {e['entity_name']} (id={e['id']})")
#         qv = entity_vectors[idx].reshape(1, -1)
#         D, I = index.search(qv, TOP_K + 1)
#         neighbor_ids = [entities[i]["id"] for i in I[0] if entities[i]["id"] != e["id"]]
#         candidates = [active_entities[nid] for nid in neighbor_ids if nid in active_entities]

#         # retrieve similar chunks for context
#         vector_index, chunk = get_chunk_by_id(e["chunk_id"], metadata)
#         similar_chunks = get_similar_chunks(vector_index, metadata, vectors, faiss_index, top_k=TOP_K)

#         prompt = build_entity_resolution_prompt(e, candidates, similar_chunks)
#         print(f"\n🟦 ENTITY RESOLUTION PROMPT for {e['id']} ({e['entity_name']}):")
#         print(prompt[:1000] + ("\n...TRUNCATED..." if len(prompt) > 1000 else ""))

#         raw = call_llm(prompt)
#         if not raw:
#             continue
#         print("🟩 Raw LLM Response (truncated):", raw[:700])
#         actions = safe_json_parse(raw)
#         execute_entity_actions(actions, active_entities, final_entities)

#         processed.add(e["id"])
#         for act in actions:
#             if act.get("action") in ("merge_entities", "rename_entity", "keep_entity"):
#                 processed.update(act.get("merged_ids", []))
#                 processed.add(act.get("entity_id", e["id"]))

#     print(f"\n💾 Saving {len(final_entities)} resolved entities to entities_resolved.jsonl")
#     with open("entities_resolved.jsonl", "w") as f:
#         for eid, e in final_entities.items():
#             json.dump(e, f)
#             f.write("\n")
#     print("✅ Entity Resolution completed.")
#     return list(final_entities.values())





# #*######################### Start ##########################
# #region:#*   # TEST RUN OF FULL PIPELINE UP TO ENTITY RESOLUTION


# # ===========================
# # run_pipeline_until_entity_resolution.py
# # ===========================

# from pprint import pprint
# import json

# # Assuming all your defined functions and imports from previous sections
# # (chunk_text, embed_chunks, save_chunks, extract_entities_from_chunk, run_entity_resolution, etc.)
# # are already in this environment or imported from modules.

# # -------------------------
# # 1️⃣ CHUNKING + EMBEDDING
# # -------------------------
# text = """
# The new library GraphGlow implements the Graph Neural Flow idea and outperforms GNNChef on small social networks.
# GNNChef (v2) is a framework; its motifX module detects recurring motifs and reports motifX-score per node.
# When using shabakeh-augment the training stability improves, but only if dropout < 0.3.
# GraphGlow also released GraphGlowLib, a set of utilities wrapping GraphGlow’s core API.
# motifX in the paper is described both as an algorithm and as a benchmark dataset (MotifX-DS).
# Despite similar names, MotifX-DS and motifX (algorithm) should not be merged; one is a dataset, the other a method.
# """

# print("\n==========================")
# print("🔹 STAGE 1: Chunking + Embedding")
# print("==========================")

# chunks = chunk_text(text, mode="disjoint", sentence_per_line=False)
# metadata, vectors = embed_chunks(chunks)
# faiss_index = build_faiss_index(vectors)

# print(f"✅ Chunking complete. Total chunks: {len(metadata)}")

# # -------------------------
# # 2️⃣ ENTITY RECOGNITION
# # -------------------------
# print("\n==========================")
# print("🔹 STAGE 2: Entity Recognition")
# print("==========================")

# all_entities = []
# for c in metadata:
#     ents = extract_entities_from_chunk(
#         chunk_id=c["id"],
#         metadata=metadata,
#         vectors=vectors,
#         faiss_index=faiss_index,
#         mode="disjoint"
#     )
#     all_entities.extend(ents)

# # Save raw entities
# with open("entities_raw.jsonl", "w") as f:
#     for e in all_entities:
#         json.dump(e, f)
#         f.write("\n")

# print(f"✅ Entity recognition complete. Extracted {len(all_entities)} entities.\n")

# # Print a preview
# print("🟢 Sample entities:")
# print("🟢 Sample entities (embeddings hidden):")
# # Hide potential large 'embedding' vectors when printing samples
# pprint([{k: v for k, v in e.items() if k != 'embedding'} for e in all_entities[:5]])

# # -------------------------
# # 3️⃣ ENTITY RESOLUTION
# # -------------------------
# print("\n==========================")
# print("🔹 STAGE 3: Entity Resolution")
# print("==========================")

# resolved_entities = run_entity_resolution(all_entities, metadata, vectors, faiss_index)


# print(f"\n✅ Entity resolution complete. Total final resolved entities: {len(resolved_entities)}\n")

# print("🟣 Sample resolved entities:")
# print("🟣 Sample resolved entities (embeddings hidden):")
# # Hide potential large 'embedding' vectors when printing samples
# pprint([{k: v for k, v in e.items() if k != 'embedding'} for e in resolved_entities[:5]])

# # -------------------------
# # Done
# # -------------------------
# print("\n💾 All stages complete up to Entity Resolution.")
# print("Files created:")
# print(" - entities_raw.jsonl")
# print(" - entities_resolved.jsonl")


# #endregion#*  Test RUN OF FULL PIPELINE UP TO ENTITY RESOLUTION
# #*#########################  End  ##########################







#endregion#?   Entity Resolution phase   |   v0.2.0
# ##########################  End  ##########################






#?######################### Start ##########################
#region:#?   Entity Resolution phase   |   v0.3


"""
Entity Resolution Phase
-----------------------
Goal:
Merge duplicate or co-referent entity mentions, rename ambiguous ones, and keep distinct ones separate.

Differences from v0.1.0:
- Adds LLM examples and clearer reasoning instructions.
- Introduces actions: merge_entities, rename_entity, keep_entity.
- Lets the LLM modify entity names and descriptions.
- Clarifies that these resolved entities will later be classified under schema-level classes,
  helping the model understand the context and avoid confusing entities with classes.

Output:
    - entities_resolved.jsonl
"""

import os
import json
import uuid
import numpy as np
import torch
import faiss
from tqdm import tqdm
from typing import List, Dict, Any, Literal
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

# ------------------------------
# CONFIG
# ------------------------------
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 6
MODEL = "gpt-4o"

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "sk-xxx"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------
# LOAD EMBEDDING MODEL
# ------------------------------
print("⚙️ Loading embedding model for entity resolution...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
model.eval()
print("✅ Model loaded on", DEVICE)

# ------------------------------
# EMBEDDING FUNCTION
# ------------------------------
@torch.no_grad()
def embed_entities(entities: List[Dict], batch_size: int = 16) -> np.ndarray:
    """Embed each entity using name + context_phrase"""
    all_vectors = []
    texts = [
        f"{e['entity_name']}. Context: {e.get('context_phrase','')}"
        for e in entities
    ]
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding entities"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, dim=1)
        all_vectors.append(emb.cpu().numpy())
    return np.vstack(all_vectors)

# ------------------------------
# BUILD FAISS INDEX
# ------------------------------
def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

# ------------------------------
# PROMPT EXAMPLES
# ------------------------------
PROMPT_EXAMPLES = r"""
===============================
### EXAMPLES FOR ENTITY RESOLUTION
===============================

1️⃣ **Merge Example**
Input group:
[
  {"entity_name": "GraphGlow", "context_phrase": "The new library GraphGlow implements Graph Neural Flow."},
  {"entity_name": "GraphGlow v2", "context_phrase": "GraphGlow v2 outperforms GNNChef."},
  {"entity_name": "the library", "context_phrase": "The library was tested on social networks."}
]

Output:
[
  {
    "action": "merge_entities",
    "canonical_id": "En_merge_01",
    "canonical_name": "GraphGlow",
    "merged_ids": ["En_001", "En_002", "En_003"],
    "new_description": "Unified mention of the GraphGlow library and its versions."
  }
]

Explanation:
All refer to the same software system (GraphGlow).

---

2️⃣ **Rename Example**
Input group:
[
  {"entity_name": "motifX", "context_phrase": "motifX detects recurring motifs in graphs."},
  {"entity_name": "MotifX", "context_phrase": "MotifX-DS is a benchmark dataset for motif detection."}
]

Output:
[
  {"action": "rename_entity", "entity_id": "En_002", "new_name": "MotifX_Dataset", "new_description": "A benchmark dataset named MotifX-DS for motif detection."},
  {"action": "rename_entity", "entity_id": "En_001", "new_name": "MotifX_Algorithm", "new_description": "An algorithm for detecting recurring motifs in graphs."}
]

Explanation:
Names are similar but represent distinct concepts; renaming avoids confusion.

---

3️⃣ **Keep Example**
Input group:
[
  {"entity_name": "GNNChef", "context_phrase": "GNNChef is a framework for GNN evaluation."}
]

Output:
[
  {"action": "keep_entity", "entity_id": "En_010"}
]

Explanation:
No merge or rename needed; keep as-is.
"""

# ------------------------------
# PROMPT BUILDER
# ------------------------------
def build_entity_resolution_prompt(focus_entity: Dict, candidate_entities: List[Dict], similar_chunks: List[Dict]) -> str:
    prompt = f"""
# ENTITY RESOLUTION TASK

You are an **Entity Resolution Agent** working in a Knowledge Graph construction pipeline.

Your job: decide which entity mentions refer to the **same real-world entity**, which should be **merged** or **renamed** for clarity, 
and which should be **kept separate**. We expect clear entity names and descriptions after your actions.

⚠️ Important Context:
- These resolved entities will later be **classified under ontology classes extracted from the same corpus**.
- Therefore, you are NOT creating classes — only deciding which *mentions* refer to the same or distinct *entities*.
- Be careful to distinguish between *entities* (individual instances) and *classes* (types/categories).

---

## RULES
- **Merge** mentions that are aliases, abbreviations, or coreferences of the same real-world object. Do not be overly stringent; minor name variations are acceptable as we care semantics much more than the wording.
- **Rename** entities that share surface form but refer to different items. Create clear new names and concise updated descriptions. We should not have two entities with same name and different description and no two entities with same description and different name.
- **Keep** distinct entities unchanged.
- Preserve all semantics from context; if uncertain, prefer keeping them separate.
- Always include a clear, one-sentence `new_description` for any merge or rename.
- Return **only JSON**, following the exact format.

---

## OUTPUT FORMAT
[
  {{
    "action": "merge_entities",
    "canonical_id": "En_merge_01",
    "canonical_name": "GraphGlow",
    "merged_ids": ["En_001","En_005"],
    "new_description": "Unified reference to the GraphGlow library."
  }},
  {{
    "action": "rename_entity",
    "entity_id": "En_002",
    "new_name": "MotifX_Algorithm",
    "new_description": "Algorithm for detecting motifs in graphs."
  }},
  {{
    "action": "keep_entity",
    "entity_id": "En_003"
  }}
]

---

{PROMPT_EXAMPLES}

---

## INPUT GROUP

### FOCUS ENTITY
{json.dumps(focus_entity, ensure_ascii=False, indent=2)}

### CANDIDATE ENTITIES
{json.dumps(candidate_entities, ensure_ascii=False, indent=2)}

### CONTEXTUALLY SIMILAR CHUNKS
{json.dumps(similar_chunks, ensure_ascii=False, indent=2)}

Return only the JSON array, no commentary.
"""
    return prompt.strip()

# ------------------------------
# LLM CALLER
# ------------------------------
def call_llm(prompt: str, max_tokens: int = 1000) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ OpenAI API Error:", e)
        return ""

# ------------------------------
# ACTION EXECUTOR
# ------------------------------
def execute_entity_actions(actions: List[Dict], active_entities: Dict[str, Dict], final_entities: Dict[str, Dict]):
    for act in actions:
        action = act.get("action")
        if action == "keep_entity":
            eid = act["entity_id"]
            if eid not in active_entities:
                continue
            e = active_entities.pop(eid)
            new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
            final_entities[new_id] = {
                "id_final": new_id,
                "label": e["entity_name"],
                "aliases": [],
                "description": e.get("context_phrase"),
                "source_chunks": [e.get("chunk_id")],
                "embedding": e.get("embedding"),
                "flag": "resolved_entity"
            }
            print(f"   ✅ keep_entity: {eid} -> {new_id}")

        elif action == "rename_entity":
            eid = act["entity_id"]
            if eid not in active_entities:
                continue
            e = active_entities.pop(eid)
            new_label = act.get("new_name", e["entity_name"])
            new_desc = act.get("new_description", e.get("context_phrase"))
            new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
            final_entities[new_id] = {
                "id_final": new_id,
                "label": new_label,
                "aliases": [e["entity_name"]] if e["entity_name"] != new_label else [],
                "description": new_desc,
                "source_chunks": [e.get("chunk_id")],
                "embedding": e.get("embedding"),
                "flag": "resolved_entity",
                "revision_log": [f"renamed_from:{e['entity_name']}"]
            }
            print(f"   🔤 rename_entity: {eid} ({e['entity_name']} -> {new_label})")

        elif action == "merge_entities":
            merge_ids = act.get("merged_ids", [])
            canonical_name = act.get("canonical_name")
            new_desc = act.get("new_description", "")
            members = [active_entities.pop(mid) for mid in merge_ids if mid in active_entities]
            if not members:
                continue
            embeddings = [np.array(m["embedding"]) for m in members if m.get("embedding") is not None]
            mean_emb = np.mean(np.vstack(embeddings), axis=0) if embeddings else None
            new_id = f"ResEnt_{uuid.uuid4().hex[:8]}"
            final_entities[new_id] = {
                "id_final": new_id,
                "label": canonical_name,
                "aliases": [m["entity_name"] for m in members if m["entity_name"] != canonical_name],
                "description": new_desc,
                "source_chunks": sorted(list(set([m["chunk_id"] for m in members]))),
                "embedding": mean_emb.tolist() if mean_emb is not None else None,
                "flag": "resolved_entity",
                "revision_log": [f"merged:{merge_ids}"]
            }
            print(f"   🔁 merge_entities: {merge_ids} -> {new_id} ({canonical_name})")

        else:
            print("⚠️ Unknown action:", action)

# ------------------------------
# SAFE PARSE
# ------------------------------
def safe_json_parse(text: str):
    txt = text.strip()
    if txt.startswith("```"):
        txt = txt.strip("`").replace("json", "").strip()
    try:
        return json.loads(txt)
    except Exception as e:
        print("⚠️ JSON parse failed:", e)
        print("Raw output:\n", text)
        return []


# ------------------------------
# Helpers + Iterative Entity Resolution (v0.3.1) - FIXED
# ------------------------------

import copy
import math

def normalize_entity_record(e: dict) -> dict:
    """
    Convert any entity-like record into the pipeline's canonical entity format:
      - entity_name (str)
      - context_phrase (str)
      - chunk_id (str)
      - id (str)
      - embedding (optional)
      - other fields preserved

    Why: final_entities use 'label'/'description' while provisional ones use 'entity_name'/'context_phrase'.
    """
    ne = dict(e)  # shallow copy to avoid mutation
    # ensure id key exists
    if "id" not in ne:
        # some final entries use id_final
        if "id_final" in ne:
            ne["id"] = ne["id_final"]
        else:
            # generate fallback id
            ne["id"] = ne.get("id", f"TEMP_{uuid.uuid4().hex[:8]}")

    # entity_name
    if "entity_name" not in ne:
        if "label" in ne:
            ne["entity_name"] = ne["label"]
        elif "name" in ne:
            ne["entity_name"] = ne["name"]
        else:
            # fallback to id
            ne["entity_name"] = ne["id"]

    # context_phrase
    if "context_phrase" not in ne:
        if "context" in ne:
            ne["context_phrase"] = ne["context"]
        elif "description" in ne:
            # use a short truncation of description as context phrase
            desc = ne["description"]
            if isinstance(desc, str):
                ne["context_phrase"] = desc if len(desc.split()) <= 12 else " ".join(desc.split()[:12]) + "..."
            else:
                ne["context_phrase"] = ""
        else:
            ne["context_phrase"] = ""

    # chunk id
    if "chunk_id" not in ne:
        # propagate first source_chunk if exists
        if "source_chunks" in ne and isinstance(ne["source_chunks"], list) and len(ne["source_chunks"])>0:
            ne["chunk_id"] = ne["source_chunks"][0]
        else:
            ne["chunk_id"] = ne.get("chunk_id", None)

    # ensure embedding is present but may be None
    if "embedding" in ne and isinstance(ne["embedding"], list):
        # keep as-is
        pass
    else:
        # leave None; will be computed by embed_entities()
        ne["embedding"] = None

    return ne

# Replace the old runner with this robust iterative version
def run_entity_resolution_iterative(
    entities: List[Dict],
    metadata: List[Dict],
    vectors: np.ndarray,
    faiss_index,
    max_iterations: int = 3,           # how many global passes to allow
    stop_if_no_change: bool = True,    # stop early when nothing changes
    min_delta: int = 1,                # minimum number of merges required to keep iterating
) -> List[Dict]:
    """
    Iterative entity resolution that:
      - normalizes entity records each pass,
      - re-embeds current entities,
      - rebuilds FAISS for entity->entity retrieval,
      - runs a one-pass entity-by-entity LLM resolution,
      - repeats until convergence or max_iterations reached.

    Returns: list of resolved entity dicts (canonical final entities).
    """
    print(f"Starting iterative entity resolution on {len(entities)} provisional entities...")
    # Normalize input entities
    current_entities = [normalize_entity_record(e) for e in entities]

    iteration = 0
    prev_count = len(current_entities)
    no_change_rounds = 0

    while iteration < max_iterations:
        iteration += 1
        print("\n" + "="*60)
        print(f"🔁 GLOBAL ENTITY RESOLUTION PASS {iteration} (entities={len(current_entities)})")
        print("="*60)

        # ---------- embed current entities ----------
        entity_vectors = embed_entities(current_entities)
        # ensure normalized vectors shape matches
        if entity_vectors.shape[0] != len(current_entities):
            raise RuntimeError(f"Embedding count mismatch: {entity_vectors.shape[0]} vs {len(current_entities)}")
        # attach embeddings back to entities
        for i, e in enumerate(current_entities):
            e["embedding"] = entity_vectors[i].tolist()

        # ---------- rebuild local FAISS index for entity->entity search ----------
        faiss.normalize_L2(entity_vectors)
        ent_index = build_faiss_index(entity_vectors)

        # ---------- per-entity resolution pass ----------
        active_entities = {e["id"]: copy.deepcopy(e) for e in current_entities}
        final_entities = {}
        processed = set()
        merges_this_pass = 0

        # iterate through snapshot list to maintain deterministic order
        snapshot_entities = list(current_entities)
        for idx, e in enumerate(snapshot_entities):
            if e["id"] in processed or e["id"] not in active_entities:
                continue

            print(f"\n🟢 Processing entity {idx+1}/{len(snapshot_entities)}: {e['entity_name']} (id={e['id']})")

            # search top-K neighbors (by embedding)
            qv = np.array(active_entities[e["id"]]["embedding"]).reshape(1, -1)
            D, I = ent_index.search(qv, TOP_K + 1)
            neighbor_ids = []
            for cand_idx in I[0]:
                if cand_idx < 0 or cand_idx >= len(snapshot_entities):
                    continue
                cand_id = snapshot_entities[cand_idx]["id"]
                if cand_id == e["id"]:
                    continue
                # only include if still active
                if cand_id in active_entities:
                    neighbor_ids.append(cand_id)

            candidates = [active_entities[nid] for nid in neighbor_ids if nid in active_entities]

            # also include simple lexical heuristics: substring/version matches
            # ensure we don't add duplicates
            lex_candidates = []
            base_name = e["entity_name"].lower()
            for aid, ae in list(active_entities.items()):
                if aid == e["id"]:
                    continue
                name = ae.get("entity_name","").lower()
                # substring / 'v' version / hyphen differences
                if base_name in name or name in base_name:
                    if aid not in [c["id"] for c in candidates]:
                        lex_candidates.append(ae)
                # version pattern, e.g., "v2"
                if (" v" in name or "v" in name) and base_name.strip() in name and aid not in [c["id"] for c in candidates]:
                    lex_candidates.append(ae)

            # merge lex candidates into candidates (avoid duplicates)
            for lc in lex_candidates:
                if lc["id"] not in [c["id"] for c in candidates]:
                    candidates.append(lc)

            # If too few candidates, optionally expand by nearest-N in embedding space
            # (we already used TOP_K, but we can request more neighbors if desired)
            # build similar chunks for context
            vector_index, chunk = get_chunk_by_id(e.get("chunk_id"), metadata) if e.get("chunk_id") else (None, None)
            similar_chunks = get_similar_chunks(vector_index, metadata, vectors, faiss_index, top_k=TOP_K) if vector_index is not None else []

            # build prompt and call LLM
            prompt = build_entity_resolution_prompt(active_entities[e["id"]], candidates, similar_chunks)
            raw = call_llm(prompt)
            if not raw:
                print("⚠️ Empty LLM response, keeping entity as-is.")
                # keep entity
                act_keep = {"action": "keep_entity", "entity_id": e["id"]}
                execute_entity_actions([act_keep], active_entities, final_entities)
                processed.add(e["id"])
                continue

            actions = safe_json_parse(raw)
            if not isinstance(actions, list):
                print("⚠️ LLM did not return a list; keeping entity.")
                act_keep = {"action": "keep_entity", "entity_id": e["id"]}
                execute_entity_actions([act_keep], active_entities, final_entities)
                processed.add(e["id"])
                continue

            # count merges in actions for convergence heuristics
            for act in actions:
                if act.get("action") == "merge_entities":
                    merges_this_pass += 1

            execute_entity_actions(actions, active_entities, final_entities)

            # mark processed ids accordingly (safe guards)
            processed.add(e["id"])
            for act in actions:
                if act.get("action") == "merge_entities":
                    for mid in act.get("merged_ids", []) or []:
                        processed.add(mid)
                if act.get("action") in ("rename_entity", "keep_entity"):
                    # entity_id might be present
                    eid = act.get("entity_id", None)
                    if eid:
                        processed.add(eid)

        # ---------- prepare for next iteration ----------
        # final_entities currently holds the resolved things from this pass
        # normalise them to canonical schema for next embedding run
        new_entities = [normalize_entity_record(fe) for fe in final_entities.values()]

        new_count = len(new_entities)
        print(f"\n📊 After pass {iteration}: {len(current_entities)} -> {new_count} entities (merges this pass: {merges_this_pass})")

        # convergence checks
        delta = abs(new_count - len(current_entities))
        if stop_if_no_change and merges_this_pass < min_delta:
            print("✅ Merge delta below min_delta; stopping early (converged).")
            current_entities = new_entities
            break

        if stop_if_no_change and new_count == len(current_entities):
            print("✅ No change in entity count; stopping early (converged).")
            current_entities = new_entities
            break

        # otherwise continue next pass
        current_entities = new_entities
        if iteration >= max_iterations:
            print(f"ℹ️ Reached max_iterations ({max_iterations}); stopping.")
            break

    # ---------- save final output ----------
    print(f"\n💾 Saving {len(current_entities)} resolved entities to entities_resolved.jsonl")
    with open("entities_resolved.jsonl", "w") as f:
        for e in current_entities:
            json.dump(e, f)
            f.write("\n")

    print(f"✅ Entity Resolution finished after {iteration} pass(es). Final entity count: {len(current_entities)}")
    return current_entities



#endregion#?   Entity Resolution phase   |   v0.3
#?#########################  End  ##########################







# ########################## Start ##########################
 #region:?   ---------------------- Concept Recognition (Class Extraction)  |   v0.1.0 : before integration with Entity Resolution

# import json
# import uuid
# from typing import List, Dict, Literal
# from openai import OpenAI
# import torch
# import numpy as np
# from tqdm import tqdm

# # ------------------------------
# # OPENAI CLIENT
# # ------------------------------
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # ------------------------------
# # PROMPT GENERATOR — Concept (Class) Extraction
# # ------------------------------

# def build_class_prompt_with_entities(
#     chunk_id: str,
#     metadata: List[Dict],
#     entities: List[Dict],
#     vectors: np.ndarray,
#     faiss_index,
#     mode: Literal["narrative", "disjoint"] = "narrative",
#     top_k: int = 4
# ) -> str:
#     """Prompt for extracting ontology classes grounded in recognized entities."""

#     vector_index, chunk = get_chunk_by_id(chunk_id, metadata)
#     vicinity_context = get_context_chunks(chunk, metadata, mode)
#     similar_chunks = get_similar_chunks(vector_index, metadata, vectors, faiss_index, top_k)

#     # Filter entities for this chunk
#     local_entities = [e for e in entities if e["chunk_id"] == chunk_id]

#     # Build context
#     prompt = f"""
# # ONTOLOGY CLASS EXTRACTION TASK (Entity-Grounded)

# You are an ontology engineer. Your job is to identify **abstract reusable classes**
# that the extracted entities in this chunk may belong to.

# Each class should represent a **type** or **category** — not a specific entity.
# Classes will later form the schema layer of a knowledge graph.

# ---

# ## FOCUS CHUNK
# Chunk ID: {chunk['id']}
# {chunk['original_text']}

# """

#     if local_entities:
#         prompt += "## RECOGNIZED ENTITIES (from previous step)\n"
#         for e in local_entities:
#             prompt += f"- {e['entity_name']} ({e.get('entity_type_hint','?')})\n"
#         prompt += "\n"
#     else:
#         prompt += "## No entities were recognized in this chunk.\n\n"

#     if mode == "narrative":
#         prompt += "## VICINITY CONTEXT (Previous / Next Chunks)\n"
#         for ctx in vicinity_context:
#             if ctx["id"] != chunk["id"]:
#                 prompt += f"[{ctx['id']}] {ctx['original_text']}\n"
#         prompt += "\n"

#     prompt += "## SEMANTICALLY SIMILAR CHUNKS (retrieved by vector similarity)\n"
#     for s in similar_chunks:
#         prompt += f"[{s['id']}] {s['original_text']}\n"
#     prompt += "\n"

#     prompt += """
# ---

# ## INSTRUCTIONS

# From the **focus chunk only**, using the listed entities and their context,
# extract all **ontology classes (concepts)** that these entities could belong to.

# Each class must:
# - Represent a **general type or category** (e.g., Film, Director, MusicAlbum, Genre)
# - Be **abstract**, not an individual instance
# - Be **reusable** across other texts
# - Optionally include high-level categories that capture relationships or properties (e.g., Collaboration as an event)

# For each class, provide:
# - `class_name`: concise and meaningful name (CamelCase)
# - `class_description`: one-sentence human-readable description
# - `context_words`: 3–10 keywords that characterize this class
# - `semantic_type_hint`: "object" | "process" | "property" | "role" | "event" | "concept"
# - `confidence_score`: float 0–1 indicating certainty

# ---

# ## OUTPUT FORMAT (strict JSON)
# [
#   {{
#     "class_name": "Film",
#     "class_description": "A cinematic work consisting of moving images and narrative structure.",
#     "context_words": ["movie", "blockbuster", "cinema", "director"],
#     "semantic_type_hint": "object",
#     "confidence_score": 0.93
#   }},
#   ...
# ]

# Return **only** the JSON array — no explanations or markdown.
# """
#     return prompt.strip()


# # ------------------------------
# # OPENAI CALL
# # ------------------------------

# def call_openai(prompt: str, model: str = "gpt-4o", max_tokens: int = 800) -> str:
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=max_tokens
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print("❌ OpenAI API Error:", e)
#         return ""

# # ------------------------------
# # MAIN WRAPPER
# # ------------------------------

# def extract_concepts_from_chunk(
#     chunk_id: str,
#     metadata: List[Dict],
#     entities: List[Dict],
#     vectors: np.ndarray,
#     faiss_index,
#     mode: Literal["narrative", "disjoint"] = "narrative",
#     top_k: int = 4
# ) -> List[Dict]:
#     """LLM-based concept extraction grounded in local entities."""
#     try:
#         _, chunk = get_chunk_by_id(chunk_id, metadata)
#     except ValueError as e:
#         print(f"❌ {e}")
#         return []

#     prompt = build_class_prompt_with_entities(chunk_id, metadata, entities, vectors, faiss_index, mode, top_k)
#     print(f"\n🟦 CLASS PROMPT for {chunk_id}:\n", prompt, "\n")

#     raw_response = call_openai(prompt)
#     if not raw_response:
#         return []

#     try:
#         txt = raw_response.strip()
#         if txt.startswith("```"):
#             txt = txt.strip("`").replace("json", "").strip()
#         class_list = json.loads(txt)
#     except Exception as e:
#         print(f"❌ JSON parse error for {chunk_id}: {e}")
#         print("Raw response:", raw_response)
#         return []

#     results = []
#     for c in class_list:
#         results.append({
#             "id": f"Cl_{uuid.uuid4().hex[:8]}",
#             "flag": "class",
#             "chunk_id": chunk_id,
#             "chunk_text": chunk["original_text"],
#             "class_name": c.get("class_name"),
#             "class_description": c.get("class_description"),
#             "context_words": c.get("context_words", []),
#             "semantic_type_hint": c.get("semantic_type_hint"),
#             "confidence_score": c.get("confidence_score"),
#             "embedding": None,
#             "cluster_id": None
#         })
#     return results


# # ------------------------------
# # EXTRACT CONCEPTS FOR ALL CHUNKS
# # ------------------------------
# all_concepts = []
# for c in metadata:
#     chunk_concepts = extract_concepts_from_chunk(
#         chunk_id=c["id"],
#         metadata=metadata,
#         entities=entities,     # from Step 1 (Entity Extraction)
#         vectors=vectors,
#         faiss_index=faiss_index,
#         mode="disjoint"
#     )
#     all_concepts.extend(chunk_concepts)

# print(f"✅ Total provisional concepts extracted: {len(all_concepts)}")


# # ------------------------------
# # EMBEDDING PROVISIONAL CONCEPTS
# # ------------------------------

# @torch.no_grad()
# def embed_concepts(concepts: List[Dict], batch_size: int = 16) -> (List[Dict], np.ndarray):
#     """Compute embeddings for each provisional concept using class_name + description + context."""
#     all_vectors = []
#     all_meta = []

#     for i in tqdm(range(0, len(concepts), batch_size), desc="Embedding Provisional Concepts"):
#         batch = concepts[i:i+batch_size]
#         texts = []
#         for c in batch:
#             context_part = ", ".join(c.get("context_words", []))
#             text_repr = f"{c['class_name']}. {c.get('class_description','')}. Keywords: {context_part}"
#             texts.append(text_repr.strip())

#         inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
#         outputs = model(**inputs)
#         emb = outputs.last_hidden_state[:, 0, :]
#         emb = torch.nn.functional.normalize(emb, dim=1)

#         for j, c in enumerate(batch):
#             vector = emb[j].cpu().numpy()
#             c["embedding"] = vector.tolist()
#             all_vectors.append(vector)
#             all_meta.append(c)

#     return all_meta, np.stack(all_vectors)

# # Run embedding
# all_concepts, concept_vectors = embed_concepts(all_concepts)
# print(f"✅ Embedded {len(all_concepts)} provisional concepts.")

# # Save for the next stage (Concept Resolution)
# with open("concepts_provisional_metadata.jsonl", "w") as f:
#     for item in all_concepts:
#         json.dump(item, f)
#         f.write("\n")
# np.save("concepts_provisional_embeddings.npy", concept_vectors)
# print("💾 Saved provisional concept metadata and embeddings.")



# #print concept concepts and its descriptions + id
# for concept in all_concepts:
#     print(f"ID: {concept['id']}, Name: {concept['class_name']}, Description: {concept['class_description']}")

 #endregion#? Concept Recognition (Class Extraction) | v0.1.0 : before integration with Entity Resolution
# ##########################  End  ##########################







#?######################### Start ##########################
#region:#?   Concept Recognition (Class Extraction)  | v0.2.0 : After Entity Resolution

"""
Concept (Class) Extraction — v0.2.0
-----------------------------------
This version works on **resolved entities** from the Entity Resolution phase
instead of raw entity mentions.

Input:
    - chunks metadata + embeddings (from chunking step)
    - entities_resolved.jsonl  (resolved entities)

Output:
    - concepts_provisional_metadata.jsonl
    - concepts_provisional_embeddings.npy
"""

import os
import json
import uuid
from typing import List, Dict, Literal
from openai import OpenAI
import torch
import numpy as np
from tqdm import tqdm

# ------------------------------
# OPENAI CLIENT
# ------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------
# PROMPT GENERATOR — Concept (Class) Extraction
# ------------------------------
def build_class_prompt_with_resolved_entities(
    chunk_id: str,
    metadata: List[Dict],
    resolved_entities: List[Dict],
    vectors: np.ndarray,
    faiss_index,
    mode: Literal["narrative", "disjoint"] = "narrative",
    top_k: int = 4
) -> str:
    """Prompt for extracting ontology classes grounded in resolved entities."""

    vector_index, chunk = get_chunk_by_id(chunk_id, metadata)
    vicinity_context = get_context_chunks(chunk, metadata, mode)
    similar_chunks = get_similar_chunks(vector_index, metadata, vectors, faiss_index, top_k)

    # Filter resolved entities belonging to this chunk (by chunk_id or source_chunks)
    local_entities = []
    for e in resolved_entities:
        src_chunks = e.get("source_chunks", [])
        if chunk_id in src_chunks or e.get("chunk_id") == chunk_id:
            local_entities.append(e)

    # Build prompt body
    prompt = f"""
# ONTOLOGY CLASS EXTRACTION TASK (Entity-Grounded)

You are an ontology engineer. Your job is to identify **abstract reusable classes**
that the extracted entities in this chunk may belong to.

Each class should represent a **type** or **category** — not a specific entity.
Classes will later form the schema layer of a knowledge graph.

---

## FOCUS CHUNK
Chunk ID: {chunk['id']}
{chunk['original_text']}

"""

    if local_entities:
        prompt += "## RESOLVED ENTITIES (after entity resolution)\n"
        for e in local_entities:
            name = e.get("label") or e.get("entity_name")
            desc = e.get("description", "")
            prompt += f"- {name}: {desc[:120]}\n"
        prompt += "\n"
    else:
        prompt += "## No resolved entities were found for this chunk.\n\n"

    if mode == "narrative":
        prompt += "## VICINITY CONTEXT (Previous / Next Chunks)\n"
        for ctx in vicinity_context:
            if ctx["id"] != chunk["id"]:
                prompt += f"[{ctx['id']}] {ctx['original_text']}\n"
        prompt += "\n"

    prompt += "## SEMANTICALLY SIMILAR CHUNKS (retrieved by vector similarity)\n"
    for s in similar_chunks:
        prompt += f"[{s['id']}] {s['original_text']}\n"
    prompt += "\n"

    prompt += """
---

## INSTRUCTIONS

From the **focus chunk only**, using the listed resolved entities and their context,
extract all **ontology classes (concepts)** that these entities could belong to.

Each class must:
- Represent a **general type or category** (e.g., Film, Director, Dataset, Framework)
- Be **abstract**, not an individual instance
- Be **reusable** across other texts
- Optionally include high-level categories that capture relationships or properties (e.g., Collaboration as an event)

For each class, provide:
- `class_name`: concise and meaningful name (CamelCase)
- `class_description`: one-sentence human-readable description
- `context_words`: 3–10 keywords that characterize this class
- `semantic_type_hint`: "object" | "process" | "property" | "role" | "event" | "concept"
- `confidence_score`: float 0–1 indicating certainty

---

## OUTPUT FORMAT (strict JSON)
[
  {{
    "class_name": "Film",
    "class_description": "A cinematic work consisting of moving images and narrative structure.",
    "context_words": ["movie", "blockbuster", "cinema", "director"],
    "semantic_type_hint": "object",
    "confidence_score": 0.93
  }},
  ...
]

Return **only** the JSON array — no explanations or markdown.
"""
    return prompt.strip()


# ------------------------------
# LLM CALLER
# ------------------------------
def call_openai(prompt: str, model: str = "gpt-4o", max_tokens: int = 800) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ OpenAI API Error:", e)
        return ""

# ------------------------------
# MAIN WRAPPER
# ------------------------------
def extract_concepts_from_chunk(
    chunk_id: str,
    metadata: List[Dict],
    resolved_entities: List[Dict],
    vectors: np.ndarray,
    faiss_index,
    mode: Literal["narrative", "disjoint"] = "narrative",
    top_k: int = 4
) -> List[Dict]:
    """LLM-based concept extraction grounded in resolved entities."""
    try:
        _, chunk = get_chunk_by_id(chunk_id, metadata)
    except ValueError as e:
        print(f"❌ {e}")
        return []

    prompt = build_class_prompt_with_resolved_entities(chunk_id, metadata, resolved_entities, vectors, faiss_index, mode, top_k)
    print(f"\n🟦 CLASS PROMPT for {chunk_id}:\n", prompt, "\n")

    raw_response = call_openai(prompt)
    if not raw_response:
        return []

    try:
        txt = raw_response.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").replace("json", "").strip()
        class_list = json.loads(txt)
    except Exception as e:
        print(f"❌ JSON parse error for {chunk_id}: {e}")
        print("Raw response:", raw_response)
        return []

    results = []
    for c in class_list:
        results.append({
            "id": f"Cl_{uuid.uuid4().hex[:8]}",
            "flag": "class",
            "chunk_id": chunk_id,
            "chunk_text": chunk["original_text"],
            "class_name": c.get("class_name"),
            "class_description": c.get("class_description"),
            "context_words": c.get("context_words", []),
            "semantic_type_hint": c.get("semantic_type_hint"),
            "confidence_score": c.get("confidence_score"),
            "embedding": None,
            "cluster_id": None
        })
    return results


# ------------------------------
# RUN EXTRACTION FOR ALL CHUNKS
# ------------------------------
def run_concept_extraction(metadata, vectors, faiss_index, resolved_entities, mode="disjoint"):
    all_concepts = []
    print("\n🚀 Starting Concept Extraction using Resolved Entities...\n")
    for c in metadata:
        chunk_concepts = extract_concepts_from_chunk(
            chunk_id=c["id"],
            metadata=metadata,
            resolved_entities=resolved_entities,
            vectors=vectors,
            faiss_index=faiss_index,
            mode=mode
        )
        all_concepts.extend(chunk_concepts)

    print(f"✅ Total provisional concepts extracted: {len(all_concepts)}")
    return all_concepts


# ------------------------------
# EMBEDDING PROVISIONAL CONCEPTS
# ------------------------------
@torch.no_grad()
def embed_concepts(concepts: List[Dict], tokenizer, model, device="cpu", batch_size: int = 16):
    """Compute embeddings for each provisional concept using class_name + description + context."""
    all_vectors = []
    all_meta = []

    for i in tqdm(range(0, len(concepts), batch_size), desc="Embedding Provisional Concepts"):
        batch = concepts[i:i+batch_size]
        texts = []
        for c in batch:
            context_part = ", ".join(c.get("context_words", []))
            text_repr = f"{c['class_name']}. {c.get('class_description','')}. Keywords: {context_part}"
            texts.append(text_repr.strip())

        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, dim=1)

        for j, c in enumerate(batch):
            vector = emb[j].cpu().numpy()
            c["embedding"] = vector.tolist()
            all_vectors.append(vector)
            all_meta.append(c)

    return all_meta, np.stack(all_vectors)


# ------------------------------
# MAIN DRIVER
# ------------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    EMBED_MODEL = "BAAI/bge-large-en-v1.5"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    model.eval()

    # Load chunks metadata and vectors
    with open("chunks_disjoint_metadata.jsonl", "r") as f:
        metadata = [json.loads(line) for line in f]
    vectors = np.load("chunks_disjoint_embeddings.npy")

    # Build FAISS index
    import faiss
    dim = vectors.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    faiss_index.add(vectors)

    # Load resolved entities
    print("📥 Loading resolved entities...")
    with open("entities_resolved.jsonl", "r") as f:
        resolved_entities = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(resolved_entities)} resolved entities.\n")

    # Run concept extraction
    all_concepts = run_concept_extraction(metadata, vectors, faiss_index, resolved_entities, mode="disjoint")

    # Embed
    all_concepts, concept_vectors = embed_concepts(all_concepts, tokenizer, model, DEVICE)
    print(f"✅ Embedded {len(all_concepts)} provisional concepts.")

    # Save
    with open("concepts_provisional_metadata.jsonl", "w") as f:
        for item in all_concepts:
            json.dump(item, f)
            f.write("\n")
    np.save("concepts_provisional_embeddings.npy", concept_vectors)
    print("💾 Saved provisional concept metadata and embeddings.\n")

    # Print preview
    for concept in all_concepts:
        print(f"ID: {concept['id']} | Name: {concept['class_name']} | Desc: {concept['class_description'][:120]}")




#endregion#? Concept Recognition (Class Extraction) | v0.2.0 : After Entity Resolution
#?######################### End ##########################






#?######################### Start ##########################
#region:#?   Class Resolution phase   |   v0.3.0


"""
concept_resolution.py
---------------------
Run after: Concept Recognition (provisional classes saved to concepts_provisional_metadata.jsonl
and concepts_provisional_embeddings.npy)

What it does (high level):
- Loads provisional classes and their embeddings
- Computes separate embeddings for `class_name` and `class_description`
- Builds FAISS indexes for names and descriptions
- Iteratively selects unresolved provisional classes, fetches union of top-K neighbors (name & desc),
  asks the LLM which function(s) to execute (merge/rename/keep), and executes them.
- Produces final_classes.jsonl with final_class entries (flag="final_class") and prints tracing info.

Usage:
    python concept_resolution.py
"""

import os
import json
import uuid
import numpy as np
import faiss
import torch
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import math
import time

# ------------------------------
# CONFIG
# ------------------------------
EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # same model used earlier
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 4            # neighbors per name or desc; union may be up to 2*TOP_K
LLM_MODEL = "gpt-4o" # keep consistent with your previous code
TEMPERATURE = 0.0

# Files (expected)
PROVISIONAL_META_PATH = "concepts_provisional_metadata.jsonl"
PROVISIONAL_EMB_PATH = "concepts_provisional_embeddings.npy"
OUTPUT_FINAL_PATH = "final_classes.jsonl"

# OpenAI client (expects OPENAI_API_KEY in env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------
# UTIL: Load provisional concepts
# ------------------------------
def load_provisional_concepts(meta_path: str, emb_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    print("🔁 Loading provisional classes metadata and embeddings...")
    with open(meta_path, "r") as f:
        concepts = [json.loads(line) for line in f]
    embeddings = np.load(emb_path)
    assert len(concepts) == embeddings.shape[0], "Mismatch between concepts and embeddings"
    # ensure each concept has an 'embedding' as list (if saved) or attach from np array
    for i, c in enumerate(concepts):
        if c.get("embedding") is None:
            c["embedding"] = embeddings[i].tolist()
    print(f"   Loaded {len(concepts)} provisional concepts.")
    return concepts, embeddings

# ------------------------------
# EMBEDDING helpers (for name and description)
# ------------------------------
print("⚙️ Loading embedding model (tokenizer + model)...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
model.eval()
print("   Model loaded on", DEVICE)

@torch.no_grad()
def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]  # CLS
        emb = torch.nn.functional.normalize(emb, dim=1)
        vectors.append(emb.cpu().numpy())
    return np.vstack(vectors)

# ------------------------------
# Build FAISS indexes for name & description embeddings
# ------------------------------
def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)  # FAISS IP expects normalized if using inner product as cosine
    idx.add(vectors)
    return idx

# ------------------------------
# LLM Prompting helpers: include examples + strict output format
# ------------------------------
PROMPT_EXAMPLES = r'''
EXAMPLES:
1) Merge example:
Input group:
- Cl_A: name="GraphLib", desc="utility wrapper around GraphGlow providing helpers", chunks=["Ch_4"]
- Cl_B: name="GraphGlowLib", desc="set of utilities wrapping GraphGlow core API", chunks=["Ch_2"]
- Cl_C: name="GraphGlowUtilities", desc="helper functions for GraphGlow", chunks=["Ch_9"]

Desired action: merge Cl_A, Cl_B, Cl_C into one final class that captures "Graph utilities for GraphGlow".
Output (structured JSON commands): one "merge_classes" command with merge_ids=[Cl_B, Cl_C], target_id=Cl_A (or chosen id), and a chosen canonical name+desc.

2) Rename example:
Input group:
- Cl_D: name="MotifX", desc="an algorithm to detect recurring motifs", chunks=["Ch_2"]
- Cl_E: name="MotifX", desc="MotifX-DS is a benchmark dataset", chunks=["Ch_5"]

Desired action: they are not the same concept. We keep them separate. Rename one to avoid confusion:
- rename Cl_D -> "MotifX_Algorithm"
- rename Cl_E -> "MotifX_Dataset"

Output: two "rename_class" commands.

3) Keep example:
Input group:
- Cl_F: name="TrainingStability", desc="the measured stability of training under augmentation", chunks=["Ch_3"]

If nothing else to do, "keep_class" for Cl_F.

OUTPUT FORMAT (MUST FOLLOW EXACTLY)
Return a JSON array of objects. Each object must be one of the following forms:

A) Merge
{ "action": "merge_classes",
  "target_id": "<the id that will remain or new id if you prefer>",
  "merge_ids": ["<id1>", "<id2>", ...],    // other ids to be absorbed (exclude target_id)
  "new_label": "<canonical name in CamelCase or snake_case>",
  "new_description": "<one-sentence unified description>" }

B) Rename
{ "action": "rename_class",
  "class_id": "<id>",
  "new_label": "<new name>",
  "new_description": "<revised description (optional)>" }

C) Keep
{ "action": "keep_class",
  "class_id": "<id>" }

Return ONLY the JSON array and nothing else. If you want to provide multiple actions for the group, include them as multiple objects in the array.
'''

def build_resolution_prompt(primary: Dict[str, Any], candidates: List[Dict[str, Any]], guidance: str = "") -> str:
    group_items = [primary] + candidates
    s = f"""
You are an class-level Schema resolution agent. You will receive a small group of provisional classes (extracted from text chunks).
Your task: decide which classes should be MERGED (they are semantically the same), which should be RENAMED (share the same surface but different meaning), or which should be KEPT as-is.

Guidelines (short):
- Focus on semantic identity, not surface form.
- Merge when two provisional classes refer to the same semantic concept across contexts.
- Rename when same surface/name but distinct semantics — propose names that disambiguate.
- Keep otherwise.
- When merging, produce a single clear canonical label + one-line canonical description that covers all merged members.
- Preserve provenance: the code will collect all source chunk ids from the merged members.

Do NOT output any commentary. Output ONLY the JSON array of actions (see exact format below).

""" + PROMPT_EXAMPLES + "\n\n" + "GROUP INPUT:\n\n"

    for c in group_items:
        s += json.dumps({
            "id": c["id"],
            "class_name": c.get("class_name"),
            "class_description": c.get("class_description"),
            "context_words": c.get("context_words", []),
            "chunk_id": c.get("chunk_id")
        }, ensure_ascii=False) + "\n"

    s += "\n\nRETURN ACTIONS:\n"
    return s

# ------------------------------
# Core loop: retrieval, LLM call, execution
# ------------------------------
def union_topk_neighbors(
    idx_name: faiss.IndexFlatIP,
    idx_desc: faiss.IndexFlatIP,
    name_vectors: np.ndarray,
    desc_vectors: np.ndarray,
    query_idx: int,
    top_k: int = TOP_K
) -> List[int]:
    # Return union of neighbor indices (excluding query_idx)
    qn = name_vectors[query_idx].reshape(1, -1).astype("float32")
    qd = desc_vectors[query_idx].reshape(1, -1).astype("float32")
    # FAISS requires pre-normalized vectors for inner product to work like cosine; ensure normalized upfront
    _, In = idx_name.search(qn, top_k + 1)
    _, Id = idx_desc.search(qd, top_k + 1)
    ids = set()
    for arr in (In[0], Id[0]):
        for i in arr:
            if int(i) != int(query_idx):
                ids.add(int(i))
    return sorted(list(ids))

def safe_parse_json(text: str) -> Any:
    # try to extract JSON even if wrapped in backticks or markdown
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
    # Find first '[' and last ']' to be robust
    a = t.find("[")
    b = t.rfind("]")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(t[a:b+1])
        except Exception:
            pass
    # fallback full parse
    try:
        return json.loads(t)
    except Exception as e:
        print("❗ Unable to parse JSON from LLM response. Raw response below:")
        print(text)
        raise e

def execute_actions(
    actions: List[Dict[str, Any]],
    active_concepts: Dict[str, Dict[str,Any]],
    final_classes: Dict[str, Dict[str,Any]],
    concept_embeddings: np.ndarray
) -> None:
    """
    actions: list of action dicts returned by LLM for this group
    active_concepts: dict id->concept for provisional/ongoing classes
    final_classes: dict id_final->final_class
    concept_embeddings: original embeddings array (rows correspond to indices mapped by 'idx_to_id' global)
    """
    global id_to_idx  # maps provisional id -> row index in concept_embeddings
    for act in actions:
        a = act.get("action")
        if a == "keep_class":
            cid = act["class_id"]
            if cid not in active_concepts:
                print(f"   ⚠️ keep_class: id {cid} not in active_concepts (might already be merged). Skipping.")
                continue
            # create final entry from single provisional
            pc = active_concepts.pop(cid)
            final_id = f"FinalCl_{uuid.uuid4().hex[:8]}"
            final_entry = {
                "id_final": final_id,
                "label": pc.get("class_name"),
                "description": pc.get("class_description"),
                "member_class_ids": [pc["id"]],
                "source_chunks": [pc.get("chunk_id")] if pc.get("chunk_id") else [],
                "semantic_type_hint": pc.get("semantic_type_hint"),
                "confidence_score": pc.get("confidence_score", 0.0),
                "embedding": pc.get("embedding"),
                "flag": "final_class",
                "revision_log": [f"kept_from:{pc['id']}"]
            }
            final_classes[final_id] = final_entry
            print(f"   ✅ keep_class executed: {cid} -> {final_id}")

        elif a == "rename_class":
            cid = act["class_id"]
            new_label = act.get("new_label")
            new_desc = act.get("new_description", None)
            if cid in active_concepts:
                # update provisional in place (keeps it available for future rounds if not finalized)
                pc = active_concepts[cid]
                old_name = pc.get("class_name")
                pc["class_name"] = new_label
                if new_desc:
                    pc["class_description"] = new_desc
                pc.setdefault("revision_log", []).append(f"renamed_from:{old_name}")
                print(f"   🔤 rename_class executed for provisional {cid}: {old_name} -> {new_label}")
            else:
                # Maybe it's already final? then update final class label
                found = False
                for fid, fe in final_classes.items():
                    if cid in fe.get("member_class_ids", []):
                        old = fe["label"]
                        fe["label"] = new_label
                        if new_desc:
                            fe["description"] = new_desc
                        fe.setdefault("revision_log", []).append(f"renamed_from:{old}")
                        print(f"   🔤 rename_class executed for final {fid}: {old} -> {new_label}")
                        found = True
                        break
                if not found:
                    print(f"   ⚠️ rename_class: id {cid} not found in active or final. Skipping.")

        elif a == "merge_classes":
            target_id = act.get("target_id")
            merge_ids = act.get("merge_ids", [])
            new_label = act.get("new_label")
            new_desc = act.get("new_description", None)

            # collect provisional objects for target + merges (they may be in active_concepts or some already final)
            members = []
            missing = []
            # target may be provisional or any of merge_ids could be provisional/final
            for hid in [target_id] + merge_ids:
                if hid in active_concepts:
                    members.append(active_concepts.pop(hid))
                else:
                    # search in final_classes to extract and remove if needed (rare)
                    found_in_final = False
                    for fid, fe in list(final_classes.items()):
                        if hid in fe.get("member_class_ids", []):
                            # take it out and add its members to new merge
                            # we will remove old final and include its member_class_ids
                            # convert to provisional-like dicts for merging
                            found_in_final = True
                            # create pseudo-provisional for merging
                            pseudo = {
                                "id": fe["member_class_ids"][0] if fe["member_class_ids"] else hid,
                                "class_name": fe["label"],
                                "class_description": fe.get("description"),
                                "context_words": [],
                                "chunk_id": None,
                                "embedding": fe.get("embedding"),
                                "semantic_type_hint": fe.get("semantic_type_hint"),
                                "confidence_score": fe.get("confidence_score", 0.0)
                            }
                            members.append(pseudo)
                            # remove old final (we'll replace it)
                            del final_classes[fid]
                            break
                    if not found_in_final:
                        missing.append(hid)
            if missing:
                print(f"   ⚠️ merge_classes: some ids not found and ignored: {missing}")

            if not members:
                print("   ⚠️ merge_classes: no members collected, skipping.")
                continue

            # create final merged entry
            final_id = f"FinalCl_{uuid.uuid4().hex[:8]}"
            member_ids = [m["id"] for m in members]
            source_chunks = []
            embeddings_to_avg = []
            semantic_hints = []
            confidences = []
            for m in members:
                if m.get("chunk_id"):
                    source_chunks.append(m["chunk_id"])
                if m.get("embedding") is not None:
                    embeddings_to_avg.append(np.array(m["embedding"], dtype=np.float32))
                if m.get("semantic_type_hint"):
                    semantic_hints.append(m.get("semantic_type_hint"))
                if m.get("confidence_score") is not None:
                    confidences.append(float(m.get("confidence_score")))

            if embeddings_to_avg:
                mean_emb = np.mean(np.vstack(embeddings_to_avg), axis=0)
                # normalize
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb = (mean_emb / norm).astype(float)
                mean_emb_list = mean_emb.tolist()
            else:
                mean_emb_list = None

            # choose canonical label/desc
            canonical_label = new_label if new_label else members[0].get("class_name")
            canonical_desc = new_desc if new_desc else members[0].get("class_description", "")

            final_entry = {
                "id_final": final_id,
                "label": canonical_label,
                "description": canonical_desc,
                "member_class_ids": member_ids,
                "source_chunks": sorted(list(set([c for c in source_chunks if c is not None]))),
                "semantic_type_hint": semantic_hints[0] if semantic_hints else None,
                "confidence_score": float(np.mean(confidences)) if confidences else 0.0,
                "embedding": mean_emb_list,
                "flag": "final_class",
                "revision_log": [f"merged:{member_ids}"]
            }
            final_classes[final_id] = final_entry
            print(f"   ✅ merge_classes executed: merged {member_ids} -> {final_id} (label={canonical_label})")

        else:
            print("   ⚠️ Unknown action from LLM:", act)

# ------------------------------
# Main driver
# ------------------------------
def run_resolution():
    # load provisional
    prov_concepts, prov_embeddings = load_provisional_concepts(PROVISIONAL_META_PATH, PROVISIONAL_EMB_PATH)
    n = len(prov_concepts)

    # create id <-> idx maps
    global id_to_idx
    id_to_idx = {c["id"]: i for i, c in enumerate(prov_concepts)}
    idx_to_id = {i: c["id"] for i, c in enumerate(prov_concepts)}

    # compute separate name / desc embeddings (so we have two views)
    print("🔁 Computing embeddings for class_name and class_description separately...")
    names = [c.get("class_name","") or "" for c in prov_concepts]
    descs = [c.get("class_description","") or "" for c in prov_concepts]

    name_vectors = embed_texts(names, batch_size=16).astype("float32")
    desc_vectors = embed_texts(descs, batch_size=16).astype("float32")
    print("   name_vectors shape:", name_vectors.shape, "desc_vectors shape:", desc_vectors.shape)

    # Build FAISS indexes (copy vectors since build_faiss_index normalizes)
    name_vectors_copy = name_vectors.copy()
    desc_vectors_copy = desc_vectors.copy()
    # Normalize for cosine via inner product
    faiss.normalize_L2(name_vectors_copy)
    faiss.normalize_L2(desc_vectors_copy)

    idx_name = build_faiss_index(name_vectors_copy)
    idx_desc = build_faiss_index(desc_vectors_copy)

    # Keep active_concepts as dict (id->concept) so we can pop/modify as we go
    active_concepts: Dict[str, Dict[str,Any]] = {c["id"]: dict(c) for c in prov_concepts}
    final_classes: Dict[str, Dict[str,Any]] = {}

    processed_ids: Set[str] = set()  # provisional ids that have been finalized (moved to final or merged)
    selection_order = list(active_concepts.keys())  # we iterate in this order, but will skip processed

    total = len(selection_order)
    i = 0
    print("\n▶️ Starting iterative resolution loop over provisional classes...")
    while True:
        # pick next unprocessed provisional id
        next_id = None
        for cid in selection_order:
            if cid not in processed_ids and cid in active_concepts:
                next_id = cid
                break
        if next_id is None:
            print("🔚 No more unprocessed provisional classes. Finishing.")
            break

        i += 1
        print("\n" + "="*80)
        print(f"Step {i}: Selected provisional class {next_id}")
        primary = active_concepts[next_id]
        q_idx = id_to_idx[next_id]

        # retrieve union neighbors
        neighbor_idxs = union_topk_neighbors(idx_name, idx_desc, name_vectors_copy, desc_vectors_copy, q_idx, TOP_K)
        neighbor_ids = [idx_to_id[idx] for idx in neighbor_idxs]
        print(f"   found {len(neighbor_ids)} neighbors (union of name & desc top-{TOP_K}): {neighbor_ids}")

        # Build candidate list objects
        candidates = []
        for nid in neighbor_ids:
            if nid == next_id:
                continue
            if nid in active_concepts:
                candidates.append(active_concepts[nid])
            else:
                print(f"   (neighbor {nid} already processed / missing; skipped)")

        # Build and print prompt
        prompt_text = build_resolution_prompt(primary, candidates)
        print("   >>> Prompt to LLM (truncated preview):")
        print(prompt_text[:1200] + ("\n...TRUNCATED...\n" if len(prompt_text) > 1200 else "\n"))

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role":"user", "content": prompt_text}],
                temperature=TEMPERATURE,
                max_tokens=800
            )
            raw = response.choices[0].message.content
            print("   <<< Raw LLM response (truncated):")
            print(raw[:1000] + ("\n...TRUNCATED...\n" if len(raw) > 1000 else "\n"))
            actions = safe_parse_json(raw)
            if not isinstance(actions, list):
                raise ValueError("LLM output JSON must be a list of action objects.")
        except Exception as e:
            print("❌ LLM call or parse error:", e)
            print("   Skipping this group and marking primary as keep (fail-open).")
            actions = [{"action":"keep_class", "class_id": next_id}]

        # Execute actions
        execute_actions(actions, active_concepts, final_classes, prov_embeddings)

        # mark all involved provisional ids in actions as processed (so we don't revisit)
        for act in actions:
            if act.get("action") == "keep_class":
                processed_ids.add(act.get("class_id"))
            elif act.get("action") == "rename_class":
                # rename doesn't finalize; keep as unresolved (unless LLM also asked keep)
                # mark it processed only if the LLM included an explicit keep or merge for it
                # We'll conservatively not mark rename-only as processed so it may be revisited in later groups
                pass
            elif act.get("action") == "merge_classes":
                # add target+merge ids to processed
                t = act.get("target_id")
                mids = act.get("merge_ids", [])
                for pid in ([t] + mids):
                    processed_ids.add(pid)
            else:
                processed_ids.add(next_id)

        # Print summary so far
        print(f"   Completed Step {i}. Active provisional left: {len([k for k in active_concepts.keys() if k not in processed_ids])}. Final classes so far: {len(final_classes)}")
        # continue loop

    # After loop: some provisional might be still active but not processed (e.g., rename-only)
    # finalize any remaining active_concepts as final classes
    remaining = [cid for cid in active_concepts.keys() if cid not in processed_ids]
    if remaining:
        print("\n⚠️ Finalizing remaining provisional classes (no LLM decision finalized them). Count:", len(remaining))
        for rcid in remaining:
            pc = active_concepts.pop(rcid)
            final_id = f"FinalCl_{uuid.uuid4().hex[:8]}"
            final_classes[final_id] = {
                "id_final": final_id,
                "label": pc.get("class_name"),
                "description": pc.get("class_description"),
                "member_class_ids": [pc["id"]],
                "source_chunks": [pc.get("chunk_id")] if pc.get("chunk_id") else [],
                "semantic_type_hint": pc.get("semantic_type_hint"),
                "confidence_score": pc.get("confidence_score", 0.0),
                "embedding": pc.get("embedding"),
                "flag": "final_class",
                "revision_log": ["auto_finalized"]
            }
            print(f"   auto-finalized {rcid} -> {final_id}")

    # Save final classes
    print(f"\n💾 Saving {len(final_classes)} final classes to {OUTPUT_FINAL_PATH} ...")
    with open(OUTPUT_FINAL_PATH, "w") as f:
        for fid, entry in final_classes.items():
            json.dump(entry, f)
            f.write("\n")
    print("✅ Done. Final classes saved.")

if __name__ == "__main__":
    start_time = time.time()
    run_resolution()
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.1f}s")






# ------------------------------
# print
# ------------------------------


# """
# inspect_classes.py
# ------------------
# Utility to print all provisional and final classes with their descriptions.
# """

# import json

# PROV_PATH = "concepts_provisional_metadata.jsonl"
# FINAL_PATH = "final_classes.jsonl"

# def print_provisional_classes(path=PROV_PATH):
#     print("\n" + "="*80)
#     print("📘 PROVISIONAL CLASSES")
#     print("="*80)
#     count = 0
#     with open(path, "r") as f:
#         for line in f:
#             c = json.loads(line)
#             count += 1
#             print(f"[{count:03d}]  {c.get('class_name','<no name>')}")
#             print(f"      → {c.get('class_description','<no description>')}")
#             print(f"      chunk_id: {c.get('chunk_id')} | conf: {c.get('confidence_score')}")
#             print("-"*80)
#     print(f"Total provisional classes: {count}")

# def print_final_classes(path=FINAL_PATH):
#     print("\n" + "="*80)
#     print("🏁 FINAL CLASSES")
#     print("="*80)
#     count = 0
#     with open(path, "r") as f:
#         for line in f:
#             c = json.loads(line)
#             count += 1
#             print(f"[{count:03d}]  {c.get('label','<no label>')}")
#             print(f"      → {c.get('description','<no description>')}")
#             print(f"      members: {len(c.get('member_class_ids',[]))} | chunks: {len(c.get('source_chunks',[]))}")
#             print("-"*80)
#     print(f"Total final classes: {count}")

# if __name__ == "__main__":
#     try:
#         print_provisional_classes()
#     except FileNotFoundError:
#         print("❌ No provisional file found:", PROV_PATH)
#     try:
#         print_final_classes()
#     except FileNotFoundError:
#         print("❌ No final file found:", FINAL_PATH)



#endregion#? Class Resolution phase  |   v0.3.0
#?#########################  End  ##########################







#?######################### Start ##########################
#region:#?   Rel Recognition phase   |   v0.1.0




#endregion#? Rel Recognition phase  |   v0.1.0
#?#########################  End  ##########################









#?######################### Start ##########################
#region:#?   Rel resolution and clustering   |   v0.1.0





#endregion#? Rel resolution and clustering  |   v0.1.0
#?#########################  End  ##########################
















#?######################### Start ##########################
#region:#?   Extracting remaining broad properties |   v0.1.0





#endregion#? Extracting remaining broad properties  |   v0.1.0
#?#########################  End  ##########################














#?######################### Start ##########################
#region:#?   Ontology Generation   |   v0.1.0




#endregion#? Ontology Generation  |   v0.1.0
#?#########################  End  ##########################





#endregion#! Schema Guided KG Generation  |   v0.1.0
#!#############################################  End Chapter  ##################################################
















#*######################### Start ##########################
#region:#*   FULL PIPELINE UP TO CONCEPT RESOLUTION (v1.0)

"""
run_full_pipeline_until_concept_resolution.py
---------------------------------------------
A test runner that executes all major steps:
  1️⃣ Chunking + Embedding
  2️⃣ Entity Recognition
  3️⃣ Entity Resolution
  4️⃣ Concept Recognition (Class Extraction)
  5️⃣ Concept Resolution

Each step saves its intermediate outputs to disk so the next can reuse them.
"""

import os
import json
import numpy as np
from pprint import pprint

# Assuming the functions below are already imported:
# chunk_text, embed_chunks, build_faiss_index, extract_entities_from_chunk,
# run_entity_resolution_iterative, run_concept_extraction, embed_concepts,
# run_resolution (concept resolution script).

# -------------------------
# 0️⃣ INPUT TEXT
# -------------------------
text = """
The new library GraphGlow implements the Graph Neural Flow idea and outperforms GNNChef on small social networks.
GNNChef (v2) is a framework; its motifX module detects recurring motifs and reports motifX-score per node.
When using shabakeh-augment the training stability improves, but only if dropout < 0.3.
GraphGlow also released GraphGlowLib, a set of utilities wrapping GraphGlow’s core API.
motifX in the paper is described both as an algorithm and as a benchmark dataset (MotifX-DS).
Despite similar names, MotifX-DS and motifX (algorithm) should not be merged; one is a dataset, the other a method.
"""

print("\n==========================")
print("🔹 STAGE 1: Chunking + Embedding")
print("==========================")

chunks = chunk_text(text, mode="disjoint", sentence_per_line=False)
metadata, vectors = embed_chunks(chunks)
faiss_index = build_faiss_index(vectors)

# Save metadata for later
with open("chunks_disjoint_metadata.jsonl", "w") as f:
    for m in metadata:
        json.dump(m, f)
        f.write("\n")
np.save("chunks_disjoint_embeddings.npy", vectors)

print(f"✅ Chunking complete. Total chunks: {len(metadata)}")

# -------------------------
# 2️⃣ ENTITY RECOGNITION
# -------------------------
print("\n==========================")
print("🔹 STAGE 2: Entity Recognition")
print("==========================")

all_entities = []
for c in metadata:
    ents = extract_entities_from_chunk(
        chunk_id=c["id"],
        metadata=metadata,
        vectors=vectors,
        faiss_index=faiss_index,
        mode="disjoint"
    )
    all_entities.extend(ents)

# Save raw entities
with open("entities_raw.jsonl", "w") as f:
    for e in all_entities:
        json.dump(e, f)
        f.write("\n")

print(f"✅ Entity recognition complete. Extracted {len(all_entities)} entities.")
pprint([{k: v for k, v in e.items() if k != 'embedding'} for e in all_entities[:5]])

# -------------------------
# 3️⃣ ENTITY RESOLUTION
# -------------------------
print("\n==========================")
print("🔹 STAGE 3: Entity Resolution")
print("==========================")

resolved_entities = run_entity_resolution_iterative(
    all_entities,
    metadata,
    vectors,
    faiss_index,
    max_iterations=3,
    stop_if_no_change=True,
    min_delta=1
)

with open("entities_resolved.jsonl", "w") as f:
    for e in resolved_entities:
        json.dump(e, f)
        f.write("\n")

print(f"✅ Entity resolution complete. Total resolved entities: {len(resolved_entities)}")
pprint([{k: v for k, v in e.items() if k != 'embedding'} for e in resolved_entities[:5]])

# -------------------------
# 4️⃣ CONCEPT RECOGNITION (Class Extraction)
# -------------------------
print("\n==========================")
print("🔹 STAGE 4: Concept Recognition (Class Extraction)")
print("==========================")

# Load BGE model
from transformers import AutoTokenizer, AutoModel
import torch
import faiss

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
model.eval()

# Ensure FAISS index exists
dim = vectors.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(vectors)
faiss_index.add(vectors)

# Run concept extraction
all_concepts = run_concept_extraction(metadata, vectors, faiss_index, resolved_entities, mode="disjoint")

# Embed provisional classes
all_concepts, concept_vectors = embed_concepts(all_concepts, tokenizer, model, DEVICE)
print(f"✅ Embedded {len(all_concepts)} provisional concepts.")

# Save for resolution
with open("concepts_provisional_metadata.jsonl", "w") as f:
    for c in all_concepts:
        json.dump(c, f)
        f.write("\n")
np.save("concepts_provisional_embeddings.npy", concept_vectors)

print("✅ Saved provisional concepts for Concept Resolution.")
pprint([{k: v for k, v in c.items() if k not in ('embedding', 'chunk_text')} for c in all_concepts[:5]])

# -------------------------
# 5️⃣ CONCEPT RESOLUTION
# -------------------------
print("\n==========================")
print("🔹 STAGE 5: Concept Resolution")
print("==========================")

# Run schema-level class resolution (from your concept_resolution.py)
final_classes = run_resolution()

# Read and preview
print("\n✅ Concept Resolution complete!")
with open("final_classes.jsonl", "r") as f:
    lines = f.readlines()

print(f"Total final classes: {len(lines)}")
for line in lines[:5]:
    c = json.loads(line)
    print(f"- {c['label']}: {c['description']}")

# -------------------------
# ✅ ALL DONE
# -------------------------
print("\n🎯 Pipeline completed up to Concept Resolution successfully!")
print("Generated files:")
print("  - chunks_disjoint_metadata.jsonl / embeddings.npy")
print("  - entities_raw.jsonl")
print("  - entities_resolved.jsonl")
print("  - concepts_provisional_metadata.jsonl / embeddings.npy")
print("  - final_classes.jsonl")

#endregion#*  FULL PIPELINE UP TO CONCEPT RESOLUTION
#*#########################  End  ##########################





















