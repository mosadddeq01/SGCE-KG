# SGCE-KG: Semantic Graph Construction and Entity Knowledge Graph

A comprehensive pipeline for extracting entities and constructing knowledge graphs from textual documents, with a focus on technical documents and standards.

## Overview

SGCE-KG is a research project that implements an end-to-end pipeline for knowledge graph construction from documents. The system processes PDF documents, extracts structured information, identifies entities, and builds context-enriched knowledge graphs suitable for downstream analysis and querying.

## Key Features

- **Document Processing**: PDF extraction and text preprocessing
- **Intelligent Chunking**: Token-driven, sentence-preserving text chunking using spaCy
- **Entity Extraction**: LLM-powered entity identification with context awareness
- **Entity Clustering**: HDBSCAN-based clustering for entity disambiguation
- **Entity Resolution**: Canonical entity resolution across document chunks
- **Knowledge Graph Construction**: Building structured knowledge graphs from extracted entities
- **Vector Embeddings**: Support for semantic embeddings using sentence transformers

## Pipeline Components

### 1. Text Chunking
The system uses a sophisticated token-driven chunking algorithm that:
- Preserves sentence boundaries
- Maintains configurable token limits per chunk
- Uses spaCy for accurate sentence splitting
- Provides flexible chunk size control

### 2. Entity Identification
- Context-aware entity extraction using OpenAI API
- Supports previous chunk context for improved accuracy
- Extracts entity types, definitions, and relationships
- Maintains provenance to source chunks

### 3. Entity Clustering & Resolution
- HDBSCAN-based clustering using semantic embeddings
- Weighted field embeddings (name, type, definition, context)
- Canonical entity resolution across clusters
- Configurable clustering parameters

### 4. Knowledge Graph Generation
- Constructs structured knowledge graphs from resolved entities
- Maintains relationships between entities
- Supports multiple graph export formats

## Environment Setup

### Prerequisites
- Python 3.12+
- Conda (recommended for environment management)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mosadddeq01/SGCE-KG.git
cd SGCE-KG
```

2. Create and activate the conda environment:
```bash
conda env create -f KGrowth_Env.yml
conda activate KGrowth
```

3. Download the spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

### OpenAI API Configuration

For entity extraction, you'll need an OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Pipeline Execution

The main pipeline is implemented in `SGCE-KG_Latest.py`. The typical workflow includes:

1. **Text Chunking**: Process documents into manageable chunks
2. **Entity Extraction**: Identify entities from chunks
3. **Entity Clustering**: Group similar entities
4. **Entity Resolution**: Create canonical entity representations
5. **Knowledge Graph Construction**: Build the final knowledge graph

### Example Workflow

```python
# 1. Chunk documents
from SGCE-KG_Latest import sentence_chunks_token_driven
chunks = sentence_chunks_token_driven(
    sections_json_path="data/pdf_to_json/kept_sections.json",
    out_path="data/Chunks/chunks_sentence.jsonl",
    max_tokens_per_chunk=350,
    min_tokens_per_chunk=250
)

# 2. Extract entities
from SGCE-KG_Latest import run_entity_extraction_on_chunks
entities = run_entity_extraction_on_chunks(
    chunks_jsonl_path="data/Chunks/chunks_sentence.jsonl",
    entities_out_path="data/Entities/entities_raw.jsonl"
)

# 3. Cluster and resolve entities
# (See SGCE-KG_Latest.py for clustering and resolution functions)
```

## Project Structure

```
SGCE-KG/
├── SGCE-KG_Latest.py          # Main pipeline implementation
├── SGCE-KG_All_Versions.py    # Development history and alternative implementations
├── KGrowth_Env.yml            # Conda environment specification
├── data/                       # Data directory
│   ├── Chunks/                 # Processed text chunks
│   ├── pdf_to_json/           # Extracted PDF content
│   └── Entities/              # Extracted and resolved entities
├── KGs_from_Essays/           # Knowledge graph outputs
└── Experiments/               # Experimental code and tests
```

## Configuration

Key parameters can be configured for different pipeline stages:

### Chunking Parameters
- `max_tokens_per_chunk`: Maximum tokens per chunk (default: 350)
- `min_tokens_per_chunk`: Minimum tokens per chunk (default: 250)
- `sentence_per_line`: Format output with one sentence per line

### Entity Extraction Parameters
- `prev_n_context`: Number of previous chunks to include as context
- OpenAI model selection (e.g., "gpt-4", "gpt-3.5-turbo")

### Clustering Parameters
- `min_cluster_size`: Minimum cluster size for HDBSCAN
- `min_samples`: Minimum samples for core points
- Embedding weights for different entity fields

## Dependencies

Core dependencies include:
- **NLP**: spaCy, transformers, sentence-transformers
- **ML/Clustering**: scikit-learn, HDBSCAN, FAISS
- **LLM**: OpenAI API
- **Document Processing**: PyMuPDF, pdfplumber
- **Data Processing**: pandas, numpy, tqdm
- **Graph Processing**: NetworkX (if used for KG representation)

See `KGrowth_Env.yml` for the complete list of dependencies.

## Research Context

This project is designed for research in knowledge graph construction from technical documents. It has been tested on:
- API standards documents (e.g., API 571)
- Technical essays and reports
- Domain-specific documentation

## Contributing

This is a research project. For questions, issues, or contributions, please open an issue on GitHub.

## License

[Add your license information here]

## Acknowledgments

Built with:
- spaCy for NLP processing
- OpenAI for entity extraction
- HDBSCAN for clustering
- Sentence Transformers for embeddings

## Citation

If you use this project in your research, please cite:

```
[Add citation information when available]
```

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
