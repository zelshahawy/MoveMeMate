# TaxMeMate

A **Retrieval-Augmented Generation (RAG)** CLI assistant for tax reduction strategies. Feed it your local PDFs and TXTs of tax forms, regulations, and guidelines—TaxMeMate will index the content and let you ask natural-language questions to discover tax-saving opportunities.

---

## Features

* **ETL Pipeline**: Extracts, cleans, and deduplicates headers from PDF/TXT documents.
* **Chunking**: Splits text into overlapping 512-token windows for context continuity.
* **Vector Indexing**: Embeds chunks with Sentence-Transformers and indexes via FAISS for sub-second retrieval.
* **RAG Chat**: Combines retrieved context with Mistral-3B-Instruct (default) or OpenAI GPT to generate concise, actionable tax advice.
* **Fine-Tuning**: Supervised fine-tuning pipeline via Unsloth + TRL’s SFTTrainer to specialize responses on custom Q\&A datasets.
* **Multi-Backend**: Supports CUDA, Apple MPS, or CPU for local inference, plus optional OpenAI API fallback.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/taxmemate.git
cd taxmemate

# Create and activate a Python 3.12+ virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install from PyPI
pip install taxmemate
```

(Or install directly from source: `pip install .`)

---

## Commands

All commands are available under the `taxmemate` entry point:

```bash
# Show global help
taxmemate --help
```

### Clean

Extract and clean raw documents into a JSONL corpus.

```bash
taxmemate clean <input_dir> <output_file>
# e.g.:
taxmemate clean ./raw_pdfs cleaned.jsonl
```

### Chunk

Tokenize and split cleaned text into overlapping passages.

```bash
taxmemate chunk cleaned.jsonl chunks.jsonl [--model MODEL]
# default MODEL: mistralai/Mistral-3B-Instruct-v0.1
```

### Index

Embed chunks and build a FAISS index plus metadata.

```bash
taxmemate index chunks.jsonl faiss.index meta.pkl [--embed-model EMBED_MODEL]
```

### Train-SFT

Fine-tune the base LLM on your own Q\&A dataset.

```bash
taxmemate train-sft sft_data.jsonl sft_model_dir [--model MODEL]
```

### Chat

Interactively ask tax questions with RAG-driven generation.

```bash
taxmemate chat faiss.index meta.pkl [--rag-model MODEL] [--openai-model MODEL] [--openai-api-key KEY]
```

---

## Usage Example

```bash
# 1. Clean source PDFs
 taxmemate clean ./raw_pdfs cleaned.jsonl

# 2. Chunk into passages
 taxmemate chunk cleaned.jsonl chunks.jsonl

# 3. Build the FAISS index
 taxmemate index chunks.jsonl faiss.index meta.pkl

# 4. (Optional) Fine-tune on your Q&A
 taxmemate train-sft my_sft.jsonl fine_tuned_model/

# 5. Chat for tax advice
 taxmemate chat faiss.index meta.pkl
```

> **Tip:** For snappier responses, use the OpenAI backend:
> `taxmemate chat faiss.index meta.pkl --openai-model gpt-3.5-turbo --openai-api-key $OPENAI_API_KEY`

---

## Configuration

* **`DEFAULT_MODEL`**: Override by passing `--model` or `--rag-model` flags.
* **Environment Variables**:

  * `TRANSFORMERS_CACHE`, `HF_HOME`, `HF_DATASETS_CACHE` for cache location.
  * `OPENAI_API_KEY` for OpenAI chat backend.

---

## Development & Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Install dev dependencies:

   ```bash
   pip install -e .[dev]
   ```
4. Run tests & linters:

   ```bash
   pytest && flake8
   ```
5. Submit a pull request

---

## License

[MIT License](LICENSE)

