"""
TaxMeMate CLI: A Retrieval-Augmented Generation (RAG) assistant for tax reduction strategies.

Commands:
  clean      Extract and clean raw PDF/TXT tax documents
  chunk      Tokenize and chunk cleaned documents
  index      Build a FAISS vector index
  train-sft  Fine-tune the model with supervised data
  chat       Interactively query TaxMeMate
"""
import os
import re
import json
import click
import pickle
from glob import glob
from tqdm import tqdm
from PyPDF2 import PdfReader

# ----- Utilities -----

def clean_text(text: str) -> str:
    """Remove unwanted characters and normalize whitespace."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        ptext = page.extract_text()
        if ptext:
            texts.append(ptext)
    return "\n".join(texts)

# ----- CLI -----
@click.group()
def cli():
    pass

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def clean(input_dir, output_file):
    """Extract text from PDFs/TXTs in INPUT_DIR, clean, and output JSONL to OUTPUT_FILE."""
    paths = glob(os.path.join(input_dir, '*'))
    with open(output_file, 'w') as out:
        for path in tqdm(paths, desc='Cleaning docs'):
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext == '.pdf':
                    raw = extract_text_from_pdf(path)
                elif ext == '.txt':
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw = f.read()
                else:
                    continue
                cleaned = clean_text(raw)
                if cleaned:
                    json.dump({'text': cleaned}, out)
                    out.write('\n')
            except Exception as e:
                click.echo(f'Warning: Failed to process {path}: {e}', err=True)
    click.echo(f'Cleaned docs written to {output_file}')

@cli.command()
@click.argument('cleaned_file', type=click.Path(exists=True))
@click.argument('chunks_file', type=click.Path())
@click.option('--model', default='meta-llama/Llama-3-8b', help='Tokenizer model')
def chunk(cleaned_file, chunks_file, model):
    """Chunk texts from CLEANED_FILE into JSONL CHUNKS_FILE."""
    # Lazy imports for faster startup
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    with open(chunks_file, 'w') as out:
        for entry in load_dataset('json', data_files=cleaned_file, split='train'):
            text = entry['text']
            enc = tokenizer(
                text,
                return_overflowing_tokens=True,
                max_length=512,
                stride=64,
                truncation=True
            )
            for ids in enc['input_ids']:
                chunk_text = tokenizer.decode(ids, clean_up_tokenization_spaces=True)
                json.dump({'chunk': chunk_text}, out)
                out.write('\n')
    click.echo(f'Chunks written to {chunks_file}')

@cli.command()
@click.argument('chunks_file', type=click.Path(exists=True))
@click.argument('index_file', type=click.Path())
@click.argument('meta_file', type=click.Path())
@click.option('--embed-model', default='all-MiniLM-L6-v2', help='Sentence embedder')
def index(chunks_file, index_file, meta_file, embed_model):
    """Build FAISS index from CHUNKS_FILE, save to INDEX_FILE and META_FILE."""
    # Lazy imports
    import faiss
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    embedder = SentenceTransformer(embed_model)
    dim = embedder.get_sentence_embedding_dimension()
    idx = faiss.IndexFlatL2(dim)
    meta = []
    for entry in load_dataset('json', data_files=chunks_file, split='train'):
        text = entry['chunk']
        vec = embedder.encode(text)
        idx.add(vec.reshape(1, -1))
        meta.append(text)
    faiss.write_index(idx, index_file)
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    click.echo(f'Index saved to {index_file}, metadata to {meta_file}')

@cli.command()
@click.argument('sft_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--model', default='meta-llama/Llama-3-8b', help='Base model')
@click.option('--batch', default=4, help='Train batch size')
@click.option('--epochs', default=3, help='Number of epochs')
def train_sft(sft_file, output_dir, model, batch, epochs):
    """Supervised fine-tune with SFTTrainer on SFT_FILE dataset."""
    # Lazy imports
    from transformers import AutoTokenizer, TrainingArguments
    from unsloth import FastLanguageModel, get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(model)
    model_fast = FastLanguageModel.from_pretrained(model)
    template = get_chat_template('alpaca')
    ds = load_dataset('json', data_files=sft_file, split='train')
    args = TrainingArguments(
        per_device_train_batch_size=batch,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        output_dir=output_dir
    )
    trainer = SFTTrainer(
        model=model_fast,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        template=template
    )
    trainer.train()
    click.echo(f'SFT model saved to {output_dir}')

@cli.command()
@click.argument('index_file', type=click.Path(exists=True))
@click.argument('meta_file', type=click.Path(exists=True))
@click.option('--rag-model', default='gpt-3.5-turbo', help='LLM for RAG')
@click.option('--embed-model', default='all-MiniLM-L6-v2', help='Embedder model')
@click.option('--sft-model', default=None, help='Path to SFT model')
def chat(index_file, meta_file, rag_model, embed_model, sft_model):
    """Interactive RAG chat session."""
    # Lazy imports
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    idx = faiss.read_index(index_file)
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    embedder = SentenceTransformer(embed_model)

    if sft_model:
        tokenizer = AutoTokenizer.from_pretrained(sft_model)
        model = AutoModelForCausalLM.from_pretrained(sft_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(rag_model)
        model = AutoModelForCausalLM.from_pretrained(rag_model)

    click.echo("TaxMeMate is ready! Ask your tax questions (type 'exit' to quit).")
    while True:
        query = input('> ')
        if query.lower() in ('exit', 'quit'):
            break
        q_vec = embedder.encode(query)
        D, I = idx.search(q_vec.reshape(1, -1), 5)
        context = "\n".join(meta[i] for i in I[0])
        prompt = f"Context:\n{context}\n\nUser: {query}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(**inputs, max_new_tokens=200)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        click.echo(answer.split('Assistant:')[-1].strip())

if __name__ == '__main__':
    cli()

