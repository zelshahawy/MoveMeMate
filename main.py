import os
import re
import json
import click
import pickle
from glob import glob
from tqdm import tqdm
from PyPDF2 import PdfReader
import multiprocessing as mp
import torch

try:
    mp.set_start_method('fork')
except RuntimeError:
    pass
# Disable tokenizer parallelism, limit threads
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# ----- Determine device (CUDA, MPS, or CPU) -----
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    click.echo(f"[Using CUDA]")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    click.echo(f"[Using MPS]")
else:
    DEVICE = torch.device('cpu')
    click.echo(f"[Using CPU]")

# Default model for RAG & SFT
DEFAULT_MODEL = 'mistralai/Mistral-3B-Instruct-v0.1'

def clean_text(text: str) -> str:
    # remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # collapse newlines
    text = re.sub(r'[\r\n]+', ' ', text)
    # collapse >2 repeats of common headers
    text = re.sub(r'(Federal income tax \d{4})(?:[ \t]*\1){2,}', r'\1', text)
    # collapse any 3+ repeats of the same 1-5 word sequence
    text = re.sub(r'(\b\w+(?: \w+){0,4}\b)(?:\s+\1){2,}', r'\1', text)
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_text_from_pdf(path: str) -> str:
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
    """Extract text, clean, dedupe headers, and output JSONL."""
    paths = glob(os.path.join(input_dir, '*'))
    with open(output_file, 'w') as out:
        for path in tqdm(paths, desc='Cleaning docs'):
            ext = os.path.splitext(path)[1].lower()
            raw = ''
            try:
                if ext == '.pdf':
                    raw = extract_text_from_pdf(path)
                elif ext == '.txt':
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw = f.read()
                if not raw:
                    continue
                cleaned = clean_text(raw)
                if cleaned:
                    json.dump({'text': cleaned}, out)
                    out.write('\n')
            except Exception as e:
                click.echo(f'Warning: failed to process {path}: {e}', err=True)
    click.echo(f'Cleaned docs written to {output_file}')

@cli.command()
@click.argument('cleaned_file', type=click.Path(exists=True))
@click.argument('chunks_file', type=click.Path())
@click.option('--model', default=DEFAULT_MODEL, help='Tokenizer model')
def chunk(cleaned_file, chunks_file, model):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    with open(chunks_file, 'w') as out:
        for entry in load_dataset('json', data_files=cleaned_file, split='train'):
            text = entry['text']
            enc = tokenizer(text, return_overflowing_tokens=True,
                             max_length=512, stride=64, truncation=True)
            for ids in enc['input_ids']:
                chunk_text = tokenizer.decode(ids, clean_up_tokenization_spaces=True)
                json.dump({'chunk': chunk_text}, out)
                out.write('\n')
    click.echo(f'Chunks written to {chunks_file}')

@cli.command()
@click.argument('chunks_file', type=click.Path(exists=True))
@click.argument('index_file', type=click.Path())
@click.argument('meta_file', type=click.Path())
@click.option('--embed-model', default='all-MiniLM-L6-v2', help='Embedder model')
def index(chunks_file, index_file, meta_file, embed_model):
    import faiss
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    embedder = SentenceTransformer(embed_model)
    dim = embedder.get_sentence_embedding_dimension()
    idx = faiss.IndexFlatL2(dim)
    meta = []
    for entry in load_dataset('json', data_files=chunks_file, split='train'):
        vec = embedder.encode(entry['chunk'])
        idx.add(vec.reshape(1, -1))
        meta.append(entry['chunk'])
    faiss.write_index(idx, index_file)
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    click.echo(f'Index saved to {index_file}, metadata to {meta_file}')

@cli.command()
@click.argument('sft_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--model', default=DEFAULT_MODEL, help='Base model')
@click.option('--batch', default=4, help='Batch size')
@click.option('--epochs', default=3, help='Epochs')
def train_sft(sft_file, output_dir, model, batch, epochs):
    from transformers import AutoTokenizer, TrainingArguments
    from unsloth import FastLanguageModel, get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    model_fast = FastLanguageModel.from_pretrained(model)
    template = get_chat_template('alpaca')
    ds = load_dataset('json', data_files=sft_file, split='train')
    args = TrainingArguments(per_device_train_batch_size=batch,
                              learning_rate=2e-5,
                              num_train_epochs=epochs,
                              output_dir=output_dir)
    trainer = SFTTrainer(model=model_fast, args=args,
                         train_dataset=ds, tokenizer=tokenizer,
                         template=template)
    trainer.train()
    click.echo(f'SFT model saved to {output_dir}')

@cli.command()
@click.argument('index_file', type=click.Path(exists=True))
@click.argument('meta_file', type=click.Path(exists=True))
@click.option('--rag-model', default=DEFAULT_MODEL, help='HF model')
@click.option('--openai-model', default=None, help='OpenAI model')
@click.option('--openai-api-key', default=None, help='OpenAI API key or env var')
@click.option('--embed-model', default='all-MiniLM-L6-v2', help='Embedder model')
@click.option('--sft-model', default=None, help='Local SFT model')
def chat(index_file, meta_file, rag_model, openai_model,
         openai_api_key, embed_model, sft_model):
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    idx = faiss.read_index(index_file)
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    embedder = SentenceTransformer(embed_model)

    using_openai = bool(openai_model)
    if using_openai:
        import openai
        openai.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
    else:
        model_name = sft_model or rag_model
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(DEVICE)
        model.eval()

    click.echo(f"TaxMeMate is ready on {DEVICE}! Ask your tax questions (type 'exit' to quit).")
    while True:
        query = input('> ')
        if query.lower() in ('exit', 'quit'):
            break

        # Retrieve top-3 contexts
        q_vec = embedder.encode(query)
        D, I = idx.search(q_vec.reshape(1, -1), 3)
        context = "\n".join(meta[i] for i in I[0])
        prompt = f"Context:\n{context}\n\nUser: {query}\nAssistant:"

        if using_openai:
            response = openai.ChatCompletion.create(
                model=openai_model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=50
            )
            answer = response.choices[0].message.content.strip()
        else:
            # Tokenize and truncate
            max_new = 50
            inp = tokenizer(prompt, return_tensors='pt')
            input_ids = inp.input_ids
            attn = inp.attention_mask
            max_len = model.config.max_position_embeddings - max_new
            if input_ids.size(1) > max_len:
                input_ids = input_ids[:, -max_len:]
                attn = attn[:, -max_len:]

            # Generate with sampling and repetition penalties
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids.to(DEVICE),
                    attention_mask=attn.to(DEVICE),
                    max_new_tokens=max_new,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id
                )
            # Decode only generated tokens
            gen_ids = outputs[0, input_ids.shape[-1]:]
            answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        click.echo(answer)

if __name__ == '__main__':
    cli()

