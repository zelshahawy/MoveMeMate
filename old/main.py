import os
import click
from pathlib import Path
from typing import List
from doc_proc import load_docs_from_folder, chunk_text
from generator import generate_answer_local_llama, generate_answer_openai

import chromadb
from sentence_transformers import SentenceTransformer

@click.command(name="build-index-cmd")
@click.option(
    "--docs-dir", default="docs", type=click.Path(exists=True),
    help="Folder containing PDF/Markdown docs to index."
)
@click.option(
    "--index-dir", default="data/chroma", type=click.Path(),
    help="Where to store the Chroma vector index."
)
def build_index_cmd(docs_dir, index_dir):
    """
    Build embeddings + vector index from raw documents.
    """
    docs_folder = Path(docs_dir)
    idx_folder = Path(index_dir)
    idx_folder.mkdir(parents=True, exist_ok=True)

    docs = load_docs_from_folder(docs_folder)
    all_chunks = []
    for doc_id, full_text in docs:
        chunks = chunk_text(full_text, chunk_size=200)
        for i, c in enumerate(chunks):
            chunk_id = f"{doc_id}::chunk_{i}"
            all_chunks.append({"id": chunk_id, "text": c, "metadata": {"source": doc_id, "chunk_index": i}})

    client = chromadb.Client()
    collection = client.create_collection(name="moveoutmate_docs")

    embed_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"))
    def embed_fn(texts: List[str]) -> List[List[float]]:
        return embed_model.encode(texts, show_progress_bar=True).tolist()

    texts = [chunk["text"] for chunk in all_chunks]
    ids = [chunk["id"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas,
        embedding_function=embed_fn,
    )

    collection.persist()
    click.echo(f"Indexed {len(all_chunks)} chunks from {len(docs)} documents into {idx_folder}")

# ---------------- LOADING INDEX ----------------
def load_chroma_index(index_dir: Path):
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="moveoutmate_docs",
        persist_directory=str(index_dir)
    )
    return collection

# ---------------- EMBED & RETRIEVE ----------------
def embed_and_retrieve(collection, question: str, top_k: int = 3) -> List[dict]:
    embed_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"))
    q_embed = embed_model.encode([question]).tolist()[0]
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=top_k,
        include=["documents", "metadatas", "ids"]
    )
    retrieved = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]
    for idx in range(len(docs)):
        retrieved.append({
            "id": ids[idx],
            "text": docs[idx],
            "metadata": metas[idx]
        })
    return retrieved

# ---------------- GENERATION (LOCAL LLaMA OR OPENAI) ----------------

# ---------------- CLI ENTRYPOINT ----------------
@click.group()
@click.option(
    "--llama-model-path", type=click.Path(exists=True),
    default=lambda: os.getenv("LLAMA_MODEL_PATH", "/models/llama-2-q4.bin"),
    help="Path to local Llama model .bin file."
)
@click.pass_context
def cli(ctx, llama_model_path):
    """MoveOutMate-CLI: RAG Q&A for student move-out."""
    ctx.ensure_object(dict)
    ctx.obj["llama_model_path"] = llama_model_path

@cli.command(name="build-index")
@click.option(
    "--docs-dir", default="docs", type=click.Path(exists=True),
    help="Folder containing raw PDF/Markdown docs."
)
@click.option(
    "--index-dir", default="data/chroma", type=click.Path(),
    help="Folder to write persisted vector index files."
)
def build_index(ctx, docs_dir, index_dir):
    build_index_cmd(docs_dir, index_dir)

@cli.command(name="ask")
@click.option(
    "--index-dir", default="data/chroma", type=click.Path(exists=True),
    help="Folder where vector index is stored."
)
@click.option("--question", "-q", default=None, type=str, help="Ask a single question and exit.")
@click.option("--interactive/--no-interactive", default=False, help="Start interactive REPL.")
@click.option("--use-openai/--no-openai", default=False, help="Use OpenAI for generation instead of local Llama.")
@click.pass_context
def ask(ctx, index_dir, question, interactive, use_openai):
    """Query MoveOutMate via CLI. Either pass a question or run interactive REPL."""
    idx_folder = Path(index_dir)
    collection = load_chroma_index(idx_folder)
    llama_path = ctx.obj.get("llama_model_path")

    def answer_query(q: str):
        retrieved = embed_and_retrieve(collection, q, top_k=3)
        if use_openai:
            ans = generate_answer_openai(retrieved, q)
        else:
            ans = generate_answer_local_llama(retrieved, q, model_path=llama_path)
        click.echo("\n--- MoveOutMate Answer ---")
        click.echo(ans)
        click.echo("--------------------------\n")

    if question:
        answer_query(question)
    elif interactive:
        click.echo("MoveOutMate CLI Interactive Mode (type 'exit' or Ctrl+C to quit)")
        while True:
            try:
                q = click.prompt("Your question")
            except (EOFError, KeyboardInterrupt):
                click.echo("\nExiting.")
                break
            if q.strip().lower() in ("exit", "quit"):
                break
            answer_query(q)
    else:
        click.echo("Error: Provide either --question or --interactive flag.")
        click.echo("E.g.: python moveoutmate.py ask --question \"How do I cancel utilities?\"")

if __name__ == "__main__":
    cli()
