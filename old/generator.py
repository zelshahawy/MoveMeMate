import llama_cpp  # if using local Llama
import openai
import os
from typing import List

def generate_answer_local_llama(retrieved_chunks: List[dict], question: str, model_path: str) -> str:
    prompt_context = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        src = chunk["metadata"]["source"]
        prompt_context += f"Excerpt {i} (source: {src}): {chunk['text']}\n\n"

    full_prompt = (
        "You are MoveOutMate, an expert on student move-out procedures. "
        "Using the following excerpts, answer the question in a concise, step-by-step manner with citations.\n\n"
        f"{prompt_context}"
        f"Question: {question}\n\nAnswer:"
    )

    llm = llama_cpp.Llama(
        model_path=model_path,
        n_ctx=int(os.getenv("LLAMA_CTX_SIZE", 2048)),
        n_threads=int(os.getenv("LLAMA_NUM_THREADS", 4)),
        temperature=float(os.getenv("LLAMA_TEMPERATURE", 0.2)),
        top_p=float(os.getenv("LLAMA_TOP_P", 0.9))
    )
    response = llm.create(
        prompt=full_prompt,
        max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", 256)),
        stop=["\n\n"]
    )
    return response.get("choices")[0].get("text", "").strip()


def generate_answer_openai(retrieved_chunks: List[dict], question: str) -> str:
    prompt_context = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        src = chunk["metadata"]["source"]
        prompt_context += f"Excerpt {i} (source: {src}): {chunk['text']}\n\n"

    user_message = (
        "You are MoveOutMate, an expert on student move-out procedures. "
        "Using the following excerpts, answer the question in a concise, step-by-step manner with citations.\n\n"
        f"{prompt_context}"
        f"Question: {question}"
    )

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_message}],
        max_tokens=256,
        temperature=0.2,
        top_p=0.9,
    )
    return response.get("choices")[0].get("message", {}).get("content", "").strip()
