from pathlib import Path
import os
from typing import List, Tuple
from pdfminer.high_level import extract_text as extract_pdf_text


def load_docs_from_folder(folder: Path) -> List[Tuple[str, str]]:
    """
    Recursively read all .pdf and .md files under `folder`.
    Return a list of (doc_id, text) tuples.
    `doc_id` can be filename + page/section indicator.
    """
    entries: List[Tuple[str, str]] = []
    for root, _, files in os.walk(folder):
        for fname in files:
            path = Path(root) / fname
            if path.suffix.lower() == ".pdf":
                full_text = extract_pdf_text(str(path))
                entries.append((fname, full_text))
            elif path.suffix.lower() in (".md", ".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                entries.append((fname, text))
    return entries


def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    """
    Naive chunking: split text into chunks of ~chunk_size words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks
