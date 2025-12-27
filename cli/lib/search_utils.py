import json
import os
import re

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
CHUNK_SIZE = 200

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
    
def dot(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("vectors must be the same length")
    total = 0.0
    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]
    return total


def euclidean_norm(vec):
    total = 0.0
    for x in vec:
        total += x**2

    return total**0.5

def chunk_text(text, chunk_size, overlap):
    if text is None or text == "":
        raise ValueError("Text cannot be None or empty")
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")
    if overlap < 0:
        raise ValueError("Overlap must be greater than 0")
    chunks = []
    words = text.split()
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:(i+chunk_size)]
        chunks.append(" ".join(chunk))

    return chunks

def semantic_chunk(text, max_chunk_size = 4, overlap = 0):
    if text is None or text == "":
        return []
    if max_chunk_size <= 0:
        raise ValueError("Max chunk size must be greater than 0")
    if overlap < 0:
        raise ValueError("Overlap must be greater than 0")

    clean_text = text.strip()
    if len(clean_text) == 0:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    step = max_chunk_size - overlap
    chunks = []
    if len(sentences) == 1 and not sentences[0].endswith(('.', '!', '?')):
        return [clean_text]
    for i in range(0, len(sentences), step):
        chunk = sentences[i:(i+max_chunk_size)]
        stripped_sentences = [s.strip() for s in chunk if s.strip()]
        if stripped_sentences:
            chunks.append(" ".join(stripped_sentences))
    return chunks