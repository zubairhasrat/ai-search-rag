import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

from lib.search_utils import dot, euclidean_norm, chunk_text, CACHE_DIR, load_movies, semantic_chunk

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2")-> None:
      self.model = SentenceTransformer(model_name)
      self.embeddings = None
      self.documents = None
      self.document_map = {}

    def build_embeddings(self, documents: list[dict]):
      self.documents = documents
      doc_list = []
      for doc in documents:
        self.document_map[doc["id"]] = doc
        doc_list.append(f"{doc['title']}: {doc['description']}")
        
      self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
      np.save(os.path.join(CACHE_DIR, "movie_embeddings.npy"), self.embeddings)
      return self.embeddings

    def load_or_create_embeddings(self, documents):
      self.documents = documents
      self.document_map = {doc["id"]: doc for doc in documents}
      if os.path.exists(os.path.join(CACHE_DIR, "movie_embeddings.npy")):
        self.embeddings = np.load(os.path.join(CACHE_DIR, "movie_embeddings.npy"))
        if len(self.embeddings) == len(documents):
          return self.embeddings
      
      return self.build_embeddings(documents)

    def generate_embedding(self, text):
      if text is None or text == "":
        raise ValueError("Text cannot be None or empty")
      return self.model.encode([text])[0]
    
    def search(self, query, limit):
      if self.embeddings is None:
          raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
      
      query_embedding = self.generate_embedding(query)
      
      scores = []
      for i, doc_embedding in enumerate(self.embeddings):
          score = cosine_similarity(query_embedding, doc_embedding)
          scores.append((score, self.documents[i]))
      
      scores.sort(key=lambda x: x[0], reverse=True)
    
      return [
          {"score": score, "title": doc["title"], "description": doc["description"]}
          for score, doc in scores[:limit]
      ]

class ChunkedSemanticSearch(SemanticSearch):
  def __init__(self, model_name="all-MiniLM-L6-v2")-> None:
    super().__init__(model_name)
    self.chunk_embeddings = None
    self.chunk_metadata = None
  
  def build_chunk_embeddings(self, documents):
    self.documents = documents
    self.document_map = {doc["id"]: doc for doc in documents}

    all_chunks = []
    chunk_metadata = []

    for movie_idx, doc in enumerate(documents):
      if doc["description"] is None or doc["description"] == "":
        continue
      chunks = semantic_chunk(doc["description"], 4, 1)
      for chunk_idx, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        chunk_metadata.append({"movie_idx": movie_idx, "chunk_idx": chunk_idx, "total_chunks": len(chunks)})
      
    embeddings = self.model.encode(all_chunks, show_progress_bar=True)
    self.chunk_embeddings = embeddings
    self.chunk_metadata = chunk_metadata
    np.save(os.path.join(CACHE_DIR, "chunk_embeddings.npy"), self.chunk_embeddings)
    with open(os.path.join(CACHE_DIR, "chunk_metadata.json"), "w") as f:
      json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
    return self.chunk_embeddings
  
  def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
    self.documents = documents
    self.document_map = {doc["id"]: doc for doc in documents}

    if os.path.exists(os.path.join(CACHE_DIR, "chunk_embeddings.npy")) and os.path.exists(os.path.join(CACHE_DIR, "chunk_metadata.json")):
      self.chunk_embeddings = np.load(os.path.join(CACHE_DIR, "chunk_embeddings.npy"))

      with open(os.path.join(CACHE_DIR, "chunk_metadata.json"), "r") as f:
        self.chunk_metadata = json.load(f)["chunks"]
      
      return self.chunk_embeddings
    else:
      return self.build_chunk_embeddings(documents)
  
  def search(self, query, limit):
    if self.chunk_embeddings is None:
      raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

  def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
    query_embedding = self.generate_embedding(query)
    chunk_scores = []

    for chunk_idx, chunk_embedding in enumerate(self.chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        chunk_scores.append({
            "chunk_idx": chunk_idx,
            "movie_idx": self.chunk_metadata[chunk_idx]["movie_idx"],
            "score": score
        })
    
    movie_scores_map = {}
    for chunk_score in chunk_scores:
        movie_idx = chunk_score["movie_idx"]
        if movie_idx not in movie_scores_map or chunk_score["score"] > movie_scores_map[movie_idx]["score"]:
            movie_scores_map[movie_idx] = chunk_score
    
    sorted_movie_scores = sorted(movie_scores_map.values(), key=lambda x: x["score"], reverse=True)[:limit]
    
    results = []
    for movie_score in sorted_movie_scores:
        movie_idx = movie_score["movie_idx"]
        doc = self.documents[movie_idx]
        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "description": doc["description"][:100],
            "score": movie_score["score"],
            "metadata": self.chunk_metadata[movie_score["chunk_idx"]]
        })

    return results
 
def semantic_chunk_command(text, max_chunk_size = 4, overlap = 0):
  chunks = semantic_chunk(text, max_chunk_size, overlap)
  return chunks

def chunk_command(text, chunk_size, overlap):
  chunks = chunk_text(text, chunk_size, overlap)
  return chunks

def semantic_search_command(query, limit):
  semantic_search = SemanticSearch()
  semantic_search.load_or_create_embeddings(load_movies())
  results = semantic_search.search(query, limit)
  return results

def verify_embeddings():
  semantic_search = SemanticSearch()
  documents = load_movies()
  semantic_search.load_or_create_embeddings(documents)
  print(f"Number of docs:   {len(documents)}")
  print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")

def embed_query_text(query):
  semantic_search = SemanticSearch()
  embedding = semantic_search.generate_embedding(query)
  print(f"Query: {query}")
  print(f"First 5 dimensions: {embedding[:5]}")
  print(f"Shape: {embedding.shape}")
  return embedding

def embed_text(text):
  semantic_search = SemanticSearch()
  embedding = semantic_search.generate_embedding(text)
  print(f"Text: {text}")
  print(f"First 3 dimensions: {embedding[:3]}")
  print(f"Dimensions: {embedding.shape[0]}")

def verify_model() -> None:
  semantic_search = SemanticSearch()
  print(f"Model loaded: {semantic_search.model}")
  print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
