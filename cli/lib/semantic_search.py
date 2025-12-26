import numpy as np
import os
from sentence_transformers import SentenceTransformer

from lib.search_utils import dot, euclidean_norm
from lib.search_utils import CACHE_DIR
from lib.search_utils import load_movies

class SemanticSearch:
    def __init__(self):
      self.model = SentenceTransformer("all-MiniLM-L6-v2")
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
