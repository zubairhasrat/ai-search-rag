import os
import pickle
import string
from collections import defaultdict, Counter
import math

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
        if doc_id not in self.doc_lengths:
            self.doc_lengths[doc_id] = len(tokens)
    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("Term must be a single token")
        token = tokens[0]
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(token, 0)

    def get_bm25_idf(self, term: str) -> float:
      tokens = tokenize_text(term)
      if len(tokens) > 1:
          raise ValueError("Term must be a single token")
      token = tokens[0]
      N = len(self.docmap)
      df = len(self.get_documents(token))
      return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def __get_avg_doc_length(self) -> float:
      if len(self.doc_lengths) == 0:
        return 0
      return sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float, b: float) -> float:
      tf = self.get_tf(doc_id, term)
      avg_doc_length = self.__get_avg_doc_length()
      length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
      return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term, k1, b) -> float:
      bm25_tf = self.get_bm25_tf(doc_id, term, k1, b)
      bm25_idf = self.get_bm25_idf(term)
      return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit, k1, b):
      tokens = tokenize_text(query)
      scores = defaultdict(float)
      
      for token in tokens:
        for doc_id in self.index[token]:
          scores[doc_id] += self.bm25(doc_id, token, k1, b)
      return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

def bm25_search_command(query, limit, k1, b) -> list[tuple[dict, float]]:
  idx = InvertedIndex()
  idx.load()
  result = idx.bm25_search(query, limit, k1, b)
  for i, res in enumerate(result, 1):
    print(f"{i}. ({res[0]}) {idx.docmap[res[0]]['title']} - {res[1]:.2f}")

def bm25_tf_command(doc_id, term, k1, b) -> float:
  idx = InvertedIndex()
  idx.load()
  return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25_idf_command(term) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)
def tfidf_command(docId, term) -> float:
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(docId, term)
    idf = idf_command(term)
    return tf * idf

def idf_command(term) -> float:
    idx = InvertedIndex()
    idx.load()
    tokens = tokenize_text(term)
    if len(tokens) > 1:
        raise ValueError("Term must be a single token")
    token = tokens[0]
    num_documents = len(idx.docmap)
    documents_with_term = len(idx.get_documents(token))
    return math.log((num_documents + 1) / (documents_with_term + 1))

def tf_command(docId, term) -> None:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(docId, term)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

