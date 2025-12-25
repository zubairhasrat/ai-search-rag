import os
from json import load
from pickle import dump, load as pickle_load

class InvertedIndex:
    def __init__(self):
      self.index = {}
      self.docmap = {}

    def build(self):
        movies = load(open("data/movies.json"))
        for movie in movies["movies"]:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def __add_document(self, doc_id, text):
        tokens = text.lower().split()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
    def get_documents(self, term):
        token = term.lower()
        doc_ids = self.index.get(token, set())
        return sorted(doc_ids)

    def save(self):
      os.makedirs("cache", exist_ok=True)
      with open("cache/index.pkl", "wb") as f:
        dump(self.index, f)
      with open("cache/docmap.pkl", "wb") as f:
        dump(self.docmap, f)

    def load(self):
      try:
        with open("cache/index.pkl", "rb") as f:
          self.index = pickle_load(f)
        with open("cache/docmap.pkl", "rb") as f:
          self.docmap = pickle_load(f)
      except FileNotFoundError:
        self.index = {}
        self.docmap = {}