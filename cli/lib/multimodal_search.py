from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
from lib.search_utils import load_movies

class MultimodalSearch:
  def __init__(self, model_name="clip-ViT-B-32", docs=[]):
    self.model = SentenceTransformer(model_name)
    self.documents = docs
    self.texts = [f"{doc['title']}: {doc['description']}" for doc in docs]
    if self.texts:
      self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    else:
      self.text_embeddings = []

  def embed_image(self, img_path):
    image = Image.open(img_path)
    return self.model.encode(image)

  def search_with_image(self, img_path):
    # Generate embedding for the image
    image = Image.open(img_path)
    img_emb = self.model.encode(image)

    # Calculate cosine similarity with each text embedding
    results = []
    for i, text_emb in enumerate(self.text_embeddings):
      sim = cosine_similarity(text_emb, img_emb)
      doc = self.documents[i]
      results.append({
        "id": doc["id"],
        "title": doc["title"],
        "description": doc["description"][:100] if doc["description"] else "",
        "score": float(sim)
      })

    # Sort by similarity score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]


def image_search_command(img_path):
  movies = load_movies()
  multimodal_search = MultimodalSearch(docs=movies)
  return multimodal_search.search_with_image(img_path)


def verify_image_embedding(img_path):
  multimodal_search = MultimodalSearch()
  embedding = multimodal_search.embed_image(img_path)
  print(f"Image: {img_path}")
  print(f"Embedding shape: {embedding.shape[0]} dimensions")
