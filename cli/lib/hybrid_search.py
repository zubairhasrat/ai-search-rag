import os
from typing import Any
import json
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import normalize_scores, load_movies, BM25_K1, BM25_B
from .llm import generate_content, system_prompt, llm_rerank_individual, llm_rerank_batch, llm_rerank_cross_encoder, evaluate_rrf


def hybrid_score(semantic_score, bm25_score, alpha):
    """Calculate hybrid score: alpha * semantic + (1-alpha) * bm25"""
    return alpha * semantic_score + (1 - alpha) * bm25_score

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def llm_rerank(query: str, doc: dict) -> float:
  return llm_rerank_individual(query, doc)

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit, BM25_K1, BM25_B)

    def weighted_search(self, query, alpha, limit=5):
        # Get 500x results to ensure enough overlap between search methods
        expanded_limit = limit * 500
        
        # Get BM25 results: list of (doc_id, score) tuples
        bm25_results = self._bm25_search(query, expanded_limit)
        
        # Get semantic results: list of dicts with 'id', 'score', etc.
        semantic_results = self.semantic_search.search_chunks(query, expanded_limit)
        
        # Extract scores for normalization
        bm25_scores = [score for _, score in bm25_results]
        semantic_scores = [res["score"] for res in semantic_results]
        
        # Normalize scores
        normalized_bm25_scores = normalize_scores(bm25_scores) if bm25_scores else []
        normalized_semantic_scores = normalize_scores(semantic_scores) if semantic_scores else []
        
        # Build a mapping of doc_id -> {doc, bm25_score, semantic_score}
        doc_score_map = {}
        
        # Add BM25 results to map
        for i, (doc_id, _) in enumerate(bm25_results):
            if doc_id not in doc_score_map:
                doc_score_map[doc_id] = {
                    "doc": self.document_map[doc_id],
                    "bm25_score": 0,
                    "semantic_score": 0
                }
            doc_score_map[doc_id]["bm25_score"] = normalized_bm25_scores[i]
        
        # Add semantic results to map
        for i, res in enumerate(semantic_results):
            doc_id = res["id"]
            if doc_id not in doc_score_map:
                doc_score_map[doc_id] = {
                    "doc": self.document_map[doc_id],
                    "bm25_score": 0,
                    "semantic_score": 0
                }
            doc_score_map[doc_id]["semantic_score"] = normalized_semantic_scores[i]
        
        # Calculate hybrid scores and build results
        results = []
        for doc_id, data in doc_score_map.items():
            h_score = hybrid_score(data["semantic_score"], data["bm25_score"], alpha)
            results.append({
                "id": doc_id,
                "title": data["doc"]["title"],
                "description": data["doc"]["description"][:100] if data["doc"]["description"] else "",
                "hybrid_score": h_score,
                "semantic_score": data["semantic_score"],
                "bm25_score": data["bm25_score"]
            })
        
        # Sort by hybrid score descending
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return results[:limit]

    def rrf_search(self, query, k, limit=10, enhance=None, rerank_method=None):
        expanded_limit = limit * 5 if rerank_method is not None else limit * 500
        
        # [DEBUG] Log original query
        print(f"[DEBUG] Original query: '{query}'")
        
        if enhance:
            final_query = generate_content(model="gemini-2.5-flash", contents=query, system_instruction=system_prompt(enhance, query))
            # [DEBUG] Log enhanced query
            print(f"[DEBUG] Enhanced query ({enhance}): '{final_query}'")
        else:
            final_query = query
            print(f"[DEBUG] No enhancement applied, using original query")

        bm25_results = self._bm25_search(final_query, expanded_limit)
        semantic_results = self.semantic_search.search_chunks(final_query, expanded_limit)
        
        # Build map of doc_id -> bm25_rank (1-indexed)
        bm25_rank_map = {}
        for i, (doc_id, _) in enumerate(bm25_results):
          if doc_id not in bm25_rank_map:
            bm25_rank_map[doc_id] = i + 1
        
        # Build map of doc_id -> semantic_rank (1-indexed)
        semantic_rank_map = {}
        for i, res in enumerate(semantic_results):
          if res["id"] not in semantic_rank_map:
            semantic_rank_map[res["id"]] = i + 1
        
        # Get all unique doc_ids from both searches
        all_doc_ids = set(bm25_rank_map.keys()) | set(semantic_rank_map.keys())
        
        # Calculate combined RRF scores
        results = []
        for doc_id in all_doc_ids:
          bm25_rank = bm25_rank_map.get(doc_id)
          semantic_rank = semantic_rank_map.get(doc_id)
          
          # Sum RRF scores (only add if rank exists)
          combined_rrf = 0
          if bm25_rank is not None:
            combined_rrf += rrf_score(bm25_rank, k)
          if semantic_rank is not None:
            combined_rrf += rrf_score(semantic_rank, k)
          
          results.append({
            "id": doc_id,
            "title": self.document_map[doc_id]["title"],
            "description": self.document_map[doc_id]["description"][:100] if self.document_map[doc_id]["description"] else "",
            "rrf_score": combined_rrf,
            "bm25_rank": bm25_rank,
            "semantic_rank": semantic_rank,
            "llm_score": 0,
          })

        # Sort by RRF score first (before any reranking)
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # [DEBUG] Log results after RRF search (before reranking)
        print(f"[DEBUG] Results after RRF search (top {min(5, len(results))}):")
        for i, res in enumerate(results[:5]):
          print(f"[DEBUG]   {i+1}. {res['title']} (RRF: {res['rrf_score']:.4f})")

        if rerank_method == "individual":
          print(f"[DEBUG] Applying '{rerank_method}' reranking...")
          llm_scores = llm_rerank_individual(final_query, results)
          for result in results:
            result["llm_score"] = llm_scores[result["id"]]
          results.sort(key=lambda x: x["llm_score"], reverse=True)
        elif rerank_method == "batch":
          print(f"[DEBUG] Applying '{rerank_method}' reranking...")
          llm_ranked_ids = llm_rerank_batch(final_query, results)
          id_to_rank = {doc_id: rank + 1 for rank, doc_id in enumerate(llm_ranked_ids)}
          for result in results:
            result["rerank_rank"] = id_to_rank.get(result["id"], len(llm_ranked_ids) + 1)
          results.sort(key=lambda x: x["rerank_rank"])
        elif rerank_method == "cross_encoder":
          print(f"[DEBUG] Applying '{rerank_method}' reranking...")
          cross_encoder_scores = llm_rerank_cross_encoder(final_query, results)
          for result in results:
            result["cross_encoder_score"] = cross_encoder_scores[result["id"]]
          results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        
        # [DEBUG] Log final results after reranking
        if rerank_method:
          print(f"[DEBUG] Final results after '{rerank_method}' reranking (top {min(5, len(results))}):")
          for i, res in enumerate(results[:5]):
            if rerank_method == "individual":
              print(f"[DEBUG]   {i+1}. {res['title']} (LLM Score: {res['llm_score']:.3f})")
            elif rerank_method == "batch":
              print(f"[DEBUG]   {i+1}. {res['title']} (Rerank Rank: {res['rerank_rank']})")
            elif rerank_method == "cross_encoder":
              print(f"[DEBUG]   {i+1}. {res['title']} (Cross Encoder: {res['cross_encoder_score']:.3f})")
        
        return results[:limit]

def normalize_command(scores):
    return normalize_scores(scores)


def weighted_search_command(query, alpha, limit):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    return hybrid_search.weighted_search(query, alpha, limit)

def rrf_search_command(query, k, limit, enhance=None, rerank_method=None):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    return hybrid_search.rrf_search(query, k, limit, enhance, rerank_method)

def evaluate_rrf_command(results, query):
    return evaluate_rrf(results, query)