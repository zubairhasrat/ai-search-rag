import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from time import sleep
import json
from sentence_transformers import CrossEncoder

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

client = genai.Client(api_key=api_key)

def generate_content(model: str, contents: str, system_instruction: str) -> str:
  response =client.models.generate_content(
    model=model,
    contents=contents,
    config=types.GenerateContentConfig(
      response_mime_type="text/plain",
      system_instruction=system_instruction,
      stopSequences=["```json", "```"],
    )
  )
  return response.text

def system_prompt(enhance: str, query: str) -> str:
  if enhance == "spell":
    return spell_enhancement_prompt(query)
  elif enhance == "rewrite":
    return query_enhancement_prompt(query)
  elif enhance == "expand":
    return expand_query_prompt(query)
  else:
    return query

def evaluate_rrf(search_results: list, query: str) -> list:
  formatted_results = [f"{i+1}. {doc['title']} - {doc['description']}" for i, doc in enumerate(search_results)]
  contents = "\n".join(formatted_results)
  response = generate_content(
    model="gemini-2.5-flash", 
    contents=contents, 
    system_instruction=evaluate_rrf_prompt(query, formatted_results)
  )
  return json.loads(response.strip())


def llm_rerank_cross_encoder(query: str, docs: list) -> dict:
  cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
  pairs = []
  for doc in docs:
    pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
  scores = cross_encoder.predict(pairs)
  return {docs[i]["id"]: float(scores[i]) for i in range(len(docs))}

def llm_rerank_individual(query: str, docs: list) -> dict:
  llm_score = {}
  for doc in docs:
    movie_info = f"{doc.get('title', '')} - {doc.get('description', '')}"
    response = generate_content(model="gemini-2.5-flash", contents=movie_info, system_instruction=rerank_query_prompt(query, doc))
    llm_score[doc["id"]] = float(response.strip())
    sleep(3)
  return llm_score

def llm_rerank_batch(query: str, docs: list) -> list:
  doc_list_str = "\n".join([f"{doc['id']}: {doc['title']} - {doc['description']}" for doc in docs])
  response = generate_content(model="gemini-2.5-flash", contents=doc_list_str, system_instruction=rerank_batch_prompt(query, doc_list_str))
  # Clean up response - strip markdown code blocks if present
  cleaned = response.strip()
  return json.loads(cleaned)

def spell_enhancement_prompt(query: str) -> str:
  return f"""
    Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""

def query_enhancement_prompt(query: str) -> str:
  return f"""Rewrite this movie search query to be more specific and searchable.

  Original: "{query}"

  Consider:
  - Common movie knowledge (famous actors, popular films)
  - Genre conventions (horror = scary, animation = cartoon)
  - Keep it concise (under 10 words)
  - It should be a google style search query that's very specific
  - Don't use boolean logic

  Examples:

  - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
  - "movie about bear in london with marmalade" -> "Paddington London marmalade"
  - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

  Rewritten query:"""

def expand_query_prompt(query: str) -> str:
  return f"""Expand this movie search query with related terms.

  Add synonyms and related concepts that might appear in movie descriptions.
  Keep expansions relevant and focused.
  This will be appended to the original query.

  Examples:

  - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
  - "action movie with bear" -> "action thriller bear chase fight adventure"
  - "comedy with bear" -> "comedy funny bear humor lighthearted"

  Query: "{query}"
  """

def rerank_query_prompt(query: str, doc: dict) -> str:
  return f"""Rate how well this movie matches the search query.

  Query: "{query}"
  Movie: {doc.get("title", "")} - {doc.get("description", "")}

  Consider:
  - Direct relevance to query
  - User intent (what they're looking for)
  - Content appropriateness

  Rate 0-10 (10 = perfect match).
  Give me ONLY the number in your response, no other text or explanation.

  Score:"""

def rerank_batch_prompt(query: str, doc_list_str: str) -> str:
  return f"""Rank these movies by relevance to the search query.

  Query: "{query}"

  Movies:
  {doc_list_str}

  Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

  [75, 12, 34, 2, 1]
  """

def evaluate_rrf_prompt(query: str, formatted_results: list) -> str:
  return f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""