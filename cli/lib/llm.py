import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

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
      system_instruction=system_instruction
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