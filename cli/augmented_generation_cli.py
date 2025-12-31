import argparse
from lib.hybrid_search import rrf_search_command
from lib.llm import generate_content

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of results to summarize (default: 5)")

    citations_parser = subparsers.add_parser(
        "citations", help="Provide citations for search results"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of results to cite (default: 5)")

    question_parser = subparsers.add_parser(
        "question", help="Conversational question-answering"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of results to use (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            search_results = rrf_search_command(query, 60, 5)
            
            # Format documents for the LLM
            docs = "\n".join([f"- {res['title']}: {res['description']}" for res in search_results])
            
            system_instruction = """Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Provide a comprehensive answer that addresses the query:"""

            contents = f"""Query: {query}

Documents:
{docs}"""

            rag_response = generate_content(model="gemini-2.5-flash", contents=contents, system_instruction=system_instruction)

            print("Search Results:")
            for res in search_results:
                print(f"  - {res['title']}")
            print(f"\nRAG Response:\n{rag_response}")
        case "summarize":
            query = args.query
            limit = args.limit
            search_results = rrf_search_command(query, 60, limit)
            
            # Format documents for the LLM
            docs = "\n".join([f"- {res['title']}: {res['description']}" for res in search_results])
            
            system_instruction = """Summarize the following search results. Provide a concise summary of the movies found."""

            contents = f"""
                Provide information useful to this query by synthesizing information from multiple search results in detail.
                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
                This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                Query: {query}
                Search Results:
                {docs}
                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
                """

            summary_response = generate_content(model="gemini-2.5-flash", contents=contents, system_instruction=system_instruction)

            print("Search Results:")
            for res in search_results:
                print(f"  - {res['title']}")
            print(f"\nSummary:\n{summary_response}")

        case "citations":
            query = args.query
            limit = args.limit
            search_results = rrf_search_command(query, 60, limit)
            
            # Format documents for the LLM
            docs = "\n".join([f"- {res['title']}: {res['description']}" for res in search_results])

            prompt = f"""Answer the question or provide information based on the provided documents.

              This should be tailored to Hoopla users. Hoopla is a movie streaming service.

              If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

              Query: {query}

              Documents:
              {docs}

              Instructions:
              - Provide a comprehensive answer that addresses the query
              - Cite sources using [1], [2], etc. format when referencing information
              - If sources disagree, mention the different viewpoints
              - If the answer isn't in the documents, say "I don't have enough information"
              - Be direct and informative

              Answer:"""
            
            citations_response = generate_content(model="gemini-2.5-flash", contents=prompt)

            print("Search Results:")
            for res in search_results:
                print(f"  - {res['title']}")
            print(f"\nCitations:\n{citations_response}")
        case "question":
            question = args.question
            limit = args.limit
            search_results = rrf_search_command(question, 60, limit)
            
            # Format documents for the LLM
            context = "\n".join([f"- {res['title']}: {res['description']}" for res in search_results])

            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

              This should be tailored to Hoopla users. Hoopla is a movie streaming service.

              Question: {question}

              Documents:
              {context}

              Instructions:
              - Answer questions directly and concisely
              - Be casual and conversational
              - Don't be cringe or hype-y
              - Talk like a normal person would in a chat conversation

              Answer:"""

            system_instruction = "You are a helpful movie recommendation assistant for Hoopla."
            answer_response = generate_content(model="gemini-2.5-flash", contents=prompt, system_instruction=system_instruction)

            print("Search Results:")
            for res in search_results:
                print(f"  - {res['title']}")
            print(f"\nAnswer:\n{answer_response}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()