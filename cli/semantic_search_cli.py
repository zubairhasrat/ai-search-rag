#!/usr/bin/env python3

import argparse
from lib.search_utils import DEFAULT_SEARCH_LIMIT, CHUNK_SIZE, load_movies

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search_command, chunk_command, semantic_chunk_command, ChunkedSemanticSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Calculate term frequency")
    tf_parser.add_argument("docId", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    subparsers.add_parser("verify_model", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query text")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("semantic_search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit the number of results")

    chunk_parse = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parse.add_argument("text", type=str, help="Text to chunk")
    chunk_parse.add_argument("chunk_size", type=int, nargs='?', default=CHUNK_SIZE, help="Chunk size")
    chunk_parse.add_argument("overlap", type=int, nargs='?', default=0, help="Overlap")

    semantic_chunk_parse = subparsers.add_parser("semantic_chunk", help="Semantic chunk text")
    semantic_chunk_parse.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parse.add_argument("max_chunk_size", type=int, nargs='?', default=4, help="Max chunk size")
    semantic_chunk_parse.add_argument("overlap", type=int, nargs='?', default=0, help="Overlap")

    embed_chunks_parse = subparsers.add_parser("embed_chunks", help="Embed chunks")
    embed_chunks_parse.add_argument("max_chunk_size", type=int, nargs='?', default=4, help="Max chunk size")
    embed_chunks_parse.add_argument("overlap", type=int, nargs='?', default=0, help="Overlap")

    search_chunked_parse = subparsers.add_parser("search_chunked", help="Search chunked")
    search_chunked_parse.add_argument("query", type=str, help="Query to search")
    search_chunked_parse.add_argument("limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit the number of results")

    args = parser.parse_args()

    match args.command:
        case "verify_model":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "semantic_search":
            results = semantic_search_command(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} - {res['score']:.2f}")
        case "chunk":
          results = chunk_command(args.text, args.chunk_size, args.overlap)
          for i, res in enumerate(results, 1):
            print(f"{i}. {res}")
        case "semantic_chunk":
          results = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
          for i, res in enumerate(results, 1):
            print(f"{i}. {res}")
        case "embed_chunks":
          movies = load_movies()
          chunk_semantic_search = ChunkedSemanticSearch()
          embeddings = chunk_semantic_search.load_or_create_chunk_embeddings(movies)
          print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
          movies = load_movies()
          chunk_semantic_search = ChunkedSemanticSearch()
          chunk_semantic_search.load_or_create_chunk_embeddings(movies)
          results = chunk_semantic_search.search_chunks(args.query, args.limit)
          for i, res in enumerate(results, 1):
            print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
            print(f"   {res['description']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()