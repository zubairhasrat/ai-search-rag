#!/usr/bin/env python3

import argparse
from lib.search_utils import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT

from lib.keyword_search import build_command, search_command, tf_command, idf_command, tfidf_command, bm25_idf_command, bm25_tf_command, bm25_search_command
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Calculate term frequency")
    tf_parser.add_argument("docId", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Calculate inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF")
    tfidf_parser.add_argument("docId", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("verify_model", help="Verify the semantic search model")

    add_vector_parser = subparsers.add_parser("add_vector", help="Add two vectors")
    add_vector_parser.add_argument("vec1", type=list, help="First vector")
    add_vector_parser.add_argument("vec2", type=list, help="Second vector")

    subtract_vector_parser = subparsers.add_parser("subtract_vector", help="Subtract two vectors")
    subtract_vector_parser.add_argument("vec1", type=list, help="First vector")
    subtract_vector_parser.add_argument("vec2", type=list, help="Second vector")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query text")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("semantic_search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit the number of results")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            result = tf_command(args.docId, args.term)
            print(result)
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(args.docId, args.term)
            print(f"TF-IDF of '{args.term}' in document {args.docId}: {tfidf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score for '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score for '{args.term}' in document {args.doc_id}: {bm25tf:.2f}")
        case "bm25search":
            bm25_search_command(args.query, DEFAULT_SEARCH_LIMIT, BM25_K1, BM25_B)
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()