#!/usr/bin/env python3

import argparse

from lib.keyword_search import build_command, search_command, tf_command, idf_command, tfidf_command


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


    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()