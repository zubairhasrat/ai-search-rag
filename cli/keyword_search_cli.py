#!/usr/bin/env python3

import argparse
from json import load
import string
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex

def clean_text(text):
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table)

def tokenize(text):
    stemmer = PorterStemmer()
    stop_words = [line.strip() for line in open("data/stopwords.txt")]
    return [stemmer.stem(word) for word in clean_text(text).split(" ") if word not in stop_words]

def search_movies(query: str):
    movies = load(open("data/movies.json"))

    query = tokenize(query)
    matching = []
    for movie in movies["movies"]:
        title = tokenize(movie["title"])
        
        if any(any(word in t for t in title) for word in query):
            matching.append(movie)

    sorted_movies = sorted(matching, key=lambda m: m["id"])[:5]
    results = [movie["title"] for movie in sorted_movies]
    print(results)
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = search_movies(args.query)
            for result in results:
                print(result)
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
            first_id = index.get_documents("merida")[0]
            print(f"Index built and saved. First document for 'merida': {first_id}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()