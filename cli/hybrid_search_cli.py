import argparse
from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_scores_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_scores_parser.add_argument("scores", type=float, nargs='+', help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted_search", help="Weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic vs BM25 (default: 0.5)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

    rrf_search_parser = subparsers.add_parser("rrf_search", help="RRF hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Query to search")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="k parameter for RRF (default: 60)")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")
    rrf_search_parser.add_argument("--enhance",type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            ns = normalize_command(args.scores)
            for score in ns:
                print(f"{score:.4f}")
        case "weighted_search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res['hybrid_score']:.3f}")
                print(f"   BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                print(f"   {res['description']}...")
        case "rrf_search":
            results = rrf_search_command(args.query, args.k, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f"   RRF Score: {res['rrf_score']:.3f}")
                print(f"   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}")
                print(f"   {res['description']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()