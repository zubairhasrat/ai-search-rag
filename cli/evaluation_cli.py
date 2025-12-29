import argparse
import json
from lib.hybrid_search import rrf_search_command


def precision_at_k(retrieved: list, relevant: list, k: int) -> float:
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for title in retrieved_at_k if title in relevant)
    return relevant_retrieved / k if k > 0 else 0.0

def recall_at_k(retrieved: list, relevant: list) -> float:
    relevant_retrieved = sum(1 for title in retrieved if title in relevant)
    return relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0

def f1_score_at_k(precision: float, recall: float) -> float:
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    rrf_k = 60  # RRF k parameter

    # run evaluation logic here
    with open("data/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)
    
    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        
        # Run RRF search with k=60 and limit from args
        results = rrf_search_command(query, rrf_k, limit)
        
        # Get the titles of returned results
        retrieved_titles = [res["title"] for res in results]

        # Calculate metrics
        precision = precision_at_k(retrieved_titles, relevant_docs, limit)
        recall = recall_at_k(retrieved_titles, relevant_docs)

        f1_score = f1_score_at_k(precision, recall)
        
        print(f"\nQuery: '{query}'")
        print(f"Precision@{limit}: {precision:.3f}")
        print(f"Recall@{limit}: {recall:.3f}")
        print(f"F1 Score@{limit}: {f1_score:.3f}")
        print(f"Expected: {relevant_docs}")
        print(f"Retrieved: {retrieved_titles}")

if __name__ == "__main__":
    main()