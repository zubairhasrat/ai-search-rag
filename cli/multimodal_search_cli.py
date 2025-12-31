import argparse

from lib.multimodal_search import verify_image_embedding
from lib.multimodal_search import image_search_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    image_search_parser = subparsers.add_parser("image_search", help="Search movies by image")
    image_search_parser.add_argument("image", type=str, help="Path to an image file")
    args = parser.parse_args()

    match args.command:
        case "image_search":
            results = image_search_command(args.image)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (similarity: {res['score']:.3f})")
                print(f"   {res['description']}...")
                print()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()