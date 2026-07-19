import argparse
from pathlib import Path
import chromadb


def main():
    parser = argparse.ArgumentParser(description="List collections in a Chroma DB directory")
    parser.add_argument(
        "--path",
        default="./chroma_db",
        help="Path to the Chroma DB directory (default: ./chroma_db)",
    )
    args = parser.parse_args()

    db_path = Path(args.path).resolve()
    print(f"Checking Chroma DB at: {db_path}")

    if not db_path.exists():
        print("Directory does not exist.")
        return 1

    client = chromadb.PersistentClient(path=str(db_path))
    collections = client.list_collections()

    if not collections:
        print("No collections found.")
        return 0

    print(f"Found {len(collections)} collection(s):")
    for idx, collection in enumerate(collections, 1):
        print(f"{idx}. {collection.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())