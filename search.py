"""
Semantic search over indexed MF Guide chunks.

Usage:
    python search.py "delegating underwriting authority"
    python search.py                   # interactive prompt
"""

import sys

import config


def search(query: str, k: int = 5) -> list:
    vector = config.EMBEDDER.embed([query])[0]
    return config.STORE.search(vector, k)


def print_results(hits: list):
    if not hits:
        print("No results found.")
        return

    for i, hit in enumerate(hits, 1):
        src   = hit["_source"]
        score = hit["_score"]
        pos   = f"{src.get('chunk_index', 0) + 1}/{src.get('total_chunks', '?')}"
        text  = src.get("text", "")

        print(f"{i}. score={score:.4f}  {src.get('chunk_id', '?')}  [{pos}]")
        if src.get("page_title"):
            print(f"   Page:    {src['page_title']}")
        if src.get("section_heading"):
            print(f"   Section: {src['section_heading']}")
        if src.get("source_url"):
            print(f"   URL:     {src['source_url']}")
        print(f"   Text:    {text[:200]}{'...' if len(text) > 200 else ''}")
        print()


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Search query: ")
    print(f"Query: {query}\n")
    print_results(search(query))


if __name__ == "__main__":
    main()
