from pathlib import Path

from documents import load_documents
from embeddings import EmbeddingModel
from search import SemanticSearch

PROJECT_DIR = Path(__file__).parent
INDEX_DIR = PROJECT_DIR / "data" / "index"


embedder = EmbeddingModel()

search = SemanticSearch(embedder)


if (INDEX_DIR / "embeddings.npy").exists() and (INDEX_DIR / "metadata.npy").exists():
    print("Loading existing search index...")
    search.load(INDEX_DIR)

else:
    print("Building search index...")

    documents = load_documents()

    search.index(documents)
    search.save(INDEX_DIR)

    print("Search index created.")


print(f"Indexed {len(search.documents)} document chunks.")
print("Semantic Search")
print("Type /exit to quit.\n")


while True:
    query = input("Search: ").strip()

    if not query:
        continue

    if query.lower() == "/exit":
        print("Goodbye!")
        break

    results = search.search(
        query=query,
        top_k=3,
    )

    print("\nResults:")

    for result in results:
        print(
            f"\n[{result.source} | chunk {result.chunk_id} | score {result.score:.4f}]"
        )

        print(result.content)

    print()
