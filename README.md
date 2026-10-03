# Semantic Search

A lightweight semantic search engine built with Python and Sentence Transformers.

This is project #3 in the `ai-engineering` series. It shows how to convert text into vector embeddings and search by _meaning_ instead of exact keyword matches. It also doubles as the retrieval layer for the RAG system we'll build in project #4.

## Architecture

```text
Documents → Chunking → Embeddings → Persistent Index → Query → Similarity → Top-k Results
```

_(Build an index once, then query it as many times as you want.)_

## Features

- Paragraph-based document chunking with source metadata
- Local embeddings via Sentence Transformers (no API calls needed)
- Cosine similarity ranking
- Persistent index (saves to disk so you don't re-embed every run)
- Top-k retrieval
- Clean split between indexing and querying

## Tech Stack

- Python
- Sentence Transformers
- NumPy

_(No vector database. We're doing it from scratch so the mechanics are obvious.)_

## Project Structure

```text
03-semantic-search/
├── main.py          # CLI entry point, decides whether to build or load the index
├── documents.py     # File discovery, reading, and paragraph chunking
├── embeddings.py    # Wrapper around the Sentence Transformer model
├── search.py        # Indexing, similarity calculation, and top-k retrieval
├── data/
│   ├── documents/   # Drop your .txt files here
│   └── index/       # Generated embeddings (gitignored)
├── README.md
└── requirements.txt
```

## Running the Project

From the repo root:

```bash
python projects/03-semantic-search/main.py
```

First run builds the index from whatever `.txt` files are in `data/documents/`. Subsequent runs just load the saved index. If you change the source docs, delete `data/index/` and run it again.

Once it's ready, try queries like:

- _"How do I create reusable code?"_
- _"How does a website communicate with a server?"_
- _"How does a computer learn from examples?"_

You'll get back the top matching chunks with similarity scores, source files, and the actual text.

## How it Works

At its core, semantic search is just math on text. Here's the quick version:

- **Embeddings:** A sentence transformer converts each chunk of text into a numerical vector (a list of floats). Texts with similar meanings end up with vectors that are close together in this space. So a query like _"How can I avoid repeating code?"_ will match a document about _"reusable blocks of code"_ even though the words are completely different.
- **Chunking:** Instead of treating each file as one giant document, we split them into paragraphs. Smaller chunks = more precise retrieval, because you get back the relevant passage instead of an entire file.
- **Similarity:** When you type a query, we embed it the same way, then compare it against all the stored chunk vectors using cosine similarity. The closest matches win.
- **Persistent index:** Embedding takes time, so we save the vectors and metadata to `.npy` files in `data/index/`. That way we only pay the compute cost once.

The default model is `all-MiniLM-L6-v2`—it's small, fast, and runs entirely locally.

### Why no vector database?

You might expect FAISS, Chroma, or Pinecone here. We're intentionally skipping those for now. The goal is to understand the raw mechanics—text → vectors → similarity → ranking—without hiding behind infrastructure. Once you get how it works under the hood, plugging in a real vector DB later is trivial.

## What's Next?

This project gives us the retrieval half of RAG. In project #4, we'll wire this up to the LLM from project #1 so the model can actually _answer questions_ grounded in real documents instead of just guessing from its training data.
