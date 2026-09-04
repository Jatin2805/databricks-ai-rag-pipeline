# Databricks RAG Pipeline

End-to-end Retrieval-Augmented Generation (RAG) pipeline using Databricks, PySpark, Delta Lake, AI Vector Search, embeddings, and Llama 3.3.

## Architecture

TXT Documents → Chunking → Embeddings → Delta Table → AI Vector Search → Retrieved Context → LLM → Answer

## Technologies

- Databricks
- PySpark
- Delta Lake
- Databricks AI Search / Vector Search
- `databricks-gte-large-en`
- `databricks-meta-llama-3-3-70b-instruct`

## Pipeline

1. Load TXT documents.
2. Split documents into 500-character chunks with 50-character overlap.
3. Generate embeddings.
4. Store chunks and embeddings in `workspace.default.embeddings`.
5. Enable Delta Change Data Feed.
6. Use the Delta Sync AI Search index `workspace.default.rag_embeddings_index`.
7. Retrieve relevant chunks using hybrid search.
8. Build context from retrieved chunks.
9. Send question + context to the LLM.
10. Generate the final grounded answer.

## Databricks Configuration

- Endpoint: `rag-search-endpoint`
- Index: `workspace.default.rag_embeddings_index`
- Source table: `workspace.default.embeddings`
- Index type: Delta Sync
- Search mode: Hybrid
- Primary key: `id`
- Text column: `content`
- Embedding column: `embedding`
- Dimension: 1024
- Update mode: Triggered

The knowledge base is synthetic and created for learning/portfolio purposes.

Do not commit credentials, API keys, tokens, passwords, or private company documents.
