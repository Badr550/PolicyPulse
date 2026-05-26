# PolicyPulse — GDPR RAG Assistant

PolicyPulse is a Retrieval-Augmented Generation (RAG) system that allows users to upload GDPR PDF documents and ask natural language questions to receive accurate, context-based answers.

The system is designed to ensure grounded and explainable responses by combining document retrieval with large language models.

---

## Features

- Upload and process GDPR PDF documents
- Token-based chunking for efficient context handling
- Semantic search using embeddings and ChromaDB
- MMR-based retrieval for relevant and diverse results
- RAG pipeline using Groq LLM (LLaMA 3.3 70B)
- Fact-checking layer to validate generated answers
- Response verification system:
  - VERIFIED → supported by retrieved context
  - WARNING → potential hallucination detected
  - NOT FOUND → insufficient relevant context
- Display of retrieved chunks with similarity scores
- Streamlit-based interactive interface

---

## Workflow

PDF Upload → Text Extraction → Chunking → Embedding → Vector Store → Retrieval → LLM Generation → Fact Checking → Final Answer

---

## Tech Stack

- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- ChromaDB
- Groq API (LLaMA 3.3 70B)

---

## How It Works

1. The user uploads a GDPR PDF document.
2. The document is split into smaller chunks.
3. Each chunk is converted into embeddings and stored in a vector database (ChromaDB).
4. User queries are processed using semantic search to retrieve relevant chunks.
5. A language model generates answers strictly based on retrieved context.
6. A fact-checking layer verifies whether the answer is supported by the source.
7. The system returns the final response along with verification status and evidence.

---

## Installation

```bash
git clone https://github.com/Badr550/PolicyPulse
cd PolicyPulse
pip install -r requirements.txt
streamlit run app/app.py
