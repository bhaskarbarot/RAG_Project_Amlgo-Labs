# RAG_Project_Amlgo-Labs
# eBay User Agreement RAG Chatbot

## 🎯 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions about eBay's User Agreement using advanced NLP techniques. The system combines document retrieval with language generation to provide accurate, source-backed responses.

## 🏗️ Architecture

### Components
- **Document Processing**: Intelligent PDF text extraction and sentence-aware chunking
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) for semantic search
- **Vector Database**: FAISS for efficient similarity search
- **Language Model**: microsoft/DialoGPT-medium for response generation
- **Interface**: Streamlit with real-time streaming responses

### Pipeline Flow
1. **Document Ingestion** → PDF extraction and cleaning
2. **Chunking** → Sentence-aware segmentation (200-300 words)
3. **Embedding** → Vector representation of text chunks
4. **Indexing** → FAISS vector database creation
5. **Retrieval** → Semantic search for relevant chunks
6. **Generation** → LLM-based response with source grounding
7. **Streaming** → Real-time response delivery

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- 8GB+ RAM recommended
- GPU optional (CUDA support)

### Installation Steps

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag-chatbot
