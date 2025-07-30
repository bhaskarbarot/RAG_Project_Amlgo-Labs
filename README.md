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

# 🤖 RAG-Based PDF Chatbot – EBay User Agreement Assistant

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to query information from legal documents (like eBay User Agreement PDFs) using semantic search and response generation.

---

## ✅ Steps to Complete RAG-Based PDF Chatbot Project

---

### 1. 🔧 Install Required Libraries

Install the necessary packages in your Python environment:

```bash
pip install streamlit faiss-cpu sentence-transformers transformers nltk chromadb langchain pypdf2
```

---

### 2. 📂 Create Folder Structure

```plaintext
rag_chatbot_project/
│
├── data/                 # Raw PDFs and cleaned text
├── embeddings/           # FAISS index and metadata storage
├── src/                  # Core Python logic
│   ├── preprocessing.py       # Data cleaning and chunking
│   ├── embedder.py            # Embedding generation
│   ├── retriever.py           # FAISS-based retriever
│   ├── generator.py           # Response generation using DialoGPT
│   └── pipeline.py            # Final end-to-end RAG pipeline
├── app/                  # Streamlit app for user interface
│   └── app.py
├── pipeline.ipynb        # Jupyter Notebook for testing pipeline
├── requirements.txt      # List of all required Python packages
└── README.md             # Project overview and setup instructions
```

---

### 3. 🧹 Preprocessing Pipeline (`src/preprocessing.py`)

- Import necessary libraries.
- Extract text from PDF using `PyPDF2`.
- Clean extracted text using regular expressions:
  - Remove headers, footers, and unwanted symbols.
- Perform intelligent chunking (e.g., sentence-based or overlapping).
- Save cleaned and chunked data to JSON/CSV format in `data/`.

---

### 4. 🧠 Embedding and FAISS Setup (`src/embedder.py`)

- Load cleaned text chunks.
- Generate sentence embeddings using `all-MiniLM-L6-v2`.
- Create FAISS vector store with metadata.
- Save FAISS index and metadata to `embeddings/`.

---

### 5. 🔍 Build Retrieval & Generation Pipeline

#### A. `src/retriever.py`
- Create a Retriever class that queries FAISS DB with user input and fetches top-k relevant chunks.

#### B. `src/generator.py`
- Load the `microsoft/DialoGPT-medium` model.
- Generate conversational responses using the retrieved context.

#### C. `src/pipeline.py`
- Integrate retriever and generator classes.
- Build an end-to-end `generate_response(query)` function.

---

### 6. 🧪 Test the Pipeline (`pipeline.ipynb`)

- Run sample queries.
- Analyze:
  - Retrieved context chunks.
  - Generated responses.
- Save results for documentation and refinement.

---

### 7. 🌐 Create Streamlit Interface (`app/app.py`)

- Build the UI:
  - Input textbox for user questions.
  - Output box for answers.
  - (Optional) Show retrieved context chunks or sources.
- Load and use `pipeline.py` in the backend.

Run using:

```bash
streamlit run app/app.py
```

---

### 8. 📦 Final Touches

- Create `requirements.txt`:

```bash
pip freeze > requirements.txt
```

- Add `README.md` with:
  - Project overview
  - Tech stack
  - Folder structure
  - Usage instructions
  - Screenshots and demo video (if any)

- Add screenshots and video files to appropriate folders for better documentation.

---

### 9. 📤 Upload to GitHub

- Create a new GitHub repository.
- Push the entire folder structure: `src/`, `app/`, `data/`, `embeddings/`, `README.md`, `requirements.txt`.
- Include screenshots and demo video links in README.

---

## 📌 Project Highlights

- 🧠 Smart chunking and semantic embedding.
- 🔍 Contextual retrieval using FAISS.
- 💬 Natural conversational responses via DialoGPT.
- 🌐 Simple user interface built with Streamlit.

---

## 📽 Demo & Screenshots

Add your screenshots and screen recording (GIF/video) here for visual demo.

---

## 👨‍💻 Author

**Bhaskar Dipakkumar Barot**  
*AI Engineer | Data Scientist | Deep Learning Enthusiast*  
🔗 [LinkedIn](https://www.linkedin.com/in/bhaskar-barot-9b2455252/) • 🌐 [Portfolio](https://bhaskarbarot.my.canva.site/)
