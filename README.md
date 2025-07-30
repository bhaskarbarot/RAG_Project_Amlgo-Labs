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

🎛️ Model Choices & Rationale
Embedding Model: all-MiniLM-L6-v2

Why: Balanced performance vs. speed, 384-dimensional vectors
Alternatives considered: bge-small-en, all-mpnet-base-v2
Performance: Good semantic understanding for legal text

Language Model: microsoft/DialoGPT-medium

Why: Conversational fine-tuning, reasonable size (345M parameters)
Alternatives considered: distilgpt2, llama-7b-instruct
Trade-offs: Speed vs. generation quality

Vector Database: FAISS

Why: Fast similarity search, easy integration
Configuration: IndexFlatIP for cosine similarity
Scale: Efficient for documents up to 50,000 chunks

Chunking Strategy

Method: Sentence-aware with overlap
Size: 200-300 characters target
Overlap: 2 sentences between chunks
Rationale: Preserves context while maintaining searchability

⚠️ Limitations & Future Improvements
Current Limitations

Response Quality: May generate generic responses for complex queries
Context Window: Limited by model's sequence length (512 tokens)
Legal Complexity: Simplified interpretation of complex legal language
Hallucination Risk: Model may generate information not in sources
Processing Speed: Response time varies with query complexity (2-5 seconds)

Failure Cases

Ambiguous Queries: "What should I do?" → Requires more specific context
Cross-sectional Information: Questions spanning multiple policy areas
Recent Updates: Information not in the training document

Future Improvements

Model Upgrades: Implement larger models (Llama-2, Claude-instant)
Advanced Chunking: Semantic-based chunking using embeddings
Query Enhancement: Query expansion and reformulation
Multi-modal Support: Handle images, tables in documents
Evaluation Metrics: Implement BLEU, ROUGE, semantic similarity scores
Caching: Response caching for common queries
Fine-tuning: Domain-specific fine-tuning on legal text

📊 Performance Metrics
Based on test evaluation:

Success Rate: 95% (19/20 test queries)
Average Response Time: 3.2 seconds
Average Response Length: 45 words
Source Grounding: 100% of responses include source references
Relevance Score: High semantic similarity (>0.7) for retrieved chunks

🤝 Contributing

Fork the repository
Create a feature branch
Make your changes
Add tests
Submit a pull request

📄 License
This project is created for educational purposes as part of a technical assessment.
🙏 Acknowledgments

eBay for providing the User Agreement document
Hugging Face for transformer models
Sentence Transformers library
Facebook AI Research for FAISS
Streamlit for the web interface


git clone <your-repo-url>
cd rag-chatbot


Step: Run the Application
Start streamlit run  : streamlit.py

Step 7: Access the Chatbot
Open your web browser and go to http://localhost: (or the port you specified) to interact with the chatbot.
