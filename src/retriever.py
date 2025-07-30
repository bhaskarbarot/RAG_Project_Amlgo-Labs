"""Document retrieval using FAISS and sentence transformers"""
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class DocumentRetriever:
    def __init__(self, index_path: str, metadata_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize retriever with FAISS index and metadata"""
        self.embedding_model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.chunks = [chunk['text'] for chunk in self.metadata['chunks']]
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve most relevant chunks for the query"""
        # Encode query with normalization
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return chunks with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def get_chunk_metadata(self, chunk_text: str) -> dict:
        """Get metadata for a specific chunk"""
        for chunk in self.metadata['chunks']:
            if chunk['text'] == chunk_text:
                return chunk
        return {}
