"""Complete RAG pipeline combining retrieval and generation"""
from src.retriever import DocumentRetriever
from .generator import ResponseGenerator
from typing import List, Tuple, Iterator

class RAGPipeline:
    def __init__(self, index_path: str, metadata_path: str, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize complete RAG pipeline"""
        print("Initializing RAG pipeline...")
        self.retriever = DocumentRetriever(index_path, metadata_path)
        self.generator = ResponseGenerator(model_name)
        print("RAG pipeline initialized!")
    
    def create_prompt(self, query: str, retrieved_chunks: List[str]) -> str:
        """Create structured prompt for RAG"""
        context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(retrieved_chunks)])
        
        prompt = f"""Based on the eBay User Agreement information provided below, answer the user's question accurately and concisely.

Context Information:
{context}

User Question: {query}

Instructions:
- Answer based only on the provided context
- Be specific and accurate
- If the context doesn't contain enough information, say so
- Keep the response concise but complete

Answer:"""
        
        return prompt
    
    def query(self, user_query: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """Process user query and return response with sources"""
        # Retrieve relevant chunks
        retrieved_results = self.retriever.retrieve(user_query, top_k)
        retrieved_chunks = [chunk for chunk, score in retrieved_results]
        
        # Create prompt
        prompt = self.create_prompt(user_query, retrieved_chunks)
        
        # Generate response
        response = self.generator.generate(prompt, max_new_tokens=200)
        
        return response, retrieved_chunks
    
    def query_streaming(self, user_query: str, top_k: int = 3) -> Tuple[Iterator[str], List[str]]:
        """Process user query and return streaming response with sources"""
        # Retrieve relevant chunks
        retrieved_results = self.retriever.retrieve(user_query, top_k)
        retrieved_chunks = [chunk for chunk, score in retrieved_results]
        
        # Create prompt
        prompt = self.create_prompt(user_query, retrieved_chunks)
        
        # Generate streaming response
        response_stream = self.generator.generate_streaming(prompt, max_new_tokens=200)
        
        return response_stream, retrieved_chunks
