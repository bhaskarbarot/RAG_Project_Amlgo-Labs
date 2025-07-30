import streamlit as st
import sys
import os
import time
import json

# Ensure src/ is in Python path
sys.path.append(os.path.abspath("src"))

from src.pipeline import RAGPipeline  

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chatbot of AMLGO LABS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stChat > div {
        padding-top: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metrics-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the pipeline
@st.cache_resource
def load_pipeline():
    try:
        return RAGPipeline(
            index_path=r"/home/petpooja/Documents/amlgo/vectordb/document_index.faiss",
            metadata_path=r"/home/petpooja/Documents/amlgo/vectordb/metadata.json"
        )
    except Exception as e:
        st.error(f"Failed to load pipeline: {str(e)}")
        return None

def display_sources(sources, title="📚 Source References"):
    if sources:
        with st.expander(f"{title} ({len(sources)} sources)"):
            for i, source in enumerate(sources, 1):
                st.markdown(f"""
                <div class="source-box">
                    <strong>Source {i}:</strong><br>
                    {source[:300]}{'...' if len(source) > 300 else ''}
                </div>
                """, unsafe_allow_html=True)

def main():
    st.title("🤖 RAG Chatbot of AMLGO LABS")
    st.markdown("Ask questions about eBay's User Agreement and get accurate answers with source references.")

    with st.sidebar:
        st.header("🔧 System Information")
        pipeline = load_pipeline()
        if pipeline is None:
            st.stop()

        try:
            # Fixed: Use the correct metadata path
            metadata_path = r"/home/petpooja/Documents/amlgo/vectordb/metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            st.markdown(f"""
            <div class="metrics-container">
                <h4>📊 System Metrics</h4>
                <ul>
                    <li><strong>Model</strong>: microsoft/DialoGPT-medium</li>
                    <li><strong>Embeddings</strong>: all-MiniLM-L6-v2</li>
                    <li><strong>Vector DB</strong>: FAISS</li>
                    <li><strong>Total Chunks</strong>: {metadata.get('total_vectors', 'N/A')}</li>
                    <li><strong>Vector Dimension</strong>: {metadata.get('vector_dimension', 'N/A')}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning("⚠️ Metadata file not found. Please check the path.")
        except json.JSONDecodeError:
            st.warning("⚠️ Invalid JSON in metadata file.")
        except Exception as e:
            st.warning(f"⚠️ Could not load system metrics: {str(e)}")

        st.subheader("💡 Sample Queries")
        sample_queries = [
            "What are eBay's return policies?",
            "How much fees does eBay charge?",
            "Can I cancel my order?",
            "What happens if I violate the agreement?",
            "How does dispute resolution work?"
        ]
        for query in sample_queries:
            if st.button(f"📜 {query}", key=f"sample_{hash(query)}"):
                st.session_state.sample_query = query

        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.subheader("⚙️ Settings")
        top_k = st.slider("Number of sources to retrieve", 1, 5, 3)
        show_scores = st.checkbox("Show relevance scores", False)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                display_sources(message["sources"])

    if hasattr(st.session_state, 'sample_query'):
        prompt = st.session_state.sample_query
        del st.session_state.sample_query
    else:
        prompt = st.chat_input("Ask me anything about eBay's User Agreement...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("🤔 Thinking..."):
                    start_time = time.time()
                    response_stream, sources = pipeline.query_streaming(prompt, top_k=top_k)
                response_placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                    time.sleep(0.03)
                response_placeholder.markdown(full_response)
                st.caption(f"⏱️ Response generated in {time.time() - start_time:.2f} seconds")
                display_sources(sources, f"📚 Retrieved Sources (Top {len(sources)})")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.exception(e)
                # Additional debugging information
                st.write("**Debug Info:**")
                st.write(f"Pipeline loaded: {pipeline is not None}")
                st.write(f"Query: {prompt}")

if __name__ == "__main__":
    main()