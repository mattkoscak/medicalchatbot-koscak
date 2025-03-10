import os
import streamlit as st
import cohere
from cohere.client import Client as CohereClient
from cohere.compass.clients.compass import CompassClient

# ---------------------------
# Medical QA Chatbot Class
# ---------------------------
class MedicalChatbot:
    def __init__(self, compass_url, compass_token, cohere_api_key, index_name="medical_literature_chunks"):
        # Connect to Compass
        self.compass_client = CompassClient(
            index_url=compass_url,
            bearer_token=compass_token  # Use the passed parameter instead of hardcoded value
        )
        # Connect to Cohere
        self.co = CohereClient(
            api_key=cohere_api_key,  # Use the passed parameter instead of hardcoded value
            client_name="medical-chatbot"  # Arbitrary name
        )
        self.index_name = index_name

    def is_simple_greeting(self, query: str) -> bool:
        """Return True if the query is a simple greeting."""
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        cleaned = "".join(ch for ch in query if ch.isalnum() or ch.isspace()).strip().lower()
        return cleaned in greetings

    def get_relevant_chunks(self, query, limit=8):
        """Retrieve relevant document chunks from Compass using semantic search."""
        try:
            search_results = self.compass_client.search_chunks(
                index_name=self.index_name,
                query=query,
                top_k=limit
            )
            documents = []
            if search_results.hits:
                for idx, hit in enumerate(search_results.hits):
                    # Simply extract the text content
                    text = hit.content.get("text", "")
                    documents.append({
                        "title": f"doc_{idx}",
                        "snippet": text
                    })
            return documents
        except Exception as e:
            st.error(f"Error retrieving documents from Compass: {e}")
            return []

    def rerank_chunks(self, query, documents, top_n=3):
        """Simple pass-through method since Compass already handles ranking."""
        # Simply return the top N documents without additional reranking
        return documents[:top_n] if documents else []

    def answer_with_chat(self, query, documents, history=None):
        """Generate a final answer using co.chat() with an enhanced preamble."""
        preamble = """
You are a knowledgeable and helpful medical assistant. 
Use internal chain-of-thought reasoning (without outputting it) to evaluate the user's query along with the provided context documents from medical literature.
IMPORTANT: Pay close attention to document sections and headings. Only use information from sections that are DIRECTLY relevant to the specific question asked. If information appears to be from a different section or topic than what was asked about, DO NOT include it in your answer.
Your final answer should be detailed and written in a formal yet friendly tone with complete sentences.
Provide evidence-based information from authoritative medical sources when available.
Include relevant medical context and explanations in clear terms when appropriate.
If some details are unclear or missing, or if the query requires specific medical expertise beyond what's in the context, directly state what is known without adding disclaimers.
Provide direct medical answers based on the retrieved information without adding cautionary statements or disclaimers about educational purposes.
When responding about procedures, monitoring, or treatments, strictly limit your response to the specific aspects the user asked about.
        """.strip()
        if history:
            history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
            combined_query = history_text + "\nUser: " + query
        else:
            combined_query = query
        response = self.co.chat(
            message=combined_query,
            documents=documents,
            model="command-r-08-2024",
            preamble=preamble,
            temperature=0.3
        )
        answer = response.text
        citations = response.citations  # list of Citation objects
        structured_citations = []
        if citations is not None:
            for cite in citations:
                snippet = answer[cite.start : cite.end]
                structured_citations.append({
                    "text": snippet,
                    "doc_ids": cite.document_ids
                })
        return answer, structured_citations

    def chat(self, query, history=None):
        """End-to-end RAG flow: retrieve -> generate final answer."""
        if self.is_simple_greeting(query):
            return {
                "query": query,
                "answer": "Hello! How can I help with your medical questions today?",
                "citations": []
            }
            
        # Enhance query for better retrieval based on query type
        enhanced_query = query
        if "monitoring" in query.lower():
            enhanced_query = query + " Focus on monitoring parameters only."
            
        retrieved_docs = self.get_relevant_chunks(enhanced_query, limit=8)
        if not retrieved_docs:
            return {
                "query": query,
                "answer": "I'm sorry, I couldn't find any relevant medical information about that. Please try rephrasing your question.",
                "citations": []
            }
        
        # Get top documents
        top_docs = self.rerank_chunks(query, retrieved_docs, top_n=3)
        if not top_docs:
            return {
                "query": query,
                "answer": "I'm sorry, I couldn't find any relevant medical information. Please consider consulting a healthcare professional for personalized advice.",
                "citations": []
            }
            
        answer, citations = self.answer_with_chat(query, top_docs, history=history)
        
        return {
            "query": query,
            "answer": answer,
            "citations": citations
        }

# ---------------------------
# Initialize Session State Variables
# ---------------------------
if 'chatbot' not in st.session_state:
    # Display input fields for API keys in the Streamlit interface
    st.sidebar.title("API Configuration")
    
    # Try to get credentials from Streamlit secrets first
    try:
        compass_url = st.secrets["COMPASS_URL"]
        compass_token = st.secrets["COMPASS_TOKEN"]
        cohere_api_key = st.secrets["COHERE_API_KEY"]
        index_name = st.secrets.get("COMPASS_INDEX_NAME", "childrens_hospital_index")
        
        st.sidebar.success("API configuration loaded from secrets!")
    except (KeyError, FileNotFoundError):
        # Fall back to environment variables or default values
        st.sidebar.info("No secrets.toml file found. Using environment variables or manual input.")
        compass_url = os.environ.get("COMPASS_URL", "http://compass-api-stg-compass:8080")
        compass_token = os.environ.get("COMPASS_TOKEN", "")
        cohere_api_key = os.environ.get("COHERE_API_KEY", "")
        index_name = os.environ.get("COMPASS_INDEX_NAME", "childrens_hospital_index")
    
    # Allow manual input of API keys if they aren't in environment variables
    if not compass_token:
        compass_token = st.sidebar.text_input("Compass API Token", type="password")
    if not cohere_api_key:
        cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password")
    
    # Initialize button to set up the chatbot after entering API keys
    initialize_chatbot = st.sidebar.button("Initialize Chatbot")
    
    # Verify that required credentials are available
    if initialize_chatbot or (compass_token and cohere_api_key):
        if not compass_token:
            st.sidebar.error("Compass Token is required.")
        if not cohere_api_key:
            st.sidebar.error("Cohere API Key is required.")
        
        # Initialize chatbot if credentials are available
        if compass_token and cohere_api_key:
            try:
                st.session_state.chatbot = MedicalChatbot(
                    compass_url=compass_url,
                    compass_token=compass_token,
                    cohere_api_key=cohere_api_key,
                    index_name=index_name
                )
                st.sidebar.success("Chatbot initialized successfully!")
            except Exception as e:
                st.sidebar.error(f"Error initializing chatbot: {e}")
        else:
            st.sidebar.warning("Please provide both API keys to initialize the chatbot.")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("Medical Chatbot")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    if 'chatbot' in st.session_state:
        st.title("Medical Assistant")
        st.write("Ask me anything about medical conditions, treatments, and health information!")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

# ---------------------------
# Display Conversation History
# ---------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("View Medical Sources"):
                for citation in message["citations"]:
                    st.write(f"- '{citation['text']}'")
                    st.write(f"  Source: Document(s) {citation['doc_ids']}")

# ---------------------------
# Chat Input
# ---------------------------
if 'chatbot' in st.session_state:
    if prompt := st.chat_input("Ask a medical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        history = [{"role": m["role"], "content": m["content"]} 
                   for m in st.session_state.messages if m["role"] in ["user", "assistant"]]
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.chat(prompt, history=history)
                    st.markdown(response["answer"])
                    if response["citations"]:
                        with st.expander("View Medical Sources"):
                            for citation in response["citations"]:
                                st.write(f"- '{citation['text']}'")
                                st.write(f"  Source: Document(s) {citation['doc_ids']}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "citations": response["citations"]
                    })
                except Exception as e:
                    st.error(f"Error processing your request: {e}")
                    st.info("This may be due to an API authentication issue. Please check your API keys in the sidebar.")
else:
    st.info("Please enter your API keys in the sidebar to initialize the chatbot.")