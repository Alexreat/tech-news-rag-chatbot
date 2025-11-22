import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
os.environ["OPENAI_API_KEY"] = "OpenAI_API_Key"
VECTOR_STORE_PATH = "faiss_index_openai"

# --- 1. SETUP THE AI ---
@st.cache_resource # This keeps the AI in memory 
def load_chain():
    """Loads the Brain and sets up the Q&A logic."""
    
    # Load the "Brain" 
    print("Loading vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    
    # Create the "Retriever" 
    retriever = vector_store.as_retriever()
    
    # Create the LLM 
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # The "Instruction Manual" for the AI
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based ONLY on the following context:
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)
    
    # Connect: Retriever -> Prompt -> LLM
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# Initialize the chain
chain = load_chain()

# --- 2. BUILD THE WEBSITE UI ---
st.title(" My Tech News Reporter")
st.caption("I read the news so you don't have to.")

# Initialize chat history in the browser
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. HANDLE USER INPUT ---
if prompt := st.chat_input("Ask me about the latest tech news..."):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get answer from AI
    with st.chat_message("assistant"):
        with st.spinner("Reading the news database..."):
            response = chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)
            
    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})