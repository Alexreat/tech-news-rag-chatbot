import os
import pandas as pd
from sqlalchemy import create_engine
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- CONFIGURATION ---
# 1. DATABASE CONNECTION
DB_USER = "postgres"
DB_PASSWORD = "1402"  
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "article_db"

# 2. OPENAI API KEY
os.environ["OPENAI_API_KEY"] = "OpenAI_API_Key" 

def load_data():
    """Reads articles from the local Postgres database."""
    print("Connecting to Database...")
    # Connect to the running Docker database
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    
    # Read the data using SQL
    query = "SELECT title, text_content, url FROM articles"
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df)} articles from database.")
    return df

def create_vector_db(df):
    """Converts text to vectors and saves them."""
    if df.empty:
        print("No data found in database!")
        return

    print("Converting articles to AI-ready documents...")
    documents = []
    for index, row in df.iterrows():
        combined_text = f"Title: {row['title']}\n\nContent: {row['text_content']}"
        
        # Create a LangChain 'Document' object
        doc = Document(
            page_content=combined_text,
            metadata={"source": row['url'], "title": row['title']}
        )
        documents.append(doc)

    print(f"Creating embeddings for {len(documents)} documents...")
    
    # "Translator" turns text into numbers
    embeddings = OpenAIEmbeddings()
    
    # (FAISS)  stores the numbers
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save the brain to disk
    print("Saving vector store to 'faiss_index_openai'...")
    vector_store.save_local("faiss_index_openai")
    print("Success! The Brain is ready.")

if __name__ == "__main__":
    df = load_data()
    create_vector_db(df)