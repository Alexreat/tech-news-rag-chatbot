#  Tech News RAG Chatbot

![LangChain](https://img.shields.io/badge/LangChain-Enabled-blue?style=for-the-badge&logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-00ADD8?style=for-the-badge)

A Full-Stack **Retrieval-Augmented Generation (RAG)** application that autonomously ingests technology news, embeds the text into a vector store, and allows users to query the data via an interactive chat interface.

This project solves the "hallucination" problem in LLMs by grounding answers in retrieved, factual data.

---

##  System Architecture

The pipeline consists of two distinct workflows:

1.  **Ingestion Pipeline (`ingest.py`)**:
    * **Load:** Scrapes/Loads raw article data.
    * **Split:** Chunks text into manageable tokens using `RecursiveCharacterTextSplitter`.
    * **Embed:** Converts text chunks into dense vector embeddings (using OpenAI/HuggingFace embeddings).
    * **Store:** Indexes vectors in a **FAISS** (Facebook AI Similarity Search) local database.

2.  **Inference Engine (`chatbot.py`)**:
    * **Retrieve:** Semantic search finds the top-k most relevant chunks for a user query.
    * **Augment:** Injects retrieved context into the LLM prompt.
    * **Generate:** The LLM produces a grounded response.

---

##  Tech Stack

* **Orchestration:** LangChain
* **Vector Database:** FAISS
* **LLM Integration:** OpenAI GPT-3.5 / HuggingFace Hub
* **Frontend:** Streamlit
* **Language:** Python 3.9+

---

##  How to Run

### 1. Installation
Clone the repo and install dependencies:
```bash
git clone [https://github.com/Alexreat/tech-news-rag-chatbot.git](https://github.com/Alexreat/tech-news-rag-chatbot.git)
cd tech-news-rag-chatbot
pip install -r requirements.txt
