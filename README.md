# LLM Portfolio

RAG chatbots and AI agents built with LangChain, Mistral, and ChromaDB.

## Projects

### 📄 DocuChat — RAG Chatbot
Upload any PDF and chat with it using AI.

**Stack:** LangChain · Mistral AI · ChromaDB · HuggingFace Embeddings · Streamlit

**Features:**
- PDF upload and processing
- Semantic search over document chunks  
- Source citations for every answer
- Clean chat UI

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your `MISTRAL_API_KEY` to a `.env` file before running.
