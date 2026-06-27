# 📄 DocuChat — AI-Powered PDF Chatbot

> Upload any PDF and have a real conversation with it using AI. Get accurate, sourced answers in seconds.

**🔗 Live Demo:** [your-app-url.streamlit.app](https://llm-portfolio-hvszmrzndcnuhemb4z3bzv.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Mistral](https://img.shields.io/badge/Mistral_AI-small--latest-orange)

---

## What It Does

Most AI chatbots answer from their training data — they can hallucinate, make things up, or give outdated information. **DocuChat answers only from your document.** Every response cites the exact chunk of text it used.

- 📤 Upload any PDF — research papers, contracts, course outlines, manuals
- 💬 Ask questions in plain English
- 📍 See exactly which part of the document was used for each answer
- 🚫 If the answer isn't in the document, it says so — no hallucination

---

## Screenshots

| Upload & Chat | Source Citations | Maintains history/memory
|---|---|
| ![Chat UI](screenshots/ss1.png) | ![Sources](screenshots/ss2.png) | ![memory](screenshots/ss3.png)


---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  PDF Upload │────▶│ Text Chunks  │────▶│   ChromaDB    │
│  (PyPDF)    │     │ (LangChain   │     │ (Vector Store)│
└─────────────┘     │  Splitter)   │     └───────┬───────┘
                    └──────────────┘             │
                                                 │ Similarity Search
┌─────────────┐     ┌──────────────┐     ┌───────▼───────┐
│   Answer    │◀────│  Mistral AI  │◀────│  Top 3 Chunks │
│ + Sources   │     │     LLM      │     │  + Question   │
└─────────────┘     └──────────────┘     └───────────────┘
```

**Embeddings:** HuggingFace `all-MiniLM-L6-v2` runs locally — no embedding API costs.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Framework | LangChain | RAG pipeline orchestration |
| LLM | Mistral AI `mistral-small-latest` | Fast, affordable, accurate |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free, runs locally |
| Vector DB | ChromaDB | Lightweight local vector store |
| Frontend | Streamlit | Clean UI with zero frontend code |
| Deployment | Streamlit Cloud | Free, one-click deploy |

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/SaraAbidHussain/llm-portfolio
cd llm-portfolio
```

**2. Create virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Add your API key**
```bash
# Create a .env file
echo "MISTRAL_API_KEY=your-key-here" > .env
```
Get a free Mistral API key at [console.mistral.ai](https://console.mistral.ai)

**4. Run**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
llm-portfolio/
├── app.py              # Main Streamlit application
├── rag_chatbot.py      # Terminal-based RAG pipeline
├── requirements.txt    # Python dependencies
├── .env                # API keys (never committed)
├── .gitignore
└── README.md
```

---

## How RAG Works

**RAG (Retrieval Augmented Generation)** is a technique that grounds LLM responses in real documents:

1. **Index** — PDF is split into ~1000 character chunks, each converted to a 384-dimensional embedding vector and stored in ChromaDB
2. **Retrieve** — when you ask a question, it's also embedded and ChromaDB finds the 3 most semantically similar chunks
3. **Generate** — those chunks + your question are sent to Mistral, which answers based only on that context

This eliminates hallucination and makes responses verifiable.

---

## Built By

**Sara Abid** — CS Sophomore at ITU Lahore, building LLM applications for freelance clients.

- 🐙 GitHub: [SaraAbidHussain](https://github.com/SaraAbidHussain)
- 💼 Available for freelance RAG chatbot and AI agent projects

---

*Built with LangChain · Mistral AI · ChromaDB · Streamlit*