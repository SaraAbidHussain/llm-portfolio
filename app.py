"""
RAG Chatbot — Streamlit Frontend
Stack: HuggingFace Embeddings (free) + Mistral + ChromaDB + LangChain
"""

import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .user-message {
        background: #2563eb; color: white;
        padding: 12px 16px; border-radius: 18px 18px 4px 18px;
        margin: 8px 0; margin-left: 20%;
        text-align: right; font-size: 15px; line-height: 1.5;
    }
    .bot-message {
        background: #f1f5f9; color: #1e293b;
        padding: 12px 16px; border-radius: 18px 18px 18px 4px;
        margin: 8px 0; margin-right: 20%;
        font-size: 15px; line-height: 1.5;
        border-left: 3px solid #2563eb;
    }
    .source-chunk {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 10px 14px; margin: 6px 0;
        font-size: 13px; color: #475569; font-family: monospace;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .status-ready { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 500; }
    .status-waiting { background: #fef9c3; color: #854d0e; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

CHROMA_PATH = "./chroma_db_streamlit"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

SYSTEM_PROMPT = """You are a professional document assistant.
Answer ONLY from the context provided.
If the answer is not in the document, say: "I don't find that information in the uploaded document."
Be concise and direct."""

# ─── SESSION STATE ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "embeddings" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = bool(os.getenv("MISTRAL_API_KEY"))

# ─── FUNCTIONS ────────────────────────────────────────────

def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nDocument context:\n{context}"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
            for doc in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def process_pdf(uploaded_file) -> bool:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.unlink(tmp_path)

        if not docs:
            st.error("Could not extract text from this PDF.")
            return False

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True
        )
        chunks = splitter.split_documents(docs)

        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=st.session_state.embeddings,
            persist_directory=CHROMA_PATH
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.chain = build_chain(vectorstore)
        return True

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False


def get_sources(query: str):
    if not st.session_state.vectorstore:
        return []
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return retriever.invoke(query)


# ─── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.title("📄 DocuChat")
    st.caption("Chat with your documents")
    st.divider()

    if st.session_state.api_key_valid:
        st.success("✓ API key loaded")
    else:
        st.error("⚠️ MISTRAL_API_KEY missing in .env")
        st.code("MISTRAL_API_KEY=your-key-here")

    st.divider()
    st.subheader("Upload Document")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            success = process_pdf(uploaded_file)
        if success:
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []
            count = st.session_state.vectorstore._collection.count()
            st.success(f"✓ Ready! ({count} chunks indexed)")

    st.divider()
    if st.session_state.pdf_name:
        st.markdown(f'<span class="status-ready">● {st.session_state.pdf_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-waiting">● No document uploaded</span>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with LangChain + Mistral + ChromaDB")


# ─── MAIN CHAT AREA ───────────────────────────────────────
st.title("💬 Document Q&A")

if not st.session_state.api_key_valid:
    st.error("Add your MISTRAL_API_KEY to the .env file and restart the app.")
    st.stop()

if not st.session_state.messages:
    if st.session_state.pdf_name:
        st.info(f"Document loaded: **{st.session_state.pdf_name}** — ask me anything!")
    else:
        st.info("👈 Upload a PDF in the sidebar to get started.")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks used)"):
                for i, source in enumerate(msg["sources"]):
                    st.markdown(
                        f'<div class="source-chunk"><strong>Chunk {i+1} · Page {source.get("page","?")}</strong>'
                        f'<br><br>{source.get("text","")}</div>',
                        unsafe_allow_html=True
                    )

# ─── CHAT INPUT ───────────────────────────────────────────
user_input = st.chat_input(
    "Ask a question about your document...",
    disabled=(not st.session_state.pdf_name or not st.session_state.api_key_valid)
)

if user_input:
    if not st.session_state.chain:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.chain.invoke(user_input)
                sources = get_sources(user_input)

                source_data = [
                    {
                        "page": doc.metadata.get("page", "?"),
                        "text": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
                    }
                    for doc in sources
                ]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_data
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}",
                    "sources": []
                })

        st.rerun()