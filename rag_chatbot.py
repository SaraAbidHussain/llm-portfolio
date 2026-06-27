"""
RAG Chatbot — Complete Implementation
Stack: HuggingFace Embeddings (free) + Mistral + ChromaDB
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 3

# Change this to your PDF path, or leave as None to use sample text
PDF_PATH = None  # e.g. "my_document.pdf"

# ─── STEP 1: Load documents ───────────────────────────────
def load_documents():
    if PDF_PATH and os.path.exists(PDF_PATH):
        print(f"Loading PDF: {PDF_PATH}")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages")
    else:
        print("No PDF provided — using built-in sample text")
        sample = """
        Retrieval Augmented Generation (RAG) is a technique that enhances LLM responses
        by retrieving relevant information from a knowledge base before generating answers.

        RAG works in two phases. First, an indexing phase where documents are split into
        chunks, converted to embeddings, and stored in a vector database like ChromaDB.
        Second, a retrieval phase where a user query is embedded and used to find the most
        similar chunks, which are then passed as context to the language model.

        Vector databases store embeddings — numerical representations of text that capture
        semantic meaning. Similar texts have similar embeddings, enabling semantic search
        that goes beyond simple keyword matching.

        LangChain is an open-source framework for building LLM-powered applications.
        It provides abstractions for document loaders, text splitters, embeddings,
        vector stores, and chains that make building RAG systems much faster.

        ChromaDB is a lightweight, open-source vector database that runs locally.
        It persists data to disk, supports metadata filtering, and integrates
        natively with LangChain through the langchain-chroma package.

        HuggingFace provides free, open-source embedding models like all-MiniLM-L6-v2
        that run locally without any API costs. This makes it ideal for development
        and for freelance projects where API costs need to be minimized.
        """
        # Write to temp file so TextLoader can read it
        with open("temp_sample.txt", "w") as f:
            f.write(sample)
        loader = TextLoader("temp_sample.txt")
        docs = loader.load()

    return docs

# ─── STEP 2: Split into chunks ────────────────────────────
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# ─── STEP 3: Build or load vector store ───────────────────
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # If ChromaDB already exists on disk — load it, don't rebuild
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("Loading existing ChromaDB from disk...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        print(f"Loaded {vectorstore._collection.count()} chunks from disk")
    else:
        # First run — build from scratch
        print("Building ChromaDB for the first time...")
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        print(f"Built and saved ChromaDB with {len(chunks)} chunks")

    return vectorstore

# ─── STEP 4: Build RAG chain ──────────────────────────────
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't have that information in my knowledge base."
Be concise and direct.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# ─── STEP 5: Terminal chat loop ───────────────────────────
def chat_loop(chain, retriever):
    print("\n" + "="*50)
    print("   RAG Chatbot Ready")
    print("   Type 'quit' to exit")
    print("   Type 'sources' after a question to see chunks")
    print("="*50 + "\n")

    last_query = None

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # Show source chunks for last query
        if user_input.lower() == "sources" and last_query:
            docs = retriever.invoke(last_query)
            print("\n--- Source chunks used ---")
            for i, doc in enumerate(docs):
                print(f"\n[{i+1}] {doc.metadata}")
                print(f"    {doc.page_content[:150]}...")
            print()
            continue

        last_query = user_input

        try:
            answer = chain.invoke(user_input)
            print(f"\nBot: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")

# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    vectorstore = get_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)
    chat_loop(chain, retriever)