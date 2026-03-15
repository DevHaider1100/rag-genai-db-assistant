# ===============================
# RAG Chatbot for GenAI Databases
# ===============================

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

import whisper
from transformers import pipeline

# ===============================
# Step 1 — Load PDF
# ===============================

def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


print("Loading PDF...")

pdf_docs = load_pdf("data/databases_genai.pdf")

print("PDF pages loaded:", len(pdf_docs))


# ===============================
# Step 2 — Transcribe Audio
# ===============================

def transcribe_audio(audio_path):

    print("Loading Whisper model...")

    model = whisper.load_model("base")

    print("Transcribing audio...")

    result = model.transcribe(audio_path)

    return result["text"]


audio_text = transcribe_audio("data/RAG_Intro.mp4")

print("Audio transcription length:", len(audio_text))


# ===============================
# Step 3 — Convert to Documents
# ===============================

audio_doc = Document(page_content=audio_text)

all_docs = pdf_docs + [audio_doc]

print("Total documents:", len(all_docs))


# ===============================
# Step 4 — Split into Chunks
# ===============================

print("Splitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(all_docs)

print("Total chunks created:", len(chunks))


# ===============================
# Step 5 — Create Embeddings
# ===============================

print("Creating embeddings...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ===============================
# Step 6 — Create Vector Database
# ===============================

print("Creating vector database...")

db = Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory="./db"
)

db.persist()

print("Vector database ready!")


# ===============================
# Step 7 — Create Retriever
# ===============================

retriever = db.as_retriever(search_kwargs={"k":3})


# ===============================
# Step 8 — Load LLM
# ===============================

print("Loading LLM...")

generator = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=generator)


# ===============================
# Step 9 — Create QA Chain
# ===============================

def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    response = generator(prompt)[0]["generated_text"]

    return response



print("\nRAG Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:

    question = input("Ask a question: ")

    if question.lower() == "exit":
        break

    answer = ask_question(question)

    print("\nAnswer:")
    print(answer)
    print("\n")