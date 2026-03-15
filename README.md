# RAG Chatbot for GenAI Databases

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that answers questions about the **Databases for GenAI** lecture materials from the AI Academy.

The system processes multiple data sources (PDF slides and lecture recordings), converts them into searchable embeddings, and retrieves relevant context to generate answers using a language model.

The goal of this project is to demonstrate an **end-to-end RAG pipeline** that integrates data ingestion, embeddings, vector databases, and LLM-based answer generation.

---

# Architecture

The system follows the standard **RAG pipeline**:

PDF Slides + Lecture Audio
↓
Text Extraction
↓
Audio Transcription (Whisper)
↓
Text Chunking
↓
Embeddings Generation
↓
Vector Database (ChromaDB)
↓
Semantic Retrieval
↓
LLM Answer Generation

---

# Features

* Extracts text from **PDF lecture slides**
* Transcribes lecture recordings using **OpenAI Whisper**
* Splits text into semantic chunks
* Generates embeddings using **SentenceTransformers**
* Stores embeddings in a **Chroma vector database**
* Retrieves the most relevant knowledge chunks
* Uses a **HuggingFace LLM** to generate final answers
* Interactive chatbot interface

---

# Technologies Used

* Python
* LangChain
* HuggingFace Transformers
* SentenceTransformers
* ChromaDB
* Whisper (speech-to-text)
* FFmpeg

---

# Project Structure

```
rag-genai-db-assistant
│
├── data/
│   ├── databases_genai.pdf
│   └── RAG_Intro.mp4
│
├── db/
│   └── vector database files
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```
git clone <your-repository-link>
cd rag-genai-db-assistant
```

Install dependencies:

```
pip install -r requirements.txt
```

Make sure **FFmpeg** is installed and available in your system PATH (required for Whisper).

---

# Running the Chatbot

Run the application:

```
python main.py
```

After the system initializes, you will see:

```
RAG Chatbot Ready!
Type 'exit' to quit.
```

You can now ask questions related to the lecture materials.

Example:

```
Ask a question: What are the production Do's for RAG?
```

---

# Example Questions

The chatbot was tested using the following questions:

1. What are the production **Do's** for RAG?
2. What is the difference between **standard retrieval and the ColPali approach**?
3. Why is **hybrid search better than vector-only search**?

The generated answers are included in **logs.txt**.

---

# Reflection

The most challenging part of this project was integrating different components into a working RAG pipeline. Extracting text from the PDF was straightforward using LangChain loaders, but transcribing the lecture audio required configuring Whisper and ensuring FFmpeg was correctly installed. Another challenge was dealing with evolving library versions, particularly LangChain, where several modules have been reorganized or deprecated.

Through this project, I learned how modern GenAI systems combine embeddings, vector databases, and large language models to retrieve and generate grounded responses from external knowledge sources. Implementing the full pipeline helped me better understand how production-grade RAG systems handle multi-format data and improve answer accuracy by retrieving relevant context before generation.

---

# Future Improvements

* Implement **hybrid search (vector + keyword retrieval)**
* Add **streaming responses**
* Improve chunking strategies
* Deploy as a **web application**
* Replace local LLM with a more powerful hosted model

---

# Author

AI Academy – Module 4 Homework
RAG Chatbot Implementation
