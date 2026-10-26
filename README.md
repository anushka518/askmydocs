# AskMyDocs

A small Retrieval-Augmented Generation (RAG) app for asking questions about a PDF. It uses Gemini to create embeddings and answers, and ChromaDB to retrieve relevant passages. Answers include source filename and page number.

## Features

- Upload a text-based PDF
- Split document text into overlapping chunks
- Create Gemini embeddings and store them in ChromaDB
- Retrieve relevant passages for a question
- Generate answers grounded in retrieved context
- Show retrieved passages with page numbers

## Requirements

- Python 3.11 or newer
- A Gemini API key from Google AI Studio

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt