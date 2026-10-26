import os

import chromadb
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing. Check that your .env file is beside app.py.")
    st.stop()

client = genai.Client(api_key=api_key)


@st.cache_resource
def get_collection():
    db = chromadb.Client()
    return db.get_or_create_collection(name="askmydocs")


def split_text(text, chunk_size=1000, overlap=150):
    text = " ".join(text.split())
    step = chunk_size - overlap

    return [
        text[start : start + chunk_size]
        for start in range(0, len(text), step)
        if text[start : start + chunk_size].strip()
    ]


def make_embedding(text, task_type):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return result.embeddings[0].values


def answer_question(question):
    collection = get_collection()

    results = collection.query(
        query_embeddings=[make_embedding(question, "RETRIEVAL_QUERY")],
        n_results=4,
    )

    passages = results["documents"][0]
    sources = results["metadatas"][0]

    context = "\n\n".join(
        f"[Source: {source['filename']}, page {source['page']}]\n{passage}"
        for passage, source in zip(passages, sources)
    )

    prompt = f"""Answer the question using only the context below.
If the context does not contain the answer, say you could not find it in the uploaded document.
Cite sources using the filename and page number shown in the context.

Context:
{context}

Question: {question}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
    except Exception as error:
        st.error(f"Gemini couldn't answer right now. Please try again in a moment. Details: {error}")
        return "No answer generated.", sources, passages

    return response.text, sources, passages


st.title("AskMyDocs")
st.write("Upload a PDF, then ask questions about what it says.")

pdf = st.file_uploader("Choose a PDF", type="pdf")

if pdf and st.button("Index PDF"):
    with st.spinner("Reading and indexing the PDF..."):
        document = pymupdf.open(
            stream=pdf.getvalue(),
            filetype="pdf",
        )
        collection = get_collection()

        # Remove previously indexed chunks before indexing this PDF.
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        ids = []
        texts = []
        vectors = []
        metadata = []

        for page_number, page in enumerate(document, start=1):
            chunks = split_text(page.get_text())

            for chunk_number, chunk in enumerate(chunks):
                ids.append(f"{pdf.name}-p{page_number}-c{chunk_number}")
                texts.append(chunk)
                vectors.append(
                    make_embedding(chunk, "RETRIEVAL_DOCUMENT")
                )
                metadata.append(
                    {
                        "filename": pdf.name,
                        "page": page_number,
                    }
                )

        if not texts:
            st.error("I couldn't find readable text in that PDF.")
        else:
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=vectors,
                metadatas=metadata,
            )
            st.session_state["indexed"] = True
            st.success(f"Indexed {len(texts)} chunks from {pdf.name}.")


if st.session_state.get("indexed"):
    st.divider()
    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("Searching the PDF and writing an answer..."):
            answer, sources, passages = answer_question(question)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Retrieved passages"):
            for source, passage in zip(sources, passages):
                st.caption(
                    f"{source['filename']} — page {source['page']}"
                )
                st.write(passage)