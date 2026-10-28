import requests
import streamlit as st
from src.pdf_processor import extract_pages, chunk_pages
from src.retriever import build_embeddings, retrieve

st.set_page_config(page_title="DocuMind AI", page_icon="📚", layout="wide")
st.title("📚 DocuMind AI")
st.caption("Ask questions about your PDFs — answers are generated locally from retrieved document passages.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model", value="llama3.2:3b")
    top_k = st.slider("Retrieved passages", min_value=2, max_value=8, value=4)
    st.markdown("**Local requirements**")
    st.markdown("- Ollama must be running\n- Download a model with `ollama pull llama3.2:3b`\n- First embedding-model load needs internet")
    st.caption("PDFs are processed in this app session. Scanned/image-only PDFs need OCR, which this version does not include.")

uploaded_files = st.file_uploader(
    "Upload one or more text-based PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if uploaded_files:
    all_pages = []
    failures = []
    for uploaded in uploaded_files:
        try:
            pages = extract_pages(uploaded.getvalue(), uploaded.name)
            all_pages.extend(pages)
        except Exception as exc:
            failures.append(f"{uploaded.name}: {exc}")

    for failure in failures:
        st.error(failure)

    chunks = chunk_pages(all_pages)
    if not chunks:
        st.warning("No extractable text found. Try a text-based PDF; scanned PDFs are not supported in this version.")
        st.stop()

    st.success(f"Loaded {len(uploaded_files)} PDF(s), {len(all_pages)} page(s), and {len(chunks)} text chunks.")
    with st.expander("Preview extracted text"):
        for chunk in chunks[:5]:
            st.markdown(f"**{chunk['source']} — page {chunk['page']}**")
            st.write(chunk["text"])

    try:
        embedder = load_embedding_model()
        with st.spinner("Creating document embeddings..."):
            doc_vectors = build_embeddings(embedder, [chunk["text"] for chunk in chunks])
    except Exception as exc:
        st.error(f"Could not load the embedding model: {exc}")
        st.info("Check your internet connection for the first download, then restart the app.")
        st.stop()

    question = st.text_input("Ask a question about your uploaded documents", placeholder="e.g., What are the main causes of overfitting?")
    ask = st.button("Generate answer", type="primary", disabled=not question.strip())

    if ask:
        with st.spinner("Searching the documents..."):
            results = retrieve(embedder, doc_vectors, chunks, question, top_k=top_k)
        if not results:
            st.warning("I couldn't find relevant passages in the uploaded documents.")
            st.stop()

        context_parts = []
        for i, result in enumerate(results, start=1):
            context_parts.append(
                f"[Source {i}: {result['source']}, Page {result['page']}]\n{result['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are DocuMind AI, a careful assistant that answers questions about uploaded documents.

Rules:
- Answer using ONLY the supplied context. Do not use outside knowledge.
- If the context does not contain enough evidence, clearly say that the uploaded documents do not provide enough information.
- Cite claims with the exact source and page labels provided, e.g. [Source: handbook.pdf, Page 4].
- Never invent a page number or source.
- Be clear and concise. If useful, answer in bullet points.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model_name.strip(),
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=300,
            )
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "").strip()
            if not answer:
                st.error("The model returned an empty answer.")
            else:
                st.subheader("Answer")
                st.markdown(answer)
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to Ollama. Open Ollama, then run `ollama pull llama3.2:3b` in a terminal.")
        except requests.exceptions.Timeout:
            st.error("The local model took too long to respond. Try a shorter question or a smaller model.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"Ollama returned an error: {exc}. Confirm the model exists with `ollama list`.")
        except Exception as exc:
            st.error(f"Unexpected error while generating the answer: {exc}")

        with st.expander("Retrieved evidence (inspect what the LLM saw)"):
            for i, result in enumerate(results, start=1):
                st.markdown(f"**{i}. {result['source']} — Page {result['page']}**")
                st.caption(f"Cosine similarity: {result['score']:.3f}")
                st.write(result["text"])
else:
    st.info("Upload a PDF to begin. Your documents are processed locally by this app; the LLM runs through Ollama on your computer.")
    st.markdown("### How it works\n1. Extract PDF text and page numbers.\n2. Split text into overlapping chunks.\n3. Embed chunks and retrieve semantically relevant passages.\n4. Ask a local LLM to answer using those passages.\n5. Display citations and the evidence used.")
