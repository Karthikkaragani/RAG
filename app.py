import streamlit as st

from ingestion import extract_document
from cleaning import clean_documents
from structure import detect_structure
from chunking import create_semantic_chunks
from embeddings import EmbeddingModel, create_embeddings


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG From Scratch",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 RAG From Scratch")

st.header("Phase 1 — Document Ingestion")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return EmbeddingModel(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a document",
    type=[
        "pdf",
        "txt",
        "docx",
        "md"
    ]
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # ========================================================
    # STEP 1 — EXTRACTION
    # ========================================================

    st.subheader(
        "1. Extraction"
    )

    extracted_documents = extract_document(
        uploaded_file
    )

    st.write(
        f"Extracted units: "
        f"{len(extracted_documents)}"
    )

    # ========================================================
    # STEP 2 — CLEANING
    # ========================================================

    st.subheader(
        "2. Cleaning"
    )

    cleaned_documents = clean_documents(
        extracted_documents
    )

    st.write(
        "Text cleaning completed."
    )

    # ========================================================
    # STEP 3 — STRUCTURE DETECTION
    # ========================================================

    st.subheader(
        "3. Document Structure"
    )

    structured_documents = detect_structure(
        cleaned_documents
    )

    headings = sum(
        1
        for unit in structured_documents
        if unit["type"] == "heading"
    )

    paragraphs = sum(
        1
        for unit in structured_documents
        if unit["type"] == "paragraph"
    )

    list_items = sum(
        1
        for unit in structured_documents
        if unit["type"] == "list_item"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Headings",
            headings
        )

    with col2:

        st.metric(
            "Paragraphs",
            paragraphs
        )

    with col3:

        st.metric(
            "List Items",
            list_items
        )

    # ========================================================
    # STEP 4 — SEMANTIC CHUNKING
    # ========================================================

    st.subheader(
        "4. Semantic Chunking"
    )

    threshold = st.slider(
        "Semantic Similarity Threshold",
        min_value=0.30,
        max_value=0.95,
        value=0.70,
        step=0.05
    )

    semantic_chunks = create_semantic_chunks(
        structured_documents,
        embedding_model.model,
        threshold
    )

    st.write(
        f"Generated chunks: "
        f"{len(semantic_chunks)}"
    )

    # ========================================================
    # DISPLAY SEMANTIC CHUNKS
    # ========================================================

    for chunk in semantic_chunks:

        chunk_id = chunk["chunk_id"]

        chunk_text = chunk["text"]

        metadata = chunk["metadata"]

        with st.expander(
            f"Chunk {chunk_id} — "
            f"{metadata.get('chunk_type', 'unknown')}"
        ):

            st.write(
                "### Text"
            )

            st.write(
                chunk_text
            )

            st.write(
                "### Metadata"
            )

            st.json(
                metadata
            )

    # ========================================================
    # STEP 5 — FINAL EMBEDDINGS
    # ========================================================

    st.subheader(
        "5. Final Embeddings"
    )

    embedded_chunks = create_embeddings(
        semantic_chunks,
        embedding_model
    )

    st.write(
        f"Generated embeddings for "
        f"{len(embedded_chunks)} chunks."
    )

    # ========================================================
    # DISPLAY EMBEDDINGS
    # ========================================================

    for chunk in embedded_chunks:

        with st.expander(
            f"Embedding — Chunk "
            f"{chunk['chunk_id']}"
        ):

            st.write(
                "### Text"
            )

            st.write(
                chunk["text"]
            )

            st.write(
                "### Metadata"
            )

            st.json(
                chunk["metadata"]
            )

            st.write(
                "### Embedding Information"
            )

            st.write(
                f"Embedding dimension: "
                f"{len(chunk['embedding'])}"
            )

            st.write(
                "First 10 values:"
            )

            st.write(
                chunk["embedding"][:10]
            )

    # ========================================================
    # FINAL DATA
    # ========================================================

    st.subheader(
        "Final Embedded Documents"
    )

    st.write(
        "These objects are now ready to be "
        "stored in a vector database."
    )

    st.json(
        embedded_chunks
    )