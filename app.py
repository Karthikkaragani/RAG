import streamlit as st

from ingestion import extract_document
from cleaning import clean_documents
from structure import detect_structure
from chunking import create_semantic_chunks
from embeddings import (
    EmbeddingModel,
    create_embeddings
)
from vector_store import VectorStore
from retriever import Retriever


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

st.header(
    "End-to-End RAG Pipeline"
)


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
# LOAD VECTOR STORE
# ============================================================

@st.cache_resource
def load_vector_store():

    return VectorStore(
        storage_dir="vector_db"
    )


vector_store = load_vector_store()


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = Retriever(
    vector_store=vector_store,
    embedding_model=embedding_model
)


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
# DOCUMENT PROCESSING
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
    # STEP 3 — DOCUMENT STRUCTURE
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
    # DISPLAY CHUNKS
    # ========================================================

    with st.expander(
        "View Semantic Chunks"
    ):

        for chunk in semantic_chunks:

            st.markdown(
                f"### Chunk {chunk['chunk_id']}"
            )

            st.write(
                chunk["text"]
            )

            st.json(
                chunk["metadata"]
            )

            st.divider()


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


    if embedded_chunks:

        embedding_dimension = len(
            embedded_chunks[0]["embedding"]
        )

        st.write(
            f"Embedding dimension: "
            f"{embedding_dimension}"
        )


        # Show first embedding as an example

        with st.expander(
            "View Sample Embedding"
        ):

            st.write(
                "Chunk ID:",
                embedded_chunks[0]["chunk_id"]
            )

            st.write(
                "Embedding dimension:",
                embedding_dimension
            )

            st.write(
                "First 10 values:"
            )

            st.write(
                embedded_chunks[0][
                    "embedding"
                ][:10]
            )


    # ========================================================
    # STEP 6 — VECTOR STORE
    # ========================================================

    st.subheader(
        "6. Vector Store"
    )


    vector_store.add_documents(
        embedded_chunks
    )


    st.success(
        "Vectors stored successfully."
    )


    st.write(
        f"Total vectors stored: "
        f"{vector_store.count()}"
    )


    if vector_store.vectors is not None:

        st.write(
            f"Vector matrix shape: "
            f"{vector_store.vectors.shape}"
        )


    # ========================================================
    # VIEW VECTOR STORE
    # ========================================================

    with st.expander(
        "View Vector Store"
    ):

        if vector_store.vectors is not None:

            st.write(
                "Stored vector matrix:"
            )

            st.write(
                vector_store.vectors
            )

            st.write(
                "Stored documents:"
            )

            st.json(
                vector_store.documents
            )


# ============================================================
# STEP 7 — RETRIEVAL
# ============================================================

st.subheader(
    "7. Dense Retrieval"
)


query = st.text_input(
    "Ask a question about your documents"
)


top_k = st.slider(
    "Number of chunks to retrieve (Top-K)",
    min_value=1,
    max_value=10,
    value=3
)


score_threshold = st.slider(
    "Similarity Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)


# ============================================================
# PERFORM RETRIEVAL
# ============================================================

if query:

    if vector_store.count() == 0:

        st.warning(
            "No vectors are stored yet. "
            "Please upload a document first."
        )

    else:

        results = retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )


        st.write(
            f"Retrieved {len(results)} chunks."
        )


        # ====================================================
        # DISPLAY RETRIEVAL RESULTS
        # ====================================================

        for rank, result in enumerate(
            results,
            start=1
        ):

            st.markdown(
                f"### Rank {rank}"
            )


            st.write(
                f"Similarity Score: "
                f"{result['score']:.4f}"
            )


            st.write(
                "#### Retrieved Chunk"
            )


            st.write(
                result["text"]
            )


            st.write(
                "#### Metadata"
            )


            st.json(
                result["metadata"]
            )


            st.divider()