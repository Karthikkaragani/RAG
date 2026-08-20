from sentence_transformers import SentenceTransformer


# ============================================================
# EMBEDDING MODEL
# ============================================================

class EmbeddingModel:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )


    # ========================================================
    # EMBED SINGLE TEXT
    # ========================================================

    def embed_text(
        self,
        text
    ):

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding


    # ========================================================
    # EMBED DOCUMENTS
    # ========================================================

    def embed_documents(
        self,
        texts
    ):

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings


    # ========================================================
    # EMBED QUERY
    # ========================================================

    def embed_query(
        self,
        query
    ):

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding


# ============================================================
# CREATE EMBEDDINGS FOR CHUNKS
# ============================================================

def create_embeddings(
    chunks,
    embedding_model
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = (
        embedding_model.embed_documents(
            texts
        )
    )

    embedded_chunks = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        embedded_chunks.append(
            {
                "chunk_id": chunk[
                    "chunk_id"
                ],

                "text": chunk[
                    "text"
                ],

                "embedding": embedding.tolist(),

                "metadata": chunk[
                    "metadata"
                ]
            }
        )

    return embedded_chunks