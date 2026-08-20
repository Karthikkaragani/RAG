import re

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# ============================================================
# SEMANTIC CHUNK
# ============================================================

def semantic_chunk(
    text,
    model,
    threshold=0.70
):

    sentences = split_into_sentences(
        text
    )

    if len(sentences) <= 1:

        return sentences

    # Generate embeddings

    embeddings = model.encode(
        sentences,
        convert_to_numpy=True
    )

    # Compare neighboring sentences

    similarities = []

    for i in range(
        len(sentences) - 1
    ):

        similarity = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]

        similarities.append(
            float(similarity)
        )

    # Build chunks

    chunks = []

    current_chunk = [
        sentences[0]
    ]

    for i, similarity in enumerate(
        similarities
    ):

        next_sentence = sentences[
            i + 1
        ]

        if similarity >= threshold:

            current_chunk.append(
                next_sentence
            )

        else:

            chunks.append(
                " ".join(
                    current_chunk
                )
            )

            current_chunk = [
                next_sentence
            ]

    # Add final chunk

    if current_chunk:

        chunks.append(
            " ".join(
                current_chunk
            )
        )

    return chunks


# ============================================================
# CREATE SEMANTIC CHUNKS
# ============================================================

def create_semantic_chunks(
    structured_documents,
    model,
    threshold=0.70
):

    chunks = []

    chunk_id = 0

    for unit in structured_documents:

        text = unit["text"]

        metadata = unit[
            "metadata"
        ].copy()

        unit_type = unit["type"]

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        if unit_type == "heading":

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": {
                        **metadata,
                        "chunk_type": "heading"
                    }
                }
            )

            chunk_id += 1

            continue

        # ----------------------------------------------------
        # Semantic chunking
        # ----------------------------------------------------

        semantic_chunks = semantic_chunk(
            text,
            model,
            threshold
        )

        # ----------------------------------------------------
        # Add metadata
        # ----------------------------------------------------

        for chunk_text in semantic_chunks:

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_type": unit_type
                    }
                }
            )

            chunk_id += 1

    return chunks