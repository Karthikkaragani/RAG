import json
import os

import numpy as np


class VectorStore:

    def __init__(
        self,
        storage_dir="vector_db"
    ):

        self.storage_dir = storage_dir

        self.vector_file = os.path.join(
            storage_dir,
            "vectors.npy"
        )

        self.metadata_file = os.path.join(
            storage_dir,
            "metadata.json"
        )

        os.makedirs(
            storage_dir,
            exist_ok=True
        )

        self.vectors = None
        self.documents = []

        self.load()


    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        embedded_chunks
    ):

        if not embedded_chunks:

            return

        new_vectors = np.array(
            [
                chunk["embedding"]
                for chunk in embedded_chunks
            ],
            dtype=np.float32
        )

        new_documents = []

        for chunk in embedded_chunks:

            new_documents.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                }
            )

        # ----------------------------------------------------
        # First insertion
        # ----------------------------------------------------

        if self.vectors is None:

            self.vectors = new_vectors

            self.documents = new_documents

        # ----------------------------------------------------
        # Add to existing vectors
        # ----------------------------------------------------

        else:

            self.vectors = np.vstack(
                [
                    self.vectors,
                    new_vectors
                ]
            )

            self.documents.extend(
                new_documents
            )

        self.save()


    # ========================================================
    # SAVE VECTOR STORE
    # ========================================================

    def save(self):

        if self.vectors is not None:

            np.save(
                self.vector_file,
                self.vectors
            )

        with open(
            self.metadata_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.documents,
                file,
                indent=4,
                ensure_ascii=False
            )


    # ========================================================
    # LOAD VECTOR STORE
    # ========================================================

    def load(self):

        if os.path.exists(
            self.vector_file
        ):

            self.vectors = np.load(
                self.vector_file
            )

        if os.path.exists(
            self.metadata_file
        ):

            with open(
                self.metadata_file,
                "r",
                encoding="utf-8"
            ) as file:

                self.documents = json.load(
                    file
                )


    # ========================================================
    # VECTOR COUNT
    # ========================================================

    def count(self):

        if self.vectors is None:

            return 0

        return len(
            self.vectors
        )


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        if self.vectors is None:

            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Since our embeddings are normalized,
        # dot product = cosine similarity
        # ----------------------------------------------------

        similarities = (
            self.vectors
            @ query_embedding
        )

        # ----------------------------------------------------
        # Get Top-K indices
        # ----------------------------------------------------

        top_k = min(
            top_k,
            len(similarities)
        )

        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        # ----------------------------------------------------
        # Build results
        # ----------------------------------------------------

        results = []

        for index in top_indices:

            document = self.documents[
                int(index)
            ].copy()

            document["score"] = float(
                similarities[index]
            )

            results.append(
                document
            )

        return results


    # ========================================================
    # CLEAR STORE
    # ========================================================

    def clear(self):

        self.vectors = None

        self.documents = []

        if os.path.exists(
            self.vector_file
        ):

            os.remove(
                self.vector_file
            )

        if os.path.exists(
            self.metadata_file
        ):

            os.remove(
                self.metadata_file
            )