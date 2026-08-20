class Retriever:

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query, top_k=5, score_threshold=None):

        # Convert user query into embedding
        query_embedding = self.embedding_model.embed_query(
            query
        )

        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # Optional similarity threshold
        if score_threshold is not None:

            results = [
                result
                for result in results
                if result["score"] >= score_threshold
            ]

        return results