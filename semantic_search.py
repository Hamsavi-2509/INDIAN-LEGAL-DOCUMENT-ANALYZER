import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


class LegalSemanticSearch:

    def __init__(
        self,
        chunks,
        model,
        embeddings
    ):

        self.chunks = chunks

        self.model = model

        self.chunk_embeddings = embeddings


    def search(
        self,
        question,
        top_k=5
    ):

        # Convert question into embedding

        question_embedding = (
            self.model.encode(
                [question],
                convert_to_numpy=True
            )
        )


        # Calculate cosine similarity

        similarities = cosine_similarity(
            question_embedding,
            self.chunk_embeddings
        )[0]


        # Rank chunks by similarity

        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]


        results = []


        for index in top_indices:

            chunk = self.chunks[index]

            results.append({

                "chunk_id":
                    chunk["chunk_id"],

                "document":
                    chunk["document"],

                "page":
                    chunk["page"],

                "text":
                    chunk["text"],

                "similarity":
                    float(
                        similarities[index]
                    )

            })


        return results