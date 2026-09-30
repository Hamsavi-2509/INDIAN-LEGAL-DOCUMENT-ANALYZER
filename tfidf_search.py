import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LegalTFIDFSearch:

    def __init__(self, chunks):

        self.chunks = chunks

        self.texts = [
            chunk["text"]
            for chunk in chunks
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                self.texts
            )
        )


    def search(
        self,
        question,
        top_k=3
    ):

        question_vector = (
            self.vectorizer.transform(
                [question]
            )
        )

        similarities = cosine_similarity(
            question_vector,
            self.tfidf_matrix
        )[0]

        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            result = self.chunks[index].copy()

            result["similarity"] = float(
                similarities[index]
            )

            results.append(
                result
            )

        return results