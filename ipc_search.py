import re
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


class IPCSearch:

    def __init__(
        self,
        data,
        model,
        embeddings
    ):

        self.data = data.reset_index(
            drop=True
        )

        self.model = model

        self.embeddings = embeddings


    def search(
        self,
        question,
        top_k=5
    ):

        question_clean = (
            question.strip().lower()
        )


        # Exact IPC section search

        section_match = re.search(
            r"(?:ipc[\s_-]*)?(?:section[\s_-]*)?(\d+)",
            question_clean
        )


        if section_match:

            section_number = (
                section_match.group(1)
            )

            target_section = (
                f"ipc_{section_number}"
            )


            exact_matches = self.data[
                self.data["Section"]
                .astype(str)
                .str.lower()
                .eq(target_section)
            ]


            if not exact_matches.empty:

                # Remove duplicate section records

                exact_matches = (
                    exact_matches
                    .drop_duplicates(
                        subset=["Section"],
                        keep="first"
                    )
                )


                results = []


                for _, row in exact_matches.iterrows():

                    results.append({

                        "Section":
                            row["Section"],

                        "Offense":
                            row["Offense"],

                        "Punishment":
                            row["Punishment"],

                        "Description":
                            row["Description"],

                        "similarity":
                            1.0

                    })


                return results[:top_k]


        # Semantic search

        question_embedding = (
            self.model.encode(
                [question],
                convert_to_numpy=True
            )
        )


        similarities = cosine_similarity(
            question_embedding,
            self.embeddings
        )[0]


        top_indices = np.argsort(
            similarities
        )[::-1]


        results = []

        seen_sections = set()


        for index in top_indices:

            row = self.data.iloc[index]

            section = str(
                row["Section"]
            )


            # Avoid duplicate sections

            if section in seen_sections:

                continue


            seen_sections.add(section)


            results.append({

                "Section":
                    row["Section"],

                "Offense":
                    row["Offense"],

                "Punishment":
                    row["Punishment"],

                "Description":
                    row["Description"],

                "similarity":
                    float(
                        similarities[index]
                    )

            })


            if len(results) >= top_k:

                break


        return results