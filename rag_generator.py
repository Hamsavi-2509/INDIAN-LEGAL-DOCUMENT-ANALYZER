import ollama


class LegalRAGGenerator:

    def __init__(
        self,
        model="llama3.2:latest"
    ):

        self.model = model


    def generate_answer(
        self,
        question,
        retrieved_chunks
    ):

        context = "\n\n".join(
            [
                (
                    f"Legal Passage {i + 1}:\n"
                    f"{chunk['text']}"
                )
                for i, chunk
                in enumerate(retrieved_chunks)
            ]
        )


        prompt = f"""
You are an Indian legal information assistant.

Answer the user's question using ONLY the
legal information provided below.

IMPORTANT RULES:

1. Use only the provided legal information.

2. Do not use outside knowledge.

3. Do not invent legal information.

4. If the user asks about a specific IPC section,
   clearly provide the section, offense, punishment,
   and relevant description when available.

5. If the user asks for the punishment, clearly
   state the punishment from the provided data.

6. If the user asks what an offense means, explain
   it using the provided description.

7. If the question asks for multiple points,
   provide them as a numbered list.

8. Give a clear and complete answer.

9. If the answer is not available in the provided
   information, say:

"The answer cannot be determined from the provided legal information."

LEGAL INFORMATION:

{context}

USER QUESTION:

{question}

ANSWER:
"""


        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            keep_alive="30m"
        )


        return (
            response["message"]["content"]
            .strip()
        )