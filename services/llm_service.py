from groq import Groq
from config import Settings

settings = Settings()


class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = "llama-3.3-70b-versatile"

    def generate_response(self, extracted_input: str, search_results: list[dict]):
        context_text = "\n\n".join(
            [
                f"Source (Page {result['page']}):\n{result['content']}"
                for result in search_results
            ]
        )

        full_prompt = f"""
        You are an expert document analyst. The user has uploaded a PDF or image.
        Your job is to:
        1. Provide a clear, structured SUMMARY of the most relevant content from the document context below.
        2. Provide actionable SUGGESTIONS or insights based on the content.
        3. Cite page numbers wherever relevant.

        If the document context does not contain enough information, say so clearly.

        --- DOCUMENT CONTEXT ---
        {context_text}

        --- EXTRACTED INPUT FROM USER UPLOAD ---
        {extracted_input}

        Respond with:
        ## Summary
        (concise summary of relevant content)

        ## Suggestions
        (actionable insights or recommendations based on the document)
        """

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta