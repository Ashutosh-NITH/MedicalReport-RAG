from google import genai
from config import Settings

settings = Settings()


class LLMService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

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

        for chunk in self.client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=full_prompt,
        ):
            yield chunk.text