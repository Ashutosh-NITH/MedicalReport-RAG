import fitz  # PyMuPDF
import base64
from google import genai
from config import Settings

settings = Settings()


class FileExtractionService:
    def __init__(self):
        # Gemini client kept solely for Vision-based image extraction
        self.vision_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def extract_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract all text from an uploaded PDF."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text().strip() + "\n"
        doc.close()
        return text.strip()

    def extract_from_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Use Gemini Vision to extract text/info from an uploaded image."""
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = self.vision_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_b64,
                            }
                        },
                        {
                            "text": "Extract and describe all text, data, tables, and information visible in this image in detail."
                        }
                    ]
                }
            ]
        )
        return response.text


