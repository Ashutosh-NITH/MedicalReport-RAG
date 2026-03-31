import asyncio
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import StreamingResponse

from services.llm_service import LLMService
from services.vector_store_service import VectorStoreService
from services.file_extraction_service import FileExtractionService

app = FastAPI()

vector_store_service = VectorStoreService()
llm_service = LLMService()
file_extraction_service = FileExtractionService()

ALLOWED_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "RAG model is running"}


@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        return {
            "error": f"Unsupported file type: {file.content_type}. Allowed: PDF, JPEG, PNG, WEBP, GIF"
        }

    contents = await file.read()

    if file.content_type == "application/pdf":
        extracted_text = file_extraction_service.extract_from_pdf(contents)
    else:
        extracted_text = file_extraction_service.extract_from_image(
            contents, file.content_type
        )

    if not extracted_text.strip():
        return {"error": "Could not extract any content from the uploaded file."}

    results = vector_store_service.query(extracted_text[:1000])

    return StreamingResponse(
        llm_service.generate_response(extracted_text, results),
        media_type="text/plain",
    )


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        file_b64 = data.get("file")
        mime_type = data.get("mime_type")

        if not file_b64 or not mime_type:
            await websocket.send_json({"type": "error", "data": "Missing file or mime_type"})
            return

        import base64
        file_bytes = base64.b64decode(file_b64)

        if mime_type == "application/pdf":
            extracted_text = file_extraction_service.extract_from_pdf(file_bytes)
        else:
            extracted_text = file_extraction_service.extract_from_image(file_bytes, mime_type)

        results = vector_store_service.query(extracted_text[:1000])
        await asyncio.sleep(0.1)
        await websocket.send_json({"type": "search_result", "data": results})

        for chunk in llm_service.generate_response(extracted_text, results):
            await asyncio.sleep(0.1)
            await websocket.send_json({"type": "content", "data": chunk})

    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_json({"type": "error", "data": str(e)})
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)