import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.ingest_service import IngestService

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        print("Example: python ingest.py ./data/my_document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    service = IngestService()
    service.ingest_pdf(pdf_path)