import fitz


def extract_pages(pdf_bytes: bytes, filename: str) -> list[dict]:
    """Extract text from a PDF, retaining a 1-based page number and filename."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "source": filename,
                    "page": page_number,
                    "text": " ".join(text.split()),
                })
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 1000, overlap: int = 150) -> list[dict]:
    """Split each page into character chunks; page/source metadata stays attached."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "source": page["source"],
                    "page": page["page"],
                    "text": piece,
                })
            if end == len(text):
                break
            start = end - overlap
    return chunks
