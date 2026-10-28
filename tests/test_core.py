import pytest
from src.pdf_processor import chunk_pages


def test_chunk_pages_preserves_metadata_and_covers_text():
    pages = [{"source": "demo.pdf", "page": 2, "text": "abcdefghij"}]
    chunks = chunk_pages(pages, chunk_size=6, overlap=2)
    assert len(chunks) == 2
    assert chunks[0]["text"] == "abcdef"
    assert chunks[1]["text"] == "efghij"
    assert all(c["source"] == "demo.pdf" and c["page"] == 2 for c in chunks)


def test_invalid_chunk_settings_raise():
    with pytest.raises(ValueError):
        chunk_pages([{"source": "x.pdf", "page": 1, "text": "hello"}], chunk_size=5, overlap=5)
