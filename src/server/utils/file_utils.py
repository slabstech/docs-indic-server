import os
import tempfile
from contextlib import contextmanager
from fastapi import HTTPException

@contextmanager
def temp_file(suffix: str = ".pdf"):
    """Create and yield a temporary file, ensuring cleanup."""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield temp
    finally:
        temp.close()
        if os.path.exists(temp.name):
            os.remove(temp.name)

def validate_pdf_file(filename: str) -> None:
    """Validate that the file is a PDF."""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported.")

def validate_png_file(content_type: str) -> None:
    """Validate that the file is a PNG image."""
    if not content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images supported.")