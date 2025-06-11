from io import BytesIO
import base64
from fastapi import HTTPException

def encode_image(image: BytesIO) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image.read()).decode("utf-8")

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences based on full stops."""
    if not text.strip():
        return []
    return [s.strip() for s in text.split('.') if s.strip()]