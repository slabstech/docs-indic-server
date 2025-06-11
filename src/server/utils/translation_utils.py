import requests
from fastapi import HTTPException
from typing import List
from .ocr_utils import split_into_sentences

def translate_text(text: str, src_lang: str, tgt_lang: str, api_url: str) -> str:
    """Translate text from source to target language."""
    try:
        sentences = split_into_sentences(text)
        payload = {
            "sentences": sentences,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        response = requests.post(
            f"{api_url}/translate?src_lang={src_lang}&tgt_lang={tgt_lang}",
            json=payload,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        return " ".join(result["translations"])
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")