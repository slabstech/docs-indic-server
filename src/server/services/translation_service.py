from fastapi import HTTPException
from config import config
from utils.translation_utils import translate_text

class TranslationService:
    def __init__(self):
        self.api_url = config.TRANSLATION_API_URL

    def validate_languages(self, src_lang: str, tgt_lang: str) -> None:
        """Validate source and target language codes."""
        if src_lang not in config.LANGUAGE_OPTIONS:
            raise HTTPException(status_code=400, detail=f"Invalid source language: {src_lang}")
        if tgt_lang not in config.LANGUAGE_OPTIONS:
            raise HTTPException(status_code=400, detail=f"Invalid target language: {tgt_lang}")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text from source to target language."""
        self.validate_languages(src_lang, tgt_lang)
        return translate_text(text, src_lang, tgt_lang, self.api_url)