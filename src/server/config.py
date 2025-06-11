from typing import Dict, List

class AppConfig:
    TRANSLATION_API_URL: str = "http://0.0.0.0:7862"
    LANGUAGE_OPTIONS: List[str] = ["kan_Knda", "eng_Latn", "hin_Deva", "tam_Taml", "tel_Telu"]
    MODEL_PORTS: Dict[str, str] = {
        "qwen3": "7880",
        "gemma3": "7881",
        "moondream": "7882",
        "qwen2.5vl": "7883",
        "sarvam-m": "7884",
        "deepseek-r1": "7885",
    }
    VALID_MODELS: List[str] = list(MODEL_PORTS.keys())
    CHAT_RATE_LIMIT: str = "10/minute"
    MAX_TOKENS: int = 500
    OPENAI_API_KEY: str = "http"
    DEFAULT_MODEL: str = "gemma3"

config = AppConfig()