from openai import OpenAI
from fastapi import HTTPException
from config import config

class OCRService:
    def __init__(self, model: str):
        self.model = model
        self.client = self._get_openai_client()

    def _get_openai_client(self) -> OpenAI:
        """Initialize OpenAI client for the specified model."""
        if self.model not in config.VALID_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {self.model}. Choose from: {', '.join(config.VALID_MODELS)}")
        base_url = f"http://0.0.0.0:{config.MODEL_PORTS[self.model]}/v1"
        return OpenAI(api_key=config.OPENAI_API_KEY, base_url=base_url)

    def ocr_image(self, img_base64: str) -> str:
        """Perform OCR on a base64-encoded image."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                            {"type": "text", "text": "Return the plain text extracted from this image."}
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

    def process_text(self, text: str, prompt: str) -> str:
        """Process extracted text with a custom prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are dwani, a helpful assistant. Summarize your answer in maximum 1 sentence. If the answer contains numerical digits, convert the digits into words"}]
                    },
                    {"role": "user", "content": f"{prompt}\n\n{text}"}
                ],
                temperature=0.3,
                max_tokens=config.MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")

    def summarize_text(self, text: str) -> str:
        """Summarize text in 3-5 sentences."""
        return self.process_text(text, "Summarize the following text in 3-5 sentences:")