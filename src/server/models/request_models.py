from pydantic import BaseModel, Field, validator
from config import config
from typing import Optional

class BasePDFRequest(BaseModel):
    page_number: int = Field(default=1, ge=1, description="Page number to process (1-based indexing)")
    model: str = Field(default=config.DEFAULT_MODEL, enum=config.VALID_MODELS, description="LLM model")

class ExtractTextRequest(BasePDFRequest):
    pass

class SummarizePDFRequest(BasePDFRequest):
    pass

class CustomPromptPDFRequest(BasePDFRequest):
    prompt: str = Field(..., description="Custom prompt for processing")

    @validator("prompt")
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v

class IndicBaseRequest(BasePDFRequest):
    source_language: str = Field(default="eng_Latn", enum=config.LANGUAGE_OPTIONS, description="Source language code")
    target_language: str = Field(default="kan_Knda", enum=config.LANGUAGE_OPTIONS, description="Target language code")

class IndicCustomPromptPDFRequest(IndicBaseRequest, CustomPromptPDFRequest):
    pass

class IndicSummarizePDFRequest(IndicBaseRequest):
    pass

class IndicExtractTextRequest(IndicBaseRequest):
    pass

class IndicVisualQueryRequest(BaseModel):
    prompt: Optional[str] = Field(default=None, description="Optional custom prompt")
    source_language: str = Field(default="eng_Latn", enum=config.LANGUAGE_OPTIONS, description="Source language code")
    target_language: str = Field(default="kan_Knda", enum=config.LANGUAGE_OPTIONS, description="Target language code")
    model: str = Field(default=config.DEFAULT_MODEL, enum=config.VALID_MODELS, description="LLM model")

    @validator("prompt")
    def prompt_not_empty_if_provided(cls, v):
        if v and not v.strip():
            raise ValueError("Prompt cannot be empty if provided")
        return v
class VisualQueryDirectRequest(BaseModel):
    prompt: Optional[str] = Field(default=None, description="Optional custom prompt")
    model: str = Field(default=config.DEFAULT_MODEL, enum=["gemma3", "moondream", "smolvla"], description="LLM model")

    @validator("prompt")
    def prompt_not_empty_if_provided(cls, v):
        if v and not v.strip():
            raise ValueError("Prompt cannot be empty if provided")
        return v
class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    src_lang: str = Field(default="eng_Latn", enum=config.LANGUAGE_OPTIONS, description="Source language code")
    tgt_lang: str = Field(default="eng_Latn", enum=config.LANGUAGE_OPTIONS, description="Target language code")
    model: str = Field(default=config.DEFAULT_MODEL, enum=config.VALID_MODELS, description="LLM model")

    @validator("prompt")
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v

class ChatDirectRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    model: list = Field(default="", description="Prompt to process")
    system_prompt: str = Field(default="", description="System Prompt")

    @validator("prompt")
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v