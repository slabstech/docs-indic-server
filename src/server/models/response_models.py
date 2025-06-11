from pydantic import BaseModel
from typing import Optional

class ExtractTextResponse(BaseModel):
    page_content: str
    processed_page: int

class IndicExtractTextResponse(ExtractTextResponse):
    translated_content: str

class SummarizePDFResponse(BaseModel):
    original_text: str
    summary: str
    processed_page: int

class IndicSummarizePDFResponse(SummarizePDFResponse):
    translated_summary: str

class CustomPromptPDFResponse(BaseModel):
    original_text: str
    response: str
    processed_page: int

class IndicCustomPromptPDFResponse(CustomPromptPDFResponse):
    translated_response: str

class IndicVisualQueryResponse(BaseModel):
    extracted_text: str
    response: Optional[str] = None
    translated_response: str

class VisualQueryDirectResponse(BaseModel):
    extracted_text: str
    response: Optional[str] = None

class ChatResponse(BaseModel):
    response: str