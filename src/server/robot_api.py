from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request, Depends, Form
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI
import base64
import json
from io import BytesIO
from PIL import Image
import requests
from typing import List, Union, Optional
from pydantic import BaseModel, Field
import argparse
import uvicorn
from time import time
from logging_config import logger
from num2words import num2words
from datetime import datetime
import pytz

# Initialize FastAPI app
app = FastAPI(
    title="dwani.ai - Document server API",
    description=(
        "API for extracting text from PDF pages and PNG images using RolmOCR, with functionality to "
        "summarize PDF content, process it with custom prompts, translate summaries to Kannada, or translate "
        "extracted Kannada text to English. Supports text extraction from a single PDF page, OCR for PNG images, "
        "summarization of a single PDF page, custom prompt-based processing, and translation between Kannada and English."
    ),
    version="1.0.0"
)

# Translation API URL
translation_api_url = "http://0.0.0.0:7862"

# Supported language codes
language_options = [
    "kan_Knda",  # Kannada
    "eng_Latn",  # English
    "hin_Deva",  # Hindi
    "tam_Taml",  # Tamil
    "tel_Telu",  # Telugu
    "deu_Latn",
]



def split_into_sentences(text):
    """Split a string into sentences based on full stops."""
    if not text.strip():
        return []
    # Split on full stops, preserving non-empty sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sentences


# Pydantic models
class ExtractTextRequest(BaseModel):
    page_number: int = Field(default=1, description="The page number to extract text from (1-based indexing).", ge=1, example=1)
    model: str = Field(default="gemma3", description="Model to use for OCR processing.", enum=["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"], example="gemma3")

class IndicVisualQueryRequest(BaseModel):
    prompt: Optional[str] = Field(default=None, description="Optional custom prompt to process the extracted text.", example="Summarize the text in 2 sentences.")
    source_language: str = Field(default="eng_Latn", description="Source language code.", examples=["eng_Latn", "hin_Deva"], enum=language_options)
    target_language: str = Field(default="kan_Knda", description="Target language code.", examples=["kan_Knda", "tam_Taml"], enum=language_options)
    model: str = Field(default="gemma3", description="Model to use for OCR and processing.", enum=["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"], example="gemma3")

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for the chat.")
    src_lang: str = Field(default="eng_Latn", description="Source language code.", enum=language_options)
    tgt_lang: str = Field(default="eng_Latn", description="Target language code.", enum=language_options)
    model: str = Field(default="gemma3", description="Model to use for chat.", enum=["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"], example="gemma3")

class ChatDirectRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for the chat.")
    model: str = Field(default="gemma3", description="Model to use for chat.", enum=["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"], example="gemma3")
    system_prompt: str = Field(default="", description="System Prompt")
 
class ChatResponse(BaseModel):
    response: str = Field(..., description="The generated or translated response.")

# Dynamic LLM client based on model
def get_openai_client(model: str) -> OpenAI:
    """Initialize OpenAI client with model-specific base URL."""
    valid_models = ["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Choose from: {', '.join(valid_models)}")
    
    model_ports = {
        "qwen3": "7880",
        "gemma3": "7890",
        "moondream": "7882",
        "qwen2.5vl": "7883",
        "sarvam-m": "7884",
        "deepseek-r1": "7885"
    }
    base_url = f"http://0.0.0.0:{model_ports[model]}/v1"

    return OpenAI(api_key="http", base_url=base_url)

def encode_image(image: BytesIO) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image.read()).decode("utf-8")

def ocr_page_with_rolm(img_base64: str, model: str) -> str:
    """Perform OCR on the provided base64 image using the specified model."""
    try:
        client = get_openai_client(model)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        },
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
    

@app.get("/")
async def root():
    return {"message": "Combined API is running."}

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images supported")

    try:
        image_bytes = await file.read()
        image = BytesIO(image_bytes)
        img_base64 = encode_image(image)
        text = ocr_page_with_rolm(img_base64, model="gemma3")
        return {"extracted_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Updated Visual Query Endpoint
@app.post("/visual-query-direct/",
          summary="Visual Query with Image",
          description="Extract text from a PNG image using OCR, optionally process it with a custom prompt",
          tags=["Chat"],
          responses={
              200: {"description": "Extracted text "},
              400: {"description": "Invalid image, prompt"},
              500: {"description": "OCR error"}
          })
async def indic_visual_query_direct(
    request: Request,
    file: UploadFile = File(..., description="PNG image file to analyze"),
    prompt: Optional[str] = Form(None, description="Optional custom prompt to process the extracted text"),
    model: str = Form("gemma3", description="LLM model", enum=["gemma3", "moondream", "smolvla"])
):
    try:
        if not file.content_type.startswith("image/png"):
            raise HTTPException(status_code=400, detail="Only PNG images supported")

        logger.debug(f"Processing indic visual query: model={model}, prompt={prompt[:50] if prompt else None}")

        image_bytes = await file.read()
        image = BytesIO(image_bytes)
        img_base64 = encode_image(image)
        extracted_text = ocr_page_with_rolm(img_base64, model)

        response = None
        text_to_translate = extracted_text
        if prompt and prompt.strip():
            client = get_openai_client(model)
            custom_response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are dwani, a helpful assistant. Summarize your answer in maximum 1 sentence. If the answer contains numerical digits, convert the digits into words"}]
                    },
                    {"role": "user", "content": f"{prompt}\n\n{extracted_text}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            response = custom_response.choices[0].message.content
            
        elif prompt and not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        result = {
            "extracted_text": extracted_text,
            "response": response
        }
        if response:
            result["response"] = response

        logger.debug(f"visual query direct successful: extracted_text_length={len(extracted_text)}")
        return JSONResponse(content=result)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error translating: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error translating: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


class Settings:
    chat_rate_limit = "10/minute"
    max_tokens = 500
    openai_api_key = "http"

def get_settings():
    return Settings()

def time_to_words():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    hour = now.hour % 12 or 12
    minute = now.minute
    hour_word = num2words(hour, to='cardinal')
    if minute == 0:
        return f"{hour_word} o'clock"
    else:
        minute_word = num2words(minute, to='cardinal')
        return f"{hour_word} hours and {minute_word} minutes"


@app.post("/chat_direct")
async def chat_direct(
    request: Request,
    chat_request: ChatDirectRequest,
    settings=Depends(get_settings)
):
    if not chat_request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    logger.debug(f"Received prompt: {chat_request.prompt},  model: {chat_request.model}")

    try:

        prompt_to_process = chat_request.prompt

        current_time = time_to_words()
        client = get_openai_client(chat_request.model)
        response = client.chat.completions.create(
            model=chat_request.model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": f"You are Dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. If the answer contains numerical digits, convert the digits into words. If user asks the time, then return answer as {current_time}"}]
                },
                {"role": "user", "content": [{"type": "text", "text": prompt_to_process}]}
            ],
            temperature=0.3,
            max_tokens=settings.max_tokens
        )
        generated_response = response.choices[0].message.content
        logger.debug(f"Generated response: {generated_response}")


        return JSONResponse(content={"response": generated_response})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    end_time = time()
    duration = end_time - start_time
    logger.info(f"Request to {request.url.path} took {duration:.3f} seconds")
    response.headers["X-Response-Time"] = f"{duration:.3f}"
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dwani.ai - Document server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7864, help="Port to bind (default: 7864)")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)