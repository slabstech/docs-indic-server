from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request, Depends, Form
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI
import base64
import json
from io import BytesIO
from PIL import Image
import tempfile
import os
import requests
from typing import List, Union, Optional
from pypdf import PdfReader
from pydantic import BaseModel, Field
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
import argparse
import uvicorn
from time import time
from logging_config import logger
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import pdfplumber
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
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
]

# Supported models and languages
SUPPORTED_MODELS = ["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]
language_options = ["kan_Knda", "eng_Latn", "hin_Deva", "tam_Taml", "tel_Telu"]


# Pydantic models
class ExtractTextRequest(BaseModel):
    page_number: int = Field(default=1, description="The page number to extract text from (1-based indexing).", ge=1, example=1)
    model: str = Field(default="gemma3", description="Model to use for OCR processing.", enum=["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"], example="gemma3")

class SummarizePDFRequest(BaseModel):
    page_number: int = Field(default=1, description="The page number to extract and summarize (1-based indexing).", ge=1, example=1)
    model: str = Field(default="gemma3", description="Model to use for summarization.", enum=["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"], example="gemma3")

class CustomPromptPDFRequest(BaseModel):
    page_number: int = Field(default=1, description="The page number to extract and process (1-based indexing).", ge=1, example=1)
    prompt: str = Field(description="The custom prompt to process the extracted text.", example="Summarize the text in 2 sentences.")
    model: str = Field(default="gemma3", description="Model to use for prompt processing.", enum=["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"], example="gemma3")

class CustomPromptPDFRequestExtended(CustomPromptPDFRequest):
    source_language: str = Field(default="eng_Latn", description="Source language code.", examples=["eng_Latn", "hin_Deva"], enum=language_options)
    target_language: str = Field(default="kan_Knda", description="Target language code.", examples=["kan_Knda", "tam_Taml"], enum=language_options)

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

class ChatResponse(BaseModel):
    response: str = Field(..., description="The generated or translated response.")

# Dynamic LLM client
def get_openai_client(model: str) -> OpenAI:
    """Initialize OpenAI client with model-specific base URL."""
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}. Supported models: {', '.join(SUPPORTED_MODELS)}")

    model_ports = {
        "qwen3": "7880",
        "gemma3": "7881",
        "moondream": "7882",
        "qwen2.5vl": "7883",
        "sarvam-m": "7884",
        "deepseek-r1": "7885"
    }
    base_url = f"http://localhost:{model_ports[model]}/v1"
    logger.debug(f"Initializing OpenAI client for model {model} with base_url: {base_url}")
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

@app.post("/extract-text/",
          summary="Extract Text from PDF",
          description="Extract text from a specified PDF page using OCR.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text"},
              400: {"description": "Invalid PDF or page number"},
              500: {"description": "OCR error"}
          })
async def extract_text_from_pdf(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(1, description="Page number to extract text from (1-based indexing)", ge=1),
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    logger.info(f"Processing extract text: page_number={page_number}, model={model} and file={file.filename}")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    logger.debug(f"Processing extract text: page_number={page_number}, model={model}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        try:
            image_base64 = render_pdf_to_base64png(temp_file_path, page_number, target_page_number=page_number)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to render PDF page: {str(e)}")

        try:
            page_content = ocr_page_with_rolm(image_base64, model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        os.remove(temp_file_path)
        return JSONResponse(content={"page_content": page_content})

    except Exception as e:
        if 'temp_file_path' in locals():
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


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

# Summarize PDF Endpoint
@app.post("/summarize-pdf",
          summary="Summarize a PDF Page",
          description="Summarize the content of a specific PDF page.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text and summary"},
              400: {"description": "Invalid PDF or page number"},
              500: {"description": "OCR or summarization error"}
          })
async def summarize_pdf(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(1, description="Page number to summarize (1-based indexing)", ge=1),
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    logger.debug(f"Processing summarize PDF: page_number={page_number}, model={model}")

    try:
        text_response = await extract_text_from_pdf(file, page_number, model)
        extracted_text = json.loads(text_response.body.decode("utf-8"))["page_content"]

        client = get_openai_client(model)
        summary_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"Summarize the following text in 3-5 sentences:\n\n{extracted_text}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        summary = summary_response.choices[0].message.content

        return JSONResponse(content={
            "original_text": extracted_text,
            "summary": summary,
            "processed_page": page_number
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Custom Prompt PDF Endpoint
@app.post("/custom-prompt-pdf",
          summary="Process a PDF with a Custom Prompt",
          description="Extract text from a PDF page and process it with a custom prompt.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text and custom response"},
              400: {"description": "Invalid PDF, page number, or prompt"},
              500: {"description": "OCR or processing error"}
          })
async def custom_prompt_pdf(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(1, description="Page number to process", ge=1),
    prompt: str = Form(..., description="Custom prompt to process"),
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    logger.debug(f"Processing custom prompt PDF: page_number={page_number}, model={model}")

    try:
        text_response = await extract_text_from_pdf(file, page_number, model)
        extracted_text = json.loads(text_response.body.decode("utf-8"))["page_content"]

        client = get_openai_client(model)
        custom_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}\n\n{extracted_text}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        response = custom_response.choices[0].message.content

        return JSONResponse(content={
            "original_text": extracted_text,
            "response": response,
            "processed_page": page_number
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@app.post("/indic-custom-prompt-pdf",
          summary="Process a PDF with Custom Prompt and Translation",
          description="Extract text from a PDF page, process it with a custom prompt, and translate the response.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text, custom response, and translated response"},
              400: {"description": "Invalid PDF, page number, prompt, or language codes"},
              500: {"description": "OCR, processing, or translation error"}
          })
async def indic_custom_prompt_pdf(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(1, description="Page number to process", ge=1),
    prompt: str = Form(..., description="Custom prompt to process"),
    src_lang: str = Form("eng_Latn", description="Source language code"),  # Default added
    tgt_lang: str = Form("kan_Knda", description="Target language code"),  # Default added
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if src_lang not in language_options or tgt_lang not in language_options:
        raise HTTPException(status_code=400, detail=f"Invalid language codes: src={src_lang}, tgt={tgt_lang}")

    logger.debug(f"Processing indic custom prompt PDF: page_number={page_number}, model={model}, src_lang={src_lang}, tgt_lang={tgt_lang}")

    try:
        text_response = await extract_text_from_pdf(file, page_number, model)
        extracted_text = json.loads(text_response.body.decode("utf-8"))["page_content"]

        client = get_openai_client(model)
        custom_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}\n\n{extracted_text}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        response = custom_response.choices[0].message.content

        translation_payload = {
            "sentences": [response],
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        translation_response = requests.post(
            f"{translation_api_url}/translate?src_lang={src_lang}&tgt_lang={tgt_lang}",
            json=translation_payload,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        translation_response.raise_for_status()
        translation_result = translation_response.json()
        translated_response = translation_result["translations"][0]

        logger.debug(f"Indic custom prompt PDF completed: response_length={len(response)}")
        return JSONResponse(content={
            "original_text": extracted_text,
            "response": response,
            "translated_response": translated_response,
            "processed_page": page_number
        })

    except requests.RequestException as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Indic Summarize PDF Endpoint (Updated)
@app.post("/indic-summarize-pdf",
          summary="Summarize and Translate a PDF Page",
          description="Summarize a PDF page and translate the summary into the target language.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text, summary, and translated summary"},
              400: {"description": "Invalid PDF, page number, or language codes"},
              500: {"description": "OCR, summarization, or translation error"}
          })
async def indic_summarize_pdf(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(1, description="Page number to extract text from (1-based indexing)", ge=1),
    src_lang: str = Form("eng_Latn", description="Source language code"),  # Default added
    tgt_lang: str = Form("kan_Knda", description="Target language code"),  # Default added
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    logger.info(f"Processing indic summarize PDF: page_number={page_number}, model={model}, src_lang={src_lang}, tgt_lang={tgt_lang}  and file={file.filename}")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    if src_lang not in language_options or tgt_lang not in language_options:
        raise HTTPException(status_code=400, detail=f"Invalid language codes: src={src_lang}, tgt={tgt_lang}")

    logger.info(f"Processing indic summarize PDF: page_number={page_number}, model={model}, src_lang={src_lang}, tgt_lang={tgt_lang}")

    try:
        text_response = await indic_extract_text_from_pdf(file, page_number, model)
        extracted_text = json.loads(text_response.body.decode("utf-8"))["page_content"]

        client = get_openai_client(model)
        summary_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"Summarize the following text in 3-5 sentences:\n\n{extracted_text}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        summary = summary_response.choices[0].message.content

        translation_payload = {
            "sentences": [summary],
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        translation_response = requests.post(
            f"{translation_api_url}/translate?src_lang={src_lang}&tgt_lang={tgt_lang}",
            json=translation_payload,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        translation_response.raise_for_status()
        translation_result = translation_response.json()
        translated_summary = translation_result["translations"][0]

        logger.info(f"Indic summarize PDF completed: summary_length={len(summary)}")
        return JSONResponse(content={
            "original_text": extracted_text,
            "summary": summary,
            "translated_summary": translated_summary,
            "processed_page": page_number
        })

    except requests.RequestException as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/indic-extract-text/")
async def indic_extract_text_from_pdf(
    file: UploadFile = File(...),
    page_number: int = Body(1, embed=True, ge=1),
    src_lang: str = Body("eng_Latn", embed=True),
    tgt_lang: str = Body("kan_Knda", embed=True),
    model: str = Body("gemma3", embed=True)
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        try:
            image_base64 = render_pdf_to_base64png(temp_file_path, page_number, target_longest_image_dim=1024)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to render PDF page: {str(e)}")

        try:
            page_content = ocr_page_with_rolm(image_base64, model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        try:
            translation_payload = {
                "sentences": [page_content],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang
            }
            translation_response = requests.post(
                f"{translation_api_url}/translate?src_lang={src_lang}&tgt_lang={tgt_lang}",
                json=translation_payload,
                headers={"accept": "application/json", "Content-Type": "application/json"}
            )
            translation_response.raise_for_status()
            translation_result = translation_response.json()
            translated_content = translation_result["translations"][0]
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error translating: {str(e)}")

        os.remove(temp_file_path)
        return JSONResponse(content={
            "page_content": page_content,
            "translated_content": translated_content,
            "processed_page": page_number
        })

    except Exception as e:
        if 'temp_file_path' in locals():
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def extract_text_with_layout(pdf_path: str, page_number: int) -> List[dict]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                raise HTTPException(status_code=400, detail=f"Invalid page number: {page_number}")
            page = pdf.pages[page_number - 1]
            chars = page.chars
            words = []
            current_word = {"text": "", "x0": None, "y0": None, "x1": None, "y1": None, "fontname": None, "size": None}
            for char in chars:
                if current_word["text"] and (
                    char["x0"] > current_word["x1"] + 2 or
                    char["fontname"] != current_word["fontname"] or
                    char["size"] != current_word["size"]
                ):
                    words.append(current_word)
                    current_word = {"text": "", "x0": None, "y0": None, "x1": None, "y1": None, "fontname": None, "size": None}
                current_word["text"] += char["text"]
                current_word["x0"] = min(current_word["x0"] or char["x0"], char["x0"])
                current_word["y0"] = min(current_word["y0"] or char["y0"], char["y0"])
                current_word["x1"] = max(current_word["x1"] or char["x1"], char["x1"])
                current_word["y1"] = max(current_word["y1"] or char["y1"], char["y1"])
                current_word["fontname"] = char["fontname"]
                current_word["size"] = char["size"]
            if current_word["text"]:
                words.append(current_word)
            return words
    except Exception as e:
        logger.warning(f"pdfplumber failed: {str(e)}. Falling back to OCR.")
        return None

def generate_pdf_with_layout(text_segments: List[dict], output_path: str, page_width: float, page_height: float):
    try:
        c = canvas.Canvas(output_path, pagesize=(page_width, page_height))
        font_map = {"Times": "Times-Roman", "Helvetica": "Helvetica", "Courier": "Courier", "Arial": "Helvetica"}
        for segment in text_segments:
            text = segment["text"]
            x0 = segment["x0"]
            y0 = page_height - segment["y1"]
            fontname = segment["fontname"].split("-")[-1] if "-" in segment["fontname"] else segment["fontname"]
            fontname = font_map.get(fontname, "Helvetica")
            fontsize = segment["size"]
            c.setFont(fontname, fontsize)
            c.drawString(x0, y0, text)
        c.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.post("/pdf-recreation/extract-text/")
async def pdf_recreation_extract_text(
    file: UploadFile = File(...),
    page_number: int = Body(1, embed=True, ge=1),
    model: str = Body("gemma3", embed=True)
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        text_segments = extract_text_with_layout(temp_file_path, page_number)
        if not text_segments:
            try:
                image_base64 = render_pdf_to_base64png(temp_file_path, page_number, target_longest_image_dim=512)
                page_content = ocr_page_with_rolm(image_base64, model)
                text_segments = [{
                    "text": page_content,
                    "x0": 50,
                    "y0": 50,
                    "x1": 550,
                    "y1": 750,
                    "fontname": "Helvetica",
                    "size": 12
                }]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        with pdfplumber.open(temp_file_path) as pdf:
            page = pdf.pages[page_number - 1]
            page_width, page_height = page.width, page.height

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            output_pdf_path = temp_pdf.name
            generate_pdf_with_layout(text_segments, output_pdf_path, page_width, page_height)

        os.remove(temp_file_path)
        return FileResponse(
            path=output_pdf_path,
            filename=f"extracted_text_page_{page_number}.pdf",
            media_type="application/pdf",
            background=lambda: os.remove(output_pdf_path)
        )

    except Exception as e:
        if 'temp_file_path' in locals():
            os.remove(temp_file_path)
        if 'output_pdf_path' in locals():
            os.remove(output_pdf_path)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/pdf-recreation/indic-extract-text/")
async def pdf_recreation_indic_extract_text(
    file: UploadFile = File(...),
    page_number: int = Body(1, embed=True, ge=1),
    src_lang: str = Body("eng_Latn", embed=True),
    tgt_lang: str = Body("kan_Knda", embed=True),
    model: str = Body("gemma3", embed=True)
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported.")
        if src_lang not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid source language: {src_lang}")
        if tgt_lang not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid target language: {tgt_lang}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        text_segments = extract_text_with_layout(temp_file_path, page_number)
        if not text_segments:
            try:
                image_base64 = render_pdf_to_base64png(temp_file_path, page_number, target_longest_image_dim=512)
                page_content = ocr_page_with_rolm(image_base64, model)
                text_segments = [{
                    "text": page_content,
                    "x0": 50,
                    "y0": 50,
                    "x1": 550,
                    "y1": 750,
                    "fontname": "Helvetica",
                    "size": 12
                }]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        translated_segments = []
        for segment in text_segments:
            try:
                translation_payload = {
                    "sentences": [segment["text"]],
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang
                }
                translation_response = requests.post(
                    f"{translation_api_url}/translate?src_lang={src_lang}&tgt_lang={tgt_lang}",
                    json=translation_payload,
                    headers={"accept": "application/json", "Content-Type": "application/json"}
                )
                translation_response.raise_for_status()
                translated_text = translation_response.json()["translations"][0]
                translated_segments.append({
                    "text": translated_text,
                    "x0": segment["x0"],
                    "y0": segment["y0"],
                    "x1": segment["x1"],
                    "y1": segment["y1"],
                    "fontname": segment["fontname"],
                    "size": segment["size"]
                })
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

        with pdfplumber.open(temp_file_path) as pdf:
            page = pdf.pages[page_number - 1]
            page_width, page_height = page.width, page.height

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            output_pdf_path = temp_pdf.name
            generate_pdf_with_layout(translated_segments, output_pdf_path, page_width, page_height)

        os.remove(temp_file_path)
        return FileResponse(
            path=output_pdf_path,
            filename=f"translated_text_page_{page_number}.pdf",
            media_type="application/pdf",
            background=lambda: os.remove(output_pdf_path)
        )

    except Exception as e:
        if 'temp_file_path' in locals():
            os.remove(temp_file_path)
        if 'output_pdf_path' in locals():
            os.remove(output_pdf_path)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def generate_pdf_from_text(text: str, output_path: str):
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        style = styles["Normal"]
        story = []
        paragraphs = text.split("\n")
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), style))
                story.append(Spacer(1, 12))
        doc.build(story)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.post("/pdf-recreation/ocr")
async def pdf_recreation_ocr_image(
    file: UploadFile = File(...),
    model: str = Body("gemma3", embed=True)
):
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images supported")

    try:
        image_bytes = await file.read()
        image = BytesIO(image_bytes)
        img_base64 = encode_image(image)
        text = ocr_page_with_rolm(img_base64, model)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            output_pdf_path = temp_pdf.name
            generate_pdf_from_text(text, output_pdf_path)

        return FileResponse(
            path=output_pdf_path,
            filename="extracted_text_image.pdf",
            media_type="application/pdf",
            background=lambda: os.remove(output_pdf_path)
        )

    except Exception as e:
        if 'output_pdf_path' in locals():
            os.remove(output_pdf_path)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Indic Visual Query Endpoint
@app.post("/indic-visual-query/",
          summary="Indic Visual Query with Image",
          description="Extract text from a PNG image using OCR, optionally process it with a custom prompt, and translate the result.",
          tags=["Chat"],
          responses={
              200: {"description": "Extracted text and translated response"},
              400: {"description": "Invalid image, prompt, or language codes"},
              500: {"description": "OCR or translation error"}
          })
async def indic_visual_query(
    request: Request,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None, description="Optional custom prompt to process the extracted text"),
    source_language: str = Form("eng_Latn", description="Source language code"),
    tgt_language: str = Form("kan_Knda", description="Target language code"),
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images supported")
    if source_language not in language_options or tgt_language not in language_options:
        raise HTTPException(status_code=400, detail=f"Invalid language codes: {source_language}, {tgt_language}")

    logger.debug(f"Processing indic visual query: model={model}, source_language={source_language}, tgt_language={tgt_language}")

    try:
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
            text_to_translate = response
        elif prompt and not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        translation_payload = {
            "sentences": [text_to_translate],
            "src_lang": source_language,
            "tgt_lang": tgt_language
        }
        translation_response = requests.post(
            f"{translation_api_url}/translate?src_lang={source_language}&tgt_lang={tgt_language}",
            json=translation_payload,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        translation_response.raise_for_status()
        translation_result = translation_response.json()
        translated_response = translation_result["translations"][0]

        result = {
            "extracted_text": extracted_text,
            "translated_response": translated_response
        }
        if response:
            result["response"] = response

        logger.debug(f"Indic visual query successful: extracted_text_length={len(extracted_text)}")
        return JSONResponse(content=result)

    except requests.RequestException as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
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

@app.post("/indic_chat")
async def indic_chat(
    request: Request,
    chat_request: ChatRequest,
    settings=Depends(get_settings)
):
    if not chat_request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    logger.debug(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}, model: {chat_request.model}")

    try:
        if chat_request.src_lang not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid source language: {chat_request.src_lang}")
        if chat_request.tgt_lang not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid target language: {chat_request.tgt_lang}")

        prompt_to_process = chat_request.prompt
        if chat_request.src_lang != "eng_Latn":
            translation_payload = {
                "sentences": [chat_request.prompt],
                "src_lang": chat_request.src_lang,
                "tgt_lang": "eng_Latn"
            }
            translation_response = requests.post(
                f"{translation_api_url}/translate?src_lang={chat_request.src_lang}&tgt_lang=eng_Latn",
                json=translation_payload,
                headers={"accept": "application/json", "Content-Type": "application/json"}
            )
            translation_response.raise_for_status()
            translation_result = translation_response.json()
            prompt_to_process = translation_result["translations"][0]
            logger.debug(f"Translated prompt to English: {prompt_to_process}")

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

        final_response = generated_response
        if chat_request.tgt_lang != "eng_Latn":
            translation_payload = {
                "sentences": [generated_response],
                "src_lang": "eng_Latn",
                "tgt_lang": chat_request.tgt_lang
            }
            translation_response = requests.post(
                f"{translation_api_url}/translate?src_lang=eng_Latn&tgt_lang={chat_request.tgt_lang}",
                json=translation_payload,
                headers={"accept": "application/json", "Content-Type": "application/json"}
            )
            translation_response.raise_for_status()
            translation_result = translation_response.json()
            final_response = translation_result["translations"][0]
            logger.debug(f"Translated response to {chat_request.tgt_lang}: {final_response}")

        return JSONResponse(content={"response": final_response})

    except requests.exceptions.RequestException as e:
        logger.error(f"Translation API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation API error: {str(e)}")
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