import argparse
import base64
import json
import os
import tempfile
import requests
from io import BytesIO
from typing import List, Union
from time import time

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from pypdf import PdfReader
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import pdfplumber

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from logging_config import logger
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI(
    title="dwani.ai - Document server API",
    description=(
        "API for extracting text from PDF pages and PNG images using RolmOCR, with functionality to "
        "summarize, process with custom prompts, translate, and generate PDFs with preserved formatting."
    ),
    version="1.0.0"
)

# Initialize OpenAI client for RolmOCR
openai_client = OpenAI(api_key="123", base_url="http://0.0.0.0:7863/v1")
rolm_model = "google/gemma-3-4b-it"
translation_api_url = "http://0.0.0.0:7862/v1/translate"
language_options = ["kan_Knda", "eng_Latn", "hin_Deva", "tam_Taml", "tel_Telu"]

# Pydantic models
class ExtractTextRequest(BaseModel):
    page_number: int = Field(default=1, description="Page number to extract (1-based).", ge=1, example=1)

class GenerateTranslatedPDFRequest(BaseModel):
    page_number: int = Field(default=1, description="Page number to extract and translate.", ge=1, example=1)
    source_language: str = Field(default="eng_Latn", description="Source language code.", examples=["eng_Latn"], enum=language_options)
    target_language: str = Field(default="kan_Knda", description="Target language code.", examples=["kan_Knda"], enum=language_options)

def encode_image(image: BytesIO) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image.read()).decode("utf-8")

def ocr_page_with_rolm(img_base64: str) -> str:
    """Perform OCR on the provided base64 image using RolmOCR."""
    try:
        response = openai_client.chat.completions.create(
            model=rolm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        {"type": "text", "text": "Return the plain text representation of this document as if you were reading it naturally.\n"},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RolmOCR processing failed: {str(e)}")

def extract_text_with_layout(pdf_path: str, page_number: int) -> List[dict]:
    """
    Extract text and layout information from a PDF page using pdfplumber.
    Returns a list of dicts with text, bounding box, and font information.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                raise HTTPException(status_code=400, detail=f"Invalid page number: {page_number}")
            page = pdf.pages[page_number - 1]
            chars = page.chars  # Extract individual characters with layout info
            # Group characters into words/lines for simplicity
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
    """
    Generate a PDF with text placed at original coordinates and approximated fonts.
    """
    try:
        c = canvas.Canvas(output_path, pagesize=(page_width, page_height))
        # Font mapping: Approximate original fonts with reportlab defaults
        font_map = {
            "Times": "Times-Roman",
            "Helvetica": "Helvetica",
            "Courier": "Courier",
            "Arial": "Helvetica",  # Fallback
        }
        for segment in text_segments:
            text = segment["text"]
            x0 = segment["x0"]
            y0 = page_height - segment["y1"]  # Flip y-coordinate (PDF origin is bottom-left)
            fontname = segment["fontname"].split("-")[-1] if "-" in segment["fontname"] else segment["fontname"]
            fontname = font_map.get(fontname, "Helvetica")
            fontsize = segment["size"]
            c.setFont(fontname, fontsize)
            c.drawString(x0, y0, text)
        c.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.post(
    "/pdf-recreation/extract-text/",
    tags=["pdf-recreation"],
    summary="Extract text from a PDF page and generate a PDF with original formatting",
    description="Extracts text from a PDF page and generates a PDF preserving the original layout and styling."
)
async def pdf_recreation_extract_text(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(default=1, embed=True, description="Page number to extract (1-based).", ge=1, example=1)
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Extract text and layout with pdfplumber
        text_segments = extract_text_with_layout(temp_file_path, page_number)

        if not text_segments:
            # Fallback to OCR for scanned PDFs
            try:
                image_base64 = render_pdf_to_base64png(temp_file_path, page_number, target_longest_image_dim=512)
                page_content = ocr_page_with_rolm(image_base64)
                # Approximate layout (basic: place text at top-left)
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

        # Get page dimensions
        with pdfplumber.open(temp_file_path) as pdf:
            page = pdf.pages[page_number - 1]
            page_width, page_height = page.width, page.height

        # Generate PDF with original layout
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
            try:
                os.remove(temp_file_path)
            except:
                pass
        if 'output_pdf_path' in locals():
            try:
                os.remove(output_pdf_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post(
    "/pdf-recreation/indic-extract-text/",
    tags=["pdf-recreation"],
    summary="Extract and translate text from a PDF page and generate a PDF with original formatting",
    description="Extracts text from a PDF page, translates it, and generates a PDF preserving the original layout."
)
async def pdf_recreation_indic_extract_text(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(default=1, embed=True, description="Page number to extract (1-based).", ge=1, example=1),
    src_lang: str = Body(default="eng_Latn", embed=True, description="Source language code.", example="eng_Latn"),
    tgt_lang: str = Body(default="kan_Knda", embed=True, description="Target language code.", example="kan_Knda")
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        if src_lang not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid source language. Choose from: {', '.join(language_options)}")
        if tgt_lang not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid target language. Choose from: {', '.join(language_options)}")

        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Extract text and layout with pdfplumber
        text_segments = extract_text_with_layout(temp_file_path, page_number)
        if not text_segments:
            # Fallback to OCR for scanned PDFs
            try:
                image_base64 = render_pdf_to_base64png(temp_file_path, page_number, target_longest_image_dim=512)
                page_content = ocr_page_with_rolm(image_base64)
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

        # Translate text segments
        translated_segments = []
        for segment in text_segments:
            try:
                translation_payload = {
                    "sentences": [segment["text"]],
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang
                }
                translation_response = requests.post(
                    translation_api_url,
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

        # Get page dimensions
        with pdfplumber.open(temp_file_path) as pdf:
            page = pdf.pages[page_number - 1]
            page_width, page_height = page.width, page.height

        # Generate PDF with translated text and original layout
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
            try:
                os.remove(temp_file_path)
            except:
                pass
        if 'output_pdf_path' in locals():
            try:
                os.remove(output_pdf_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Placeholder for other endpoints (unchanged)
@app.get("/")
async def root():
    return {"message": "Combined OCR API is running"}

# Add Timing Middleware
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

    uvicorn.run(app=app, host=args.host, port=args.port)