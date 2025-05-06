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
from openai import OpenAI
from PIL import Image
from pypdf import PdfReader
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from logging_config import logger

# Initialize FastAPI app with enhanced metadata
app = FastAPI(
    title="dwani.ai - Document server API",
    description=(
        "API for extracting text from PDF pages and PNG images using RolmOCR, with functionality to "
        "summarize PDF content, process it with custom prompts, translate summaries to Kannada, or translate "
        "extracted Kannada text to English. Supports text extraction from a single PDF page, OCR for PNG images, "
        "summarization of a single PDF page, custom prompt-based processing, translation between Kannada and English, "
        "and generation of translated PDFs."
    ),
    version="1.0.0"
)

# Initialize OpenAI client for RolmOCR
openai_client = OpenAI(api_key="123", base_url="http://0.0.0.0:7863/v1")

rolm_model = "google/gemma-3-4b-it"  # for A100 only

# Translation API endpoint
translation_api_url = "http://0.0.0.0:7862/v1/translate"

# Supported language codes
language_options = [
    "kan_Knda",  # Kannada
    "eng_Latn",  # English
    "hin_Deva",  # Hindi
    "tam_Taml",  # Tamil
    "tel_Telu",  # Telugu
]

# Pydantic models for request parameters
class ExtractTextRequest(BaseModel):
    page_number: int = Field(
        default=1,
        description="The page number to extract text from (1-based indexing). Must be a positive integer.",
        ge=1,
        example=1
    )

class SummarizePDFRequest(BaseModel):
    page_number: int = Field(
        default=1,
        description="The page number to extract and summarize (1-based indexing). Must be a positive integer.",
        ge=1,
        example=1
    )

class CustomPromptPDFRequest(BaseModel):
    page_number: int = Field(
        default=1,
        description="The page number to extract and process with the custom prompt (1-based indexing). Must be a positive integer.",
        ge=1,
        example=1
    )
    prompt: str = Field(
        description="The custom prompt to process the extracted text. For example, 'Summarize in 2 sentences' or 'List key points'.",
        example="Summarize the text in 2 sentences."
    )

class CustomPromptPDFRequestExtended(CustomPromptPDFRequest):
    source_language: str = Field(
        default="eng_Latn",
        description="Source language code for translation (e.g., 'eng_Latn' for English).",
        examples=["eng_Latn", "hin_Deva"],
        enum=language_options
    )
    target_language: str = Field(
        default="kan_Knda",
        description="Target language code for translation (e.g., 'kan_Knda' for Kannada).",
        examples=["kan_Knda", "tam_Taml"],
        enum=language_options
    )

class GenerateTranslatedPDFRequest(BaseModel):
    page_number: int = Field(
        default=1,
        description="The page number to extract and translate (1-based indexing). Must be a positive integer.",
        ge=1,
        example=1
    )
    source_language: str = Field(
        default="eng_Latn",
        description="Source language code for translation (e.g., 'eng_Latn' for English).",
        examples=["eng_Latn", "hin_Deva"],
        enum=language_options
    )
    target_language: str = Field(
        default="kan_Knda",
        description="Target language code for translation (e.g., 'kan_Knda' for Kannada).",
        examples=["kan_Knda", "tam_Taml"],
        enum=language_options
    )

def encode_image(image: BytesIO) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image.read()).decode("utf-8")

def ocr_page_with_rolm(img_base64: str) -> str:
    """Perform OCR on the provided base64 image using RolmOCR via OpenAI API."""
    try:
        response = openai_client.chat.completions.create(
            model=rolm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                        {
                            "type": "text",
                            "text": "Return the plain text representation of this document as if you were reading it naturally.\n",
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RolmOCR processing failed: {str(e)}")

def generate_pdf_from_text(text: str, output_path: str):
    """Generate a PDF from the given text using reportlab."""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        style = styles["Normal"]
        story = []

        # Split text into paragraphs (simple split by newlines)
        paragraphs = text.split("\n")
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), style))
                story.append(Spacer(1, 12))

        doc.build(story)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.get(
    "/",
    summary="Health check endpoint",
    description="Returns a simple message to confirm that the Combined OCR API is running.",
    response_description="A JSON object with a confirmation message."
)
async def root():
    return {"message": "Combined OCR API is running"}

@app.post(
    "/extract-text/",
    response_model=dict,
    summary="Extract text from a PDF page",
    description=(
        "Extracts text from a specific page of a PDF file using RolmOCR. The page is rendered as an image, "
        "and OCR is performed to extract the text content."
    ),
    response_description="A JSON object containing the extracted text from the specified page."
)
async def extract_text_from_pdf(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(
        default=1,
        embed=True,
        description=ExtractTextRequest.model_fields["page_number"].description,
        ge=1,
        example=1
    )
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        try:
            image_base64 = render_pdf_to_base64png(
                temp_file_path, page_number, target_longest_image_dim=1024
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to render PDF page: {str(e)}")

        try:
            page_content = ocr_page_with_rolm(image_base64)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        os.remove(temp_file_path)
        return JSONResponse(content={"page_content": page_content})

    except Exception as e:
        if 'temp_file_path' in locals():
            try:
                os.remove(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post(
    "/ocr",
    response_model=dict,
    summary="Extract text from a PNG image",
    description=(
        "Performs OCR on a PNG image using RolmOCR to extract text content. The image is encoded to base64 "
        "and processed via the OpenAI API."
    ),
    response_description="A JSON object containing the extracted text from the image."
)
async def ocr_image(file: UploadFile = File(..., description="The PNG image to process. Must be a valid PNG.")):
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images are supported")

    try:
        image_bytes = await file.read()
        image = BytesIO(image_bytes)
        img_base64 = encode_image(image)
        text = ocr_page_with_rolm(img_base64)
        return {"extracted_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post(
    "/summarize-pdf",
    response_model=dict,
    summary="Extract and summarize text from a single PDF page",
    description=(
        "Extracts text from a specified page of a PDF file using RolmOCR and generates a 3-5 sentence summary "
        "using chat completion."
    ),
    response_description=(
        "A JSON object containing the extracted text, a summary, and the processed page number."
    )
)
async def summarize_pdf(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(
        default=1,
        embed=True,
        description=SummarizePDFRequest.model_fields["page_number"].description,
        ge=1,
        example=1
    )
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        text_response = await extract_text_from_pdf(file, page_number)
        extracted_text = text_response.body.decode("utf-8")
        extracted_json = json.loads(extracted_text)
        extracted_text = extracted_json["page_content"]

        summary_response = openai_client.chat.completions.create(
            model=rolm_model,
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize the following text in 3-5 sentences:\n\n{extracted_text}"
                }
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
        raise HTTPException(status_code=500, detail=f"Error processing PDF or generating summary: {str(e)}")

@app.post(
    "/custom-prompt-pdf",
    response_model=dict,
    summary="Extract and process text from a single PDF page with a custom prompt",
    description=(
        "Extracts text from a specified page of a PDF file using RolmOCR and processes it with a user-provided prompt "
        "using chat completion. The custom prompt allows flexible text processing, such as summarization, key point extraction, or translation."
    ),
    response_description=(
        "A JSON object containing the extracted text, the response generated by the custom prompt, and the processed page number."
    )
)
async def custom_prompt_pdf(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(
        default=1,
        embed=True,
        description=CustomPromptPDFRequest.model_fields["page_number"].description,
        ge=1,
        example=1
    ),
    prompt: str = Body(
        ...,
        embed=True,
        description=CustomPromptPDFRequest.model_fields["prompt"].description,
        examples=CustomPromptPDFRequest.model_fields["prompt"].examples
    )
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        text_response = await extract_text_from_pdf(file, page_number)
        extracted_text = text_response.body.decode("utf-8")
        extracted_json = json.loads(extracted_text)
        extracted_text = extracted_json["page_content"]

        custom_response = openai_client.chat.completions.create(
            model=rolm_model,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n{extracted_text}"
                }
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
        raise HTTPException(status_code=500, detail=f"Error processing PDF or generating response: {str(e)}")

@app.post(
    "/indic-custom-prompt-pdf",
    response_model=dict,
    summary="Extract, process, and translate text from a single PDF page with a custom prompt",
    description=(
        "Extracts text from a specified page of a PDF file using RolmOCR, processes it with a user-provided prompt "
        "using chat completion, and translates the response into a target language."
    ),
    response_description=(
        "A JSON object containing the extracted text, the response generated by the custom prompt, the translated response, "
        "and the processed page number."
    )
)
async def indic_custom_prompt_pdf(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(
        default=1,
        embed=True,
        description=CustomPromptPDFRequestExtended.model_fields["page_number"].description,
        ge=1,
        example=1
    ),
    prompt: str = Body(
        ...,
        embed=True,
        description=CustomPromptPDFRequestExtended.model_fields["prompt"].description,
        examples=CustomPromptPDFRequestExtended.model_fields["prompt"].examples
    ),
    source_language: str = Body(
        default="eng_Latn",
        embed=True,
        description=CustomPromptPDFRequestExtended.model_fields["source_language"].description,
        examples=["eng_Latn", "hin_Deva"]
    ),
    target_language: str = Body(
        default="kan_Knda",
        embed=True,
        description=CustomPromptPDFRequestExtended.model_fields["target_language"].description,
        examples=["kan_Knda", "tam_Taml"]
    )
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        if source_language not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid source language. Choose from: {', '.join(language_options)}")
        if target_language not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid target language. Choose from: {', '.join(language_options)}")

        text_response = await extract_text_from_pdf(file, page_number)
        extracted_text = text_response.body.decode("utf-8")
        extracted_json = json.loads(extracted_text)
        extracted_text = extracted_json["page_content"]

        custom_response = openai_client.chat.completions.create(
            model=rolm_model,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n{extracted_text}"
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
        response = custom_response.choices[0].message.content

        translation_payload = {
            "sentences": [response],
            "src_lang": source_language,
            "tgt_lang": target_language
        }
        translation_response = requests.post(
            translation_api_url,
            json=translation_payload,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        translation_response.raise_for_status()
        translation_result = translation_response.json()
        translated_response = translation_result["translations"][0]

        customPDFResponse  = JSONResponse(content={
            "original_text": extracted_text,
            "response": response,
            "translated_response": translated_response,
            "processed_page": page_number
        })

        logger.debug(customPDFResponse)
        return customPDFResponse

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error querying translation API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF or generating response: {str(e)}")

@app.post(
    "/indic-summarize-pdf",
    response_model=dict,
    summary="Extract, summarize, and translate text from a single PDF page",
    description=(
        "Extracts text from a specified page of a PDF file using RolmOCR, generates a 3-5 sentence summary "
        "using chat completion, and translates the summary into a target language."
    ),
    response_description=(
        "A JSON object containing the extracted text, summary, translated summary, and processed page number."
    )
)
async def indic_summarize_pdf(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(
        default=1,
        embed=True,
        description="The page number to extract and summarize (1-based indexing).",
        ge=1,
        example=1
    ),
    src_lang: str = Body(
        default="eng_Latn",
        embed=True,
        description="Source language for translation (e.g., 'eng_Latn' for English).",
        example="eng_Latn"
    ),
    tgt_lang: str = Body(
        default="kan_Knda",
        embed=True,
        description="Target language for translation (e.g., 'kan_Knda' for Kannada).",
        example="kan_Knda"
    )
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        text_response = await extract_text_from_pdf(file, page_number)
        extracted_text = text_response.body.decode("utf-8")
        extracted_json = json.loads(extracted_text)
        extracted_text = extracted_json["page_content"]

        summary_response = openai_client.chat.completions.create(
            model=rolm_model,
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize the following text in 3-5 sentences:\n\n{extracted_text}"
                }
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
            translation_api_url,
            json=translation_payload,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        translation_response.raise_for_status()
        translation_result = translation_response.json()
        translated_summary = translation_result["translations"][0]

        return JSONResponse(content={
            "original_text": extracted_text,
            "summary": summary,
            "translated_summary": translated_summary,
            "processed_page": page_number
        })

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error querying translation API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF or generating summary: {str(e)}")

@app.post(
    "/indic-extract-text/",
    response_model=dict,
    summary="Extract and translate text from a PDF page",
    description=(
        "Extracts text from a specific page of a PDF file using RolmOCR and translates it into a target language. "
        "The page is rendered as an image, and OCR is performed to extract the text content."
    ),
    response_description="A JSON object containing the extracted text, translated text, and processed page number."
)
async def indic_extract_text_from_pdf(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(
        default=1,
        embed=True,
        description="The page number to extract text from (1-based indexing).",
        ge=1,
        example=1
    ),
    src_lang: str = Body(
        default="eng_Latn",
        embed=True,
        description="Source language for translation (e.g., 'eng_Latn' for English).",
        example="eng_Latn"
    ),
    tgt_lang: str = Body(
        default="kan_Knda",
        embed=True,
        description="Target language for translation (e.g., 'kan_Knda' for Kannada).",
        example="kan_Knda"
    )
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        try:
            image_base64 = render_pdf_to_base64png(
                temp_file_path, page_number, target_longest_image_dim=1024
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to render PDF page: {str(e)}")

        try:
            page_content = ocr_page_with_rolm(image_base64)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        try:
            translation_payload = {
                "sentences": [page_content],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang
            }
            translation_response = requests.post(
                translation_api_url,
                json=translation_payload,
                headers={"accept": "application/json", "Content-Type": "application/json"}
            )
            translation_response.raise_for_status()
            translation_result = translation_response.json()
            translated_content = translation_result["translations"][0]
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error querying translation API: {str(e)}")

        try:
            os.remove(temp_file_path)
        except:
            pass

        return JSONResponse(content={
            "page_content": page_content,
            "translated_content": translated_content,
            "processed_page": page_number
        })

    except Exception as e:
        if 'temp_file_path' in locals():
            try:
                os.remove(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post(
    "/indic-generate-translated-pdf/",
    summary="Extract, translate, and generate a PDF from a single PDF page",
    description=(
        "Extracts text from a specified page of a PDF file using RolmOCR, translates it into a target language, "
        "and generates a new PDF containing the translated text. The generated PDF is returned as a downloadable file."
    ),
    response_description="A downloadable PDF file containing the translated text from a single page."
)
async def indic_generate_translated_pdf(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(
        default=1,
        embed=True,
        description=GenerateTranslatedPDFRequest.model_fields["page_number"].description,
        ge=1,
        example=1
    ),
    src_lang: str = Body(
        GenerateTranslatedPDFRequest.model_fields["source_language"].default,
        embed=True,
        description=GenerateTranslatedPDFRequest.model_fields["source_language"].description,
        examples=GenerateTranslatedPDFRequest.model_fields["source_language"].examples
    ),
    tgt_lang: str = Body(
        default=GenerateTranslatedPDFRequest.model_fields["target_language"].default,
        embed=True,
        description=GenerateTranslatedPDFRequest.model_fields["target_language"].description,
        examples=GenerateTranslatedPDFRequest.model_fields["target_language"].examples
    )
):
    """
    Extract text from a specified page of a PDF file, translate it, and generate a new PDF with the translated text.

    Args:
        file (UploadFile): The PDF file to process.
        page_number (int): The page number to extract and translate (1-based indexing). Defaults to 1.
        src_lang (str): Source language for translation (e.g., 'eng_Latn'). Defaults to English.
        tgt_lang (str): Target language for translation (e.g., 'kan_Knda'). Defaults to Kannada.

    Returns:
        FileResponse: A downloadable PDF file containing the translated text.

    Raises:
        HTTPException: If the file is not a PDF, the page number is invalid, language codes are invalid, or processing fails.

    Example:
        Upload a PDF file, specify page_number=1, src_lang="eng_Latn", tgt_lang="kan_Knda".
        Response: A downloadable PDF file named "translated_page_1.pdf" containing the translated text.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Validate language codes
        if src_lang not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid source language. Choose from: {', '.join(language_options)}")
        if tgt_lang not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid target language. Choose from: {', '.join(language_options)}")

        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Render the specified page to an image
        try:
            image_base64 = render_pdf_to_base64png(
                temp_file_path, page_number, target_longest_image_dim=1024
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to render PDF page: {str(e)}")

        # Perform OCR using RolmOCR
        try:
            page_content = ocr_page_with_rolm(image_base64)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        # Translate the extracted text
        try:
            translation_payload = {
                "sentences": [page_content],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang
            }
            translation_response = requests.post(
                translation_api_url,
                json=translation_payload,
                headers={"accept": "application/json", "Content-Type": "application/json"}
            )
            translation_response.raise_for_status()
            translation_result = translation_response.json()
            translated_content = translation_result["translations"][0]
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error querying translation API: {str(e)}")

        # Generate PDF with translated text
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                output_pdf_path = temp_pdf.name
                generate_pdf_from_text(translated_content, output_pdf_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

        # Clean up temporary input file
        try:
            os.remove(temp_file_path)
        except:
            pass

        # Return the generated PDF as a downloadable file
        return FileResponse(
            path=output_pdf_path,
            filename=f"translated_page_{page_number}.pdf",
            media_type="application/pdf",
            background=lambda: os.remove(output_pdf_path)  # Clean up after response
        )

    except Exception as e:
        # Clean up temporary files in case of error
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