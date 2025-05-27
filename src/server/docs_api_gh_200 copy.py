from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI
import base64
import json
from io import BytesIO
from PIL import Image
import tempfile
import os
import requests
from typing import List, Union
from pypdf import PdfReader
from pydantic import BaseModel, Field
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
import argparse
import uvicorn
from time import time
from logging_config import logger

import argparse
import base64
import json
import os
import tempfile
import requests
from io import BytesIO
from typing import List, Union
from time import time

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import pdfplumber

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from openai import OpenAI


# Initialize FastAPI app with enhanced metadata
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


llm_base_url = os.getenv("DWANI_API_BASE_URL_LLM")


# Initialize OpenAI client for RolmOCR
#openai_client = OpenAI(api_key="123", base_url=llm_base_url)
openai_client = OpenAI(api_key="123", base_url="http://0.0.0.0:7860/v1")


#translation_api_url = os.getenv("DWANI_API_BASE_URL_TRANSLATE")


translation_api_url = "http://0.0.0.0:7862/v1/translate"


#rolm_model = "google/gemma-3-12b-it"   # for H100 only
rolm_model = "google/gemma-3-4b-it"   # for A100 only

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

@app.get(
    "/",
    summary="Health check endpoint",
    description="Returns a simple message to confirm that the Combined OCR API is running.",
    response_description="A JSON object with a confirmation message."
)
async def root():
    """
    Root endpoint for health check.

    Returns:
        dict: A JSON object containing a message indicating the API status.

    Example:
        ```json
        {"message": "Combined OCR API is running"}
        ```
    """
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
    """
    Extract text from a specific page of a PDF file using RolmOCR.

    Args:
        file (UploadFile): The PDF file to process.
        page_number (int): The page number to extract text from (1-based indexing). Defaults to 1.

    Returns:
        JSONResponse: A dictionary containing:
            - page_content: The extracted text from the specified page.

    Raises:
        HTTPException: If the file is not a PDF, the page number is invalid, or processing fails.

    Example:
        ```json
        {"page_content": "Extracted text from the PDF page"}
        ```
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

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

        # Clean up temporary file
        os.remove(temp_file_path)

        return JSONResponse(content={"page_content": page_content})

    except Exception as e:
        # Clean up in case of error
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
    """
    Upload a PNG image and extract text using RolmOCR.

    Args:
        file (UploadFile): The PNG image to process.

    Returns:
        dict: A dictionary containing:
            - extracted_text: The text extracted from the image.

    Raises:
        HTTPException: If the file is not a PNG or processing fails.

    Example:
        ```json
        {"extracted_text": "Text extracted from the PNG image"}
        ```
    """
    # Validate file type
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images are supported")

    try:
        # Read image file
        image_bytes = await file.read()
        image = BytesIO(image_bytes)
        
        # Encode to base64
        img_base64 = encode_image(image)
        
        # Perform OCR
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
    """
    Extract text from a specified page of a PDF file and generate a summary using RolmOCR and chat completion.

    Args:
        file (UploadFile): The PDF file to process.
        page_number (int): The page number to extract and summarize (1-based indexing). Defaults to 1.

    Returns:
        JSONResponse: A dictionary containing:
            - original_text: Text extracted from the specified page.
            - summary: A 3-5 sentence summary of the extracted text.
            - processed_page: The page number processed.

    Raises:
        HTTPException: If the file is not a PDF, the page number is invalid, or processing fails.

    Example:
        ```json
        {
            "original_text": "Text from page 1",
            "summary": "The document discusses... [3-5 sentence summary]",
            "processed_page": 1
        }
        ```
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Extract text using existing endpoint logic
        text_response = await extract_text_from_pdf(file, page_number)
        extracted_text = text_response.body.decode("utf-8")
        extracted_json = json.loads(extracted_text)
        extracted_text = extracted_json["page_content"]

        # Generate summary using OpenAI chat completion
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
    """
    Extract text from a specified page of a PDF file and process it with a custom prompt using RolmOCR and chat completion.

    Args:
        file (UploadFile): The PDF file to process.
        page_number (int): The page number to extract and process (1-based indexing). Defaults to 1.
        prompt (str): The custom prompt to process the extracted text (e.g., "Summarize in 2 sentences" or "List key points").

    Returns:
        JSONResponse: A dictionary containing:
            - original_text: Text extracted from the specified page.
            - response: The output generated by the custom prompt.
            - processed_page: The page number processed.

    Raises:
        HTTPException: If the file is not a PDF, the page number is invalid, the prompt is empty, or processing fails.

    Example:
        ```json
        {
            "original_text": "Text from page 1",
            "response": "The text summarizes... [2-sentence summary]",
            "processed_page": 1
        }
        ```
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Validate prompt
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        # Extract text using existing endpoint logic
        text_response = await extract_text_from_pdf(file, page_number)
        extracted_text = text_response.body.decode("utf-8")
        extracted_json = json.loads(extracted_text)
        extracted_text = extracted_json["page_content"]

        # Process text with custom prompt using OpenAI chat completion
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



# Assuming these are defined elsewhere in your codebase


# Translation API endpoint (from reference)

# Supported language codes (based on reference)
language_options = [
    "kan_Knda",  # Kannada
    "eng_Latn",  # English
    "hin_Deva",  # Hindi
    "tam_Taml",  # Tamil
    "tel_Telu",  # Telugu
]

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

@app.post(
    "/indic-custom-prompt-pdf",
    response_model=dict,
    summary="Extract, process, and translate text from a single PDF page with a custom prompt",
    description=(
        "Extracts text from a specified page of a PDF file using RolmOCR, processes it with a user-provided prompt "
        "using chat completion, and translates the response into a target language. The custom prompt allows flexible "
        "text processing, such as summarization, key point extraction, or translation."
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
    """
    Extract text from a specified page of a PDF file, process it with a custom prompt, and translate the response.

    Args:
        file (UploadFile): The PDF file to process.
        page_number (int): The page number to extract and process (1-based indexing). Defaults to 1.
        prompt (str): The custom prompt to process the extracted text (e.g., "Summarize in 2 sentences").
        source_language (str): Source language code for translation (e.g., 'eng_Latn' for English).
        target_language (str): Target language code for translation (e.g., 'kan_Knda' for Kannada).

    Returns:
        JSONResponse: A dictionary containing:
            - original_text: Text extracted from the specified page.
            - response: The output generated by the custom prompt.
            - translated_response: The translated version of the response.
            - processed_page: The page number processed.

    Raises:
        HTTPException: If the file is not a PDF, the page number is invalid, the prompt is empty, 
                       the language codes are invalid, or processing/translation fails.

    Example:
        ```json
        {
            "original_text": "Text from page 1",
            "response": "The text summarizes... [2-sentence summary]",
            "translated_response": "Translated summary in Kannada",
            "processed_page": 1
        }
        ```
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Validate prompt
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        # Validate language codes
        if source_language not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid source language. Choose from: {', '.join(language_options)}")
        if target_language not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid target language. Choose from: {', '.join(language_options)}")

        # Extract text using existing endpoint logic
        text_response = await extract_text_from_pdf(file, page_number)
        extracted_text = text_response.body.decode("utf-8")
        extracted_json = json.loads(extracted_text)
        extracted_text = extracted_json["page_content"]

        # Process text with custom prompt using OpenAI chat completion
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

        # Translate the response
        translation_payload = {
            "sentences": [response],
            "src_lang": source_language,
            "tgt_lang": target_language
        }
        translation_response = requests.post(
            f"{translation_api_url}/translate?src_lang={source_language}&tgt_lang={target_language}",
            json=translation_payload,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        translation_response.raise_for_status()  # Raise exception for bad status codes
        translation_result = translation_response.json()
        translated_response = translation_result["translations"][0]

        customPDFResponse = JSONResponse(content={
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
    """
    Extract text from a specified page of a PDF file, generate a summary, and translate it.

    Args:
        file (UploadFile): The PDF file to process.
        page_number (int): The page number to extract and summarize (1-based indexing). Defaults to 1.
        src_lang (str): Source language for translation (e.g., 'eng_Latn'). Defaults to English.
        tgt_lang (str): Target language for translation (e.g., 'kan_Knda'). Defaults to Kannada.

    Returns:
        JSONResponse: A dictionary containing:
            - original_text: Text extracted from the specified page.
            - summary: A 3-5 sentence summary of the extracted text.
            - translated_summary: The summary translated into the target language.
            - processed_page: The page number processed.

    Raises:
        HTTPException: If the file is not a PDF, the page number is invalid, or processing/translation fails.

    Example:
        ```json
        {
            "original_text": "Text from page 1",
            "summary": "The document discusses... [3-5 sentence summary]",
            "translated_summary": "Translated summary in target language",
            "processed_page": 1
        }
        ```
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Extract text using existing endpoint logic
        text_response = await extract_text_from_pdf(file, page_number)
        extracted_text = text_response.body.decode("utf-8")
        extracted_json = json.loads(extracted_text)
        extracted_text = extracted_json["page_content"]

        # Generate summary using OpenAI chat completion
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

        # Translate the summary
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
    """
    Extract text from a specific page of a PDF file using RolmOCR and translate it.

    Args:
        file (UploadFile): The PDF file to process.
        page_number (int): The page number to extract text from (1-based indexing). Defaults to 1.
        src_lang (str): Source language for translation (e.g., 'eng_Latn'). Defaults to English.
        tgt_lang (str): Target language for translation (e.g., 'kan_Knda'). Defaults to Kannada.

    Returns:
        JSONResponse: A dictionary containing:
            - page_content: The extracted text from the specified page.
            - translated_content: The extracted text translated into the target language.
            - processed_page: The page number processed.

    Raises:
        HTTPException: If the file is not a PDF, the page number is invalid, or processing/translation fails.

    Example:
        ```json
        {
            "page_content": "Extracted text from the PDF page",
            "translated_content": "Translated text in target language",
            "processed_page": 1
        }
        ```
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

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
                f"{translation_api_url}/translate?src_lang={src_lang}&tgt_lang={tgt_lang}",
                json=translation_payload,
                headers={"accept": "application/json", "Content-Type": "application/json"}
            )
            translation_response.raise_for_status()
            translation_result = translation_response.json()
            translated_content = translation_result["translations"][0]
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error querying translation API: {str(e)}")

        # Clean up temporary file
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
        # Clean up in case of error
        if 'temp_file_path' in locals():
            try:
                os.remove(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




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


@app.post(
    "/pdf-recreation/ocr",
    tags=["pdf-recreation"],
    summary="Extract text from a PNG image and generate a PDF",
    description=(
        "Performs OCR on a PNG image using RolmOCR and generates a PDF containing the extracted text."
    ),
    response_description="A downloadable PDF file containing the extracted text from the image."
)
async def pdf_recreation_ocr_image(file: UploadFile = File(..., description="The PNG image to process. Must be a valid PNG.")):
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images are supported")

    try:
        image_bytes = await file.read()
        image = BytesIO(image_bytes)
        img_base64 = encode_image(image)
        text = ocr_page_with_rolm(img_base64)

        # Generate PDF with extracted text
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
            try:
                os.remove(output_pdf_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

from typing import Optional
class IndicVisualQueryRequest(BaseModel):
    prompt: Optional[str] = Field(
        default=None,
        description="Optional custom prompt to process the extracted text (e.g., 'Summarize in 2 sentences' or 'List key points'). If not provided, only extraction and translation are performed.",
        example="Summarize the text in 2 sentences."
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

@app.post(
    "/indic-visual-query/",
    response_model=dict,
    summary="Extract, process, and translate text from a PNG image",
    description=(
        "Extracts text from a PNG image using RolmOCR, optionally processes it with a custom prompt "
        "using chat completion, and translates the result into a target language. Supports flexible "
        "text processing such as summarization or key point extraction."
    ),
    response_description=(
        "A JSON object containing the extracted text, optional processed response (if a prompt is provided), "
        "and the translated response."
    )
)
async def indic_visual_query(
    file: UploadFile = File(..., description="The PNG image to process. Must be a valid PNG."),
    prompt: Optional[str] = Body(
        default=None,
        embed=True,
        description=IndicVisualQueryRequest.model_fields["prompt"].description,
        examples=IndicVisualQueryRequest.model_fields["prompt"].examples
    ),
    source_language: str = Body(
        default="eng_Latn",
        embed=True,
        description=IndicVisualQueryRequest.model_fields["source_language"].description,
        examples=["eng_Latn", "hin_Deva"]
    ),
    target_language: str = Body(
        default="kan_Knda",
        embed=True,
        description=IndicVisualQueryRequest.model_fields["target_language"].description,
        examples=["kan_Knda", "tam_Taml"]
    )
):
    """
    Extract text from a PNG image, optionally process it with a custom prompt, and translate the result.

    Args:
        file (UploadFile): The PNG image to process.
        prompt (Optional[str]): Optional custom prompt to process the extracted text (e.g., "Summarize in 2 sentences").
        source_language (str): Source language code for translation (e.g., 'eng_Latn' for English).
        target_language (str): Target language code for translation (e.g., 'kan_Knda' for Kannada).

    Returns:
        JSONResponse: A dictionary containing:
            - extracted_text: Text extracted from the image.
            - response (optional): The output generated by the custom prompt, if provided.
            - translated_response: The translated version of the response or extracted text.
    
    Raises:
        HTTPException: If the file is not a PNG, the prompt is empty (when provided), 
                       the language codes are invalid, or processing/translation fails.

    Example:
        ```json
        {
            "extracted_text": "Text extracted from the PNG image",
            "response": "The text summarizes... [2-sentence summary]",
            "translated_response": "Translated summary in Kannada"
        }
        ```
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/png"):
            raise HTTPException(status_code=400, detail="Only PNG images are supported")

        # Validate language codes
        if source_language not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid source language. Choose from: {', '.join(language_options)}")
        if target_language not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid target language. Choose from: {', '.join(language_options)}")

        # Read and encode image
        image_bytes = await file.read()
        image = BytesIO(image_bytes)
        img_base64 = encode_image(image)

        # Perform OCR using RolmOCR
        extracted_text = ocr_page_with_rolm(img_base64)

        # Process with custom prompt if provided
        response = None
        text_to_translate = extracted_text
        if prompt and prompt.strip():
            custom_response = openai_client.chat.completions.create(
                model=rolm_model,
                messages=[
                {
                    "role": "system",
                "content": [{"type": "text", "text": "You are dwani, a helpful assistant. Summarize your answer in maximum 1 sentence. If the answer contains numerical digits, convert the digits into words"}]
                },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n{extracted_text}"
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            response = custom_response.choices[0].message.content
            text_to_translate = response
        elif prompt and not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        # Translate the text (either extracted text or prompt response)
        translation_payload = {
            "sentences": [text_to_translate],
            "src_lang": source_language,
            "tgt_lang": target_language
        }
        translation_response = requests.post(
            f"{translation_api_url}/translate?src_lang={source_language}&tgt_lang={target_language}",
            json=translation_payload,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        translation_response.raise_for_status()
        translation_result = translation_response.json()
        translated_response = translation_result["translations"][0]

        # Build response
        result = {
            "extracted_text": extracted_text,
            "translated_response": translated_response
        }
        if response:
            result["response"] = response

        return JSONResponse(content=result)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error querying translation API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image or generating response: {str(e)}")



# Configuration from environment variables
#llm_base_url = os.getenv("DWANI_API_BASE_URL_LLM")
llm_base_url = "http://0.0.0.0:7860/v1"
#translation_api_url = os.getenv("DWANI_API_BASE_URL_TRANSLATE")
translation_api_url = "http://0.0.0.0:7862"
rolm_model = "google/gemma-3-4b-it"  # As per reference

# Supported language codes
language_options = [
    "kan_Knda",  # Kannada
    "eng_Latn",  # English
    "hin_Deva",  # Hindi
    "tam_Taml",  # Tamil
    "tel_Telu",  # Telugu
]

# Mock settings (replace with actual configuration)
class Settings:
    chat_rate_limit = "10/minute"
    max_tokens = 500
    openai_api_key = "123"  # Replace with actual key

def get_settings():
    return Settings()

from fastapi import APIRouter, HTTPException, Body, Request, Depends
from num2words import num2words
from datetime import datetime
import pytz

def time_to_words():
    """Convert current IST time to words (e.g., '4:04' to 'four hours and four minutes', '4:00' to 'four o'clock')."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    hour = now.hour % 12 or 12  # Convert 24-hour to 12-hour format (0 -> 12)
    minute = now.minute
    
    # Convert hour to words
    hour_word = num2words(hour, to='cardinal')
    
    # Handle minutes
    if minute == 0:
        return f"{hour_word} o'clock"
    else:
        minute_word = num2words(minute, to='cardinal')
        return f"{hour_word} hours and {minute_word} minutes"


llm_base_url="http://0.0.0.0:7860/v1"
# Initialize OpenAI client
openai_client = OpenAI(api_key="asdas", base_url=llm_base_url)

# Pydantic models
class ChatRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for the chat.")
    src_lang: str = Field(default="eng_Latn", description="Source language code.", enum=language_options)
    tgt_lang: str = Field(default="eng_Latn", description="Target language code.", enum=language_options)

class ChatResponse(BaseModel):
    response: str = Field(..., description="The generated or translated response.")

# Indic chat endpoint
@app.post(
    "/indic_chat",
    response_model=ChatResponse,
    summary="Process and translate chat prompts",
    description=(
        "Processes a user-provided prompt using OpenAI Chat Completions, with optional translation of the prompt "
        "to English (if not in English) and translation of the response to a target language."
    ),
    response_description="A JSON object containing the generated or translated response."
)
async def indic_chat(
    request: Request,
    chat_request: ChatRequest,
    settings=Depends(get_settings)
):
    """
    Process a chat prompt, optionally translating it to English for processing and translating the response to a target language.

    Args:
        request (Request): The FastAPI request object.
        chat_request (ChatRequest): The chat request containing prompt, source language, and target language.
        settings: Application settings for rate limiting and max tokens.

    Returns:
        JSONResponse: A dictionary containing the generated or translated response.

    Raises:
        HTTPException: If the prompt is empty, language codes are invalid, or processing/translation fails.

    Example:
        ```json
        {"response": "Generated or translated response"}
        ```
    """
    if not chat_request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    logger.debug(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}")

    try:
        # Validate language codes
        if chat_request.src_lang not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid source language. Choose from: {', '.join(language_options)}")
        if chat_request.tgt_lang not in language_options:
            raise HTTPException(status_code=400, detail=f"Invalid target language. Choose from: {', '.join(language_options)}")

        # Translate prompt to English if source language is not English
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
        else:
            logger.debug("Prompt in English, no translation needed")

        current_time = time_to_words()
        response = openai_client.chat.completions.create(
            model=rolm_model,
            messages=[
                {
                    "role": "system",
            "content": [{"type": "text", "text": f"You are Dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. If the answer contains numerical digits, convert the digits into words. If user asks the time, then return answer as {current_time}"}]                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_to_process}]
                }
            ],
            temperature=0.3,
            max_tokens=settings.max_tokens
        )
        generated_response = response.choices[0].message.content
        logger.debug(f"Generated response: {generated_response}")

        # Translate response to target language if not English
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
        else:
            logger.debug("Response in English, no translation needed")

        return JSONResponse(content={"response": final_response})

    except requests.exceptions.RequestException as e:
        logger.error(f"Translation API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation API error: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
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

    uvicorn.run(app, host=args.host, port=args.port)