from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
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

# Initialize FastAPI app with enhanced metadata
app = FastAPI(
    title="Combined OCR API",
    description=(
        "API for extracting text from PDF pages and PNG images using RolmOCR, with functionality to "
        "summarize PDF content, process it with custom prompts, translate summaries to Kannada, or translate "
        "extracted Kannada text to English. Supports text extraction from a single PDF page, OCR for PNG images, "
        "summarization of a single PDF page, custom prompt-based processing, and translation between Kannada and English."
    ),
    version="1.0.0"
)

# Initialize OpenAI client for RolmOCR
openai_client = OpenAI(api_key="123", base_url="http://0.0.0.0:7863/v1")

#rolm_model = "google/gemma-3-12b-it"   - for H100 only
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7864)