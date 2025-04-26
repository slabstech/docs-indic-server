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
openai_client = OpenAI(api_key="123", base_url="http://localhost:8000/v1")

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

class SummarizePDFKannadaRequest(BaseModel):
    page_number: int = Field(
        default=1,
        description="The page number to extract, summarize, and translate to Kannada (1-based indexing). Must be a positive integer.",
        ge=1,
        example=1
    )

class TranslatePDFKannadaToEnglishRequest(BaseModel):
    page_number: int = Field(
        default=1,
        description="The page number to extract and translate from Kannada to English (1-based indexing). Must be a positive integer.",
        ge=1,
        example=1
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
    "/summarize-pdf-kannada/",
    response_model=dict,
    summary="Extract, summarize, and translate summary to Kannada from a single PDF page",
    description=(
        "Extracts text from a specified page of a PDF file using RolmOCR, generates a 3-5 sentence summary "
        "in English using chat completion, and translates the summary to Kannada using an external translation API."
    ),
    response_description=(
        "A JSON object containing the extracted text, the English summary, the Kannada summary, and the processed page number."
    )
)
async def summarize_pdf_kannada(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(
        default=1,
        embed=True,
        description=SummarizePDFKannadaRequest.model_fields["page_number"].description,
        ge=1,
        example=1
    )
):
    """
    Extract text from a specified page of a PDF file, summarize it, and translate the summary to Kannada.

    Args:
        file (UploadFile): The PDF file to process.
        page_number (int): The page number to extract, summarize, and translate (1-based indexing). Defaults to 1.

    Returns:
        JSONResponse: A dictionary containing:
            - original_text: Text extracted from the specified page.
            - english_summary: A 3-5 sentence summary in English.
            - kannada_summary: The English summary translated to Kannada.
            - processed_page: The page number processed.

    Raises:
        HTTPException: If the file is not a PDF, the page number is invalid, translation fails, or processing fails.

    Example:
        ```json
        {
            "original_text": "Text from page 1",
            "english_summary": "The document discusses... [3-5 sentence summary]",
            "kannada_summary": "ದಾಖಲೆಯು ಚರ್ಚಿಸುತ್ತದೆ... [translated summary]",
            "processed_page": 1
        }
        ```
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Extract text and generate English summary using existing endpoint logic
        summary_response = await summarize_pdf(file, page_number)
        summary_content = summary_response.body.decode("utf-8")
        summary_json = json.loads(summary_content)
        extracted_text = summary_json["original_text"]
        english_summary = summary_json["summary"]

        # Translate English summary to Kannada using external API
        translation_url = "http://0.0.0.0:7861/translate?src_lang=eng_Latn&tgt_lang=kan_Knda"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        payload = {
            "sentences": [english_summary],
            "src_lang": "eng_Latn",
            "tgt_lang": "kan_Knda"
        }
        try:
            response = requests.post(translation_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for bad status codes
            translation_data = response.json()
            kannada_summary = translation_data["translations"][0]
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Translation API request failed: {str(e)}")
        except (KeyError, IndexError) as e:
            raise HTTPException(status_code=500, detail=f"Invalid translation API response: {str(e)}")

        return JSONResponse(content={
            "original_text": extracted_text,
            "english_summary": english_summary,
            "kannada_summary": kannada_summary,
            "processed_page": page_number
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF, generating summary, or translating: {str(e)}")

@app.post(
    "/translate-pdf-kannada-to-english/",
    response_model=dict,
    summary="Extract and translate text from Kannada to English from a single PDF page",
    description=(
        "Extracts text from a specified page of a PDF file using RolmOCR and translates it from Kannada to English "
        "using an external translation API."
    ),
    response_description=(
        "A JSON object containing the extracted text (in Kannada), the translated text (in English), and the processed page number."
    )
)
async def translate_pdf_kannada_to_english(
    file: UploadFile = File(..., description="The PDF file to process. Must be a valid PDF."),
    page_number: int = Body(
        default=1,
        embed=True,
        description=TranslatePDFKannadaToEnglishRequest.model_fields["page_number"].description,
        ge=1,
        example=1
    )
):
    """
    Extract text from a specified page of a PDF file and translate it from Kannada to English.

    Args:
        file (UploadFile): The PDF file to process.
        page_number (int): The page number to extract and translate (1-based indexing). Defaults to 1.

    Returns:
        JSONResponse: A dictionary containing:
            - original_text: Text extracted from the specified page (in Kannada).
            - english_text: The extracted text translated to English.
            - processed_page: The page number processed.

    Raises:
        HTTPException: If the file is not a PDF, the page number is invalid, translation fails, or processing fails.

    Example:
        ```json
        {
            "original_text": "ದಾಖಲೆಯು ಚರ್ಚಿಸುತ್ತದೆ... [Kannada text]",
            "english_text": "The document discusses... [translated text]",
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

        # Translate extracted text from Kannada to English using external API
        translation_url = "http://0.0.0.0:7861/translate?src_lang=kan_Knda&tgt_lang=eng_Latn"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        payload = {
            "sentences": [extracted_text],
            "src_lang": "kan_Knda",
            "tgt_lang": "eng_Latn"
        }
        try:
            response = requests.post(translation_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for bad status codes
            translation_data = response.json()
            english_text = translation_data["translations"][0]
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Translation API request failed: {str(e)}")
        except (KeyError, IndexError) as e:
            raise HTTPException(status_code=500, detail=f"Invalid translation API response: {str(e)}")

        return JSONResponse(content={
            "original_text": extracted_text,
            "english_text": english_text,
            "processed_page": page_number
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF or translating: {str(e)}")
    

@app.post(
    "/visual-query/",
    response_model=dict,
    summary="Extract text from a PNG image using visual query",
    description=(
        "Processes a PNG image using an external visual query API to extract a description of the image content. "
        "The query 'describe the image' is used to process the image."
    ),
    response_description="A JSON object containing the extracted text from the visual query API."
)
async def visual_query(file: UploadFile = File(..., description="The PNG image to process. Must be a valid PNG.")):
    """
    Extract text from a PNG image using an external visual query API.

    Args:
        file (UploadFile): The PNG image to process.

    Returns:
        dict: A dictionary containing:
            - extracted_text: The text extracted from the image via the visual query API.

    Raises:
        HTTPException: If the file is not a PNG or the external API request fails.

    Example:
        ```json
        {
            "extracted_text": "Here’s a summary of the image in one sentence:\\n\\nThe image displays an AWS cost and usage report..."
        }
        ```
    """
    # Validate file type
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images are supported")

    try:
        # Read image file
        image_bytes = await file.read()

        # Prepare multipart/form-data for the external API
        files = {
            "file": (file.filename, image_bytes, "image/png")
        }
        data = {
            "query": "describe the image",
            "src_lang": "eng_Latn",
            "tgt_lang": "eng_Latn"
        }

        # Make POST request to the external visual query API
        visual_query_url = "http://0.0.0.0:7862/v1/visual_query/"
        headers = {
            "accept": "application/json"
        }
        try:
            response = requests.post(visual_query_url, headers=headers, files=files, data=data)
            response.raise_for_status()  # Raise exception for bad status codes
            response_data = response.json()
            extracted_text = response_data.get("answer", "")
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Visual query API request failed: {str(e)}")
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=500, detail=f"Invalid visual query API response: {str(e)}")

        return {"extracted_text": extracted_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post(
    "/extract-text/",
    response_model=dict,
    summary="Extract text from a PDF page using visual query",
    description=(
        "Extracts text from a specific page of a PDF file by rendering it as an image and processing it with an external visual query API. "
        "The query 'describe the image' is used to generate a description of the page content."
    ),
    response_description="A JSON object containing the extracted text from the specified page."
)
async def extract_text_visual_query(
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
    Extract text from a specific page of a PDF file using an external visual query API.

    Args:
        file (UploadFile): The PDF file to process.
        page_number (int): The page number to extract text from (1-based indexing). Defaults to 1.

    Returns:
        JSONResponse: A dictionary containing:
            - page_content: The extracted text from the specified page via the visual query API.

    Raises:
        HTTPException: If the file is not a PDF, the page number is invalid, or processing fails.

    Example:
        ```json
        {"page_content": "Here’s a summary of the page in one sentence:\\n\\nThe page displays..."}
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

        # Decode base64 image to bytes for visual query API
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to decode image: {str(e)}")

        # Prepare multipart/form-data for the external visual query API
        files = {
            "file": ("page.png", image_bytes, "image/png")
        }
        data = {
            "query": "describe the image",
            "src_lang": "eng_Latn",
            "tgt_lang": "eng_Latn"
        }

        # Make POST request to the external visual query API
        visual_query_url = "http://0.0.0.0:7862/v1/visual_query/"
        headers = {
            "accept": "application/json"
        }
        try:
            response = requests.post(visual_query_url, headers=headers, files=files, data=data)
            response.raise_for_status()  # Raise exception for bad status codes
            response_data = response.json()
            page_content = response_data.get("answer", "")
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Visual query API request failed: {str(e)}")
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=500, detail=f"Invalid visual query API response: {str(e)}")

        # Clean up temporary Old temporary file
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

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import io
from typing import List   



# Model initialization
MODEL_ID = "google/gemma-3-4b-it"
device = "cuda"  # Force CUDA usage

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

# Initialize processor WITHOUT .to(device) - processors don't need GPU
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    use_fast=True  # Critical optimization
)

class MessageContentItem(BaseModel):
    type: str
    text: str = None
    image: str = None

class Message(BaseModel):
    role: str
    content: List[MessageContentItem]

class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3b-it"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 200

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    try:
        # Convert messages to processor format
        hf_messages = []
        for msg in request.messages:
            content_items = []
            for item in msg.content:
                if item.type == "text":
                    content_items.append({"type": "text", "text": item.text})
                elif item.type == "image":
                    content_items.append({"type": "image", "image": item.image})
            hf_messages.append({"role": msg.role, "content": content_items})

        # Process and generate
        inputs = processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)  # Move TENSORS to GPU

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True
            )
            generation = generation[0][input_len:]

        response = processor.decode(generation, skip_special_tokens=True)

        return {
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/vision/completions")
async def vision_completion(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    max_tokens: int = Form(200),
    temperature: float = Form(0.7)
):
    try:
        # Validate and process image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

        # Create message with image tensor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process and move tensors to GPU
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)  # Critical: Move processed tensors to GPU

        input_len = inputs["input_ids"].shape[-1]

        # CUDA-accelerated generation
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            generation = generation[0][input_len:]

        response = processor.decode(generation, skip_special_tokens=True)

        return {
            "object": "vision.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)