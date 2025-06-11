from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from time import time
import uvicorn
import argparse
from logging_config import logger
from config import config
from services.ocr_service import OCRService
from services.translation_service import TranslationService
from services.pdf_service import PDFService
from models.request_models import (
    ExtractTextRequest, SummarizePDFRequest, CustomPromptPDFRequest,
    IndicExtractTextRequest, IndicSummarizePDFRequest, IndicCustomPromptPDFRequest,
    IndicVisualQueryRequest, ChatRequest, ChatDirectRequest, VisualQueryDirectRequest
)
from models.response_models import (
    ExtractTextResponse, IndicExtractTextResponse, SummarizePDFResponse,
    IndicSummarizePDFResponse, CustomPromptPDFResponse, IndicCustomPromptPDFResponse,
    IndicVisualQueryResponse, ChatResponse, VisualQueryDirectResponse
)
from utils.file_utils import temp_file, validate_png_file
from utils.ocr_utils import encode_image
from utils.pdf_utils import generate_pdf_with_layout
from datetime import datetime
import pytz
from num2words import num2words
from io import BytesIO

app = FastAPI(
    title="dwani.ai - Document Server API",
    description="API for document processing, OCR, summarization, and translation.",
    version="1.0.0"
)

def get_ocr_service(model: str = config.DEFAULT_MODEL) -> OCRService:
    return OCRService(model)

def get_translation_service() -> TranslationService:
    return TranslationService()

def time_to_words():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    hour = now.hour % 12 or 12
    minute = now.minute
    hour_word = num2words(hour, to='cardinal')
    if minute == 0:
        return f"{hour_word} o'clock"
    minute_word = num2words(minute, to='cardinal')
    return f"{hour_word} hours and {minute_word} minutes"

@app.get("/")
async def root():
    return {"message": "dwani.ai API is running."}

@app.post("/extract-text/", response_model=ExtractTextResponse)
async def extract_text_from_pdf(
    file: UploadFile = File(...),
    request: ExtractTextRequest = Depends(),
    pdf_service: PDFService = Depends(get_ocr_service)
):
    text = await pdf_service.extract_text(file, request.page_number)
    return ExtractTextResponse(page_content=text, processed_page=request.page_number)

@app.post("/indic-extract-text/", response_model=IndicExtractTextResponse)
async def indic_extract_text_from_pdf(
    file: UploadFile = File(...),
    request: IndicExtractTextRequest = Depends(),
    pdf_service: PDFService = Depends(get_ocr_service),
    translation_service: TranslationService = Depends(get_translation_service)
):
    text = await pdf_service.extract_text(file, request.page_number)
    translated = translation_service.translate(text, request.source_language, request.target_language)
    return IndicExtractTextResponse(
        page_content=text,
        translated_content=translated,
        processed_page=request.page_number
    )

@app.post("/summarize-pdf/", response_model=SummarizePDFResponse)
async def summarize_pdf(
    file: UploadFile = File(...),
    request: SummarizePDFRequest = Depends(),
    pdf_service: PDFService = Depends(get_ocr_service)
):
    text = await pdf_service.extract_text(file, request.page_number)
    summary = pdf_service.ocr_service.summarize_text(text)
    return SummarizePDFResponse(
        original_text=text,
        summary=summary,
        processed_page=request.page_number
    )

@app.post("/indic-summarize-pdf/", response_model=IndicSummarizePDFResponse)
async def indic_summarize_pdf(
    file: UploadFile = File(...),
    request: IndicSummarizePDFRequest = Depends(),
    pdf_service: PDFService = Depends(get_ocr_service),
    translation_service: TranslationService = Depends(get_translation_service)
):
    text = await pdf_service.extract_text(file, request.page_number)
    summary = pdf_service.ocr_service.summarize_text(text)
    translated = translation_service.translate(summary, request.source_language, request.target_language)
    return IndicSummarizePDFResponse(
        original_text=text,
        summary=summary,
        translated_summary=translated,
        processed_page=request.page_number
    )

@app.post("/custom-prompt-pdf/", response_model=CustomPromptPDFResponse)
async def custom_prompt_pdf(
    file: UploadFile = File(...),
    request: CustomPromptPDFRequest = Depends(),
    pdf_service: PDFService = Depends(get_ocr_service)
):
    text = await pdf_service.extract_text(file, request.page_number)
    response = pdf_service.ocr_service.process_text(text, request.prompt)
    return CustomPromptPDFResponse(
        original_text=text,
        response=response,
        processed_page=request.page_number
    )

@app.post("/indic-custom-prompt-pdf/", response_model=IndicCustomPromptPDFResponse)
async def indic_custom_prompt_pdf(
    file: UploadFile = File(...),
    request: IndicCustomPromptPDFRequest = Depends(),
    pdf_service: PDFService = Depends(get_ocr_service),
    translation_service: TranslationService = Depends(get_translation_service)
):
    text = await pdf_service.extract_text(file, request.page_number)
    response = pdf_service.ocr_service.process_text(text, request.prompt)
    translated = translation_service.translate(response, request.source_language, request.target_language)
    return IndicCustomPromptPDFResponse(
        original_text=text,
        response=response,
        translated_response=translated,
        processed_page=request.page_number
    )

@app.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    validate_png_file(file.content_type)
    img_bytes = await file.read()
    img_base64 = encode_image(BytesIO(img_bytes))
    text = ocr_service.ocr_image(img_base64)
    return {"extracted_text": text}

@app.post("/pdf-recreation/extract-text/")
async def pdf_recreation_extract_text(
    file: UploadFile = File(...),
    request: ExtractTextRequest = Depends(),
    pdf_service: PDFService = Depends(get_ocr_service)
):
    output_path = await pdf_service.recreate_pdf(file, request.page_number)
    return FileResponse(
        path=output_path,
        filename=f"extracted_text_page_{request.page_number}.pdf",
        media_type="application/pdf",
        background=lambda: temp_file(suffix=".pdf").__exit__(None, None, None)
    )

@app.post("/pdf-recreation/indic-extract-text/")
async def pdf_recreation_indic_extract_text(
    file: UploadFile = File(...),
    request: IndicExtractTextRequest = Depends(),
    pdf_service: PDFService = Depends(get_ocr_service),
    translation_service: TranslationService = Depends(get_translation_service)
):
    text_segments = await pdf_service.extract_text_with_layout(file, request.page_number)
    translated_segments = []
    for segment in text_segments:
        translated_text = translation_service.translate(segment["text"], request.source_language, request.target_language)
        translated_segments.append({**segment, "text": translated_text})
    with temp_file(suffix=".pdf") as temp:
        temp.write(await file.read())
        with pdfplumber.open(temp.name) as pdf:
            page = pdf.pages[request.page_number - 1]
            page_width, page_height = page.width, page.height
        with temp_file(suffix=".pdf") as temp_pdf:
            output_path = temp_pdf.name
            generate_pdf_with_layout(translated_segments, output_path, page_width, page_height)
            return FileResponse(
                path=output_path,
                filename=f"translated_text_page_{request.page_number}.pdf",
                media_type="application/pdf",
                background=lambda: temp_file(suffix=".pdf").__exit__(None, None, None)
            )

@app.post("/pdf-recreation/ocr")
async def pdf_recreation_ocr_image(
    file: UploadFile = File(...),
    request: ExtractTextRequest = Depends(),
    pdf_service: PDFService = Depends(get_ocr_service)
):
    output_path = await pdf_service.recreate_pdf_from_image(file)
    return FileResponse(
        path=output_path,
        filename="extracted_text_image.pdf",
        media_type="application/pdf",
        background=lambda: temp_file(suffix=".pdf").__exit__(None, None, None)
    )

@app.post("/indic-visual-query/", response_model=IndicVisualQueryResponse)
async def indic_visual_query(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    source_language: str = Form("eng_Latn"),
    target_language: str = Form("kan_Knda"),
    model: str = Form(config.DEFAULT_MODEL),
    ocr_service: OCRService = Depends(get_ocr_service),
    translation_service: TranslationService = Depends(get_translation_service)
):
    validate_png_file(file.content_type)
    img_bytes = await file.read()
    img_base64 = encode_image(BytesIO(img_bytes))
    extracted_text = ocr_service.ocr_image(img_base64)
    text_to_translate = extracted_text
    response = None
    if prompt:
        response = ocr_service.process_text(extracted_text, prompt)
        text_to_translate = response
    translated = translation_service.translate(text_to_translate, source_language, target_language)
    return IndicVisualQueryResponse(
        extracted_text=extracted_text,
        response=response,
        translated_response=translated
    )

@app.post("/visual-query-direct/", response_model=VisualQueryDirectResponse)
async def visual_query_direct(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    model: str = Form(config.DEFAULT_MODEL),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    validate_png_file(file.content_type)
    img_bytes = await file.read()
    img_base64 = encode_image(BytesIO(img_bytes))
    extracted_text = ocr_service.ocr_image(img_base64)
    response = None
    if prompt:
        response = ocr_service.process_text(extracted_text, prompt)
    return VisualQueryDirectResponse(
        extracted_text=extracted_text,
        response=response
    )

@app.post("/indic_chat", response_model=ChatResponse)
async def indic_chat(
    chat_request: ChatRequest,
    ocr_service: OCRService = Depends(get_ocr_service),
    translation_service: TranslationService = Depends(get_translation_service)
):
    prompt = chat_request.prompt
    if chat_request.src_lang != "eng_Latn":
        prompt = translation_service.translate(prompt, chat_request.src_lang, "eng_Latn")
    current_time = time_to_words()
    response = ocr_service.process_text(
        prompt,
        f"You are Dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. If the answer contains numerical digits, convert the digits into words. If user asks the time, then return answer as {current_time}."
    )
    if chat_request.tgt_lang != "eng_Latn":
        response = translation_service.translate(response, "eng_Latn", chat_request.tgt_lang)
    return ChatResponse(response=response)

@app.post("/chat", response_model=ChatResponse)
async def chat_direct(
    chat_request: ChatDirectRequest,
    ocr_service: OCRService = Depends(get_ocr_service)
):
    current_time = time_to_words()
    response = ocr_service.process_text(
        chat_request.prompt,
        f"You are Dwani, a prompt to process {chat_request.system_prompt}. Answer questions considering India as base country and Karnataka as base state. Provide a response in one sentence. If the answer contains numerical digits, convert them into words. If user asks the time, then return answer as {current_time}."
    )
    return JSONResponse(content=response)

@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    duration = time() - start_time
    logger.info(f"Request to {request.url.path} took {duration:.3f} seconds")
    response.headers["X-Response-Time"] = str(duration)
    return JSONResponse(content=response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dwani.ai FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=7864, help="Port to bind")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)