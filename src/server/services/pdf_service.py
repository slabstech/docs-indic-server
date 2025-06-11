from fastapi import UploadFile, HTTPException
from typing import List, Dict
from .ocr_service import OCRService
from utils.file_utils import temp_file, validate_pdf_file
from utils.pdf_utils import render_pdf_page_to_base64, extract_text_with_layout, generate_pdf_with_layout, generate_pdf_from_text
import pdfplumber

class PDFService:
    def __init__(self, ocr_service: OCRService):
        self.ocr_service = ocr_service

    async def extract_text(self, file: UploadFile, page_number: int) -> str:
        """Extract text from a PDF page using OCR."""
        validate_pdf_file(file.filename)
        with temp_file(suffix=".pdf") as temp:
            temp.write(await file.read())
            img_base64 = render_pdf_page_to_base64(temp.name, page_number)
            return self.ocr_service.ocr_image(img_base64)

    async def extract_text_with_layout(self, file: UploadFile, page_number: int) -> List[Dict]:
        """Extract text with layout information or fall back to OCR."""
        validate_pdf_file(file.filename)
        with temp_file(suffix=".pdf") as temp:
            temp.write(await file.read())
            text_segments = extract_text_with_layout(temp.name, page_number)
            if not text_segments:
                img_base64 = render_pdf_page_to_base64(temp.name, page_number, target_dim=512)
                page_content = self.ocr_service.ocr_image(img_base64)
                text_segments = [{
                    "text": page_content,
                    "x0": 50,
                    "y0": 50,
                    "x1": 550,
                    "y1": 750,
                    "fontname": "Helvetica",
                    "size": 12
                }]
            return text_segments

    async def recreate_pdf(self, file: UploadFile, page_number: int) -> str:
        """Recreate a PDF page with extracted text."""
        validate_pdf_file(file.filename)
        with temp_file(suffix=".pdf") as temp:
            temp.write(await file.read())
            text_segments = await self.extract_text_with_layout(file, page_number)
            with pdfplumber.open(temp.name) as pdf:
                page = pdf.pages[page_number - 1]
                page_width, page_height = page.width, page.height
            with temp_file(suffix=".pdf") as temp_pdf:
                output_path = temp_pdf.name
                generate_pdf_with_layout(text_segments, output_path, page_width, page_height)
                return output_path

    async def recreate_pdf_from_image(self, file: UploadFile) -> str:
        """Recreate a PDF from an image using OCR."""
        from utils.file_utils import validate_png_file
        validate_png_file(file.content_type)
        img_bytes = await file.read()
        img_base64 = encode_image(BytesIO(img_bytes))
        text = self.ocr_service.ocr_image(img_base64)
        with temp_file(suffix=".pdf") as temp_pdf:
            output_path = temp_pdf.name
            generate_pdf_from_text(text, output_path)
            return output_path