from olmocr.data.renderpdf import render_pdf_to_base64png
from fastapi import HTTPException
import pdfplumber
from typing import List, Dict
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def render_pdf_page_to_base64(pdf_path: str, page_number: int, target_dim: int = 1024) -> str:
    """Render a PDF page to base64-encoded PNG."""
    try:
        return render_pdf_to_base64png(pdf_path, page_number, target_longest_image_dim=target_dim)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render PDF page: {str(e)}")

def extract_text_with_layout(pdf_path: str, page_number: int) -> List[Dict]:
    """Extract text with layout information using pdfplumber."""
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
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

def generate_pdf_with_layout(text_segments: List[Dict], output_path: str, page_width: float, page_height: float):
    """Generate a PDF with text segments at specified positions."""
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

def generate_pdf_from_text(text: str, output_path: str):
    """Generate a simple PDF from plain text."""
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