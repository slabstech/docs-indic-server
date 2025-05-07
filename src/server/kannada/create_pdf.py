from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4

# Register the Google Kannada font
pdfmetrics.registerFont(TTFont('NotoSansKannada', 'NotoSansKannada-Regular.ttf'))

# Create PDF canvas
w, h = A4
c = canvas.Canvas("google_kannada_font.pdf", pagesize=A4)

# Set the font
c.setFont('NotoSansKannada', 24)

# Kannada text example
kannada_text = "ನಮಸ್ಕಾರ ವರ್ಲ್ಡ್"  # "Hello World" in Kannada

# Draw the text
c.drawString(100, h - 100, kannada_text)

c.save()
