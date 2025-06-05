FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    sudo \
    wget libvips\
    poppler-utils fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*
#ttf-mscorefonts-installer msttcorefonts 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#COPY pixtral_requirements.txt .
#RUN pip install --no-cache-dir -r pixtral_requirements.txt

COPY . .

RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# Use absolute path for clarity
CMD ["python", "/app/src/server/docs_api.py", "--host", "0.0.0.0", "--port", "7860"]