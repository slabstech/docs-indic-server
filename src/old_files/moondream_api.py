import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, OrderedDict, List
import torch
from fastapi import Body, FastAPI, HTTPException, Response, UploadFile, File
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed, AutoModelForCausalLM
import numpy as np
from config import SPEED, ResponseFormat, config
from logger import logger
import uvicorn
import argparse
from fastapi.responses import RedirectResponse, StreamingResponse
import io
import zipfile
import os
import logging
from PIL import Image

# https://github.com/huggingface/parler-tts?tab=readme-ov-file#usage
if torch.cuda.is_available():
    device = "cuda:0"
    logger.info("GPU will be used for inference")
else:
    device = "cpu"
    logger.info("CPU will be used for inference")
torch_dtype = torch.float16 if device != "cpu" else torch.float32

# Check CUDA availability and version
cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda if cuda_available else None

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    compute_capability_float = float(f"{capability[0]}.{capability[1]}")
    print(f"CUDA version: {cuda_version}")
    print(f"CUDA Compute Capability: {compute_capability_float}")
else:
    print("CUDA is not available on this system.")

app = FastAPI()

# Lazy loading of the model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map={"": device}
    )
    yield
    # Clean up the model, if needed
    model = None

app.router.lifespan_context = lifespan

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/caption/")
async def caption_image(file: UploadFile = File(...), length: str = "normal"):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image = Image.open(file.file)
    if length == "short":
        caption = model.caption(image, length="short")["caption"]
    else:
        caption = model.caption(image, length="normal")
    return {"caption": caption}

@app.post("/visual_query/")
async def visual_query(file: UploadFile = File(...), query: str = Body(...)):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image = Image.open(file.file)
    answer = model.query(image, query)["answer"]
    return {"answer": answer}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...), object_type: str = "face"):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image = Image.open(file.file)
    objects = model.detect(image, object_type)["objects"]
    return {"objects": objects}

@app.post("/point/")
async def point_objects(file: UploadFile = File(...), object_type: str = "person"):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image = Image.open(file.file)
    points = model.point(image, object_type)["points"]
    return {"points": points}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server for TTS.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)