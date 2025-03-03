import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, OrderedDict,List
import torch
from fastapi import Body, FastAPI, HTTPException, Response
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import numpy as np
from config import SPEED, ResponseFormat, config
from logger import logger
import uvicorn
import argparse
from fastapi.responses import RedirectResponse
import io
import zipfile
from fastapi.responses import StreamingResponse
from typing import List
import numpy as np
import soundfile as sf
import time
import os
import logging



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


class ModelManager:
    def __init__(self):
        self.model_tokenizer: OrderedDict[
            str, tuple[ParlerTTSForConditionalGeneration, AutoTokenizer]
        ] = OrderedDict()

    def load_model(
        self, model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, AutoTokenizer]:
        logger.debug(f"Loading {model_name}...")
        start = time.perf_counter()
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(  # type: ignore
            device,  # type: ignore
            dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

        logger.info(
            f"Loaded {model_name} and tokenizer in {time.perf_counter() - start:.2f} seconds"
        )
        return model, tokenizer, description_tokenizer

    def get_or_load_model(
        self, model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, Any]:
        if model_name not in self.model_tokenizer:
            logger.info(f"Model {model_name} isn't already loaded")
            if len(self.model_tokenizer) == config.max_models:
                logger.info("Unloading the oldest loaded model")
                del self.model_tokenizer[next(iter(self.model_tokenizer))]
            self.model_tokenizer[model_name] = self.load_model(model_name)
        return self.model_tokenizer[model_name]


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.lazy_load_model:
        model_manager.get_or_load_model(config.model)
    yield


app = FastAPI(lifespan=lifespan)

from fastapi.responses import StreamingResponse
import io

@app.post("/v1/media/image")
async def describe_image(
    input: Annotated[str, Body()] = config.input,
    voice: Annotated[str, Body()] = config.voice,
    model: Annotated[str, Body()] = config.model,
    response_format: Annotated[ResponseFormat, Body(include_in_schema=False)] = config.response_format,
    speed: Annotated[float, Body(include_in_schema=False)] = SPEED,
) -> StreamingResponse:
    tts, tokenizer, description_tokenizer = model_manager.get_or_load_model(model)
    if speed != SPEED:
        logger.warning(
            "Specifying speed isn't supported by this model. Audio will be generated with the default speed"
        )
    start = time.perf_counter()

    # Tokenize the voice description
    input_ids = description_tokenizer(voice, return_tensors="pt").input_ids.to(device)

    # Tokenize the input text
    prompt_input_ids = tokenizer(input, return_tensors="pt").input_ids.to(device)

    # Generate the audio
    generation = tts.generate(
        input_ids=input_ids, prompt_input_ids=prompt_input_ids
    ).to(torch.float32)
    audio_arr = generation.cpu().numpy().squeeze()

    # Ensure device is a string
    device_str = str(device)

    logger.info(
        f"Took {time.perf_counter() - start:.2f} seconds to generate audio for {len(input.split())} words using {device_str.upper()}"
    )

    # Create an in-memory file
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_arr, tts.config.sampling_rate, format=response_format)
    audio_buffer.seek(0)

    return StreamingResponse(audio_buffer, media_type=f"audio/{response_format}")

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server for TTS.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    args = parser.parse_args()

    #asr_manager = ASRModelManager(device_type=args.device)
    uvicorn.run(app, host=args.host, port=args.port)