# Docs-Indic-Server

## Overview
Document parser for Indian languages

## Table of Contents
- [Usage](#usage)
- [Getting Started - Development](#getting-started---development)
- [Downloading Model](#downloading-indic-model)
- [Running with FastAPI Server](#running-with-fastapi-server)
- [Evaluating Results](#evaluating-results)
- [Citations](#citations)



### For Development (Local)
- **Prerequisites**: Python 3.6+
- **Steps**:
  1. **Create a virtual environment**:
  ```bash
  python -m venv venv
  ```
  2. **Activate the virtual environment**:
  ```bash
  source venv/bin/activate
  ```
  On Windows, use:
  ```bash
  venv\Scripts\activate
  ```
  3. **Install dependencies**:
  - For GPU
      ```bash
      pip install -r requirements.txt
      ```
  - For CPU only
      ```bash
      pip install -r cpu-requirements.txt
      ```

## Download Pixtral 

```bash download_model.sh
huggingface_cli download mistralai/Pixtral-12B-2409
```

## Download Moondream2
```bash
huggingface_cli  vikhyatk/moondream2
```
Model Size - 4GB

## Running with FastAPI Server
Run the server using FastAPI
- for GPU
  ```bash
  python src/docs_api.py --port 7860 --host 0.0.0.0 --device gpu
  ```
- for CPU only
  ```bash
  python src/docs_api.py --port 7860 --host 0.0.0.0 --device cpu
  ```



## Getting Started - Development

- For moondream, libvips system library is required 
  ```
  sudo apt-get update && sudo apt-get install libvips
  ```

## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

Also you can join the [discord group](https://discord.gg/WZMCerEZ2P) to collaborate

- Reference
    - [HF - moondream](https://huggingface.co/vikhyatk/moondream2)
    - [source - moondream](https://github.com/vikhyat/moondream)
    - [moondream-blog](https://moondream.ai/blog/introducing-a-new-moondream-1-9b-and-gpu-support)
    - [pixtral-12-b-2409](https://huggingface.co/mistralai/Pixtral-12B-2409)


<!-- 

## Evaluating Results
You can evaluate the ASR transcription results using `curl` commands. Below are examples for Kannada audio samples.

#### Kannada

```bash kannada_example.sh
curl -s -H "content-type: application/json" localhost:7860/v1/audio/speech -d '{"input": "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ."}' -o audio_kannada.mp3
```

#### Hindi

```bash hindi_example.sh
curl -s -H "content-type: application/json" localhost:7860/v1/audio/speech -d '{"input": "अरे, तुम आज कैसे हो?"}' -o audio_hindi.mp3
```

### Specifying a Different Format

```bash specify_format.sh
curl -s -H "content-type: application/json" localhost:7860/v1/audio/speech -d '{"input": "Hey, how are you?", "response_type": "wav"}' -o audio.wav
```



### For Production (Docker)
- **Prerequisites**: Docker and Docker Compose
- **Steps**:
  1. **Start the server**:
  For GPU
  ```bash
  docker compose -f compose.yaml up -d
  ```
  For CPU only
  ```bash
  docker compose -f cpu-compose.yaml up -d
  ```




## Building Docker Image
Build the Docker image locally:
```bash
docker build -t slabstech/docs_indic_server -f Dockerfile .
```

### Run the Docker Image
```bash
docker run --gpus all -it --rm -p 7860:7860 slabstech/docs_indic_server
```


-->