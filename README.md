# Docs-Indic-Server

## Overview
Document parser for Indian languages

## Table of Contents
- [Features](#features)
- [Getting Started - Development](#getting-started---development)
- [Downloading Model](#downloading-indic-model)
- [Running with FastAPI Server](#running-with-fastapi-server)
- [Evaluating Results](#evaluating-results)
- [Citations](#citations)

### Features 
  - Extract text from PDF - Single Page, Multiple, Full
  - Extract text from Image
  - Summary text from Image/PDF
    - English
    - Kannada
    - German
  - Recreate PDF -> Scanned doc to clean PDF
  - Convert PDF ->
    - English to Kannada
    - Kannada to English


### For Development 
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
  - ```bash
    pip install -r requirements.txt
    ```

- Backend Server  - Select based on GPU VRAM
  - ```bash
    vllm serve google/gemma-3-4b-it   
    ```
  - ```bash
    vllm serve reducto/RolmOCR
    ```
  - ```bash
    vllm serve google/gemma-3-12b-it
    ```

- for H100 only
  - google/gemma-3-12b-it

- for A100 only
  - google/gemma-3-12b-it


### Running with FastAPI Server
- 
  ```bash
  python src/server/docs_api.py --port 7860 --host 0.0.0.0
  ```




wget https://github.com/slabstech/docs-indic-server/blob/01e811210d56e655091313c1df8481d11e7640a6/install-script.sh
chmod +x install-script.sh
bash install-script.sh


### GPU server setup
  - Terminal 1 
    ```bash
    git clone https://github.com/slabstech/docs-indic-server.git
    cd docs-indic-server
    chmod +x install-script.sh
    bash install-script.sh
    export HF_TOKEN='YOUR-HF-TOKEN'
    export HF_HOME=/home/ubuntu/data-dhwani-models
    vllm serve google/gemma-3-4b-it
    ```
  - Terminal 2
    ```bash
    cd docs-indic-server
    source venv/bin/activate
    export HF_TOKEN='YOUR-HF-TOKEN'
    export HF_HOME=/home/ubuntu/data-dhwani-models
    python src/server/docs_api.py --port 7860 --host 0.0.0.0
    ```
  - Terminal 3
    ```bash
    git clone https://github.com/slabstech/indic-translate-server
    cd indic-translate-server
    python3.10 -m venv venv
    source venv/bin/activate
    pip install -r server-requirements.txt
    export HF_TOKEN='YOUR-HF-TOKEN'
    export HF_HOME=/home/ubuntu/data-dhwani-models
    huggingface-cli download ai4bharat/indictrans2-indic-en-dist-200M
    huggingface-cli download ai4bharat/indictrans2-en-indic-dist-200M
    python src/server/translate_api.py --port 7861 --host 0.0.0.0 --device cuda --use_distilled
    ```


-- 

For Translation 



## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

Also you can join the [discord group](https://discord.gg/WZMCerEZ2P) to collaborate

- Reference
    - [HF - moondream](https://huggingface.co/vikhyatk/moondream2)
    - [source - moondream](https://github.com/vikhyat/moondream)
    - [moondream-blog](https://moondream.ai/blog/introducing-a-new-moondream-1-9b-and-gpu-support)
    - [pixtral-12-b-2409](https://huggingface.co/mistralai/Pixtral-12B-2409)




<!-- 

## Download Qwen VL

```bash download_model.sh
huggingface_cli download google/gemma-3-4b-it
```

## Download Gemma

```bash download_model.sh
huggingface_cli download google/gemma-3-4b-it
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

## Getting Started - Development

- For moondream, libvips system library is required 
  ```
  sudo apt-get update && sudo apt-get install libvips
  ```



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

#  - vllm serve vikhyatk/moondream2 --trust-remote-code


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