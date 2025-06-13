git clone https://github.com/slabstech/docs-indic-server.git

cd docs-indic-server

sudo apt-get update

sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

python3.10 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

export HF_HOME=/home/ubuntu/data-dhwani-models
#export VLLM_USE_V1=1