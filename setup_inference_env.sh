#!/bin/bash

pip install networkx==2.5
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch "transformers>=4.33.2,<5.0" datasets flax nltk
sudo apt-get install golang -y
pip install jax-smi

pip install sacremoses pandas regex mock mosestokenizer
pip install bitsandbytes scipy accelerate 
pip install sentencepiece sacrebleu 
pip install "huggingface_hub[cli]<1.0"

mkdir -p flax_weights

git clone https://www.github.com/kathir-ks/IndicTransTokenizer
cd IndicTransTokenizer
pip install --no-deps -e ./
pip install git+https://github.com/VarunGumma/indic_nlp_library

# Optional: download model from HuggingFace
if [ -n "${MODEL_REPO:-}" ]; then
    echo "Downloading model from $MODEL_REPO..."
    huggingface-cli download "$MODEL_REPO" --local-dir ../flax_weights/200m
fi

# sudo shutdown -r now