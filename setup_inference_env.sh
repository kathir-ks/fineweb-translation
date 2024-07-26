#!/bin/bash

pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch transformers datasets flax nltk fsspec gcsfs
sudo apt-get install golang -y
pip install jax-smi

pip install sacremoses pandas regex mock "transformers>=4.33.2" mosestokenizer
pip install bitsandbytes scipy accelerate 
pip install sentencepiece sacrebleu 
pip install -U "huggingface_hub[cli]"

mkdir flax_weights

git clone https://www.github.com/kathir-ks/IndicTransTokenizer
cd IndicTransTokenizer
pip install --editable ./

sudo shutdown -r now