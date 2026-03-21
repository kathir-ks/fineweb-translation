#!/bin/bash

# Install required packages
pip install networkx==2.5
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install nltk sacremoses pandas regex mock "transformers>=4.33.2,<5.0" mosestokenizer
pip install bitsandbytes scipy accelerate datasets
pip install sentencepiece sacrebleu 
pip install "huggingface_hub[cli]<1.0"

# Install IndicTransTokenizer
# forked from https://github.com/VarunGumma/IndicTransTokenizer and the indicprocessor is modified according to 
# the indicprocessor from setu-translate https://github.com/AI4Bharat/setu-translate/blob/433723c52678cb79e54a04749e3d8a58737a2b35/IndicTransTokenizer/IndicTransTokenizer/utils.py#L189
# to get and add placeholder entity maps that is not present in the main repo

git clone https://www.github.com/kathir-ks/IndicTransTokenizer
cd IndicTransTokenizer
pip install --no-deps -e ./
pip install git+https://github.com/VarunGumma/indic_nlp_library

# sudo shutdown -r now