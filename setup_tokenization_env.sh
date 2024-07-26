#!/bin/bash

# Install required packages
pip install nltk sacremoses pandas regex mock "transformers>=4.33.2" mosestokenizer
pip install bitsandbytes scipy accelerate datasets
pip install sentencepiece sacrebleu 
pip install fsspec gcsfs
pip install -U "huggingface_hub[cli]"

# Install IndicTransTokenizer
# forked from https://github.com/VarunGumma/IndicTransTokenizer and the indicprocessor is modified according to 
# the indicprocessor from setu-translate https://github.com/AI4Bharat/setu-translate/blob/433723c52678cb79e54a04749e3d8a58737a2b35/IndicTransTokenizer/IndicTransTokenizer/utils.py#L189
# to get and add placeholder entity maps that is not present in the main repo

git clone https://www.github.com/kathir-ks/IndicTransTokenizer
cd IndicTransTokenizer
pip install --editable ./

# sudo shutdown -r now