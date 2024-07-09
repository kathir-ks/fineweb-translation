#!/bin/bash

pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch transformers datasets flax nltk fsspec gcsfc
sudo apt-get install golang -y
pip install jax-smi

mkdir flax_weights
