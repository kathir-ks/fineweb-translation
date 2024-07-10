#!/bin/bash

pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch transformers datasets flax nltk fsspec gcsfs
sudo apt-get install golang -y
pip install jax-smi
pip install flax nltk

mkdir flax_weights