git clone https://github.com/fsspec/gcsfs/
cd gcsfs/
pip install .

cd ..

git clone https://github.com/alvations/sacremoses
cd sacremoses/
pip install -e .

cd ..

git clone https://github.com/pandas-dev/pandas
cd pandas/
python -m pip install -r requirements-dev.txt


git clone https://github.com/mrabarnett/mrab-regex
cd mrab-regex/
pip install .

git clone https://github.com/nltk/nltk.git
cd nltk/
pip install .

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

git clone https://github.com/google/jax
cd jax
python build/build.py
pip install dist/*.whl  # installs jaxlib (includes XLA)

