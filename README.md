# fineweb-translation

Translate the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset to Indian languages using [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) on Google Cloud TPUs.

Supports single-host (v3-8, v4-8) and multihost (v4-256) TPU topologies.

## Pipeline overview

```
 HuggingFace Dataset
        │
        ▼
 ┌──────────────┐
 │ Tokenization │  tokenization_parallel.py
 │  (CPU VM)     │  Streams parquet files, splits into sentences,
 │              │  tokenizes with IndicTransTokenizer, writes
 │              │  shards to GCS partitioned by node.
 └──────┬───────┘
        │  GCS: {bucket}/{name}/{subset}/{node_id}/tokenized/{shard}.json
        ▼
 ┌──────────────┐
 │  Inference   │  inference.py
 │  (TPU VM)    │  Loads shards, runs IndicTrans2 Flax model via
 │              │  jax.pmap across all TPU chips, writes output
 │              │  back to GCS.
 └──────┬───────┘
        │  GCS: {bucket}/{name}/{subset}/{node_id}/output/{shard}.json
        ▼
 ┌──────────────┐
 │   Decode     │  decode.py (called by inference.py, or standalone)
 │              │  Decodes token IDs back to text, merges sentences
 │              │  into per-document translations.
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │  Upload      │  upload_to_hub.py
 │              │  Reads decoded output from GCS, pushes to
 │              │  HuggingFace Hub as a Dataset.
 └──────────────┘
```

## Setup

### Inference environment (TPU VM)

```bash
chmod +x setup_inference_env.sh
./setup_inference_env.sh
```

### Tokenization environment (CPU VM)

```bash
chmod +x setup_tokenization_env.sh
./setup_tokenization_env.sh
```

## Usage

### 1. Tokenization

```bash
python3 tokenization_parallel.py \
    --name HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --tokenization_batch_size 64 \
    --bucket gs://my-bucket \
    --shard_size 64000 \
    --total_nodes 4 \
    --total_files 99 \
    --start_file 0 --end_file 10
```

### 2. Inference

```bash
# Single-host TPU (e.g. v4-8)
python3 inference.py \
    --name HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --batch_size 256 \
    --bucket gs://my-bucket \
    --lang hin_Deva

# Multihost TPU — run on every worker (jax.distributed handles coordination)
# node_id and total_nodes default to jax.process_index() and jax.process_count()
python3 inference.py \
    --name HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --batch_size 256 \
    --bucket gs://my-bucket \
    --lang hin_Deva
```

### 3. Upload to HuggingFace Hub

```bash
python3 upload_to_hub.py \
    --name HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --bucket gs://my-bucket \
    --total_nodes 4 \
    --start 0 --end 3
```

## Infrastructure scripts

Launcher scripts that create TPU VMs, set up the environment, and handle preemption retries:

```bash
# Inference — single-host
./run_inference.sh --vm_name worker-0 --region us-central2-b \
    --accelerator_type v4-8 --bucket gs://my-bucket \
    --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
    --batch_size 256 --lang hin_Deva

# Inference — multihost (e.g. v4-256)
./run_inference.sh --vm_name main-1 --region us-central2-b \
    --accelerator_type v4-256 --multihost \
    --bucket gs://my-bucket --dataset HuggingFaceFW/fineweb-edu \
    --subset sample-10BT --batch_size 256 --lang hin_Deva

# Tokenization
./run_tokenization.sh --vm_name tpu-tok-0 --region us-central2-b \
    --accelerator_type v4-8 --bucket gs://my-bucket \
    --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --tokenization_batch_size 64 --shard_size 64000 \
    --total_nodes 4 --total_files 99 \
    --start_file 0 --end_file 10
```

## Project structure

```
├── inference.py                 # TPU inference pipeline (multihost-aware)
├── tokenization_parallel.py     # Sentence splitting + tokenization
├── decode.py                    # Token decoding + sentence merging
├── upload_to_hub.py             # Push results to HuggingFace Hub
├── storage.py                   # GCS I/O helpers (fsspec)
├── utils.py                     # Text cleaning, sentence splitting, tokenization
├── modeling_flax_indictrans.py  # Flax IndicTrans2 model
├── configuration_indictrans.py  # Model configuration
├── setup_inference_env.sh       # TPU inference env setup
├── setup_tokenization_env.sh    # Tokenization env setup
├── run_inference.sh             # TPU VM inference launcher
└── run_tokenization.sh          # TPU VM tokenization launcher
```
