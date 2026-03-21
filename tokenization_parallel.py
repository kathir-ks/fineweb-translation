"""
Tokenization pipeline for the fineweb-translation project.

Streams the FineWeb-Edu dataset, splits documents into sentences,
tokenizes them with IndicTransTokenizer, and writes shards to a local
directory.

Supports multiprocessing (one process per parquet file) and resume
via per-file checkpoints.

Usage example
-------------
    python tokenization_parallel.py \
        --name HuggingFaceFW/fineweb-edu \
        --subset sample-10BT \
        --src_lang eng_Latn --tgt_lang hin_Deva \
        --tokenization_batch_size 64 \
        --data_dir ~/data \
        --shard_size 64000 \
        --total_nodes 4 \
        --total_files 99 \
        --start_file 0 --end_file 10
"""

import argparse
import logging
import os
from multiprocessing import Pool

from datasets import load_dataset
from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor

from utils import split_into_sentences, preprocess_and_tokenize
from storage import (
    write_tokenized_shard,
    save_tokenization_checkpoint,
    load_tokenization_checkpoint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and tokenize FineWeb-Edu for IndicTrans2 translation",
    )
    parser.add_argument("--name", default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, required=True,
                        help="Subset of the dataset")
    parser.add_argument("--streaming", default=True, type=bool,
                        help="Whether to stream the dataset")
    parser.add_argument("--src_lang", type=str, required=True,
                        help="Source language code (e.g. eng_Latn)")
    parser.add_argument("--tgt_lang", type=str, required=True,
                        help="Target language code (e.g. hin_Deva)")
    parser.add_argument("--tokenization_batch_size", type=int, required=True,
                        help="Batch size for tokenization")
    parser.add_argument("--data_dir", type=str, default="~/data",
                        help="Local directory to store shards")
    parser.add_argument("--shard_size", type=int, default=64000,
                        help="Number of sentences per shard")
    parser.add_argument("--resume", type=bool, default=False,
                        help="Resume from last checkpoint")
    parser.add_argument("--total_nodes", type=int, required=True,
                        help="Number of inference nodes (for shard partitioning)")
    parser.add_argument("--total_files", type=int, required=True,
                        help="Number of shards to split the dataset into")
    parser.add_argument("--start_file", type=int, default=0,
                        help="First file index to process (inclusive, default 0)")
    parser.add_argument("--end_file", type=int, default=None,
                        help="Last file index to process (exclusive, default total_files)")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(name, subset, streaming, file_no, total_files, split="train"):
    """Load a single parquet shard from the dataset by index.

    Uses the HuggingFace named config (subset) so the correct data_files
    are resolved automatically, then selects a single shard by index.
    """
    ds = load_dataset(
        name, subset, streaming=streaming, split=split,
    )
    # When streaming, shard() picks every total_files-th example starting at file_no
    return ds.shard(num_shards=total_files, index=file_no)


# ---------------------------------------------------------------------------
# Per-shard tokenization
# ---------------------------------------------------------------------------

def tokenize_shard(
    sentences,
    temp_ids,
    meta_data,
    src_lang,
    tgt_lang,
    tokenization_batch_size,
    name,
    subset,
    data_dir,
    shard,
    total_nodes,
    row,
):
    """Tokenize a shard of sentences and write the result to disk."""
    tokenized_inputs = []
    ids = []

    assert len(sentences) == len(temp_ids)

    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction='en-indic')

    for i in range(0, len(sentences), tokenization_batch_size):
        try:
            tokenized_inputs.append(
                preprocess_and_tokenize(
                    tokenizer, ip,
                    sentences[i:i + tokenization_batch_size],
                    src_lang, tgt_lang,
                )
            )
            ids.append(temp_ids[i:i + tokenization_batch_size])
        except TimeoutError as exc:
            ip.get_placeholder_entity_maps(clear_ple_maps=True)
            logger.warning("Tokenization timed out at offset %d: %s", i, exc)

    assert len(tokenized_inputs) == len(ids)

    data = {
        'tokenized_inputs': tokenized_inputs,
        'ids': ids,
        'row': row,
        'shard': shard,
        'meta_data': meta_data,
    }
    write_tokenized_shard(data_dir, name, subset, shard, total_nodes, data)


# ---------------------------------------------------------------------------
# Per-file worker (one per multiprocessing.Pool process)
# ---------------------------------------------------------------------------

def process_file(args_tuple):
    """Process a single parquet file: split into sentences, tokenize, shard."""
    (
        name, subset, src_lang, tgt_lang, streaming,
        tokenization_batch_size, data_dir, shard_size,
        total_nodes, file_no, total_files, resume,
    ) = args_tuple

    sentences = []
    temp_ids = []
    meta_data = []

    row = 0
    shard_start = file_no * 2500 + 1
    shard = shard_start
    tokenized_rows = 0

    # --- Resume from checkpoint if available ---
    checkpoint = load_tokenization_checkpoint(data_dir, name, subset, file_no)
    if checkpoint is not None:
        resume = True
        tokenized_rows = checkpoint['row']
        shard = checkpoint['shard']

    # --- Stream the parquet file ---
    data = load_data(name, subset, streaming, file_no, total_files)

    for d in data:
        if resume and row < (tokenized_rows - 1):
            row += 1
            continue

        sents = split_into_sentences(d['text'])
        temp_ids.extend([d['id']] * len(sents))
        meta_data.append({
            'id': d['id'], 'dump': d['dump'],
            'url': d['url'], 'file_path': d['file_path'],
        })
        sentences.extend(sents)
        row += 1

        if len(sentences) >= shard_size:
            tokenize_shard(
                sentences[:shard_size], temp_ids[:shard_size], meta_data,
                src_lang, tgt_lang, tokenization_batch_size,
                name, subset, data_dir, shard, total_nodes, row,
            )
            sentences = sentences[shard_size:]
            temp_ids = temp_ids[shard_size:]
            meta_data = [meta_data[-1]] if temp_ids else []

            assert len(sentences) == len(temp_ids)
            if meta_data:
                assert meta_data[0]['id'] == temp_ids[0]
                assert meta_data[0]['id'] == temp_ids[-1]
            shard += 1

        # Periodic checkpoint
        if row % 1000 == 0:
            save_tokenization_checkpoint(
                data_dir, name, subset, file_no, row, shard,
            )

    # Flush remaining sentences
    if sentences:
        tokenize_shard(
            sentences, temp_ids, meta_data,
            src_lang, tgt_lang, tokenization_batch_size,
            name, subset, data_dir, shard, total_nodes, row,
        )

    # Final checkpoint
    save_tokenization_checkpoint(data_dir, name, subset, file_no, row, shard)
    logger.info("File %d complete — final shard %d, rows %d", file_no, shard, row)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(args):
    data_dir = os.path.expanduser(args.data_dir)
    start = args.start_file
    end = args.end_file if args.end_file is not None else args.total_files
    num_files = end - start

    process_args = [
        (
            args.name, args.subset, args.src_lang, args.tgt_lang,
            args.streaming, args.tokenization_batch_size,
            data_dir, args.shard_size, args.total_nodes,
            i, args.total_files, args.resume,
        )
        for i in range(start, end)
    ]

    logger.info(
        "Starting tokenization — files %d to %d (%d files), pool size %d",
        start, end - 1, num_files, num_files,
    )
    with Pool(processes=num_files) as pool:
        pool.map(process_file, process_args)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(args)