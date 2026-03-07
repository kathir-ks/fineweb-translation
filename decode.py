"""
Decoding and sentence-merging for the fineweb-translation pipeline.

Takes raw model output tokens, decodes them via IndicTransTokenizer,
and reconstructs per-document translated text by merging sentences
that share the same document UUID.
"""

import argparse
import json
import logging

import numpy as np

from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor
from storage import get_fs, read_json, write_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def decode(data, ip: IndicProcessor, tokenizer: IndicTransTokenizer, lang: str):
    """Decode output token arrays back to text for a single shard."""
    assert len(data['outputs']) == len(data['ids']) == len(data['placeholder_entity_maps'])

    sentences = []
    for output, pe_map in zip(data['outputs'], data['placeholder_entity_maps']):
        output = tokenizer.batch_decode(np.asarray(output), src=False)
        output = ip.postprocess_batch(output, lang=lang, placeholder_entity_maps=pe_map)
        sentences.append(output)

    return {
        'sentences': sentences,
        'ids': data['ids'],
        'row': data['row'],
        'shard': data['shard'],
        'meta_data': data['meta_data'],
    }


def merge(_sentences, _ids, _meta_data, row, shard):
    """Merge sentence-level translations back into per-document text."""
    sentences = [s for batch in _sentences for s in batch]
    ids = [i for batch in _ids for i in batch]

    assert len(sentences) == len(ids)

    meta_lookup = {md['id']: md for md in _meta_data}

    uuid = []
    text = []
    meta_data = []
    prev_uuid = None

    for sentence, doc_id in zip(sentences, ids):
        if doc_id == prev_uuid:
            text[-1].append(sentence)
        else:
            prev_uuid = doc_id
            text.append([sentence])
            uuid.append(doc_id)
            if meta_lookup:
                meta_data.append(meta_lookup[doc_id])

    assert len(text) == len(uuid)
    if _meta_data:
        assert len(meta_data) == len(text)

    return {
        'text': text, 'uuid': uuid,
        'row': row, 'shard': shard,
        'meta_data': meta_data,
    }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Decode output tokens to target language using IndicTransTokenizer",
    )
    parser.add_argument("--name", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, required=True, help="Dataset subset")
    parser.add_argument("--direction", type=str, default='en-indic',
                        help="IndicTransTokenizer direction")
    parser.add_argument("--lang", type=str, required=True, help="Target language code")
    parser.add_argument("--bucket", type=str, required=True, help="GCS bucket URI")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--_from", type=int, required=True, help="Start shard index")
    parser.add_argument("--to", type=int, required=True, help="End shard index")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    fs = get_fs(args.bucket)
    files = fs.ls(f'{args.bucket}/{args.name}/{args.subset}')
    total_shards = len(files)

    curr_shard = 1
    if args.resume:
        left, right = curr_shard, total_shards
        while left <= right:
            mid = left + (right - left) // 2
            if fs.isfile(f'{args.bucket}/{args.name}/{args.subset}/{mid}/sentences.json'):
                left = mid + 1
            else:
                right = mid - 1
        curr_shard = left

    logger.info("Starting decode from shard %d", curr_shard)

    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction=args.direction)

    end = min(args.to, total_shards)

    for i in range(args._from, end + 1):
        try:
            output = read_json(fs, f'{args.bucket}/{args.name}/{args.subset}/{i}/output.json')
            if not output:
                continue

            sentences = decode(output, ip, tokenizer, args.lang)
            sentences = merge(
                sentences['sentences'], sentences['ids'],
                sentences['meta_data'], sentences['row'], sentences['shard'],
            )

            write_json(fs, f'{args.bucket}/{args.name}/{args.subset}/{i}/sentences.json', sentences)
            write_json(fs, f'{args.bucket}/{args.name}/{args.subset}/{i}/output.json', [])

        except Exception as exc:
            logger.error("Failed to decode shard %d: %s", i, exc)