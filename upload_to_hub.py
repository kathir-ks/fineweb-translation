"""
Upload translated output from GCS to the HuggingFace Hub.

Reads decoded sentence shards, flattens them into a HuggingFace Dataset,
and pushes with a timestamped name.  Cleans up output files after upload.
"""

import argparse
import logging
from datetime import datetime

from datasets import Dataset

from storage import get_fs, read_json

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload translated data to HuggingFace Hub")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--total_nodes", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    fs = get_fs(args.bucket)
    dataset = []

    for node in range(args.start, args.end + 1):
        try:
            files = fs.ls(f'{args.bucket}/{args.name}/{args.subset}/{node}/output')
            shards = sorted(int(f.split('.')[-2].split('/')[-1]) for f in files)

            for shard_id in shards:
                sentences = read_json(
                    fs,
                    f'{args.bucket}/{args.name}/{args.subset}/{node}/output/{shard_id}.json',
                )
                for text, uuid, meta in zip(
                    sentences['text'], sentences['uuid'], sentences['meta_data'],
                ):
                    dataset.append({'text': text, 'uuid': uuid, 'meta_data': meta})

        except Exception as exc:
            logger.error("Error reading node %d: %s", node, exc)

    if dataset:
        ds = Dataset.from_list(dataset)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        hub_name = f'{args.subset}_row_wise_{tag}'
        ds.push_to_hub(hub_name)
        logger.info("Pushed %d rows to %s", len(dataset), hub_name)

    # Cleanup
    for node in range(args.start, args.end + 1):
        try:
            files = fs.ls(f'{args.bucket}/{args.name}/{args.subset}/{node}/output')
            shards = sorted(int(f.split('.')[-2].split('/')[-1]) for f in files)
            for shard_id in shards:
                fs.rm(f'{args.bucket}/{args.name}/{args.subset}/{node}/output/{shard_id}.json')
        except Exception as exc:
            logger.error("Cleanup error for node %d: %s", node, exc)