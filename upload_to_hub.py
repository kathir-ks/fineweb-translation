"""
Upload translated output to the HuggingFace Hub.

Reads decoded sentence shards from local disk, flattens them into a
HuggingFace Dataset, and pushes with a timestamped name.  Cleans up
output files after upload.
"""

import argparse
import logging
import os
from datetime import datetime

from datasets import Dataset

from storage import read_json

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload translated data to HuggingFace Hub")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="~/data",
                        help="Local directory for data")
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

    data_dir = os.path.expanduser(args.data_dir)
    dataset = []

    for node in range(args.start, args.end + 1):
        try:
            output_dir = os.path.join(data_dir, args.name, args.subset, str(node), "output")
            files = os.listdir(output_dir)
            shards = sorted(int(f.split('.')[0]) for f in files if f.endswith('.json'))

            for shard_id in shards:
                shard_path = os.path.join(output_dir, f"{shard_id}.json")
                sentences = read_json(shard_path)
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
            output_dir = os.path.join(data_dir, args.name, args.subset, str(node), "output")
            files = os.listdir(output_dir)
            shards = sorted(int(f.split('.')[0]) for f in files if f.endswith('.json'))
            for shard_id in shards:
                os.remove(os.path.join(output_dir, f"{shard_id}.json"))
        except Exception as exc:
            logger.error("Cleanup error for node %d: %s", node, exc)