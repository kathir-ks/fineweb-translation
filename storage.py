"""
Centralised storage helpers for reading/writing JSON to the local filesystem.

All file I/O that previously lived inline in tokenization, inference,
and decode scripts is now routed through these helpers.
"""

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic JSON I/O
# ---------------------------------------------------------------------------

def write_json(path: str, data: Any) -> None:
    """Serialise *data* as JSON and write it to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)
    logger.debug("Wrote %s", path)


def read_json(path: str) -> Any:
    """Read and deserialise JSON from *path*."""
    with open(path, 'r') as f:
        data = json.load(f)
    logger.debug("Read %s", path)
    return data


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def shard_path(data_dir: str, name: str, subset: str, node_id: int, kind: str, shard: int) -> str:
    """Return the local path for a tokenized or output shard.

    *kind* is typically ``"tokenized"`` or ``"output"``.
    """
    return os.path.join(data_dir, name, subset, str(node_id), kind, f"{shard}.json")


# ---------------------------------------------------------------------------
# Tokenization-specific helpers
# ---------------------------------------------------------------------------

def write_tokenized_shard(
    data_dir: str,
    name: str,
    subset: str,
    shard: int,
    total_nodes: int,
    data: dict,
) -> None:
    """Write a tokenized shard JSON to its node-partitioned path."""
    path = shard_path(data_dir, name, subset, shard % total_nodes, "tokenized", shard)
    write_json(path, data)
    logger.info("Saved tokenized shard %d -> %s", shard, path)


def save_tokenization_checkpoint(
    data_dir: str,
    name: str,
    subset: str,
    file_no: int,
    row: int,
    shard: int,
) -> None:
    """Persist a tokenization progress checkpoint."""
    path = os.path.join(data_dir, name, subset, f"tokenization_meta_data_{file_no}.json")
    write_json(path, {'row': row, 'shard': shard, 'file': file_no})


def load_tokenization_checkpoint(
    data_dir: str,
    name: str,
    subset: str,
    file_no: int,
) -> Optional[dict]:
    """Load a tokenization checkpoint, or return ``None`` if it doesn't exist."""
    path = os.path.join(data_dir, name, subset, f"tokenization_meta_data_{file_no}.json")
    if os.path.exists(path):
        data = read_json(path)
        logger.info("Resuming file %d from row %d, shard %d", file_no, data['row'], data['shard'])
        return data
    return None


# ---------------------------------------------------------------------------
# Inference-specific helpers
# ---------------------------------------------------------------------------

def find_shards(
    data_dir: str,
    name: str,
    subset: str,
    node_id: int,
) -> list[int]:
    """List available tokenized shard indices for a given node."""
    shard_dir = os.path.join(data_dir, name, subset, str(node_id), "tokenized")
    try:
        files = os.listdir(shard_dir)
        shards = sorted(
            int(f.split('.')[0]) for f in files if f.endswith('.json')
        )
        return shards
    except Exception as exc:
        logger.warning("Could not list shards for node %d: %s", node_id, exc)
        return []


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def delete_file(path: str) -> None:
    """Remove a file if it exists."""
    try:
        os.remove(path)
        logger.debug("Deleted %s", path)
    except FileNotFoundError:
        pass
