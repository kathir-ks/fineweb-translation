"""
Centralised storage helpers for reading/writing JSON to GCS (or any
fsspec-compatible filesystem).

All file I/O that previously lived inline in tokenization, inference,
and decode scripts is now routed through these helpers.
"""

import json
import logging
from typing import Any, Optional

import fsspec
from fsspec import AbstractFileSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filesystem bootstrap
# ---------------------------------------------------------------------------

def get_fs(bucket: str) -> AbstractFileSystem:
    """Return an fsspec filesystem object for *bucket*."""
    return fsspec.core.url_to_fs(bucket)[0]


# ---------------------------------------------------------------------------
# Generic JSON I/O
# ---------------------------------------------------------------------------

def write_json(fs: AbstractFileSystem, path: str, data: Any) -> None:
    """Serialise *data* as JSON and write it to *path*."""
    with fs.open(path, 'w') as f:
        json.dump(data, f)
    logger.debug("Wrote %s", path)


def read_json(fs: AbstractFileSystem, path: str) -> Any:
    """Read and deserialise JSON from *path*."""
    with fs.open(path, 'r') as f:
        data = json.load(f)
    logger.debug("Read %s", path)
    return data


# ---------------------------------------------------------------------------
# Tokenization-specific helpers
# ---------------------------------------------------------------------------

def write_tokenized_shard(
    fs: AbstractFileSystem,
    bucket: str,
    name: str,
    subset: str,
    shard: int,
    total_nodes: int,
    data: dict,
) -> None:
    """Write a tokenized shard JSON to its node-partitioned path."""
    path = f'{bucket}/{name}/{subset}/{shard % total_nodes}/tokenized/{shard}.json'
    write_json(fs, path, data)
    logger.info("Saved tokenized shard %d -> %s", shard, path)


def save_tokenization_checkpoint(
    fs: AbstractFileSystem,
    bucket: str,
    name: str,
    subset: str,
    file_no: int,
    row: int,
    shard: int,
) -> None:
    """Persist a tokenization progress checkpoint."""
    path = f'{bucket}/{name}/{subset}/tokenization_meta_data_{file_no}.json'
    write_json(fs, path, {'row': row, 'shard': shard, 'file': file_no})


def load_tokenization_checkpoint(
    fs: AbstractFileSystem,
    bucket: str,
    name: str,
    subset: str,
    file_no: int,
) -> Optional[dict]:
    """Load a tokenization checkpoint, or return ``None`` if it doesn't exist."""
    path = f'{bucket}/{name}/{subset}/tokenization_meta_data_{file_no}.json'
    if fs.exists(path):
        data = read_json(fs, path)
        logger.info("Resuming file %d from row %d, shard %d", file_no, data['row'], data['shard'])
        return data
    return None


# ---------------------------------------------------------------------------
# Inference-specific helpers
# ---------------------------------------------------------------------------

def find_shards(
    fs: AbstractFileSystem,
    bucket: str,
    name: str,
    subset: str,
    node_id: int,
) -> list[int]:
    """List available tokenized shard indices for a given node."""
    try:
        files = fs.ls(f'{bucket}/{name}/{subset}/{node_id}/tokenized')
        shards = sorted(
            int(f.split('.')[-2].split('/')[-1]) for f in files
        )
        return shards
    except Exception as exc:
        logger.warning("Could not list shards for node %d: %s", node_id, exc)
        return []
