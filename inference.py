"""
Inference pipeline for fineweb-translation.

Loads tokenized shards from GCS, runs the IndicTrans2 Flax model on TPUs
via JAX pmap, decodes the output, and writes translated sentences back to GCS.

Supports multihost TPU setups (e.g. v4-256) via jax.distributed.initialize().
"""

import os
import jax
jax.distributed.initialize()

import argparse
import logging
import time

import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax_smi import initialise_tracking

from modeling_flax_indictrans import FlaxIndicTransForConditionalGeneration
from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor
from decode import decode, merge
from storage import get_fs, read_json, write_json, find_shards

logger = logging.getLogger(__name__)

initialise_tracking()


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def padding_fn(batch, keys_to_pad=None):
    """Pad variable-length sequences in *batch* to the longest length.

    Returns ``None`` if any sequence exceeds 260 tokens (safety guard).
    """
    if keys_to_pad is None:
        keys_to_pad = [("input_ids", 1), ("attention_mask", 0)]

    batch_out = {key: list(batch[key]) for key in batch}

    for key, pad_value in keys_to_pad:
        lengths = [len(x) for x in batch_out[key]]
        max_len = max(lengths)

        if max_len > 260:
            logger.warning("Skipping batch — max length %d exceeds 260", max_len)
            return None

        padded = []
        for x in batch_out[key]:
            if len(x) < max_len:
                padded.append(
                    np.concatenate([np.full(max_len - len(x), pad_value), np.array(x)])
                )
            else:
                padded.append(np.array(x))
        batch_out[key] = np.stack(padded)

    return batch_out


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model(model_path):
    """Load the IndicTrans2 model and create a pmap'd generate function.

    Returns (replicated_params, p_generate).
    """
    model = FlaxIndicTransForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=jnp.float16,
    )
    logger.info("Model loaded from %s", model_path)

    params = replicate(model.params, devices=jax.local_devices())
    logger.info("Params replicated across %d devices", jax.local_device_count())

    def generate(batch, params):
        model.params = params
        return model.generate(
            **batch,
            num_beams=1,
            num_return_sequences=1,
            max_length=256,
            do_sample=False,
        ).sequences

    p_generate = jax.pmap(generate, devices=jax.local_devices())

    return params, p_generate


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(p_generate, params, data, batch_size):
    """Run model inference on a single tokenized shard."""
    t = time.time()
    ldc = jax.local_device_count()

    row = data['row']
    _shard = data['shard']

    # Flatten all tokenized batches into contiguous lists
    input_ids = []
    attention_mask = []
    _placeholder_entity_maps = []
    _ids = []

    for i in data['ids']:
        _ids.extend(i)
    for i in data['tokenized_inputs']:
        input_ids.extend(i['batch']['input_ids'])
        attention_mask.extend(i['batch']['attention_mask'])
        _placeholder_entity_maps.extend(i['placeholder_entity_maps'])

    assert len(_ids) == len(input_ids) == len(attention_mask) == len(_placeholder_entity_maps)

    # Re-batch for inference
    inputs = []
    placeholder_entity_maps = []
    ids = []

    for i in range(0, len(input_ids), batch_size):
        inp = padding_fn({
            "input_ids": input_ids[i:i + batch_size],
            "attention_mask": attention_mask[i:i + batch_size],
        })
        if inp and len(inp['input_ids']) % ldc == 0:
            inputs.append(inp)
            placeholder_entity_maps.append(_placeholder_entity_maps[i:i + batch_size])
            ids.append(_ids[i:i + batch_size])

    del _placeholder_entity_maps, _ids

    assert len(inputs) == len(placeholder_entity_maps) == len(ids)

    def inference_step(batch):
        try:
            input_batch = {
                "input_ids": shard(jnp.array(batch["input_ids"])),
                "attention_mask": shard(jnp.array(batch["attention_mask"])),
            }
            output = p_generate(input_batch, params)
            output = output.block_until_ready()
            if ldc != 1:
                output = output.reshape(-1, *output.shape[2:])
            else:
                output = output[0]
            return output
        except Exception as exc:
            logger.error("Inference step failed: %s", exc)
            return []

    outputs = []
    _placeholder_entity_maps = []
    _ids = []

    for inp, pe_map, id_list in zip(inputs, placeholder_entity_maps, ids):
        output = inference_step(inp)
        if len(output) > 0:
            outputs.append(output.tolist())
            _placeholder_entity_maps.append(pe_map)
            _ids.append(id_list)

    assert len(_placeholder_entity_maps) == len(_ids) == len(outputs)

    logger.info("Inference done in %.1fs", time.time() - t)

    meta_data = data.get('meta_data', [])

    return {
        'outputs': outputs,
        'placeholder_entity_maps': _placeholder_entity_maps,
        'ids': _ids,
        'meta_data': meta_data,
        'row': row,
        'shard': _shard,
    }


# ---------------------------------------------------------------------------
# Per-node processing loop
# ---------------------------------------------------------------------------

def process_shards(shards, fs, p_generate, params, bucket, name, subset, node_id, batch_size, lang):
    """Run inference on each shard, decode, and save output."""
    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction='en-indic')

    for shard_id in shards:
        data = read_json(fs, f'{bucket}/{name}/{subset}/{node_id}/tokenized/{shard_id}.json')

        output = run_inference(p_generate, params, data, batch_size)
        sentences = decode(output, ip, tokenizer, lang)
        sentences = merge(
            sentences['sentences'], sentences['ids'],
            sentences['meta_data'], sentences['row'], sentences['shard'],
        )

        write_json(fs, f'{bucket}/{name}/{subset}/{node_id}/output/{shard_id}.json', sentences)
        fs.rm(f'{bucket}/{name}/{subset}/{node_id}/tokenized/{shard_id}.json')

        del data, sentences


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translate tokenized sentences using IndicTrans2")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--node_id", type=int, default=-1)
    parser.add_argument("--total_nodes", type=int, default=-1)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    fs = get_fs(args.bucket)
    pid = jax.process_index()
    logger.info("JAX process index: %d", pid)

    curr_dir = os.getcwd()
    model_path = f'{curr_dir}/flax_weights/200m'

    if not os.path.isdir(model_path):
        os.makedirs('flax_weights', exist_ok=True)
        os.system(f'gsutil cp -R {args.bucket}/IndicTrans2/flax_weights/200m {curr_dir}/flax_weights/')

    # Load model and create pmap'd generate — once for all shards
    params, p_generate = load_model(model_path)

    node_id = args.node_id if args.node_id != -1 else pid
    total_nodes = args.total_nodes if args.total_nodes != -1 else jax.process_count()

    shard_list = find_shards(fs, args.bucket, args.name, args.subset, node_id)

    while shard_list:
        logger.info("Processing shards: %s", shard_list)
        process_shards(
            shard_list, fs, p_generate, params, args.bucket,
            args.name, args.subset, node_id, args.batch_size, args.lang,
        )

        # Check for newly arrived shards
        new_shards = find_shards(fs, args.bucket, args.name, args.subset, node_id)
        shard_list = [s for s in new_shards if s not in shard_list]
