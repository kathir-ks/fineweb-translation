import os
import jax
# Initialize jax distributed
jax.distributed.initialize()

import jax.numpy as jnp
import numpy as np
import argparse
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from modeling_flax_indictrans import FlaxIndicTransForConditionalGeneration
from jax_smi import initialise_tracking

import json
import nltk
nltk.download('punkt')
import time
import fsspec
from fsspec import AbstractFileSystem


# start tracing
initialise_tracking()

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def padding_fn(
        batch,
        keys_to_pad=[
                ("input_ids", 1),
                ("attention_mask", 0),
            ]
        ):

        batch_out = {key: [] for key in batch.keys()}
    
        for key in batch_out.keys():
            batch_out[key] += batch[key]
    
        for key, value_to_pad_with in keys_to_pad:

            len_list = list(map(lambda x: len(x), batch_out[key]))

            padding_length = max(len_list)

            if padding_length > 256:
                
                print(padding_length)

                return None
            
            array_list = []
            for i, x in enumerate(batch_out[key]):

                if len(x) < padding_length:
                    padded_array = np.concatenate([np.full((padding_length - len(x)), value_to_pad_with), np.array(x)])
                    array_list.append(padded_array)
                else:
                    array_list.append(np.array(x))

            batch_out[key] = np.stack(array_list)

        return batch_out


def main(model, params, data, batch_size):
        
    t = time.time()
    local_device_count = jax.local_device_count()
    inputs = []

    # make an extended list of input_ids, attention_mask , placeholder_entity_maps and ids and then create batches 
    # because inference_batch_size and tokenization_batch_size may differ 

    row = data['row']
    _shard = data['shard']
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

    del data

    assert len(_ids) == len(input_ids)
    assert len(input_ids) == len(attention_mask)
    assert len(attention_mask) == len(_placeholder_entity_maps)

    placeholder_entity_maps = []
    ids = []

    for i in range(0, len(input_ids), batch_size):
        
        input = {
            "input_ids": input_ids[i : i + batch_size],
            "attention_mask": attention_mask[i : i + batch_size]
        }
        
        input = padding_fn(input)
        if input and len(input['input_ids']) % local_device_count==0:
            inputs.append(input)
            placeholder_entity_maps.append(_placeholder_entity_maps[i : i + batch_size])
            ids.append(_ids[i : i + batch_size])

    del _placeholder_entity_maps
    del _ids

    assert len(inputs) == len(placeholder_entity_maps)
    assert len(placeholder_entity_maps) == len(ids)

    def generate(
            batch,
            params,
        ):
            model.params = params
            return model.generate(
                **batch,
                num_beams=1,
                num_return_sequences=1,
                max_length=256,
                do_sample=False,
            ).sequences

    p_generate = jax.pmap(generate) 

    # no need to jit the generate function because in jax by default pmapped functions are jitted!

    def run_inference_step(batch, params, run_ds):
        
        output = None
        try:
            input_batch = {
                "input_ids": shard(jnp.array(batch["input_ids"])),
                "attention_mask": shard(jnp.array(batch["attention_mask"]))
            }
            output = p_generate(input_batch, params)
            output = output.block_until_ready()
            if local_device_count != 1:
                output = output.reshape(-1, *output.shape[2:])
            else:
                output = output[0]
        except:
            print("!Error in inference step")

        return output

    outputs = []
    _placeholder_entity_maps = []
    _ids = []

    for input, placeholder_entity_map, id in zip(inputs, placeholder_entity_maps, ids):
        output = run_inference_step(input, params, None)
        if output is not None:
            outputs.append(output.tolist())
            _placeholder_entity_maps.append(placeholder_entity_map)
            _ids.append(id)

    assert len(_placeholder_entity_maps) == len(_ids)
    assert len(_ids) == len(outputs)

    print("Inference completed!")
    print(time.time() - t)
    
    return {'outputs' : outputs, 'placeholder_entity_maps' : placeholder_entity_maps, 'ids' : ids, 'row' : row, 'shard': _shard}

if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="Tanslate tokenized sentences")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--subset", type=str, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--tokenization_batch_size", type=int, default=64, required=False)
    parser.add_argument("--bucket", type=str, required=True)
    
    args = parser.parse_args()
    name = args.name
    subset = args.subset
    batch_size = args.batch_size
    bucket = args.bucket

    fs : AbstractFileSystem = fsspec.core.url_to_fs(bucket)[0]

    pid = jax.process_index()

    print(pid)

    global_devices = jax.device_count()
    local_devices = jax.local_device_count()

    process_count = jax.process_count()

    # print(global_devices)
    
    curr_dir = os.getcwd()
    model_path = f'{curr_dir}/flax_weights/200m'
    
    if not os.path.isdir(model_path):
        os.system("mkdir flax_weights")
        os.system(f'gsutil cp -R gs://indic-llama-data/indic-llama/flax_weights/200m {curr_dir}/flax_weights/')

    curr_shard = 1
    files = fs.ls(f'{bucket}/{name}/{subset}')
    total_shards = len(files)

    for i in range(1, total_shards + 1, 1):
        if fs.isfile(f'{bucket}/{name}/{subset}/{i}/output.json'):
            curr_shard = i + 1
        else:
            break
    
    curr_shard = curr_shard + pid
    
    print("starting from shard ",curr_shard)

    for i in range(curr_shard, total_shards + 1, process_count):
        
        model = FlaxIndicTransForConditionalGeneration.from_pretrained(model_path, local_files_only=True,dtype=jnp.float16,)
        print("model loaded!")
        params = replicate(model.params)
        print("model replicated!")

        if fs.isfile(f'{bucket}/{name}/{subset}/{i}/output.json'):
            continue

        if not fs.isfile(f'{bucket}/{name}/{subset}/{i}/data.json'):
            print(f'{bucket}/{name}/{subset}/{i}/data.json not found')
            break

        with fs.open(f'{bucket}/{name}/{subset}/{i}/data.json', 'r') as f:
            data = json.load(f)

        output = main(model, params, data, batch_size)

        with fs.open(f'{bucket}/{name}/{subset}/{i}/output.json', 'w') as f:
            json.dump(output, f)
            
        del model, params

    