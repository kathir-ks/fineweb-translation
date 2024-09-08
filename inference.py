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
from decode import decode, merge

import json
import nltk
nltk.download('punkt')
import time
import fsspec
from fsspec import AbstractFileSystem
from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor

# start tracing
initialise_tracking()

def find_shards(fs, bucket, name, subset, node_id):

    shards = []
    try:
        files = fs.ls(f'{bucket}/{name}/{subset}/{node_id}/tokenized')

        for file in files:
            shards.append(int(file.split('.')[-2].split('/')[-1]))

        shards.sort()
        return shards
    
    except Exception as e:
        print(e)
        return []

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

            if padding_length > 260:
                
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

    @jax.jit()
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

    @jax.jit()
    def run_inference_step(batch, params, run_ds):

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

            return output
        
        except Exception as e:
            
            print(f"!Error in inference step: {e}")
            return []

    outputs = []
    _placeholder_entity_maps = []
    _ids = []

    for input, placeholder_entity_map, id in zip(inputs, placeholder_entity_maps, ids):
        output = run_inference_step(input, params, None)
        if len(output) > 0:
            outputs.append(output.tolist())
            _placeholder_entity_maps.append(placeholder_entity_map)
            _ids.append(id)

    assert len(_placeholder_entity_maps) == len(_ids)
    assert len(_ids) == len(outputs)

    print("Inference completed!")
    print(time.time() - t)
    
    meta_data = []
    if 'meta_data' in data.keys():
        meta_data = data['meta_data']

    return {'outputs' : outputs, 'placeholder_entity_maps' : _placeholder_entity_maps, 'ids' : _ids,'meta_data':meta_data ,'row' : row, 'shard': _shard}


def _main(shards, fs : AbstractFileSystem, model_path, bucket, name, subset, node_id, batch_size, lang):

    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction='en-indic')

    for i in shards:

        with fs.open(f'{bucket}/{name}/{subset}/{node_id}/tokenized/{i}.json', 'r') as f:
            data = json.load(f)

        model = FlaxIndicTransForConditionalGeneration.from_pretrained(model_path, local_files_only=True,dtype=jnp.float16,)
        print("model loaded!")

        params = replicate(model.params)
        print("model replicated!")

        output = main(model, params, data, batch_size)

        sentences = decode(output, ip, tokenizer, lang)

        sentences = merge(sentences['sentences'], sentences['ids'],sentences['meta_data'], sentences['row'], sentences['shard'])

        with fs.open(f'{bucket}/{name}/{subset}/{node_id}/output/{i}.json', 'w') as f:
            json.dump(sentences, f)

        fs.rm(f'{bucket}/{name}/{subset}/{node_id}/tokenized/{i}.json')
            
        del data, sentences

if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="Tanslate tokenized sentences")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--subset", type=str, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--tokenization_batch_size", type=int, default=64, required=False)
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--node_id", type=int, default=-1)
    parser.add_argument("--total_nodes", type=int, default=-1)
    parser.add_argument("--lang", type=str, required=True)

    args = parser.parse_args()
    name = args.name
    subset = args.subset
    batch_size = args.batch_size
    bucket = args.bucket
    node_id = args.node_id
    total_nodes = args.total_nodes
    lang = args.lang

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
        os.system(f'gsutil cp -R {bucket}/IndicTrans2/flax_weights/200m {curr_dir}/flax_weights/')

    curr_shard = 1

    if node_id == -1 and total_nodes == -1:
        node_id = pid 
        total_nodes = process_count

    shards = find_shards(fs, bucket, name, subset, node_id)
    _shards = []

    while(len(shards) > 0):

        print(shards)
        _main(shards, fs, model_path, bucket, name, subset, node_id, batch_size, lang)

        _shards = find_shards(fs, bucket, name, subset, node_id)
        
        updated_shards = []
        for _shard in _shards:
            if _shard not in shards:
                updated_shards.append(_shard)

        shards = updated_shards[:]
        updated_shards = []
        _shards = []


    # shards = []

    # for i in shards:

    #     if fs.isfile(f'{bucket}/{name}/{subset}/{i}/sentences.json'):
    #         continue

    #     if fs.isfile(f'{bucket}/{name}/{subset}/{i}/data.json'):
    #         with fs.open(f'{bucket}/{name}/{subset}/{i}/data.json', 'r') as f:
    #             data = json.load(f)
    #     else:
    #         continue

    #     model = FlaxIndicTransForConditionalGeneration.from_pretrained(model_path, local_files_only=True,dtype=jnp.float16,)
    #     print("model loaded!")
    #     params = replicate(model.params)
    #     print("model replicated!")

    #     output = main(model, params, data, batch_size)

    #     sentences = decode(output, ip, tokenizer, lang)

    #     sentences = merge(sentences['sentences'], sentences['ids'],sentences['meta_data'], sentences['row'], sentences['shard'])

    #     with fs.open(f'{bucket}/{name}/{subset}/{i}/sentences.json', 'w') as f:
    #         json.dump(sentences, f)
            
    #     with fs.open(f'{bucket}/{name}/{subset}/{i}/data.json', 'w') as f:
    #         json.dump({'row':data['row'], 'shard':data['shard']}, f)

    #     del model, params, data, sentences

    