import os
import json
import argparse
import fsspec
from fsspec import AbstractFileSystem
import numpy as np

from IndicTransTokenizer import IndicTransTokenizer , IndicProcessor

def decode(data , ip : IndicProcessor, tokenizer : IndicTransTokenizer, lang : str):
    
    assert len(data['outputs']) == len(data['ids'])
    assert len(data['ids']) == len(data['placeholder_entity_maps'])
    
    row = data['row']
    shard = data['shard']
    ids = data['ids']
    
    sentences = []
    for output, placeholder_entity_map in zip(data['outputs'], data['placeholder_entity_maps']):
        output = tokenizer.batch_decode(np.asarray(output), src=False)
        output = ip.postprocess_batch(output, lang=lang, placeholder_entity_maps=placeholder_entity_map)
        sentences.append(output)

    return {'sentences': sentences, 'ids':ids, 'row':row, 'shard':shard}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Decode the output tokens to the desired language using indictranstokenizer")
    parser.add_argument("--name", type=str, required=True, help='name of the dataset huggingface')
    parser.add_argument("--subset", type=str, required=True, help='subset of the dataset')
    parser.add_argument("--direction", type=str, required=False, default='en-indic', help='direction of the indictranstokenizer')
    parser.add_argument("--lang", type=str, required=True, help='the target lanaguage')  # the IndicTransTokenizer only needs the target language
    parser.add_argument("--bucket", type=str, required=True, help='the gcs bucket in which the data is present')
    parser.add_argument("--resume", type=bool, default=False, required=False)

    args = parser.parse_args()

    name = args.name
    subset = args.subset
    direction = args.direction
    lang = args.lang
    bucket = args.bucket
    resume = args.resume

    fs : AbstractFileSystem = fsspec.core.url_to_fs(bucket)[0]

    files = fs.ls(f'{bucket}/{name}/{subset}')

    total_shards = len(files)
    curr_shard = 1

    # perform a binary search to reach the files that are yet to be decoded
    if resume:

        left = curr_shard
        right = total_shards

        while(left<=right):
            mid = left + int((right - left)/2)
            if fs.isfile(f'{bucket}/{name}/{subset}/{mid}/sentences.json'):
                left = mid + 1
            else: 
                right = mid -1
        
        curr_shard = left

    print(curr_shard)
    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction=direction)

    for i in range(curr_shard, total_shards + 1, 1):

        try:
            with fs.open(f'{bucket}/{name}/{subset}/{i}/output.json', 'r') as f:
                output = json.load(f)

            sentences = decode(output, ip, tokenizer, lang)

            with fs.open(f'{bucket}/{name}/{subset}/{i}/sentences.json', 'w') as f:
                json.dump(sentences, f) 

        except:
            print(f'The file {bucket}/{name}/{subset}/{i}/output.json is not available')