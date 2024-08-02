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

    return {'sentences': sentences, 'ids':ids, 'row':row, 'shard':shard,'meta_data': data['meta_data']}

def merge(_sentences, _ids,meta_data, row, shard):
    sentences = []
    for sentence in _sentences:
        sentences.extend(sentence)
    
    ids = []
    for id in _ids:
        ids.extend(id)

    assert len(sentences) == len(ids)

    uuid = []
    text = []
    prev_uuid = -1
    for sentence, id in zip(sentences, ids):
        if id == prev_uuid:
            text[-1].append(sentence)
        else:
            prev_uuid = id
            text.append([sentence])
            uuid.append(id)

    assert len(text) == len(uuid)
    if len(meta_data) > 0:
        assert len(meta_data) == len(text)
    
    return {'text':text, 'uuid':uuid, 'row':row, 'shard':shard, 'meta_data':meta_data}
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Decode the output tokens to the desired language using indictranstokenizer")
    parser.add_argument("--name", type=str, required=True, help='name of the dataset huggingface')
    parser.add_argument("--subset", type=str, required=True, help='subset of the dataset')
    parser.add_argument("--direction", type=str, required=False, default='en-indic', help='direction of the indictranstokenizer')
    parser.add_argument("--lang", type=str, required=True, help='the target lanaguage')  # the IndicTransTokenizer only needs the target language
    parser.add_argument("--bucket", type=str, required=True, help='the gcs bucket in which the data is present')
    parser.add_argument("--resume", type=bool, default=False, required=False)
    parser.add_argument("--_from",type=int)
    parser.add_argument("--to", type=int)

    args = parser.parse_args()

    name = args.name
    subset = args.subset
    direction = args.direction
    lang = args.lang
    bucket = args.bucket
    resume = args.resume
    _from = args._from
    to = args.to
    
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

    if to > total_shards:
        to = total_shards
     
    for i in range(_from, to + 1, 1):

        try:
            with fs.open(f'{bucket}/{name}/{subset}/{i}/output.json', 'r') as f:
                output = json.load(f)

            if len(output) == 0:
                continue

            sentences = decode(output, ip, tokenizer, lang)

            sentences = merge(sentences['sentences'], sentences['ids'],sentences['meta_data'] , sentences['row'], sentences['shard'])

            with fs.open(f'{bucket}/{name}/{subset}/{i}/sentences.json', 'w') as f:
                json.dump(sentences, f) 
            
            # empyt the output file
            with fs.open(f'{bucket}/{name}/{subset}/{i}/output.json', 'w') as f:
                json.dump([], f)

        except:
            print(f'The file {bucket}/{name}/{subset}/{i}/output.json is not available')