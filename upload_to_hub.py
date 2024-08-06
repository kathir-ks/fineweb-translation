from datasets import Dataset
import argparse
import fsspec
import json
from fsspec import AbstractFileSystem
from datetime import datetime

parser = argparse.ArgumentParser(description="")
parser.add_argument("--name", type=str)
parser.add_argument("--subset", type=str)
parser.add_argument("--bucket", type=str)
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)


args = parser.parse_args()

name = args.name
subset = args.subset
start = args.start
end = args.end
bucket = args.bucket

fs : AbstractFileSystem = fsspec.core.url_to_fs(f'{bucket}')[0]

dataset = []
files = fs.ls(f'{bucket}/{name}/{subset}')

shards = []
max_shard = -1
for file in files:
    shard_no = int(file.split('/')[-1])
    shards.append(shard_no)
    if shard_no > max_shard and shard_no <= end:
        max_shard = shard_no

shards.sort()

_row = -1
_shard = -1

for shard in shards:
    
    if shard < start:
          
          continue
    if shard > max_shard:
        break
    
    if fs.isfile(f'{bucket}/{name}/{subset}/{shard}/sentences.json'):
        try:
            with fs.open(f'{bucket}/{name}/{subset}/{shard}/sentences.json', 'r') as f:
                sentences = json.load(f)
                _row = sentences['row']
                _shard = sentences['shard']
                if 'meta_data' in sentences.keys():
                    if len(sentences['meta_data']) > 0:
                        for i, j, k in zip(sentences['text'], sentences['uuid'], sentences['meta_data']):
                            dataset.append({'text':i, 'uuid':j, 'meta_data':k})
                    else:
                        for i, j in zip(sentences['text'], sentences['uuid']):
                            dataset.append({'text':i, 'uuid':j })
        
                else:
                    for i, j in zip(sentences['text'], sentences['uuid']):
                        dataset.append({'text':i, 'uuid':j })   

        except Exception as e:
            print(e)         

if len(dataset) > 0:    
    dataset_to_upload = Dataset.from_list(dataset)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_to_upload.push_to_hub(f'{subset}_{start}_{end}_row_wise_{current_time}')


for shard in shards:

    if shard < start:
        continue

    if shard == max_shard:
        try:
            if fs.isfile(f'{bucket}/{name}/{subset}/{shard}/sentences.json'):
               with fs.open(f'{bucket}/{name}/{subset}/{shard}/sentences.json', 'w') as f: 
                    json.dump({'row':_row, 'shard':_shard})
            break
        except Exception as e:
            print(e)

    if shard > max_shard:
        break

    if fs.isfile(f'{bucket}/{name}/{subset}/{shard}/sentences.json'):
        fs.rm(f'{bucket}/{name}/{subset}/{shard}/sentences.json')
        fs.rm(f'{bucket}/{name}/{subset}/{shard}/data.json')
        fs.rmdir(f'{bucket}/{name}/{subset}/{shard}')
            