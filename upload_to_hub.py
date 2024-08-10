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
parser.add_argument("--total_nodes", type=int)
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)

args = parser.parse_args()

name = args.name
subset = args.subset
bucket = args.bucket
total_nodes= args.total_nodes
start = args.start
end = args.end

fs : AbstractFileSystem = fsspec.core.url_to_fs(f'{bucket}')[0]

dataset = []

for node in range(start, end + 1):
    try:
        files = fs.ls(f'{bucket}/{name}/{subset}/{node}/output')
        shards = []
        for file in files:
            shards.append(int(file.split('.')[-2].split('/')[-1]))
        shards.sort()

        for shard in shards:
            with fs.open(f'{bucket}/{name}/{subset}/{node}/output/{shard}.json', 'r') as f:
                sentences = json.load(f)
                for i, j, k in zip(sentences['text'], sentences['uuid'], sentences['meta_data']):
                    dataset.append({'text':i, 'uuid':j, 'meta_data':k})
    
    except Exception as e:
        print(e)
                         
if len(dataset) > 0:    
    dataset_to_upload = Dataset.from_list(dataset)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_to_upload.push_to_hub(f'{subset}_row_wise_{current_time}')
            
for i in range(start, end + 1):
    try:
        files = fs.ls(f'{bucket}/{name}/{subset}/{i}/output')
        shards = []
        for file in files:
            shards.append(int(file.split('.')[-2].split('/')[-1]))
        shards.sort()

        for shard in shards:
            fs.rm(f'{bucket}/{name}/{subset}/{i}/output/{shard}.json')

    except Exception as e:
        print(e)