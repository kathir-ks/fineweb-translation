from datasets import Dataset
import argparse
import fsspec
import json
from fsspec import AbstractFileSystem

parser = argparse.ArgumentParser(description="")
# parser.add_argument("--name", type=str)
parser.add_argument("--subset", type=str)
# parser.add_argument("--bucket", type=str)
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)


args = parser.parse_args()
name = 'HuggingFaceFW/fineweb-edu'
subset = args.subset
start = args.start
end = args.end
bucket = 'gs://indic-llama-data'

fs : AbstractFileSystem = fsspec.core.url_to_fs(f'{bucket}')[0]

files_per_shard = 200

file_count = 0
dataset = []
shard = 1

for i in range(start, end + 1, 1):
    
    if fs.isfile(f'{bucket}/{name}/{subset}/{i}/sentences.json'):
        with fs.open(f'{bucket}/{name}/{subset}/{i}/sentences.json', 'r') as f:
            sentences = json.load(f)
            for i, j in zip(sentences['text'], sentences['uuid']):
                dataset.append({'text':i, 'uuid':j})            
            file_count += 1

            if file_count % files_per_shard == 0:
                dataset_to_upload = Dataset.from_list(dataset)
                dataset_to_upload.push_to_hub(f'{subset}_{start}_{end}_shard_{shard}_row_wise')
                dataset = []
                shard += 1

if len(dataset) > 0:
    dataset_to_upload = Dataset.from_list(dataset)
    dataset_to_upload.push_to_hub(f'{subset}_{start}_{end}_shard_{shard}_row_wise')




