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

dataset = []

for i in range(start, end+1, 1):

    data = {}
    if fs.isfile(f'{bucket}/{name}/{subset}/{i}/sentences.json'):
        with fs.open(f'{bucket}/{name}/{subset}/{i}/sentences.json', 'r') as f:
            sentences = json.load(f)
            data['shard'] = i
            data['sentences'] = sentences
            dataset.append(data)


dataset_to_upload = Dataset.from_list(dataset)

dataset_to_upload.push_to_hub(f'{subset}_{start}_{end}')
