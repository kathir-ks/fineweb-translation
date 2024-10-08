import re
import argparse
import nltk
nltk.download('punkt')

# from nltk.tokenize import sent_tokenize
from unicodedata import normalize
from datasets import load_dataset
from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor
import os
import json
import signal
import fsspec
from fsspec import AbstractFileSystem


def parse_args():

    parser = argparse.ArgumentParser(description="Performs preprocessing and tokenization for fineweb")
    parser.add_argument("--name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, required=True, help='subset of the dataset')
    parser.add_argument("--streaming", default=True, type=bool, required=False, help='whether to stream or download the dataset')
    parser.add_argument("--src_lang", type=str, required=True, help='source language (i.e) the language of the dataset')
    parser.add_argument("--tgt_lang", type=str, required=True, help='target language')
    parser.add_argument("--tokenization_batch_size", type=int, required=True, help='batch size to perform tokenization')
    parser.add_argument("--bucket", type=str, required=True, help='gcs bucket to store the shards')
    parser.add_argument("--shard_size", type=int,default=64000, required=False, help='sharding based on no of sentences')
    parser.add_argument("--resume",type=bool , required=False, default=False)
    parser.add_argument("--total_nodes", type=int, required=True, help="to split the shards based on the no of nodes")
    parser.add_argument("--total_files", type=int, required=True, help='total no of files in the datasets')

    args = parser.parse_args()
    return args


# timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

signal.signal(signal.SIGALRM, timeout_handler)

# Decorator to apply timeout
def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.alarm(seconds)  # Set the alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable the alarm
            return result
        return wrapper
    return decorator


def save_data(subset, shard, data):
     
     with open(f'{subset}_{shard}.json', 'w') as f:
          json.dump(data, f)

def save_data_and_push_to_gcs(subset, shard, data, bucket):
     
    with open(f'{subset}_{shard}.json', 'w') as f:
        json.dump(data, f)
    
    cwd = os.getcwd()
    # push the file to gcs
    os.system(f'gsutil cp {subset}_{shard}.json {bucket}/{subset}/')
    # remove the file from disk
    os.system(f'rm {subset}_{shard}.json')


def load_data(name,subset, streaming, file_no, total_files , split="train"):
    
    data =  load_dataset(name, data_files={f'data/{subset}/train-{str(file_no).zfill(5)}-of-{str(total_files).zfill(5)}.parquet'}, streaming=streaming, split=split)
    # data = data['text']
    return data


# ref: https://github.com/AI4Bharat/setu-translate/blob/433723c52678cb79e54a04749e3d8a58737a2b35/stages/document.py#L75

def clean_string(s):  
    
        # Remove all symbols and numbers from beginning and end of the string
        stripped_s = s.strip("@#$^&*-_+=[]{}|\\<>/\n")
        stripped_s = stripped_s.strip() # Stripping left-over whitespaces if any

        # Strip all types of bullet points
        pattern = r'^\s*(\•|\○|\*|\-|[0-9]+\.)\s*'
        stripped_s = re.sub(pattern, '', stripped_s)
        stripped_s = stripped_s.strip() # Stripping left-over whitespaces if any

        return stripped_s

def split_with_delimiter(
        text,
        # delimiter_pattern=r'[.?!।|॥؟۔](?:\n+)?'
        delimiter_pattern=r'(?<!\d)\.(?!\d)|(?<!\w)\.(?!\w)|[?!।|॥؟۔\n](?:\n+)?', 
    ):
        lines = re.split(f'({delimiter_pattern})', text)
        if len(lines) % 2 == 0:
            iter_range = range(0, len(lines), 2)
            out = [lines[i]+lines[i+1] for i in iter_range]
        else:
            iter_range = range(0, len(lines) - 1, 2)
            out = [lines[i]+lines[i+1] for i in iter_range] + [lines[-1]]
        return out 

def split_into_sentences(text, method="regex"):
        split_methods = {
            "regex": split_with_delimiter,
        }
        text = normalize('NFKC', text).lower()
        sents = [clean_string(sent.text if not isinstance(sent, str) else sent) for sent in split_methods[method](text) if len(sent)]
        sents = [sent for sent in sents if len(sent)]
        # return remove_duplicate_string(sents)
        return sents

@timeout(1)
def preprocess_and_tokenize(tokenizer, ip, batch, src_lang, tgt_lang):
    
    batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
    batch = tokenizer(batch, padding="longest", truncation=True, max_length=256,src=True, return_tensors="pt",return_attention_mask=True)
    batch = {key: value.tolist() for key, value in batch.items()}
    placeholder_entity_maps = ip.get_placeholder_entity_maps(clear_ple_maps=True)
    return {"batch":batch, "placeholder_entity_maps":placeholder_entity_maps}



def _main(sentences, temp_ids, meta_data, src_lang, tgt_lang, tokenization_batch_size, name, subset, bucket, shard, total_nodes, fs, row):

    tokenized_inputs = []
    ids = []

    assert len(sentences) == len(temp_ids)

    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction='en-indic')
    
    for i in range(0, len(sentences), tokenization_batch_size):
        try:    
            tokenized_inputs.append(preprocess_and_tokenize(tokenizer, ip, sentences[i : i + tokenization_batch_size], src_lang, tgt_lang))
            ids.append(temp_ids[i : i + tokenization_batch_size])
        except TimeoutError as e:
            ip.get_placeholder_entity_maps(clear_ple_maps=True)
            print(e)
    
    assert len(tokenized_inputs)  == len(ids)

    data = {'tokenized_inputs':tokenized_inputs, "ids":ids, "row": row, "shard":shard, 'meta_data':meta_data}

    with fs.open(f'{bucket}/{name}/{subset}/{shard % total_nodes}/tokenized/{shard}.json' ,'w') as f:
        json.dump(data, f)


def main(args):

    name = args.name
    subset = args.subset
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    streaming = args.streaming
    tokenization_batch_size = args.tokenization_batch_size
    # rows_per_shard = args.rows_per_shard
    bucket = args.bucket
    shard_size = args.shard_size
    resume = args.resume
    total_nodes = args.total_nodes
    total_files = args.total_files


    sentences = []
    temp_ids = []
    meta_data = []

    row = 0
    shard = 1
    tokenized_rows = 0
    tokenized_shards = 0
    start_file = 0

    fs : AbstractFileSystem = fsspec.core.url_to_fs(bucket)[0]

    if resume:
        if fs.isfile(f'{bucket}/{name}/{subset}/tokenization_meta_data.json'):
            with fs.open(f'{bucket}/{name}/{subset}/tokenization_meta_data.json', 'r') as f:
                tokenization_meta_data = json.load(f)
                tokenized_rows = tokenization_meta_data['row']
                tokenized_shards = tokenization_meta_data['shard']
                tokenized_files = tokenization_meta_data['file']


    if tokenized_shards != 0:
        # print(file)
        print("tokenized rows", tokenized_rows)
        shard = tokenized_shards + 1
        start_file = tokenized_files
        print("tokenized shards", tokenized_shards)

    for i in range(start_file, total_files, 1):
        row = 0
        data = load_data(name, subset, streaming, i, total_files)
        for d in data:
            if row < (tokenized_rows-1):
                row += 1
                continue
            tokenized_rows = 0
            sents = split_into_sentences(d['text'])
            temp_ids.extend([d['id']] * len(sents))
            meta_data.append({'id':d['id'], 'dump':d['dump'], 'url':d['url'],'file_path':d['file_path']})
            sentences.extend(sents)
            row += 1    
            if len(sentences) >= shard_size:
                _main(sentences[: shard_size], temp_ids[: shard_size], meta_data, src_lang, tgt_lang, tokenization_batch_size, name, subset, bucket, shard, total_nodes, fs, row)
                sentences = sentences[shard_size : ]
                temp_ids = temp_ids[shard_size : ]
                if len(temp_ids) > 0:
                    meta_data = [meta_data[-1]]
                else:
                    meta_data = []
                
                assert len(sentences) == len(temp_ids)
                if len(meta_data) > 0:
                    assert meta_data[0]['id'] == temp_ids[0]
                    assert meta_data[0]['id'] == temp_ids[-1]
                shard += 1

                if shard % 50 == 0:
                    with fs.open(f'{bucket}/{name}/{subset}/tokenization_meta_data.json', 'w') as f:
                        json.dump({'row':row, 'shard': shard, 'file':i}, f)

if __name__ == '__main__':
    
    args = parse_args()

    main(args)
    