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


def parse_args():

    parser = argparse.ArgumentParser(description="Performs preprocessing and tokenization for fineweb")

    parser.add_argument("--name", default="HuggingFaceFW/fineweb-edu")

    parser.add_argument("--subset", type=str, required=True)

    parser.add_argument("--streaming", default=True, type=bool, required=False)

    parser.add_argument("--src_lang", type=str, required=True)

    parser.add_argument("--tgt_lang", type=str, required=True)

    parser.add_argument("--tokenization_batch_size", type=int, required=True)

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


def load_data(name, subset,streaming, split="train"):
    
    data =  load_dataset(name, subset, streaming=streaming, split=split)
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

def split_into_sentences(text,method="regex"):
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



if __name__ == '__main__':
    
    args = parse_args()

    name = args.name
    subset = args.subset
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    streaming = args.streaming
    tokenzation_batch_size = args.tokenization_batch_size

    data = load_data(name, subset, streaming)

    sentences = []

    for d in data:
        sentences.extend(split_into_sentences(d['text']))
    
    del data

    tokenized_inputs = []

    ip = IndicProcessor(inference=True)
    tokenizer = IndicTransTokenizer(direction='en-indic')

    count = 1
    for i in range(0, len(sentences), tokenzation_batch_size):
        try:    
            tokenized_inputs.append(preprocess_and_tokenize(tokenizer, ip, sentences[i : i + tokenzation_batch_size], src_lang, tgt_lang))
        except TimeoutError as e:
            ip.get_placeholder_entity_maps(clear_ple_maps=True)
            print(e)
        count += 1

    with open(f'{subset}.json','w') as f:
        json.dump(tokenized_inputs,f)