import os
import subprocess
import argparse
import fsspec
from fsspec import AbstractFileSystem
import time

parser = argparse.ArgumentParser(description="parser node_id and subset")
parser.add_argument("--node_id", type=int)
parser.add_argument("--subset", type=str)

args = parser.parse_args()
node_id = args.node_id
subset = args.subset

fs : AbstractFileSystem = fsspec.core.url_to_fs('gs://indic-llama-data')[0]

files = fs.ls(f'gs://indic-llama-data/HuggingFaceFW/fineweb-edu/{subset}')

l = len(files)

i = node_id

while i < l + 1:

    if fs.isfile(f'gs://indic-llama-data/HuggingFaceFW/fineweb-edu/{subset}/{i}/output.json'):
        i += 32
        continue

    output = ''

    try:
        output = subprocess.check_output(
                ['gcloud', 'compute', 'tpus', 'tpu-vm', 'describe', f'main-{node_id}', '--zone=us-central2-b'],
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                universal_newlines=True
            )

    except subprocess.CalledProcessError as e:
            output = e.output

    while 'READY' not in output:

        try:
            output = subprocess.check_output(
                ['gcloud', 'compute', 'tpus', 'tpu-vm', 'create', f'main-{node_id}', '--zone=us-central2-b', '--accelerator-type=v4-8', '--version=tpu-ubuntu2204-base', '--internal-ips', '--preemptible'],
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                universal_newlines=True
            )
            os.system(f'''gcloud alpha compute tpus tpu-vm ssh main-{node_id} --zone=us-central2-b --tunnel-through-iap 
                  --command=\'git clone https://github.com/kathir-ks/fineweb-translation;
                  cd fineweb-translation;
                  chmod +x setup_inference_env.sh;
                  ./setup_inference_env.sh\'''')
            
        except subprocess.CalledProcessError as e:
            output = e.output
            time.sleep(60)

    # if 'READY' in output:
    #     print("node is already running")
    # else:
    #     os.system(f'gcloud compute tpus tpu-vm create main-{node_id} --zone=us-central2-b --internal-ips --accelerator-type=v4-8 --version=tpu-ubuntu2204-base --preemptible')
    #     os.system(f'''gcloud alpha compute tpus tpu-vm ssh node-{node_id} --zone=us-central2-b --tunnel-through-iap 
    #               --command=\'git clone https://github.com/kathir-ks/fineweb-translation;
    #               cd fineweb-translation;
    #               chmod +x setup_inference_env.sh;
    #               ./setup_inference_env.sh\'''')
        
        # time.sleep(10)

    os.system(f'gcloud alpha compute tpus tpu-vm ssh main-{node_id} --zone=us-central2-b --tunnel-through-iap --command=\'cd fineweb-translation;python3 inference.py --name HuggingFaceFW/fineweb-edu --subset {subset} --batch_size 2048 --bucket gs://indic-llama-data --node_id {i} --total_nodes 32\'')

    if fs.isfile(f'gs://indic-llama-data/HuggingFaceFW/fineweb-edu/{subset}/{i}/output.json'):
        i += 32
    
print("inference completed")