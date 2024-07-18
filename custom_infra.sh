#!/bin/bash

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --node_id) node_id="$2"; shift ;;
        --subset) subset="$2"; shift ;;
        --region) region="$2"; shift ;;
        --accelerator_type) accelerator_type="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure node_id, subset, region, and accelerator_type are provided
if [ -z "$node_id" ] || [ -z "$subset" ] || [ -z "$region" ] || [ -z "$accelerator_type" ]; then
    echo "Usage: $0 --node_id NODE_ID --subset SUBSET --region REGION --accelerator_type ACCELERATOR_TYPE"
    exit 1
fi

# Get the list of files from the GCS bucket
files=$(gsutil ls "gs://indic-llama-data/HuggingFaceFW/fineweb-edu/$subset/")
l=$(echo "$files" | wc -l)
i=$node_id

while [ $i -lt $((l + 1)) ]; do

    # Check if output.json file exists
    if gsutil -q stat "gs://indic-llama-data/HuggingFaceFW/fineweb-edu/$subset/$i/output.json"; then
        i=$((i + 32))
        continue
    fi

    output=$(gcloud compute tpus tpu-vm describe "main-$node_id" --zone=$region 2>&1)

    while [[ $output != *"READY"* ]]; do
        gcloud compute tpus tpu-vm create "main-$node_id" --zone=$region --accelerator-type=$accelerator_type --version=tpu-ubuntu2204-base --preemptible
        gcloud compute tpus tpu-vm ssh "main-$node_id" --zone=$region --command='
            git clone https://github.com/kathir-ks/fineweb-translation;
            cd fineweb-translation;
            chmod +x setup_inference_env.sh;
            ./setup_inference_env.sh'
        sleep 150
        output=$(gcloud compute tpus tpu-vm describe "main-$node_id" --zone=$region 2>&1)
    done

    gcloud compute tpus tpu-vm ssh "main-$node_id" --zone=$region --command="
        cd fineweb-translation;
        python3 inference.py --name HuggingFaceFW/fineweb-edu --subset $subset --batch_size 1024 --bucket gs://indic-llama-data --node_id $i --total_nodes 32"

    files=$(gsutil ls "gs://indic-llama-data/HuggingFaceFW/fineweb-edu/$subset/")
    l=$(echo "$files" | wc -l)
    
    if gsutil -q stat "gs://indic-llama-data/HuggingFaceFW/fineweb-edu/$subset/$i/output.json"; then
        i=$((i + 32))
    fi
done

echo "Inference completed"
