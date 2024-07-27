#!/bin/bash

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --subset) subset="$2"; shift ;;
        --region) region="$2"; shift ;;
        --accelerator_type) accelerator_type="$2"; shift ;;
        --bucket) bucket="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --lang) lang="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure all required arguments are provided
if [ -z "$subset" ] || [ -z "$region" ] || [ -z "$accelerator_type" ] || [ -z "$bucket" ] || [ -z "$dataset" ] || [ -z "$batch_size" ] || [ -z "$lang" ]; then
    echo "Usage: $0 --subset SUBSET --region REGION --accelerator_type ACCELERATOR_TYPE --bucket BUCKET --dataset DATASET --batch_size BATCH_SIZE --lang LANG"
    exit 1
fi

while true; do
    output=$(gcloud compute tpus tpu-vm describe main-1 --zone=$region 2>&1)

    if [[ $output != *"READY"* ]]; then
        echo "Creating TPU VM 'main-1'..."
        gcloud compute tpus tpu-vm create main-1 --zone=$region --accelerator-type=$accelerator_type --version=tpu-ubuntu2204-base --preemptible
        sleep 10

        gcloud compute tpus tpu-vm ssh main-1 --zone=$region --worker=all --command='
            git clone https://github.com/kathir-ks/fineweb-translation;
            cd fineweb-translation;
            chmod +x setup_inference_env.sh;
            ./setup_inference_env.sh'
        sleep 20
    fi

    echo "Starting inference..."
    gcloud compute tpus tpu-vm ssh main-1 --zone=$region --worker=all --command="
        cd fineweb-translation;
        python3 inference.py --name $dataset --subset $subset --batch_size $batch_size --bucket $bucket --lang $lang"

    if [ $? -eq 0 ]; then
        echo "Inference completed successfully"
        break
    else
        echo "Inference failed or TPU preempted, retrying..."
        sleep 100
    fi
done
