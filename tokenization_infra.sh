#!/bin/bash

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --subset) subset="$2"; shift ;;
        --region) region="$2"; shift ;;
        --accelerator_type) accelerator_type="$2"; shift ;;
        --bucket) bucket="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --src_lang) src_lang="$2"; shift ;;
        --tgt_lang) tgt_lang="$2"; shift ;;
        --tokenization_batch_size) tokenization_batch_size="$2"; shift ;;
        --shard_size) shard_size="$2"; shift ;;
        --total_nodes) total_nodes="$2"; shift ;;
        --total_files) total_files="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure all required arguments are provided
if [ -z "$subset" ] || [ -z "$region" ] || [ -z "$accelerator_type" ] || [ -z "$bucket" ] || [ -z "$dataset" ] || [ -z "$src_lang" ] || [ -z "$tgt_lang" ] || [ -z "$tokenization_batch_size" ] || [ -z "$shard_size" ] || [ -z "$total_nodes"] || [ -z "$total_files "]; then
    echo "Usage: $0 --subset SUBSET --region REGION --accelerator_type ACCELERATOR_TYPE --bucket BUCKET --dataset DATASET --src_lang SRC_LANG --tgt_lang TGT_LANG --tokenization_batch_size TOKENIZATION_BATCH_SIZE --shard_size SHARD_SIZE"
    exit 1
fi

# Function to create and set up the TPU VM
setup_tpu_vm() {

    gcloud compute tpus tpu-vm create tokenizer --zone=$region --accelerator-type=$accelerator_type --version=tpu-ubuntu2204-base --preemptible
    gcloud compute tpus tpu-vm ssh tokenizer --zone=$region --command='
        git clone https://github.com/kathir-ks/fineweb-translation;
        cd fineweb-translation;
        chmod +x setup_tokenization_env.sh;
        ./setup_tokenization_env.sh'
}

# Function to run the tokenization process
run_tokenization() {

    gcloud compute tpus tpu-vm ssh tokenizer --zone=$region --command="
        cd fineweb-translation;
        python3 _tokenization.py --name $dataset --subset $subset --src_lang $src_lang --tgt_lang $tgt_lang --tokenization_batch_size $tokenization_batch_size --bucket $bucket --shard_size $shard_size --resume True --total_nodes $total_nodes --total_files $total_files"
}

# Main script
node_id=0

# Run tokenization in a loop to handle potential preemption
while true; do
    output=$(gcloud compute tpus tpu-vm describe tokenizer --zone=$region 2>&1)

    while [[ $output != *"READY"* ]]; do
        echo "TPU VM is not ready, setting up the TPU VM"
        setup_tpu_vm 
        sleep 60
        output=$(gcloud compute tpus tpu-vm describe tokenizer --zone=$region 2>&1)
    done

    run_tokenization 

    if [ $? -eq 0 ]; then
        echo "Tokenization completed successfully"
        break
    else
        echo "TPU VM was preempted or tokenization failed, waiting and trying again"
        sleep 120
    fi
done

echo "Tokenization process finished"
