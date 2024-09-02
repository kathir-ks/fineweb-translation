#!/bin/bash

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --node_id) node_id="$2"; shift;;
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
if [ -z "$node_id" ] || [ -z "$subset" ] || [ -z "$region" ] || [ -z "$accelerator_type" ] || [ -z "$bucket" ] || [ -z "$dataset" ] || [ -z "$src_lang" ] || [ -z "$tgt_lang" ] || [ -z "$tokenization_batch_size" ] || [ -z "$shard_size" ] || [ -z "$total_nodes" ] || [ -z "$total_files" ]; then
    echo "Usage: $0 --node_id NODE_ID --subset SUBSET --region REGION --accelerator_type ACCELERATOR_TYPE --bucket BUCKET --dataset DATASET --src_lang SRC_LANG --tgt_lang TGT_LANG --tokenization_batch_size TOKENIZATION_BATCH_SIZE --shard_size SHARD_SIZE --total_nodes TOTAL_NODES --total_files TOTAL_FILES"
    exit 1
fi

# Function to create and set up the TPU VM
setup_tpu_vm() {
    local tpu_name="tpu-vm-${node_id}"
    
    if ! gcloud compute tpus tpu-vm create $tpu_name --zone=$region --accelerator-type=$accelerator_type --version=tpu-ubuntu2204-base --preemptible; then
        echo "Failed to create TPU VM"
        return 1
    fi

    if ! gcloud compute tpus tpu-vm ssh $tpu_name --zone=$region --command='
        git clone https://github.com/kathir-ks/fineweb-translation;
        cd fineweb-translation;
        chmod +x setup_tokenization_env.sh;
        ./setup_tokenization_env.sh'; then
        echo "Failed to set up TPU VM environment"
        return 1
    fi

    return 0
}

# Function to run the tokenization process
run_tokenization() {
    local tpu_name="tpu-vm-${node_id}"

    if ! gcloud compute tpus tpu-vm ssh $tpu_name --zone=$region --command="
        cd fineweb-translation;
        python3 tokenization_parallel.py --name $dataset --subset $subset --src_lang $src_lang --tgt_lang $tgt_lang --tokenization_batch_size $tokenization_batch_size --bucket $bucket --shard_size $shard_size --resume True --total_nodes $total_nodes --total_files $total_files"; then
        echo "Tokenization process failed"
        return 1
    fi

    return 0
}

# Main script
# Run tokenization in a loop to handle potential preemption
while true; do
    tpu_name="tpu-vm-${node_id}"
    output=$(gcloud compute tpus tpu-vm describe $tpu_name --zone=$region 2>&1)

    if [[ $output != *"READY"* ]]; then
        echo "TPU VM is not ready, setting up the TPU VM"
        if setup_tpu_vm; then
            echo "TPU VM setup completed successfully"
        else
            echo "TPU VM setup failed, retrying in 60 seconds"
            sleep 60
            continue
        fi
    fi

    if run_tokenization; then
        echo "Tokenization completed successfully"
        break
    else
        echo "TPU VM was preempted or tokenization failed, waiting 120 seconds before trying again"
        sleep 20
    fi
done

echo "Tokenization process finished"