#!/bin/bash
#
# Tokenization launcher for TPU VMs (standalone, local storage).
#
# Usage:
#   ./run_tokenization.sh --vm_name tpu-tok-0 --zone us-central2-b \
#       --accelerator_type v4-8 --data_dir ~/data \
#       --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
#       --src_lang eng_Latn --tgt_lang hin_Deva \
#       --tokenization_batch_size 64 --shard_size 64000 \
#       --total_nodes 4 --total_files 99 \
#       --start_file 0 --end_file 10

set -euo pipefail

preemptible=true
data_dir="~/data"

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --vm_name) vm_name="$2"; shift ;;
        --zone) zone="$2"; shift ;;
        --accelerator_type) accelerator_type="$2"; shift ;;
        --data_dir) data_dir="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --subset) subset="$2"; shift ;;
        --src_lang) src_lang="$2"; shift ;;
        --tgt_lang) tgt_lang="$2"; shift ;;
        --tokenization_batch_size) tokenization_batch_size="$2"; shift ;;
        --shard_size) shard_size="$2"; shift ;;
        --total_nodes) total_nodes="$2"; shift ;;
        --total_files) total_files="$2"; shift ;;
        --start_file) start_file="$2"; shift ;;
        --end_file) end_file="$2"; shift ;;
        --no-preemptible) preemptible=false ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required args
for var in vm_name zone accelerator_type dataset subset src_lang tgt_lang tokenization_batch_size shard_size total_nodes total_files; do
    if [ -z "${!var:-}" ]; then
        echo "Error: --$var is required"
        exit 1
    fi
done

create_flags="--accelerator-type=$accelerator_type --version=tpu-ubuntu2204-base"
if [ "$preemptible" = true ]; then
    create_flags="$create_flags --preemptible"
fi

setup_vm() {
    echo "Creating TPU VM '$vm_name'..."
    gcloud compute tpus tpu-vm create "$vm_name" --zone="$zone" $create_flags
    sleep 10

    echo "Setting up environment..."
    gcloud compute tpus tpu-vm ssh "$vm_name" --zone="$zone" --command='
        git clone https://github.com/kathir-ks/fineweb-translation
        cd fineweb-translation
        chmod +x setup_tokenization_env.sh
        ./setup_tokenization_env.sh'
    sleep 10
}

# Main loop — handles preemption
while true; do
    output=$(gcloud compute tpus tpu-vm describe "$vm_name" --zone="$zone" 2>&1 || true)

    if [[ $output != *"READY"* ]]; then
        if ! setup_vm; then
            echo "VM setup failed. Retrying in 60s..."
            sleep 60
            continue
        fi
    fi

    echo "Starting tokenization on '$vm_name'..."
    tok_cmd="cd fineweb-translation && python3 tokenization_parallel.py \
            --name $dataset --subset $subset \
            --src_lang $src_lang --tgt_lang $tgt_lang \
            --tokenization_batch_size $tokenization_batch_size \
            --data_dir $data_dir --shard_size $shard_size \
            --resume True --total_nodes $total_nodes \
            --total_files $total_files"
    if [ -n "${start_file:-}" ]; then
        tok_cmd="$tok_cmd --start_file $start_file"
    fi
    if [ -n "${end_file:-}" ]; then
        tok_cmd="$tok_cmd --end_file $end_file"
    fi

    if gcloud compute tpus tpu-vm ssh "$vm_name" --zone="$zone" --command="$tok_cmd"; then
        echo "Tokenization completed successfully"
        break
    else
        echo "Tokenization failed or TPU preempted. Retrying in 60s..."
        sleep 60
    fi
done

echo "Done."
