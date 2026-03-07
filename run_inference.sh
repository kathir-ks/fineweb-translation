#!/bin/bash
#
# Unified inference launcher for single-host and multihost TPU VMs.
#
# Usage:
#   # Single-host (e.g. v3-8, v4-8):
#   ./run_inference.sh --vm_name worker-0 --region us-central2-b \
#       --accelerator_type v4-8 --bucket gs://my-bucket \
#       --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
#       --batch_size 256 --lang hin_Deva
#
#   # Multihost (e.g. v4-256):
#   ./run_inference.sh --vm_name main-1 --region us-central2-b \
#       --accelerator_type v4-256 --multihost \
#       --bucket gs://my-bucket --dataset HuggingFaceFW/fineweb-edu \
#       --subset sample-10BT --batch_size 256 --lang hin_Deva

set -euo pipefail

# Defaults
multihost=false
preemptible=true

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --vm_name) vm_name="$2"; shift ;;
        --region) region="$2"; shift ;;
        --accelerator_type) accelerator_type="$2"; shift ;;
        --bucket) bucket="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --subset) subset="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --lang) lang="$2"; shift ;;
        --node_id) node_id="$2"; shift ;;
        --total_nodes) total_nodes="$2"; shift ;;
        --multihost) multihost=true ;;
        --no-preemptible) preemptible=false ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required args
for var in vm_name region accelerator_type bucket dataset subset batch_size lang; do
    if [ -z "${!var:-}" ]; then
        echo "Error: --$var is required"
        exit 1
    fi
done

# Build SSH command prefix (--worker=all for multihost)
ssh_prefix="gcloud compute tpus tpu-vm ssh $vm_name --zone=$region"
if [ "$multihost" = true ]; then
    ssh_prefix="$ssh_prefix --worker=all"
fi

# Build inference command
inference_cmd="cd fineweb-translation && python3 inference.py --name $dataset --subset $subset --batch_size $batch_size --bucket $bucket --lang $lang"
if [ -n "${node_id:-}" ]; then
    inference_cmd="$inference_cmd --node_id $node_id"
fi
if [ -n "${total_nodes:-}" ]; then
    inference_cmd="$inference_cmd --total_nodes $total_nodes"
fi

# Build create flags
create_flags="--accelerator-type=$accelerator_type --version=tpu-ubuntu2204-base"
if [ "$preemptible" = true ]; then
    create_flags="$create_flags --preemptible"
fi

# Main loop — handles preemption by recreating the VM
while true; do
    output=$(gcloud compute tpus tpu-vm describe "$vm_name" --zone="$region" 2>&1 || true)

    if [[ $output != *"READY"* ]]; then
        echo "TPU VM '$vm_name' not ready. Creating..."
        gcloud compute tpus tpu-vm create "$vm_name" --zone="$region" $create_flags
        sleep 20

        echo "Setting up environment..."
        $ssh_prefix --command='
            git clone https://github.com/kathir-ks/fineweb-translation
            cd fineweb-translation
            chmod +x setup_inference_env.sh
            ./setup_inference_env.sh'
        sleep 20
    fi

    echo "Starting inference on '$vm_name'..."
    if $ssh_prefix --command="$inference_cmd"; then
        echo "Inference completed successfully"
        break
    else
        echo "Inference failed or TPU preempted. Retrying in 60s..."
        sleep 60
    fi
done

echo "Done."
