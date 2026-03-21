#!/bin/bash
#
# Unified inference launcher for single-host and multihost TPU VMs.
# Runs tokenization + inference on the same VM using local storage.
#
# Usage:
#   # Single-host (e.g. v4-8, preemptible):
#   ./run_inference.sh --vm_name worker-0 --zone us-central2-b \
#       --accelerator_type v4-8 --data_dir ~/data \
#       --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
#       --batch_size 256 --lang hin_Deva \
#       --src_lang eng_Latn --tgt_lang hin_Deva \
#       --total_files 99 --start_file 0 --end_file 1 \
#       --total_nodes 1 --shard_size 64000 \
#       --tokenization_batch_size 64
#
#   # Multihost (e.g. v5litepod-64, us-central1):
#   ./run_inference.sh --vm_name main-1 --zone us-central1-a \
#       --accelerator_type v5litepod-64 --multihost \
#       --data_dir ~/data --dataset HuggingFaceFW/fineweb-edu \
#       --subset sample-10BT --batch_size 256 --lang hin_Deva \
#       --src_lang eng_Latn --tgt_lang hin_Deva \
#       --total_files 99 --total_nodes 8 --shard_size 64000 \
#       --tokenization_batch_size 64
#
# Zone defaults:
#   Multihost:    us-central1-a  (v5litepod-64)
#   Single-host:  us-central2-b  (preemptible v4-8)

set -euo pipefail

# Defaults
multihost=false
preemptible=true
data_dir="~/data"
model_repo=""
total_nodes=1

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --vm_name) vm_name="$2"; shift ;;
        --zone) zone="$2"; shift ;;
        --accelerator_type) accelerator_type="$2"; shift ;;
        --data_dir) data_dir="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --subset) subset="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --lang) lang="$2"; shift ;;
        --src_lang) src_lang="$2"; shift ;;
        --tgt_lang) tgt_lang="$2"; shift ;;
        --total_files) total_files="$2"; shift ;;
        --start_file) start_file="$2"; shift ;;
        --end_file) end_file="$2"; shift ;;
        --total_nodes) total_nodes="$2"; shift ;;
        --shard_size) shard_size="$2"; shift ;;
        --tokenization_batch_size) tokenization_batch_size="$2"; shift ;;
        --node_id) node_id="$2"; shift ;;
        --model_repo) model_repo="$2"; shift ;;
        --multihost) multihost=true ;;
        --no-preemptible) preemptible=false ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required args
for var in vm_name zone accelerator_type dataset subset batch_size lang src_lang tgt_lang total_files shard_size tokenization_batch_size; do
    if [ -z "${!var:-}" ]; then
        echo "Error: --$var is required"
        exit 1
    fi
done

# Build SSH command prefix (--worker=all for multihost)
ssh_prefix="gcloud compute tpus tpu-vm ssh $vm_name --zone=$zone"
if [ "$multihost" = true ]; then
    ssh_prefix="$ssh_prefix --worker=all"
fi

# Build tokenization command
tok_cmd="cd fineweb-translation && python3 tokenization_parallel.py \
    --name $dataset --subset $subset \
    --src_lang $src_lang --tgt_lang $tgt_lang \
    --tokenization_batch_size $tokenization_batch_size \
    --data_dir $data_dir --shard_size $shard_size \
    --resume True --total_nodes $total_nodes \
    --total_files $total_files"

# For multihost, compute per-worker file ranges from GCE metadata
if [ "$multihost" = true ]; then
    tok_cmd="cd fineweb-translation && \
NODE_ID=\$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number 2>/dev/null || echo 0) && \
FILES_PER_NODE=\$(( $total_files / $total_nodes )) && \
START=\$(( NODE_ID * FILES_PER_NODE )) && \
END=\$(( START + FILES_PER_NODE )) && \
python3 tokenization_parallel.py \
    --name $dataset --subset $subset \
    --src_lang $src_lang --tgt_lang $tgt_lang \
    --tokenization_batch_size $tokenization_batch_size \
    --data_dir $data_dir --shard_size $shard_size \
    --resume True --total_nodes $total_nodes \
    --total_files $total_files \
    --start_file \$START --end_file \$END"
else
    if [ -n "${start_file:-}" ]; then
        tok_cmd="$tok_cmd --start_file $start_file"
    fi
    if [ -n "${end_file:-}" ]; then
        tok_cmd="$tok_cmd --end_file $end_file"
    fi
fi

# Build inference command
inference_cmd="cd fineweb-translation && python3 inference.py --name $dataset --subset $subset --batch_size $batch_size --data_dir $data_dir --lang $lang"
if [ -n "${model_repo:-}" ]; then
    inference_cmd="$inference_cmd --model_repo $model_repo"
fi
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
    output=$(gcloud compute tpus tpu-vm describe "$vm_name" --zone="$zone" 2>&1 || true)

    if [[ $output != *"READY"* ]]; then
        echo "TPU VM '$vm_name' not ready. Creating..."
        gcloud compute tpus tpu-vm create "$vm_name" --zone="$zone" $create_flags
        sleep 20

        echo "Setting up environment..."
        $ssh_prefix --command='
            git clone https://github.com/kathir-ks/fineweb-translation
            cd fineweb-translation
            chmod +x setup_inference_env.sh
            ./setup_inference_env.sh'
        sleep 20
    fi

    echo "Running tokenization on '$vm_name'..."
    if ! $ssh_prefix --command="$tok_cmd"; then
        echo "Tokenization failed or TPU preempted. Retrying in 60s..."
        sleep 60
        continue
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
