#!/bin/bash

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --node_id) node_id="$2"; shift ;;
        --subset) subset="$2"; shift ;;
        --region) region="$2"; shift ;;
        --accelerator_type) accelerator_type="$2"; shift ;;
        --total_nodes) total_nodes="$2"; shift ;;
        --bucket) bucket="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --lang) lang="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure all required arguments are provided
if [ -z "$node_id" ] || [ -z "$subset" ] || [ -z "$region" ] || [ -z "$accelerator_type" ] || [ -z "$total_nodes" ] || [ -z "$bucket" ] || [ -z "$dataset" ] || [ -z "$batch_size" ] || [ -z "$lang" ]; then
    echo "Usage: $0 --node_id NODE_ID --subset SUBSET --region REGION --accelerator_type ACCELERATOR_TYPE --total_nodes TOTAL_NODES --bucket BUCKET --dataset DATASET --batch_size BATCH_SIZE --lang LANG"
    exit 1
fi

while true; do
    output=$(gcloud compute tpus tpu-vm describe "worker-$node_id" --zone=$region 2>&1)

    if [[ $output != *"READY"* ]]; then
        echo "TPU VM is not ready. Creating or recreating..."
        gcloud compute tpus tpu-vm create "worker-$node_id" --zone=$region --accelerator-type=$accelerator_type --version=tpu-ubuntu2204-base --preemptible
        sleep 20
        gcloud compute tpus tpu-vm ssh "worker-$node_id" --zone=$region --command='
            git clone https://github.com/kathir-ks/fineweb-translation;
            cd fineweb-translation;
            chmod +x setup_inference_env.sh;
            ./setup_inference_env.sh'
        sleep 20
    else
        echo "TPU VM is ready. Running inference..."
        if gcloud compute tpus tpu-vm ssh "worker-$node_id" --zone=$region --command="
            cd fineweb-translation;
            python3 inference.py --name $dataset --subset $subset --batch_size $batch_size --bucket $bucket --node_id $node_id --total_nodes $total_nodes --lang $lang"; then
            echo "Inference completed successfully"
            break
        else
            echo "Inference failed or TPU was preempted. Retrying..."
            sleep 60
        fi
    fi
done

echo "Deleting TPU VM..."
gcloud compute tpus tpu-vm delete "worker-$node_id" --zone=$region --quiet

echo "Script execution completed"