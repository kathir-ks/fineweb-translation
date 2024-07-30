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
if [ -z "$node_id" ] || [ -z "$subset" ] || [ -z "$region" ] || [ -z "$accelerator_type" ] || [ -z "$total_nodes" ] || [ -z "$bucket" ] || [ -z "$dataset" ] || [ -z "$batch_size" ]; then
    echo "Usage: $0 --node_id NODE_ID --subset SUBSET --region REGION --accelerator_type ACCELERATOR_TYPE --total_nodes TOTAL_NODES --bucket BUCKET --dataset DATASET --batch_size BATCH_SIZE"
    exit 1
fi

# Function to create TPU and run inference
create_tpu_and_run_inference() {
    local node_id=$1
    local i=$2

    output=$(gcloud compute tpus tpu-vm describe "main-$node_id" --zone=$region 2>&1)

    while [[ $output != *"READY"* ]]; do
        gcloud compute tpus tpu-vm create "main-$node_id" --zone=$region --accelerator-type=$accelerator_type --version=tpu-ubuntu2204-base --preemptible
        sleep 10
        gcloud compute tpus tpu-vm ssh "main-$node_id" --zone=$region --command='
            git clone https://github.com/kathir-ks/fineweb-translation;
            cd fineweb-translation;
            chmod +x setup_inference_env.sh;
            ./setup_inference_env.sh'
        sleep 20
        output=$(gcloud compute tpus tpu-vm describe "main-$node_id" --zone=$region 2>&1)
    done

    gcloud compute tpus tpu-vm ssh "main-$node_id" --zone=$region --command="
        cd fineweb-translation;
        python3 inference.py --name $dataset --subset $subset --batch_size $batch_size --bucket $bucket --node_id $i --total_nodes $total_nodes --lang $lang"
}

# Get the list of files and extract the indices
indices=$(gsutil ls "$bucket/$dataset/$subset/" | sed -n 's|.*/\([0-9]\+\)$|\1|p' | sort -n)

# Get the lowest and highest indices
lowest_index=$(echo "$indices" | head -n 1)
highest_index=$(echo "$indices" | tail -n 1)

# Array to store background job PIDs
pids=()

# Start TPU creation and inference for each node
for ((node=0; node<total_nodes; node++)); do
    i=$((lowest_index + node))
    while [ $i -le $highest_index ]; do
        # Check if output.json file exists
        if ! gsutil -q stat "$bucket/$dataset/$subset/$i/sentences.json"; then
            create_tpu_and_run_inference $node $i &
            pids+=($!)
        fi
        i=$((i + total_nodes))
    done
done

# Wait for all background jobs to complete
for pid in "${pids[@]}"; do
    wait $pid
done

echo "Inference completed for all nodes"