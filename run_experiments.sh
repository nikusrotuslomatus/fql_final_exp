#!/bin/bash

# Comprehensive experiment runner with multiple seeds
# Usage: ./run_experiments.sh ENV_NAME EXPERIMENT_NAME

ENV_NAME=${1:-"antmaze-medium-play-v0"}
EXPERIMENT_NAME=${2:-"comprehensive_test"}
OFFLINE_STEPS=${3:-500000}
NUM_SEEDS=5

echo "Running comprehensive experiments for $ENV_NAME"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Seeds: $NUM_SEEDS"

# Create experiment configurations
declare -A CONFIGS=(
    ["baseline"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=False"
    ["kl_only"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=False"
    ["awfm_only"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
    ["combined_conservative"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
    ["combined_aggressive"]="--agent.kl_coeff=0.5 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.5"
)

# Function to run single experiment
run_experiment() {
    local config_name=$1
    local config_args=$2
    local seed=$3
    
    echo "Starting: $config_name with seed $seed"
    
    python main.py \
        --env_name=$ENV_NAME \
        --offline_steps=$OFFLINE_STEPS \
        --online_steps=0 \
        --seed=$seed \
        --run_group="${EXPERIMENT_NAME}_${config_name}" \
        --eval_episodes=50 \
        --eval_interval=50000 \
        --log_interval=1000 \
        $config_args &
    
    # Limit parallel jobs
    if (( $(jobs -r | wc -l) >= 1 )); then
        wait -n
    fi
}

# Run all configurations
for config_name in "${!CONFIGS[@]}"; do
    config_args="${CONFIGS[$config_name]}"
    echo "Configuration: $config_name -> $config_args"
    
    for seed in $(seq 0 $((NUM_SEEDS-1))); do
        run_experiment "$config_name" "$config_args" "$seed"
    done
done

# Wait for all jobs to complete
wait

echo "All experiments completed!"
echo ""
echo "To analyze results, run:"
echo "python analyze_results.py --run_group=${EXPERIMENT_NAME}_baseline"
echo "python analyze_results.py --run_group=${EXPERIMENT_NAME}_kl_only"
echo "python analyze_results.py --run_group=${EXPERIMENT_NAME}_awfm_only"
echo "python analyze_results.py --run_group=${EXPERIMENT_NAME}_combined_conservative"
echo "python analyze_results.py --run_group=${EXPERIMENT_NAME}_combined_aggressive"
echo ""
echo "Or analyze all together:"
echo "python analyze_results.py --run_group=${EXPERIMENT_NAME} --hyperparams kl_coeff advantage_weighted adv_weight_coeff"
