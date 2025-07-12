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

# Extended hyperparameter grid for optimal tuning
declare -A CONFIGS=(
    # Baseline comparisons
    ["baseline"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=False --agent.flow_steps=10"
    ["midpoint_only"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=False --agent.flow_steps=15"
    
    # KL pessimism sweep
    ["kl_light"]="--agent.kl_coeff=0.1 --agent.advantage_weighted=False"
    ["kl_moderate"]="--agent.kl_coeff=0.2 --agent.advantage_weighted=False"
    ["kl_medium"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=False"
    ["kl_strong"]="--agent.kl_coeff=0.5 --agent.advantage_weighted=False"
    ["kl_heavy"]="--agent.kl_coeff=0.8 --agent.advantage_weighted=False"
    ["kl_extreme"]="--agent.kl_coeff=1.0 --agent.advantage_weighted=False"
    
    # Advantage weighting sweep
    ["awfm_light"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=True --agent.adv_weight_coeff=0.5"
    ["awfm_moderate"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
    ["awfm_strong"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.5"
    ["awfm_heavy"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=True --agent.adv_weight_coeff=2.0"
    ["awfm_extreme"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=True --agent.adv_weight_coeff=3.0"
    
    # Combined configurations - conservative
    ["combined_kl01_adv05"]="--agent.kl_coeff=0.1 --agent.advantage_weighted=True --agent.adv_weight_coeff=0.5"
    ["combined_kl01_adv10"]="--agent.kl_coeff=0.1 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
    ["combined_kl02_adv05"]="--agent.kl_coeff=0.2 --agent.advantage_weighted=True --agent.adv_weight_coeff=0.5"
    ["combined_kl02_adv10"]="--agent.kl_coeff=0.2 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
    ["combined_kl02_adv15"]="--agent.kl_coeff=0.2 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.5"
    
    # Combined configurations - moderate
    ["combined_kl03_adv05"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=0.5"
    ["combined_kl03_adv10"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
    ["combined_kl03_adv15"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.5"
    ["combined_kl03_adv20"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=2.0"
    
    # Combined configurations - aggressive
    ["combined_kl05_adv10"]="--agent.kl_coeff=0.5 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
    ["combined_kl05_adv15"]="--agent.kl_coeff=0.5 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.5"
    ["combined_kl05_adv20"]="--agent.kl_coeff=0.5 --agent.advantage_weighted=True --agent.adv_weight_coeff=2.0"
    ["combined_kl08_adv10"]="--agent.kl_coeff=0.8 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
    ["combined_kl08_adv15"]="--agent.kl_coeff=0.8 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.5"
    
    # Flow steps variations with best combinations
    ["best_flow12"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.flow_steps=12"
    ["best_flow18"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.flow_steps=18"
    ["best_flow20"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.flow_steps=20"
    
    # Alpha variations (for different environments)
    ["alpha_low"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.alpha=5.0"
    ["alpha_high"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.alpha=15.0"
    
    # KL sample variations
    ["kl_samples_5"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.kl_num_samples=5"
    ["kl_samples_15"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.kl_num_samples=15"
    ["kl_samples_20"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.kl_num_samples=20"
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