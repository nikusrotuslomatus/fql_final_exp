#!/bin/bash

# Smart staged hyperparameter search
# Usage: ./smart_hyperparameter_search.sh ENV_NAME EXPERIMENT_NAME [STAGE]

ENV_NAME=${1:-"antmaze-medium-play-v0"}
EXPERIMENT_NAME=${2:-"smart_search"}
STAGE=${3:-"all"}
OFFLINE_STEPS=${4:-500000}
NUM_SEEDS=3  # Use fewer seeds for initial exploration

echo "Smart Hyperparameter Search for $ENV_NAME"
echo "Experiment: $EXPERIMENT_NAME, Stage: $STAGE"

# Function to run experiment with error handling
run_experiment() {
    local config_name=$1
    local config_args=$2
    local seed=$3
    local stage=$4
    
    echo "[$stage] Starting: $config_name (seed $seed)"
    
    python main.py \
        --env_name=$ENV_NAME \
        --offline_steps=$OFFLINE_STEPS \
        --online_steps=0 \
        --seed=$seed \
        --run_group="${EXPERIMENT_NAME}_stage${stage}_${config_name}" \
        --eval_episodes=30 \
        --eval_interval=50000 \
        --log_interval=2000 \
        $config_args &
    
    # Limit parallel jobs based on system
    local max_jobs=2
    if [[ $(nproc) -gt 8 ]]; then
        max_jobs=4
    fi
    
    if (( $(jobs -r | wc -l) >= $max_jobs )); then
        wait -n
    fi
}

# Stage 1: Baseline and individual components
run_stage1() {
    echo "=== STAGE 1: Baseline and Component Analysis ==="
    
    declare -A STAGE1_CONFIGS=(
        ["baseline_euler"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=False --agent.flow_steps=10"
        ["baseline_midpoint"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=False --agent.flow_steps=15"
        ["kl_light"]="--agent.kl_coeff=0.1 --agent.advantage_weighted=False"
        ["kl_medium"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=False"
        ["kl_strong"]="--agent.kl_coeff=0.5 --agent.advantage_weighted=False"
        ["awfm_light"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=True --agent.adv_weight_coeff=0.5"
        ["awfm_medium"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
        ["awfm_strong"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=True --agent.adv_weight_coeff=2.0"
    )
    
    for config_name in "${!STAGE1_CONFIGS[@]}"; do
        config_args="${STAGE1_CONFIGS[$config_name]}"
        for seed in $(seq 0 $((NUM_SEEDS-1))); do
            run_experiment "$config_name" "$config_args" "$seed" "1"
        done
    done
    
    wait
    echo "Stage 1 completed. Analyze results with:"
    echo "python analyze_results.py --run_group=${EXPERIMENT_NAME}_stage1"
}

# Stage 2: Promising combinations
run_stage2() {
    echo "=== STAGE 2: Promising Combinations ==="
    
    declare -A STAGE2_CONFIGS=(
        # Conservative combinations
        ["combo_conservative1"]="--agent.kl_coeff=0.1 --agent.advantage_weighted=True --agent.adv_weight_coeff=0.5"
        ["combo_conservative2"]="--agent.kl_coeff=0.1 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
        ["combo_conservative3"]="--agent.kl_coeff=0.2 --agent.advantage_weighted=True --agent.adv_weight_coeff=0.5"
        
        # Moderate combinations
        ["combo_moderate1"]="--agent.kl_coeff=0.2 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
        ["combo_moderate2"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=0.5"
        ["combo_moderate3"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
        ["combo_moderate4"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.5"
        
        # Aggressive combinations
        ["combo_aggressive1"]="--agent.kl_coeff=0.5 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
        ["combo_aggressive2"]="--agent.kl_coeff=0.5 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.5"
        ["combo_aggressive3"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=2.0"
    )
    
    for config_name in "${!STAGE2_CONFIGS[@]}"; do
        config_args="${STAGE2_CONFIGS[$config_name]}"
        for seed in $(seq 0 $((NUM_SEEDS-1))); do
            run_experiment "$config_name" "$config_args" "$seed" "2"
        done
    done
    
    wait
    echo "Stage 2 completed. Analyze results with:"
    echo "python analyze_results.py --run_group=${EXPERIMENT_NAME}_stage2"
}

# Stage 3: Fine-tuning best configurations
run_stage3() {
    echo "=== STAGE 3: Fine-tuning Best Configurations ==="
    echo "Note: Manually update this stage based on Stage 1-2 results"
    
    # These should be updated based on previous stage results
    declare -A STAGE3_CONFIGS=(
        # Assume kl_coeff=0.3, adv_weight_coeff=1.0 was best from stage 2
        ["best_flow12"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.flow_steps=12"
        ["best_flow18"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.flow_steps=18"
        ["best_flow20"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.flow_steps=20"
        
        ["best_alpha5"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.alpha=5.0"
        ["best_alpha15"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.alpha=15.0"
        
        ["best_klsamp5"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.kl_num_samples=5"
        ["best_klsamp15"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.kl_num_samples=15"
        ["best_klsamp20"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.kl_num_samples=20"
        
        # Fine variations around best
        ["best_kl025"]="--agent.kl_coeff=0.25 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
        ["best_kl035"]="--agent.kl_coeff=0.35 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
        ["best_adv08"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=0.8"
        ["best_adv12"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.2"
    )
    
    # Use more seeds for final evaluation
    local final_seeds=5
    
    for config_name in "${!STAGE3_CONFIGS[@]}"; do
        config_args="${STAGE3_CONFIGS[$config_name]}"
        for seed in $(seq 0 $((final_seeds-1))); do
            run_experiment "$config_name" "$config_args" "$seed" "3"
        done
    done
    
    wait
    echo "Stage 3 completed. Analyze results with:"
    echo "python analyze_results.py --run_group=${EXPERIMENT_NAME}_stage3"
}

# Stage 4: Final validation with more seeds
run_stage4() {
    echo "=== STAGE 4: Final Validation ==="
    echo "Note: Update with best configuration from Stage 3"
    
    # This should be the single best configuration
    local BEST_CONFIG="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
    local final_seeds=10
    
    echo "Running final validation with $final_seeds seeds"
    
    for seed in $(seq 0 $((final_seeds-1))); do
        run_experiment "final_best" "$BEST_CONFIG" "$seed" "4"
    done
    
    wait
    echo "Stage 4 completed. Final analysis:"
    echo "python analyze_results.py --run_group=${EXPERIMENT_NAME}_stage4_final_best"
}

# Agent comparison (FQL vs IFQL)
run_agent_comparison() {
    echo "=== AGENT COMPARISON: FQL vs IFQL ==="
    
    local BEST_CONFIG="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0"
    
    for seed in $(seq 0 $((NUM_SEEDS-1))); do
        # FQL
        run_experiment "fql_best" "$BEST_CONFIG --agent=agents/fql.py" "$seed" "compare"
        
        # IFQL  
        run_experiment "ifql_best" "$BEST_CONFIG --agent=agents/ifql.py" "$seed" "compare"
    done
    
    wait
    echo "Agent comparison completed. Analyze with:"
    echo "python analyze_results.py --run_group=${EXPERIMENT_NAME}_stagecompare"
}

# Main execution
case $STAGE in
    "1"|"stage1")
        run_stage1
        ;;
    "2"|"stage2")
        run_stage2
        ;;
    "3"|"stage3")
        run_stage3
        ;;
    "4"|"stage4")
        run_stage4
        ;;
    "compare"|"comparison")
        run_agent_comparison
        ;;
    "all")
        echo "Running all stages sequentially..."
        run_stage1
        echo "Stage 1 done. Please analyze results and update Stage 3 configs if needed."
        echo "Continue with: ./smart_hyperparameter_search.sh $ENV_NAME $EXPERIMENT_NAME 2"
        ;;
    *)
        echo "Usage: $0 ENV_NAME EXPERIMENT_NAME [1|2|3|4|compare|all]"
        echo "Stages:"
        echo "  1: Baseline and individual components"
        echo "  2: Promising combinations"
        echo "  3: Fine-tuning (update based on stage 1-2 results)"
        echo "  4: Final validation with more seeds"
        echo "  compare: FQL vs IFQL comparison"
        echo "  all: Run stage 1 only (then analyze before continuing)"
        exit 1
        ;;
esac 