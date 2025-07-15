#!/bin/bash

# Comprehensive FQL experiments for three key environments
# Each experiment tests: baseline, KL pessimism, and AWFM (advantage-weighted flow matching)
# 200k offline steps per experiment, CSV logging only (no WandB)

set -e  # Exit on any error

echo "🚀 Starting FQL Three-Environment Experiment Suite"
echo "=================================================="

# Configuration
OFFLINE_STEPS=200000
EVAL_EPISODES=30
EVAL_INTERVAL=50000
LOG_INTERVAL=10000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
RESULTS_DIR="experiment_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Function to run single experiment
run_experiment() {
    local env_name=$1
    local config_name=$2
    local config_args=$3
    local seed=$4
    local exp_group=$5
    
    echo "🧪 Running: $env_name | $config_name | seed $seed"
    
    # Create unique run name
    local run_name="${env_name}_${config_name}_seed${seed}"
    
    python main.py \
        --env_name="$env_name" \
        --offline_steps=$OFFLINE_STEPS \
        --online_steps=0 \
        --seed=$seed \
        --run_group="$exp_group" \
        --eval_episodes=$EVAL_EPISODES \
        --eval_interval=$EVAL_INTERVAL \
        --log_interval=$LOG_INTERVAL \
        --use_wandb=False \
        $config_args \
        > "$RESULTS_DIR/${run_name}.log" 2>&1 &
    
    # Store PID for monitoring
    local pid=$!
    echo "  Started PID: $pid"
    
    # Limit parallel jobs (adjust based on your system)
    local max_parallel=2
    if (( $(jobs -r | wc -l) >= $max_parallel )); then
        echo "  Waiting for slot..."
        wait -n
    fi
}

# Function to run all configurations for an environment
run_environment_experiments() {
    local env_name=$1
    local exp_group=$2
    local alpha_value=$3
    local extra_args=$4
    
    echo ""
    echo "🔬 Starting experiments for: $env_name"
    echo "   Group: $exp_group"
    echo "   Alpha: $alpha_value"
    echo "   Extra args: $extra_args"
    echo "----------------------------------------"
    
    # Define configurations
    declare -A configs=(
        ["baseline"]="--agent.kl_coeff=0.0 --agent.advantage_weighted=False --agent.alpha=$alpha_value"
        ["kl_pessimism"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=False --agent.alpha=$alpha_value"
        ["awfm"]="--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0 --agent.alpha=$alpha_value"
    )
    
    # Run each configuration with 3 seeds
    for config_name in "${!configs[@]}"; do
        config_args="${configs[$config_name]} $extra_args"
        
        for seed in 0 1 2; do
            run_experiment "$env_name" "$config_name" "$config_args" "$seed" "$exp_group"
        done
    done
}

# Experiment 1: Puzzle (OGBench, multi-step puzzle)
echo "🧩 EXPERIMENT 1: Puzzle Tasks"
echo "Testing long-horizon manipulation tasks with sparse rewards"

# Puzzle 3x3 (easier)
run_environment_experiments \
    "puzzle-3x3-play-singletask-v0" \
    "Puzzle3x3_Experiment" \
    "1000" \
    ""

# Puzzle 4x4 (harder)
run_environment_experiments \
    "puzzle-4x4-play-singletask-v0" \
    "Puzzle4x4_Experiment" \
    "1000" \
    ""

# Experiment 2: AntMaze-Large (D4RL, complex maze)
echo "🐜 EXPERIMENT 2: AntMaze-Large"
echo "Testing navigation with distribution shift and sparse rewards"

run_environment_experiments \
    "antmaze-large-play-v2" \
    "AntMazeLarge_Experiment" \
    "3" \
    "--agent.q_agg=min"

# Experiment 3: HumanoidMaze (OGBench, humanoid control)
echo "🤖 EXPERIMENT 3: HumanoidMaze"
echo "Testing high-DOF humanoid navigation through complex maze"

# HumanoidMaze Medium (21 DOF)
run_environment_experiments \
    "humanoidmaze-medium-navigate-singletask-v0" \
    "HumanoidMaze_Experiment" \
    "30" \
    "--agent.discount=0.995"

# Wait for all experiments to complete
echo ""
echo "⏳ Waiting for all experiments to complete..."
wait

echo ""
echo "✅ All experiments completed!"
echo "=================================================="

# Create summary report
SUMMARY_FILE="$RESULTS_DIR/experiment_summary.txt"
echo "📊 FQL Three-Environment Experiment Summary" > "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "Offline Steps: $OFFLINE_STEPS" >> "$SUMMARY_FILE"
echo "Seeds per config: 3" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Experiments:" >> "$SUMMARY_FILE"
echo "1. Puzzle Tasks (3x3 and 4x4)" >> "$SUMMARY_FILE"
echo "   - Success rate of puzzle completion" >> "$SUMMARY_FILE"
echo "   - Average percentage of subtasks completed" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "2. AntMaze-Large (D4RL)" >> "$SUMMARY_FILE"
echo "   - Success rate reaching goal" >> "$SUMMARY_FILE"
echo "   - Performance under distribution shift" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "3. HumanoidMaze-Medium (OGBench)" >> "$SUMMARY_FILE"
echo "   - Success rate navigating maze" >> "$SUMMARY_FILE"
echo "   - Mean return performance" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Configurations tested:" >> "$SUMMARY_FILE"
echo "- Baseline: No KL penalty, no advantage weighting" >> "$SUMMARY_FILE"
echo "- KL Pessimism: KL coefficient = 0.3" >> "$SUMMARY_FILE"
echo "- AWFM: Advantage-weighted flow matching" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Count log files
echo "Log files created:" >> "$SUMMARY_FILE"
ls -1 "$RESULTS_DIR"/*.log | wc -l >> "$SUMMARY_FILE"

echo "CSV data location: csv_logs/" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Display summary
cat "$SUMMARY_FILE"

echo ""
echo "📁 Results saved in: $RESULTS_DIR/"
echo "📈 CSV metrics saved in: csv_logs/"
echo ""
echo "🎯 Next steps:"
echo "1. Analyze CSV files in csv_logs/ directory"
echo "2. Run visualization scripts for each experiment"
echo "3. Compare success rates across configurations"
echo ""
echo "Example visualization commands:"
echo "  python csv_logs/plot_*_Puzzle3x3_*.py"
echo "  python csv_logs/plot_*_AntMazeLarge_*.py"
echo "  python csv_logs/plot_*_HumanoidMaze_*.py"
echo ""
echo "🔍 To check experiment status:"
echo "  tail -f $RESULTS_DIR/*.log"
echo "  ls -la csv_logs/" 