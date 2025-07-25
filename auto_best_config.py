#!/usr/bin/env python3
"""
Automatic best configuration selector for staged hyperparameter search.
Usage: python auto_best_config.py --run_group=experiment_stage1 --output_stage=3
"""

import argparse
import json
import wandb
import numpy as np
from typing import Dict, List, Tuple
from utils.evaluation_metrics import log_seed_statistics


def collect_stage_results(run_group: str, project: str = "fql") -> List[Dict]:
    """Collect results from a specific stage."""
    api = wandb.Api()
    runs = api.runs(f"{project}", filters={"group": run_group})
    
    results = []
    for run in runs:
        if run.state == "finished":
            # Get comprehensive metrics
            history = run.scan_history(keys=[
                "comprehensive/success_rate_percent",
                "comprehensive/d4rl_score", 
                "comprehensive/auc_learning_curve",
                "comprehensive/divergence_rate_percent",
                "comprehensive/kl_policy_data",
                "comprehensive/awfm_ess",
            ])
            
            # Get final values
            final_metrics = {}
            for row in history:
                for key, value in row.items():
                    if key != "_step" and value is not None:
                        final_metrics[key.replace("comprehensive/", "")] = value
            
            if final_metrics:
                # Extract hyperparameters from config
                config = run.config
                final_metrics.update({
                    'kl_coeff': config.get('agent.kl_coeff', 0.0),
                    'advantage_weighted': config.get('agent.advantage_weighted', False),
                    'adv_weight_coeff': config.get('agent.adv_weight_coeff', 1.0),
                    'flow_steps': config.get('agent.flow_steps', 15),
                    'alpha': config.get('agent.alpha', 10.0),
                    'kl_num_samples': config.get('agent.kl_num_samples', 10),
                    'seed': config.get('seed', 0),
                    'run_id': run.id,
                    'run_name': run.name
                })
                results.append(final_metrics)
    
    return results


def group_by_hyperparams(results: List[Dict], hyperparams: List[str]) -> Dict[str, List[Dict]]:
    """Group results by hyperparameter combinations."""
    grouped = {}
    
    for result in results:
        # Create key from hyperparameters
        param_values = []
        for param in hyperparams:
            value = result.get(param, "unknown")
            if isinstance(value, float):
                value = round(value, 3)  # Round floats for grouping
            param_values.append(f"{param}={value}")
        
        key = ", ".join(param_values)
        
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    
    return grouped


def compute_composite_score(stats: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> float:
    """Compute a composite score for ranking configurations."""
    score = 0.0
    total_weight = 0.0
    
    for metric, weight in weights.items():
        if metric in stats and 'mean' in stats[metric]:
            value = stats[metric]['mean']
            
            # Handle different metric types
            if metric == 'divergence_rate_percent':
                # Lower is better for divergence
                score += weight * max(0, 100 - value) / 100
            elif metric in ['success_rate_percent', 'd4rl_score', 'auc_learning_curve', 'awfm_ess']:
                # Higher is better
                if metric == 'success_rate_percent':
                    score += weight * value / 100  # Normalize to 0-1
                elif metric == 'd4rl_score':
                    score += weight * value / 100  # Normalize to 0-1
                else:
                    score += weight * min(1.0, value)  # Cap at 1.0
            elif metric == 'kl_policy_data':
                # Lower is generally better for KL, but not too low
                optimal_kl = 0.1  # Reasonable KL divergence
                kl_score = 1.0 - min(1.0, abs(value - optimal_kl) / optimal_kl)
                score += weight * kl_score
            
            total_weight += weight
    
    return score / total_weight if total_weight > 0 else 0.0


def analyze_stage_results(results: List[Dict], stage: int) -> Dict[str, any]:
    """Analyze results from a specific stage."""
    if not results:
        return {"error": "No results found"}
    
    # Define hyperparameters to group by based on stage
    if stage == 1:
        hyperparams = ['kl_coeff', 'advantage_weighted', 'adv_weight_coeff']
    else:
        hyperparams = ['kl_coeff', 'advantage_weighted', 'adv_weight_coeff', 'flow_steps', 'alpha']
    
    # Group results
    grouped = group_by_hyperparams(results, hyperparams)
    
    # Compute statistics for each group
    group_stats = {}
    for group_key, group_results in grouped.items():
        metrics = ['success_rate_percent', 'd4rl_score', 'auc_learning_curve', 
                  'divergence_rate_percent', 'kl_policy_data', 'awfm_ess']
        stats = log_seed_statistics(group_results, metrics)
        
        # Compute composite score
        weights = {
            'success_rate_percent': 0.4,
            'd4rl_score': 0.3,
            'auc_learning_curve': 0.1,
            'divergence_rate_percent': 0.1,
            'awfm_ess': 0.1
        }
        composite_score = compute_composite_score(stats, weights)
        
        group_stats[group_key] = {
            'statistics': stats,
            'composite_score': composite_score,
            'num_seeds': len(group_results),
            'config': group_results[0]  # Representative config
        }
    
    # Rank by composite score
    ranked_configs = sorted(group_stats.items(), 
                           key=lambda x: x[1]['composite_score'], 
                           reverse=True)
    
    return {
        'total_configs': len(grouped),
        'total_runs': len(results),
        'ranked_configs': ranked_configs[:10],  # Top 10
        'all_stats': group_stats
    }


def suggest_next_stage_configs(analysis: Dict[str, any], target_stage: int) -> List[Dict[str, any]]:
    """Suggest configurations for the next stage based on current results."""
    if 'ranked_configs' not in analysis:
        return []
    
    top_configs = analysis['ranked_configs'][:3]  # Top 3 configurations
    suggestions = []
    
    for i, (config_name, config_data) in enumerate(top_configs):
        base_config = config_data['config']
        
        if target_stage == 2:
            # Stage 2: Explore combinations around promising individual components
            kl_coeff = base_config.get('kl_coeff', 0.0)
            adv_coeff = base_config.get('adv_weight_coeff', 1.0)
            
            # Generate variations
            variations = []
            if kl_coeff > 0:  # If KL was promising
                variations.extend([
                    {'kl_coeff': kl_coeff * 0.8, 'advantage_weighted': True, 'adv_weight_coeff': 0.5},
                    {'kl_coeff': kl_coeff, 'advantage_weighted': True, 'adv_weight_coeff': 1.0},
                    {'kl_coeff': kl_coeff * 1.2, 'advantage_weighted': True, 'adv_weight_coeff': 1.5},
                ])
            
            if base_config.get('advantage_weighted', False):  # If AWFM was promising
                variations.extend([
                    {'kl_coeff': 0.1, 'advantage_weighted': True, 'adv_weight_coeff': adv_coeff},
                    {'kl_coeff': 0.3, 'advantage_weighted': True, 'adv_weight_coeff': adv_coeff},
                    {'kl_coeff': 0.5, 'advantage_weighted': True, 'adv_weight_coeff': adv_coeff},
                ])
            
            suggestions.extend(variations)
        
        elif target_stage == 3:
            # Stage 3: Fine-tune around best combinations
            base_params = {
                'kl_coeff': base_config.get('kl_coeff', 0.3),
                'advantage_weighted': base_config.get('advantage_weighted', True),
                'adv_weight_coeff': base_config.get('adv_weight_coeff', 1.0),
            }
            
            # Fine variations
            kl_base = base_params['kl_coeff']
            adv_base = base_params['adv_weight_coeff']
            
            variations = [
                # Flow steps variations
                {**base_params, 'flow_steps': 12},
                {**base_params, 'flow_steps': 18},
                {**base_params, 'flow_steps': 20},
                
                # Alpha variations
                {**base_params, 'alpha': 5.0},
                {**base_params, 'alpha': 15.0},
                
                # KL samples variations
                {**base_params, 'kl_num_samples': 5},
                {**base_params, 'kl_num_samples': 15},
                {**base_params, 'kl_num_samples': 20},
                
                # Fine parameter tuning
                {**base_params, 'kl_coeff': kl_base * 0.9},
                {**base_params, 'kl_coeff': kl_base * 1.1},
                {**base_params, 'adv_weight_coeff': adv_base * 0.8},
                {**base_params, 'adv_weight_coeff': adv_base * 1.2},
            ]
            
            suggestions.extend(variations)
    
    # Remove duplicates
    unique_suggestions = []
    seen = set()
    for suggestion in suggestions:
        key = tuple(sorted(suggestion.items()))
        if key not in seen:
            seen.add(key)
            unique_suggestions.append(suggestion)
    
    return unique_suggestions[:15]  # Limit to 15 suggestions


def generate_stage_script(suggestions: List[Dict], stage: int, experiment_name: str) -> str:
    """Generate bash script for the next stage."""
    script_lines = [
        f"#!/bin/bash",
        f"# Auto-generated configurations for Stage {stage}",
        f"# Based on analysis of previous stage results",
        f"",
        f"ENV_NAME=${{1:-\"antmaze-medium-play-v0\"}}",
        f"EXPERIMENT_NAME=${{2:-\"{experiment_name}\"}}",
        f"NUM_SEEDS=3",
        f"",
        f"declare -A STAGE{stage}_CONFIGS=(",
    ]
    
    for i, config in enumerate(suggestions):
        config_name = f"auto_config_{i+1:02d}"
        
        # Build config string
        config_parts = []
        for key, value in config.items():
            if key == 'advantage_weighted':
                config_parts.append(f"--agent.{key}={'True' if value else 'False'}")
            else:
                config_parts.append(f"--agent.{key}={value}")
        
        config_str = " ".join(config_parts)
        script_lines.append(f'    ["{config_name}"]="{config_str}"')
    
    script_lines.extend([
        ")",
        "",
        "# Run experiments",
        "for config_name in \"${!STAGE" + str(stage) + "_CONFIGS[@]}\"; do",
        f"    config_args=\"${{STAGE{stage}_CONFIGS[$config_name]}}\"",
        "    echo \"Running: $config_name -> $config_args\"",
        "    ",
        "    for seed in $(seq 0 $((NUM_SEEDS-1))); do",
        "        python main.py \\",
        "            --env_name=$ENV_NAME \\",
        "            --offline_steps=500000 \\",
        "            --online_steps=0 \\",
        "            --seed=$seed \\",
        f"            --run_group=\"${{EXPERIMENT_NAME}}_stage{stage}_${{config_name}}\" \\",
        "            --eval_episodes=30 \\",
        "            --eval_interval=50000 \\",
        "            --log_interval=2000 \\",
        "            $config_args &",
        "        ",
        "        # Limit parallel jobs",
        "        if (( $(jobs -r | wc -l) >= 2 )); then",
        "            wait -n",
        "        fi",
        "    done",
        "done",
        "",
        "wait",
        f"echo \"Stage {stage} completed. Analyze with:\"",
        f"echo \"python analyze_results.py --run_group=${{EXPERIMENT_NAME}}_stage{stage}\"",
    ])
    
    return "\n".join(script_lines)


def main():
    parser = argparse.ArgumentParser(description="Auto-select best configurations for next stage")
    parser.add_argument("--run_group", required=True, help="W&B run group to analyze")
    parser.add_argument("--project", default="fql", help="W&B project name")
    parser.add_argument("--output_stage", type=int, required=True, help="Target stage number for suggestions")
    parser.add_argument("--experiment_name", default="auto_search", help="Base experiment name")
    parser.add_argument("--output_script", help="Output bash script file")
    
    args = parser.parse_args()
    
    print(f"Analyzing results from: {args.run_group}")
    
    # Collect results
    results = collect_stage_results(args.run_group, args.project)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} completed runs")
    
    # Analyze results
    current_stage = 1  # Infer from run_group if possible
    if "stage2" in args.run_group:
        current_stage = 2
    elif "stage3" in args.run_group:
        current_stage = 3
    
    analysis = analyze_stage_results(results, current_stage)
    
    # Print top configurations
    print(f"\n{'='*80}")
    print(f"TOP CONFIGURATIONS FROM STAGE {current_stage}")
    print(f"{'='*80}")
    
    for i, (config_name, config_data) in enumerate(analysis['ranked_configs'][:5]):
        stats = config_data['statistics']
        score = config_data['composite_score']
        
        print(f"\n{i+1}. {config_name}")
        print(f"   Composite Score: {score:.3f}")
        print(f"   Seeds: {config_data['num_seeds']}")
        
        if 'success_rate_percent' in stats:
            sr = stats['success_rate_percent']
            print(f"   Success Rate: {sr['mean']:.1f}% ± {sr['std']:.1f}%")
        
        if 'd4rl_score' in stats:
            d4rl = stats['d4rl_score']
            print(f"   D4RL Score: {d4rl['mean']:.1f} ± {d4rl['std']:.1f}")
        
        if 'divergence_rate_percent' in stats:
            div = stats['divergence_rate_percent']
            print(f"   Divergence Rate: {div['mean']:.1f}%")
    
    # Generate suggestions for next stage
    suggestions = suggest_next_stage_configs(analysis, args.output_stage)
    
    print(f"\n{'='*80}")
    print(f"SUGGESTED CONFIGURATIONS FOR STAGE {args.output_stage}")
    print(f"{'='*80}")
    
    for i, config in enumerate(suggestions[:10]):
        print(f"{i+1:2d}. {config}")
    
    # Generate script
    if args.output_script:
        script_content = generate_stage_script(suggestions, args.output_stage, args.experiment_name)
        
        with open(args.output_script, 'w') as f:
            f.write(script_content)
        
        print(f"\nGenerated script: {args.output_script}")
        print(f"Make executable with: chmod +x {args.output_script}")
    
    # Save detailed analysis
    output_file = f"stage{current_stage}_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'analysis': analysis,
            'suggestions': suggestions,
            'args': vars(args)
        }, f, indent=2, default=str)
    
    print(f"Detailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main() 