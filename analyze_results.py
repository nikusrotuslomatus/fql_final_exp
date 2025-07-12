#!/usr/bin/env python3
"""
Script to analyze comprehensive metrics across multiple seeds.
Usage: python analyze_results.py --run_group=your_experiment_group
"""

import argparse
import json
import os
from typing import Dict, List
import wandb
import numpy as np
from utils.evaluation_metrics import log_seed_statistics


def collect_wandb_results(run_group: str, project: str = "fql") -> List[Dict]:
    """Collect results from W&B runs with the specified run group."""
    api = wandb.Api()
    runs = api.runs(f"{project}", filters={"group": run_group})
    
    results = []
    for run in runs:
        if run.state == "finished":
            # Get final comprehensive metrics
            history = run.scan_history(keys=[
                "comprehensive/success_rate_percent",
                "comprehensive/d4rl_score", 
                "comprehensive/auc_learning_curve",
                "comprehensive/divergence_detected",
                "comprehensive/divergence_rate_percent",
                "comprehensive/kl_policy_data",
                "comprehensive/awfm_ess",
                "comprehensive/mean_return",
                "comprehensive/std_return"
            ])
            
            # Get final values
            final_metrics = {}
            for row in history:
                for key, value in row.items():
                    if key != "_step" and value is not None:
                        final_metrics[key.replace("comprehensive/", "")] = value
            
            if final_metrics:
                # Add run metadata
                final_metrics['seed'] = run.config.get('seed', 0)
                final_metrics['run_id'] = run.id
                final_metrics['run_name'] = run.name
                results.append(final_metrics)
    
    return results


def analyze_hyperparameter_sweep(results: List[Dict], 
                                hyperparams: List[str]) -> Dict[str, Dict]:
    """Analyze results across different hyperparameter settings."""
    
    # Group results by hyperparameter combinations
    grouped_results = {}
    for result in results:
        # Create key from hyperparameters
        param_key = tuple(result.get(param, "unknown") for param in hyperparams)
        if param_key not in grouped_results:
            grouped_results[param_key] = []
        grouped_results[param_key].append(result)
    
    # Analyze each group
    analysis = {}
    for param_key, group_results in grouped_results.items():
        param_dict = dict(zip(hyperparams, param_key))
        
        # Compute statistics for this parameter combination
        metrics = [
            'success_rate_percent', 'd4rl_score', 'auc_learning_curve',
            'divergence_rate_percent', 'kl_policy_data', 'awfm_ess'
        ]
        
        stats = log_seed_statistics(group_results, metrics)
        
        analysis[str(param_dict)] = {
            'params': param_dict,
            'num_seeds': len(group_results),
            'statistics': stats
        }
    
    return analysis


def print_summary_table(analysis: Dict[str, Dict]):
    """Print a nice summary table of results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS ANALYSIS")
    print("="*80)
    
    # Header
    print(f"{'Config':<30} {'Seeds':<6} {'Success%':<12} {'D4RL':<8} {'AUC':<8} {'Div%':<6} {'KL':<8} {'ESS':<8}")
    print("-" * 80)
    
    # Sort by success rate (descending)
    sorted_configs = sorted(analysis.items(), 
                          key=lambda x: x[1]['statistics'].get('success_rate_percent', {}).get('mean', 0),
                          reverse=True)
    
    for config_name, data in sorted_configs:
        params = data['params']
        stats = data['statistics']
        num_seeds = data['num_seeds']
        
        # Format parameter string
        param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k != 'seed'])
        if len(param_str) > 28:
            param_str = param_str[:25] + "..."
        
        # Extract key metrics with confidence intervals
        success = stats.get('success_rate_percent', {})
        d4rl = stats.get('d4rl_score', {})
        auc = stats.get('auc_learning_curve', {})
        div = stats.get('divergence_rate_percent', {})
        kl = stats.get('kl_policy_data', {})
        ess = stats.get('awfm_ess', {})
        
        def format_metric(metric_dict):
            if not metric_dict:
                return "N/A"
            mean = metric_dict.get('mean', 0)
            std = metric_dict.get('std', 0)
            return f"{mean:.1f}±{std:.1f}"
        
        print(f"{param_str:<30} {num_seeds:<6} {format_metric(success):<12} "
              f"{format_metric(d4rl):<8} {format_metric(auc):<8} "
              f"{format_metric(div):<6} {format_metric(kl):<8} {format_metric(ess):<8}")
    
    print("="*80)


def save_detailed_analysis(analysis: Dict[str, Dict], output_file: str):
    """Save detailed analysis to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nDetailed analysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze comprehensive metrics across seeds")
    parser.add_argument("--run_group", required=True, help="W&B run group to analyze")
    parser.add_argument("--project", default="fql", help="W&B project name")
    parser.add_argument("--hyperparams", nargs="+", default=["kl_coeff", "advantage_weighted", "adv_weight_coeff"],
                       help="Hyperparameters to group by")
    parser.add_argument("--output", default="analysis_results.json", help="Output file for detailed results")
    
    args = parser.parse_args()
    
    print(f"Collecting results from W&B project '{args.project}', group '{args.run_group}'...")
    
    # Collect results from W&B
    results = collect_wandb_results(args.run_group, args.project)
    
    if not results:
        print("No finished runs found!")
        return
    
    print(f"Found {len(results)} completed runs")
    
    # Analyze results
    analysis = analyze_hyperparameter_sweep(results, args.hyperparams)
    
    # Print summary
    print_summary_table(analysis)
    
    # Save detailed results
    save_detailed_analysis(analysis, args.output)
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    # Find best configuration
    best_config = max(analysis.items(), 
                     key=lambda x: x[1]['statistics'].get('success_rate_percent', {}).get('mean', 0))
    
    print(f"🏆 Best configuration: {best_config[1]['params']}")
    best_stats = best_config[1]['statistics']
    if 'success_rate_percent' in best_stats:
        success_mean = best_stats['success_rate_percent']['mean']
        success_ci = (best_stats['success_rate_percent']['ci_95_lower'], 
                     best_stats['success_rate_percent']['ci_95_upper'])
        print(f"   Success Rate: {success_mean:.1f}% (95% CI: {success_ci[0]:.1f}-{success_ci[1]:.1f}%)")
    
    # Check for divergence issues
    diverged_configs = [name for name, data in analysis.items() 
                       if data['statistics'].get('divergence_rate_percent', {}).get('mean', 0) > 10]
    if diverged_configs:
        print(f"⚠️  Configurations with >10% divergence rate: {len(diverged_configs)}")
    
    # ESS analysis for advantage weighting
    awfm_configs = [data for data in analysis.values() 
                   if data['params'].get('advantage_weighted', False)]
    if awfm_configs:
        avg_ess = np.mean([data['statistics'].get('awfm_ess', {}).get('mean', 0) 
                          for data in awfm_configs])
        print(f"📊 Average ESS for advantage weighting: {avg_ess:.3f}")
        if avg_ess < 0.5:
            print("   ⚠️  Low ESS suggests too aggressive advantage weighting")


if __name__ == "__main__":
    main() 