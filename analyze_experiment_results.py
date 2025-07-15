#!/usr/bin/env python3
"""
Analysis script for FQL three-environment experiments.
Analyzes CSV data and creates comparative visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from typing import Dict, List, Tuple
import argparse

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def find_csv_files(csv_dir: str = "csv_logs") -> Dict[str, List[str]]:
    """Find all CSV files organized by experiment type."""
    
    experiments = {
        'puzzle': [],
        'antmaze': [],
        'humanoid': []
    }
    
    if not os.path.exists(csv_dir):
        print(f"❌ CSV directory not found: {csv_dir}")
        return experiments
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # Categorize by experiment type
        if 'puzzle' in filename.lower():
            experiments['puzzle'].append(file_path)
        elif 'antmaze' in filename.lower():
            experiments['antmaze'].append(file_path)
        elif 'humanoid' in filename.lower():
            experiments['humanoid'].append(file_path)
    
    return experiments

def load_metric_data(csv_files: List[str], metric: str) -> pd.DataFrame:
    """Load and combine metric data from multiple CSV files."""
    
    all_data = []
    
    for file_path in csv_files:
        if metric in os.path.basename(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Extract metadata from filename
                filename = os.path.basename(file_path)
                parts = filename.replace('.csv', '').split('_')
                
                # Try to extract experiment info
                env_name = "unknown"
                config = "unknown"
                seed = "unknown"
                
                for i, part in enumerate(parts):
                    if 'puzzle' in part.lower():
                        env_name = part
                    elif 'antmaze' in part.lower():
                        env_name = part
                    elif 'humanoid' in part.lower():
                        env_name = part
                    elif part in ['baseline', 'kl', 'awfm']:
                        config = part
                    elif part.startswith('seed'):
                        seed = part
                
                df['environment'] = env_name
                df['configuration'] = config
                df['seed'] = seed
                df['source_file'] = filename
                
                all_data.append(df)
                
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def analyze_success_rates(experiments: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """Analyze success rates for each experiment."""
    
    results = {}
    
    for exp_name, csv_files in experiments.items():
        print(f"\n📊 Analyzing success rates for {exp_name}...")
        
        # Load success rate data
        success_data = load_metric_data(csv_files, 'success_rate')
        
        if success_data.empty:
            print(f"  ❌ No success rate data found for {exp_name}")
            continue
        
        # Calculate statistics
        stats = success_data.groupby(['environment', 'configuration']).agg({
            'Value': ['mean', 'std', 'count', 'max']
        }).round(4)
        
        stats.columns = ['mean_success', 'std_success', 'count', 'max_success']
        stats = stats.reset_index()
        
        results[exp_name] = stats
        
        print(f"  ✅ Found {len(success_data)} success rate measurements")
        print(f"  📈 Best configuration: {stats.loc[stats['mean_success'].idxmax(), 'configuration']}")
    
    return results

def analyze_mean_returns(experiments: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """Analyze mean returns for each experiment."""
    
    results = {}
    
    for exp_name, csv_files in experiments.items():
        print(f"\n📊 Analyzing mean returns for {exp_name}...")
        
        # Load mean return data
        return_data = load_metric_data(csv_files, 'mean_return')
        
        if return_data.empty:
            print(f"  ❌ No mean return data found for {exp_name}")
            continue
        
        # Calculate statistics
        stats = return_data.groupby(['environment', 'configuration']).agg({
            'Value': ['mean', 'std', 'count', 'max']
        }).round(4)
        
        stats.columns = ['mean_return', 'std_return', 'count', 'max_return']
        stats = stats.reset_index()
        
        results[exp_name] = stats
        
        print(f"  ✅ Found {len(return_data)} return measurements")
        print(f"  📈 Best configuration: {stats.loc[stats['mean_return'].idxmax(), 'configuration']}")
    
    return results

def create_comparison_plots(experiments: Dict[str, List[str]], output_dir: str = "analysis_plots"):
    """Create comparative plots for all experiments."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration colors
    config_colors = {
        'baseline': '#1f77b4',
        'kl': '#ff7f0e', 
        'awfm': '#2ca02c',
        'kl_pessimism': '#ff7f0e',
        'unknown': '#d62728'
    }
    
    for exp_name, csv_files in experiments.items():
        print(f"\n🎨 Creating plots for {exp_name}...")
        
        # Load data
        success_data = load_metric_data(csv_files, 'success_rate')
        return_data = load_metric_data(csv_files, 'mean_return')
        
        if success_data.empty and return_data.empty:
            print(f"  ❌ No data found for {exp_name}")
            continue
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'FQL Experiment Results: {exp_name.upper()}', fontsize=16, fontweight='bold')
        
        # Plot 1: Success Rate Over Time
        if not success_data.empty:
            ax1 = axes[0, 0]
            for config in success_data['configuration'].unique():
                config_data = success_data[success_data['configuration'] == config]
                if not config_data.empty:
                    ax1.plot(config_data['Step'], config_data['Value'], 
                            label=config, color=config_colors.get(config, '#d62728'),
                            linewidth=2, alpha=0.8)
            
            ax1.set_title('Success Rate Over Training')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Success Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean Return Over Time
        if not return_data.empty:
            ax2 = axes[0, 1]
            for config in return_data['configuration'].unique():
                config_data = return_data[return_data['configuration'] == config]
                if not config_data.empty:
                    ax2.plot(config_data['Step'], config_data['Value'],
                            label=config, color=config_colors.get(config, '#d62728'),
                            linewidth=2, alpha=0.8)
            
            ax2.set_title('Mean Return Over Training')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Mean Return')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Final Success Rate Comparison
        if not success_data.empty:
            ax3 = axes[1, 0]
            final_success = success_data.groupby('configuration')['Value'].agg(['mean', 'std']).reset_index()
            
            bars = ax3.bar(final_success['configuration'], final_success['mean'],
                          yerr=final_success['std'], capsize=5,
                          color=[config_colors.get(c, '#d62728') for c in final_success['configuration']])
            
            ax3.set_title('Final Success Rate by Configuration')
            ax3.set_ylabel('Success Rate')
            ax3.set_xlabel('Configuration')
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, final_success['mean']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Final Return Comparison
        if not return_data.empty:
            ax4 = axes[1, 1]
            final_return = return_data.groupby('configuration')['Value'].agg(['mean', 'std']).reset_index()
            
            bars = ax4.bar(final_return['configuration'], final_return['mean'],
                          yerr=final_return['std'], capsize=5,
                          color=[config_colors.get(c, '#d62728') for c in final_return['configuration']])
            
            ax4.set_title('Final Mean Return by Configuration')
            ax4.set_ylabel('Mean Return')
            ax4.set_xlabel('Configuration')
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, final_return['mean']):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{exp_name}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved plot: {plot_path}")
        
        plt.close()

def create_summary_report(experiments: Dict[str, List[str]], output_file: str = "experiment_analysis_summary.txt"):
    """Create a comprehensive summary report."""
    
    print(f"\n📝 Creating summary report: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("FQL Three-Environment Experiment Analysis Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Analyze each experiment
        success_results = analyze_success_rates(experiments)
        return_results = analyze_mean_returns(experiments)
        
        for exp_name in experiments.keys():
            f.write(f"\n{exp_name.upper()} EXPERIMENT RESULTS\n")
            f.write("-" * 30 + "\n")
            
            # Success rate analysis
            if exp_name in success_results:
                f.write("\nSuccess Rate Analysis:\n")
                success_df = success_results[exp_name]
                f.write(success_df.to_string(index=False))
                f.write("\n")
                
                # Find best configuration
                best_config = success_df.loc[success_df['mean_success'].idxmax()]
                f.write(f"Best Configuration: {best_config['configuration']} ")
                f.write(f"(Success Rate: {best_config['mean_success']:.3f} ± {best_config['std_success']:.3f})\n")
            
            # Return analysis
            if exp_name in return_results:
                f.write("\nMean Return Analysis:\n")
                return_df = return_results[exp_name]
                f.write(return_df.to_string(index=False))
                f.write("\n")
                
                # Find best configuration
                best_config = return_df.loc[return_df['mean_return'].idxmax()]
                f.write(f"Best Configuration: {best_config['configuration']} ")
                f.write(f"(Mean Return: {best_config['mean_return']:.1f} ± {best_config['std_return']:.1f})\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("OVERALL CONCLUSIONS\n")
        f.write("=" * 50 + "\n")
        f.write("1. Compare success rates across environments\n")
        f.write("2. Evaluate KL pessimism vs AWFM effectiveness\n")
        f.write("3. Identify best configuration per environment\n")
        f.write("4. Check for consistent patterns across experiments\n")
        f.write("\nFor detailed visualizations, see analysis_plots/ directory\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze FQL experiment results')
    parser.add_argument('--csv_dir', default='csv_logs', help='Directory containing CSV files')
    parser.add_argument('--output_dir', default='analysis_plots', help='Output directory for plots')
    parser.add_argument('--summary_file', default='experiment_analysis_summary.txt', help='Summary report file')
    
    args = parser.parse_args()
    
    print("🔍 FQL Experiment Results Analysis")
    print("=" * 40)
    
    # Find CSV files
    experiments = find_csv_files(args.csv_dir)
    
    total_files = sum(len(files) for files in experiments.values())
    print(f"📁 Found {total_files} CSV files across {len(experiments)} experiment types")
    
    for exp_name, files in experiments.items():
        print(f"  {exp_name}: {len(files)} files")
    
    if total_files == 0:
        print("❌ No CSV files found. Make sure experiments have been run.")
        return
    
    # Create plots
    create_comparison_plots(experiments, args.output_dir)
    
    # Create summary report
    create_summary_report(experiments, args.summary_file)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Plots saved in: {args.output_dir}/")
    print(f"📝 Summary report: {args.summary_file}")
    print(f"\n🎯 Key files to check:")
    print(f"  - {args.summary_file}")
    print(f"  - {args.output_dir}/puzzle_analysis.png")
    print(f"  - {args.output_dir}/antmaze_analysis.png")
    print(f"  - {args.output_dir}/humanoid_analysis.png")

if __name__ == "__main__":
    main() 