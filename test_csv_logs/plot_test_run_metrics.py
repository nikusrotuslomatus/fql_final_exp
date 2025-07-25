#!/usr/bin/env python3
"""
Matplotlib visualization script for FQL training metrics.
Auto-generated script to plot CSV logged metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Configuration
LOG_DIR = "test_csv_logs"
RUN_NAME = "test_run"

def plot_metric(csv_file, title, ylabel, save_name):
    """Plot a single metric from CSV file."""
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Step'], df['Value'], linewidth=2, alpha=0.8)
    
    # Add confidence intervals if available
    if 'Min' in df.columns and 'Max' in df.columns:
        plt.fill_between(df['Step'], df['Min'], df['Max'], alpha=0.2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_metrics():
    """Plot all available metrics."""
    
    # Define metrics to plot
    metrics_config = {
        'mean_return': ('Mean Episode Return', 'Return'),
        'success_rate': ('Success Rate', 'Success Rate'),
        'd4rl_score': ('D4RL Normalized Score', 'D4RL Score'),
        'td_loss': ('TD Loss', 'Loss'),
        'kl_divergence': ('KL Divergence', 'KL Divergence'),
        'q_values': ('Q Values', 'Q Value'),
        'advantage_weights': ('Advantage Weights', 'Weight'),
        'effective_sample_size': ('Effective Sample Size', 'Sample Size')
    }
    
    print("📊 Plotting FQL training metrics...")
    
    for metric_name, (title, ylabel) in metrics_config.items():
        csv_file = os.path.join(LOG_DIR, f"{RUN_NAME}_{metric_name}.csv")
        save_name = f"{RUN_NAME}_{metric_name}_plot"
        plot_metric(csv_file, title, ylabel, save_name)
    
    print("✅ All plots generated!")

def plot_summary():
    """Plot summary of key metrics in subplots."""
    summary_file = os.path.join(LOG_DIR, f"{RUN_NAME}_summary.csv")
    
    if not os.path.exists(summary_file):
        print(f"Summary file not found: {summary_file}")
        return
    
    df = pd.read_csv(summary_file)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('FQL Training Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean Return
    if 'Mean Return' in df.columns:
        axes[0, 0].plot(df['Step'], df['Mean Return'], 'b-', linewidth=2)
        axes[0, 0].set_title('Mean Return')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    if 'Success Rate' in df.columns:
        axes[0, 1].plot(df['Step'], df['Success Rate'], 'g-', linewidth=2)
        axes[0, 1].set_title('Success Rate')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: TD Loss
    if 'TD Loss' in df.columns:
        axes[1, 0].plot(df['Step'], df['TD Loss'], 'r-', linewidth=2)
        axes[1, 0].set_title('TD Loss')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: D4RL Score
    if 'D4RL Score' in df.columns:
        axes[1, 1].plot(df['Step'], df['D4RL Score'], 'm-', linewidth=2)
        axes[1, 1].set_title('D4RL Score')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Set x-labels
    for ax in axes.flat:
        ax.set_xlabel('Training Step')
    
    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("🎨 FQL Metrics Visualization")
    print("=" * 50)
    
    # Plot individual metrics
    plot_all_metrics()
    
    # Plot summary
    plot_summary()
    
    print("\n📈 Visualization complete!")
