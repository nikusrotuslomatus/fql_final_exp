#!/usr/bin/env python3
"""
CSV Logger for FQL training metrics.
Saves key metrics to CSV files for easy matplotlib visualization.
"""

import os
import csv
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np


class CSVMetricsLogger:
    """Logs training and evaluation metrics to CSV files."""
    
    def __init__(self, log_dir: str = "csv_logs", run_name: Optional[str] = None):
        """
        Initialize CSV logger.
        
        Args:
            log_dir: Directory to save CSV files
            run_name: Name of the run (used as prefix for files)
        """
        self.log_dir = log_dir
        self.run_name = run_name or f"run_{int(time.time())}"
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize data storage
        self.metrics_data = defaultdict(list)
        self.steps = []
        
        # Define key metrics to track
        self.key_metrics = {
            'training_loss': 'Training Loss',
            'mean_return': 'Mean Return',
            'success_rate': 'Success Rate',
            'kl_divergence': 'KL Divergence',
            'q_values': 'Q Values',
            'd4rl_score': 'D4RL Score',
            'td_loss': 'TD Loss',
            'policy_loss': 'Policy Loss',
            'critic_loss': 'Critic Loss',
            'advantage_weights': 'Advantage Weights',
            'effective_sample_size': 'Effective Sample Size'
        }
        
        # File handles for continuous writing
        self.csv_files = {}
        self.csv_writers = {}
        
        print(f"📊 CSV Logger initialized: {log_dir}/{run_name}_*.csv")
    
    def _get_csv_path(self, metric_name: str) -> str:
        """Get CSV file path for a metric."""
        return os.path.join(self.log_dir, f"{self.run_name}_{metric_name}.csv")
    
    def _init_csv_file(self, metric_name: str):
        """Initialize CSV file for a metric."""
        if metric_name not in self.csv_files:
            csv_path = self._get_csv_path(metric_name)
            self.csv_files[metric_name] = open(csv_path, 'w', newline='')
            self.csv_writers[metric_name] = csv.writer(self.csv_files[metric_name])
            
            # Write header
            header = ['Step', 'Value', 'Timestamp']
            if metric_name in ['mean_return', 'success_rate', 'd4rl_score']:
                header.extend(['Min', 'Max', 'Std', 'Count'])
            
            self.csv_writers[metric_name].writerow(header)
            self.csv_files[metric_name].flush()
    
    def log_training_metrics(self, step: int, metrics: Dict[str, Any]):
        """
        Log training metrics.
        
        Args:
            step: Training step
            metrics: Dictionary of metrics
        """
        timestamp = time.time()
        
        # Training loss (TD loss)
        if 'td_loss' in metrics:
            self._log_scalar('td_loss', step, metrics['td_loss'], timestamp)
        
        # Policy loss
        if 'policy_loss' in metrics:
            self._log_scalar('policy_loss', step, metrics['policy_loss'], timestamp)
        
        # Critic loss
        if 'critic_loss' in metrics:
            self._log_scalar('critic_loss', step, metrics['critic_loss'], timestamp)
        
        # KL divergence
        if 'kl_divergence' in metrics:
            self._log_scalar('kl_divergence', step, metrics['kl_divergence'], timestamp)
        
        # Q values
        if 'q_values' in metrics:
            self._log_scalar('q_values', step, metrics['q_values'], timestamp)
        
        # Advantage weights
        if 'advantage_weights' in metrics:
            self._log_scalar('advantage_weights', step, metrics['advantage_weights'], timestamp)
        
        # Effective sample size
        if 'effective_sample_size' in metrics:
            self._log_scalar('effective_sample_size', step, metrics['effective_sample_size'], timestamp)
    
    def log_evaluation_metrics(self, step: int, returns: List[float], 
                             success_rate: float, d4rl_score: float):
        """
        Log evaluation metrics.
        
        Args:
            step: Training step
            returns: List of episode returns
            success_rate: Success rate (0-1)
            d4rl_score: D4RL normalized score
        """
        timestamp = time.time()
        
        # Mean return with statistics
        if returns:
            mean_return = np.mean(returns)
            min_return = np.min(returns)
            max_return = np.max(returns)
            std_return = np.std(returns)
            count = len(returns)
            
            self._log_scalar_with_stats('mean_return', step, mean_return, 
                                      min_return, max_return, std_return, count, timestamp)
        
        # Success rate
        self._log_scalar('success_rate', step, success_rate, timestamp)
        
        # D4RL score
        self._log_scalar('d4rl_score', step, d4rl_score, timestamp)
    
    def _log_scalar(self, metric_name: str, step: int, value: float, timestamp: float):
        """Log a scalar metric."""
        self._init_csv_file(metric_name)
        
        row = [step, value, timestamp]
        self.csv_writers[metric_name].writerow(row)
        self.csv_files[metric_name].flush()
        
        # Track steps for summary
        self.steps.append(step)
        self.metrics_data[metric_name].append((step, value))
    
    def _log_scalar_with_stats(self, metric_name: str, step: int, value: float,
                              min_val: float, max_val: float, std_val: float, 
                              count: int, timestamp: float):
        """Log a scalar metric with statistics."""
        self._init_csv_file(metric_name)
        
        row = [step, value, timestamp, min_val, max_val, std_val, count]
        self.csv_writers[metric_name].writerow(row)
        self.csv_files[metric_name].flush()
        
        # Track steps for summary
        self.steps.append(step)
        self.metrics_data[metric_name].append((step, value))
    
    def create_summary_csv(self):
        """Create a summary CSV with all key metrics."""
        summary_path = self._get_csv_path('summary')
        
        # Collect all data
        all_steps = sorted(set(self.steps))
        if not all_steps:
            print("⚠️  No data to create summary CSV")
            return
        
        # Create summary data
        summary_data = []
        for step in all_steps:
            row = {'Step': step}
            
            # Add metrics for this step
            for metric_name in self.key_metrics:
                if metric_name in self.metrics_data:
                    # Find closest step
                    metric_steps = [s for s, _ in self.metrics_data[metric_name]]
                    if step in metric_steps:
                        idx = metric_steps.index(step)
                        row[self.key_metrics[metric_name]] = self.metrics_data[metric_name][idx][1]
                    else:
                        row[self.key_metrics[metric_name]] = None
            
            summary_data.append(row)
        
        # Write summary CSV
        if summary_data:
            with open(summary_path, 'w', newline='') as f:
                fieldnames = ['Step'] + list(self.key_metrics.values())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
            
            print(f"📋 Summary CSV created: {summary_path}")
    
    def close(self):
        """Close all CSV files."""
        for f in self.csv_files.values():
            f.close()
        
        # Create summary
        self.create_summary_csv()
        
        print(f"📊 CSV logging completed. Files saved in: {self.log_dir}/")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except:
            pass


def create_matplotlib_script(log_dir: str = "csv_logs", run_name: str = "run"):
    """Create a matplotlib script to visualize the logged metrics."""
    
    script_content = f'''#!/usr/bin/env python3
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
LOG_DIR = "{log_dir}"
RUN_NAME = "{run_name}"

def plot_metric(csv_file, title, ylabel, save_name):
    """Plot a single metric from CSV file."""
    if not os.path.exists(csv_file):
        print(f"File not found: {{csv_file}}")
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
    plt.savefig(f"{{save_name}}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_metrics():
    """Plot all available metrics."""
    
    # Define metrics to plot
    metrics_config = {{
        'mean_return': ('Mean Episode Return', 'Return'),
        'success_rate': ('Success Rate', 'Success Rate'),
        'd4rl_score': ('D4RL Normalized Score', 'D4RL Score'),
        'td_loss': ('TD Loss', 'Loss'),
        'kl_divergence': ('KL Divergence', 'KL Divergence'),
        'q_values': ('Q Values', 'Q Value'),
        'advantage_weights': ('Advantage Weights', 'Weight'),
        'effective_sample_size': ('Effective Sample Size', 'Sample Size')
    }}
    
    print("📊 Plotting FQL training metrics...")
    
    for metric_name, (title, ylabel) in metrics_config.items():
        csv_file = os.path.join(LOG_DIR, f"{{RUN_NAME}}_{{metric_name}}.csv")
        save_name = f"{{RUN_NAME}}_{{metric_name}}_plot"
        plot_metric(csv_file, title, ylabel, save_name)
    
    print("✅ All plots generated!")

def plot_summary():
    """Plot summary of key metrics in subplots."""
    summary_file = os.path.join(LOG_DIR, f"{{RUN_NAME}}_summary.csv")
    
    if not os.path.exists(summary_file):
        print(f"Summary file not found: {{summary_file}}")
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
    plt.savefig(f"{{RUN_NAME}}_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("🎨 FQL Metrics Visualization")
    print("=" * 50)
    
    # Plot individual metrics
    plot_all_metrics()
    
    # Plot summary
    plot_summary()
    
    print("\\n📈 Visualization complete!")
'''
    
    script_path = os.path.join(log_dir, f"plot_{run_name}_metrics.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"📈 Matplotlib script created: {script_path}")
    print(f"   Run with: python {script_path}") 