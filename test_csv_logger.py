#!/usr/bin/env python3
"""
Test script for CSV logger functionality.
"""

import numpy as np
import time
from utils.csv_logger import CSVMetricsLogger, create_matplotlib_script

def test_csv_logger():
    """Test the CSV logger with sample data."""
    print("🧪 Testing CSV Logger...")
    
    # Initialize logger
    logger = CSVMetricsLogger(log_dir="test_csv_logs", run_name="test_run")
    
    # Simulate training data
    print("📝 Logging training metrics...")
    for step in range(0, 1000, 100):
        # Simulate training metrics
        metrics = {
            'td_loss': 0.5 * np.exp(-step / 500) + 0.1 * np.random.random(),
            'critic_loss': 0.3 * np.exp(-step / 400) + 0.05 * np.random.random(),
            'policy_loss': 0.2 * np.exp(-step / 600) + 0.02 * np.random.random(),
            'kl_divergence': 0.1 * np.exp(-step / 300) + 0.01 * np.random.random(),
            'q_values': -10 + 5 * (1 - np.exp(-step / 800)) + 0.5 * np.random.random(),
            'advantage_weights': 0.5 + 0.3 * np.sin(step / 200) + 0.1 * np.random.random(),
            'effective_sample_size': 100 + 50 * np.random.random()
        }
        
        logger.log_training_metrics(step, metrics)
        time.sleep(0.01)  # Small delay to simulate real training
    
    # Simulate evaluation data
    print("📊 Logging evaluation metrics...")
    for step in range(0, 1000, 200):
        # Simulate episode returns
        base_return = -5 + 10 * (1 - np.exp(-step / 600))
        returns = [base_return + 2 * np.random.random() - 1 for _ in range(10)]
        
        # Simulate success rate (improving over time)
        success_rate = min(0.9, step / 1000 + 0.1 * np.random.random())
        
        # Simulate D4RL score
        d4rl_score = min(100, step / 10 + 5 * np.random.random())
        
        logger.log_evaluation_metrics(step, returns, success_rate, d4rl_score)
        time.sleep(0.01)
    
    # Close logger
    logger.close()
    
    # Create matplotlib script
    create_matplotlib_script(log_dir="test_csv_logs", run_name="test_run")
    
    print("✅ CSV Logger test completed!")
    print("📁 Check test_csv_logs/ directory for CSV files")
    print("📈 Run: python test_csv_logs/plot_test_run_metrics.py")

if __name__ == "__main__":
    test_csv_logger() 