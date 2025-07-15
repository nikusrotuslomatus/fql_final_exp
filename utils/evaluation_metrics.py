"""
Comprehensive evaluation metrics for offline RL experiments.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple
import wandb


def compute_d4rl_score(returns: np.ndarray, env_name: str) -> float:
    """Compute D4RL normalized score (0=random, 100=expert)."""
    
    # D4RL score ranges (approximate values, adjust per environment)
    score_ranges = {
        # AntMaze
        'antmaze-umaze-v0': {'min': 0.0, 'max': 1.0},
        'antmaze-umaze-diverse-v0': {'min': 0.0, 'max': 1.0},
        'antmaze-medium-play-v0': {'min': 0.0, 'max': 1.0},
        'antmaze-medium-diverse-v0': {'min': 0.0, 'max': 1.0},
        'antmaze-large-play-v0': {'min': 0.0, 'max': 1.0},
        'antmaze-large-diverse-v0': {'min': 0.0, 'max': 1.0},
        
        # Locomotion
        'halfcheetah-medium-v2': {'min': -280.0, 'max': 12135.0},
        'halfcheetah-medium-replay-v2': {'min': -280.0, 'max': 12135.0},
        'halfcheetah-medium-expert-v2': {'min': -280.0, 'max': 12135.0},
        'walker2d-medium-v2': {'min': 1.6, 'max': 4592.3},
        'walker2d-medium-replay-v2': {'min': 1.6, 'max': 4592.3},
        'walker2d-medium-expert-v2': {'min': 1.6, 'max': 4592.3},
        'hopper-medium-v2': {'min': -20.3, 'max': 3234.3},
        'hopper-medium-replay-v2': {'min': -20.3, 'max': 3234.3},
        'hopper-medium-expert-v2': {'min': -20.3, 'max': 3234.3},
        
        # Kitchen
        'kitchen-complete-v0': {'min': 0.0, 'max': 4.0},
        'kitchen-partial-v0': {'min': 0.0, 'max': 4.0},
        'kitchen-mixed-v0': {'min': 0.0, 'max': 4.0},
        
        # Default fallback
        'default': {'min': 0.0, 'max': 1.0}
    }
    
    # Get score range for environment
    range_key = env_name.lower()
    for key in score_ranges:
        if key in range_key:
            score_range = score_ranges[key]
            break
    else:
        score_range = score_ranges['default']
    
    mean_return = np.mean(returns)
    min_score = score_range['min']
    max_score = score_range['max']
    
    # Normalize to 0-100 scale
    if max_score > min_score:
        normalized_score = 100.0 * (mean_return - min_score) / (max_score - min_score)
    else:
        normalized_score = 0.0
    
    return float(np.clip(normalized_score, 0.0, 100.0))


def compute_success_rate(info_dicts: List[Dict]) -> float:
    """Compute success rate from episode info dictionaries."""
    successes = []
    for info in info_dicts:
        if 'success' in info:
            successes.append(float(info['success']))
        elif 'is_success' in info:
            successes.append(float(info['is_success']))
        elif 'goal_achieved' in info:
            successes.append(float(info['goal_achieved']))
        # For AntMaze: success if reached goal position
        elif 'goal_distance' in info:
            successes.append(float(info['goal_distance'] < 0.5))
    
    if successes:
        return float(np.mean(successes)) * 100.0  # Convert to percentage
    else:
        return 0.0


def compute_auc(values: List[float], steps: List[int], max_steps: int) -> float:
    """Compute Area Under Curve for learning progress."""
    if not values or not steps:
        return 0.0
    
    # Filter to max_steps
    filtered_pairs = [(s, v) for s, v in zip(steps, values) if s <= max_steps]
    if not filtered_pairs:
        return 0.0
    
    steps_filtered, values_filtered = zip(*filtered_pairs)
    
    # Interpolate to regular grid
    step_grid = np.linspace(0, max_steps, 100)
    values_interp = np.interp(step_grid, steps_filtered, values_filtered)
    
    # Compute AUC using trapezoidal rule
    auc = float(np.trapz(values_interp, step_grid) / max_steps)
    return auc


def detect_divergence(losses: List[float], threshold: float = 1e3) -> Tuple[bool, float]:
    """Detect training divergence based on loss values."""
    if not losses:
        return False, 0.0
    
    # Check for NaN or infinite values
    has_nan = any(np.isnan(loss) or np.isinf(loss) for loss in losses)
    
    # Check for explosion (loss > threshold)
    recent_losses = losses[-10:]  # Check last 10 values
    has_explosion = any(loss > threshold for loss in recent_losses)
    
    # Compute divergence rate over recent history
    diverged_count = sum(1 for loss in recent_losses if np.isnan(loss) or np.isinf(loss) or loss > threshold)
    divergence_rate = float(diverged_count / len(recent_losses)) * 100.0
    
    return has_nan or has_explosion, divergence_rate


def compute_policy_data_kl(policy_actions: jnp.ndarray, data_actions: jnp.ndarray, 
                          bandwidth: float = 0.1) -> float:
    """Compute KL divergence between policy and data action distributions."""
    # Simple approximation using Gaussian kernel density estimation
    
    def gaussian_kde_logprob(x, samples, bandwidth):
        """Compute log probability under Gaussian KDE."""
        diff = x[None, :] - samples[:, None, :]  # [n_samples, n_points, action_dim]
        sq_dist = jnp.sum(diff ** 2, axis=-1)  # [n_samples, n_points]
        log_weights = -0.5 * sq_dist / (bandwidth ** 2)
        log_weights = log_weights - jnp.log(bandwidth * jnp.sqrt(2 * jnp.pi))
        return jax.scipy.special.logsumexp(log_weights, axis=0) - jnp.log(len(samples))
    
    # Compute log probabilities
    policy_logprob = gaussian_kde_logprob(policy_actions, policy_actions, bandwidth)
    data_logprob = gaussian_kde_logprob(policy_actions, data_actions, bandwidth)
    
    # KL divergence: E_policy[log(policy) - log(data)]
    kl_div = jnp.mean(policy_logprob - data_logprob)
    return float(kl_div)


def compute_effective_sample_size(weights: jnp.ndarray) -> float:
    """Compute effective sample size for advantage weighting."""
    if len(weights) == 0:
        return 0.0
    
    # ESS = (sum(w))^2 / sum(w^2)
    sum_w = jnp.sum(weights)
    sum_w2 = jnp.sum(weights ** 2)
    
    if sum_w2 > 0:
        ess = (sum_w ** 2) / sum_w2
        # Normalize by total number of samples
        ess_normalized = float(ess / len(weights))
        return ess_normalized
    else:
        return 0.0


class MetricsTracker:
    """Tracks and computes comprehensive evaluation metrics."""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.returns = []
        self.success_info = []
        self.eval_scores = []
        self.eval_steps = []
        self.td_losses = []
        self.policy_actions_buffer = []
        self.data_actions_buffer = []
        self.weights_buffer = []
        
    def add_evaluation(self, returns: List[float], info_dicts: List[Dict], step: int):
        """Add evaluation results."""
        if returns:
            self.returns.extend(returns)
            print(f"Step {step}: Added {len(returns)} returns, mean={np.mean(returns):.3f}")
        
        if info_dicts:
            self.success_info.extend(info_dicts)
        
        # Track evaluation score over time
        if returns:
            mean_return = np.mean(returns)
            self.eval_scores.append(mean_return)
            self.eval_steps.append(step)
            print(f"Step {step}: Tracking eval score {mean_return:.3f}")
    
    def add_training_metrics(self, td_loss: float, policy_actions: Optional[jnp.ndarray] = None,
                           data_actions: Optional[jnp.ndarray] = None, 
                           weights: Optional[jnp.ndarray] = None):
        """Add training step metrics."""
        if not np.isnan(td_loss) and not np.isinf(td_loss):
            self.td_losses.append(td_loss)
        
        if policy_actions is not None:
            self.policy_actions_buffer.append(policy_actions)
            # Keep buffer size manageable
            if len(self.policy_actions_buffer) > 100:
                self.policy_actions_buffer = self.policy_actions_buffer[-50:]
        
        if data_actions is not None:
            self.data_actions_buffer.append(data_actions)
            if len(self.data_actions_buffer) > 100:
                self.data_actions_buffer = self.data_actions_buffer[-50:]
                
        if weights is not None:
            self.weights_buffer.append(weights)
            if len(self.weights_buffer) > 100:
                self.weights_buffer = self.weights_buffer[-50:]
    
    def compute_comprehensive_metrics(self, max_steps: int = 500000) -> Dict[str, float]:
        """Compute all comprehensive metrics."""
        metrics = {}
        
        # 1. Success Rate
        if self.success_info:
            try:
                metrics['success_rate_percent'] = compute_success_rate(self.success_info)
            except Exception as e:
                print(f"Warning: Could not compute success rate: {e}")
                metrics['success_rate_percent'] = 0.0
        else:
            metrics['success_rate_percent'] = 0.0
        
        # 2. D4RL Normalized Score
        if self.returns:
            try:
                metrics['d4rl_score'] = compute_d4rl_score(np.array(self.returns), self.env_name)
                metrics['mean_return'] = float(np.mean(self.returns))
                metrics['std_return'] = float(np.std(self.returns))
                print(f"Computed mean_return: {metrics['mean_return']:.3f}")
            except Exception as e:
                print(f"Warning: Could not compute D4RL score: {e}")
                metrics['d4rl_score'] = 0.0
                metrics['mean_return'] = 0.0
                metrics['std_return'] = 0.0
        else:
            metrics['d4rl_score'] = 0.0
            metrics['mean_return'] = 0.0
            metrics['std_return'] = 0.0
        
        # 3. Area Under Learning Curve
        if self.eval_scores and self.eval_steps:
            try:
                metrics['auc_learning_curve'] = compute_auc(self.eval_scores, self.eval_steps, max_steps)
            except Exception as e:
                print(f"Warning: Could not compute AUC: {e}")
                metrics['auc_learning_curve'] = 0.0
        else:
            metrics['auc_learning_curve'] = 0.0
        
        # 4. Divergence Detection
        if self.td_losses:
            try:
                is_diverged, div_rate = detect_divergence(self.td_losses)
                metrics['divergence_detected'] = float(is_diverged)
                metrics['divergence_rate_percent'] = div_rate
                metrics['final_td_loss'] = float(self.td_losses[-1]) if self.td_losses else 0.0
            except Exception as e:
                print(f"Warning: Could not compute divergence: {e}")
                metrics['divergence_detected'] = 0.0
                metrics['divergence_rate_percent'] = 0.0
                metrics['final_td_loss'] = 0.0
        else:
            metrics['divergence_detected'] = 0.0
            metrics['divergence_rate_percent'] = 0.0
            metrics['final_td_loss'] = 0.0
        
        # 5. Policy-Data KL Divergence
        if self.policy_actions_buffer and self.data_actions_buffer:
            try:
                policy_actions = jnp.concatenate(self.policy_actions_buffer[-10:], axis=0)
                data_actions = jnp.concatenate(self.data_actions_buffer[-10:], axis=0)
                metrics['kl_policy_data'] = compute_policy_data_kl(policy_actions, data_actions)
            except Exception as e:
                print(f"Warning: Could not compute KL divergence: {e}")
                metrics['kl_policy_data'] = 0.0
        else:
            metrics['kl_policy_data'] = 0.0
        
        # 6. Effective Sample Size (AWFM)
        if self.weights_buffer:
            try:
                recent_weights = jnp.concatenate(self.weights_buffer[-10:], axis=0)
                metrics['awfm_ess'] = compute_effective_sample_size(recent_weights)
            except Exception as e:
                print(f"Warning: Could not compute ESS: {e}")
                metrics['awfm_ess'] = 1.0
        else:
            metrics['awfm_ess'] = 1.0
        
        return metrics
    
    def log_to_wandb(self, step: int, prefix: str = "comprehensive"):
        """Log comprehensive metrics to W&B."""
        metrics = self.compute_comprehensive_metrics()
        
        # Add prefix to metric names
        wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to wandb
        wandb.log(wandb_metrics, step=step)
        return metrics


def log_seed_statistics(results_across_seeds: List[Dict[str, float]], 
                       metric_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute statistics across multiple seeds."""
    seed_stats = {}
    
    for metric in metric_names:
        values = [result.get(metric, 0.0) for result in results_across_seeds]
        if values:
            seed_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'ci_95_lower': float(np.percentile(values, 2.5)),
                'ci_95_upper': float(np.percentile(values, 97.5)),
            }
    
    return seed_stats 