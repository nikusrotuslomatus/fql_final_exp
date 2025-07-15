import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import platform

# Universal compatibility patch for D4RL/mujoco_py issues
try:
    # Apply patches before importing D4RL-dependent modules
    from patch_colab_compatibility import apply_all_patches
    apply_all_patches()
except ImportError as e:
    print(f"Warning: Could not apply compatibility patches: {e}")
    print("Proceeding without patches - some environments may have issues with D4RL imports")

import json
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.evaluation_metrics import MetricsTracker
from utils.csv_logger import CSVMetricsLogger, create_matplotlib_script
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
 
FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_boolean('use_wandb', True, 'Whether to use Weights & Biases logging.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    if FLAGS.use_wandb:
        setup_wandb(project='fql', group=FLAGS.run_group, name=exp_name)
    else:
        # Initialize wandb in offline mode for compatibility
        wandb.init(mode='offline', project='fql', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    if FLAGS.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    
    # Initialize comprehensive metrics tracker
    metrics_tracker = MetricsTracker(FLAGS.env_name)
    
    # Initialize CSV logger
    run_name = f"{FLAGS.env_name}_{FLAGS.run_group}_{int(time.time())}"
    csv_logger = CSVMetricsLogger(log_dir="csv_logs", run_name=run_name)
    
    # Create matplotlib script for visualization
    create_matplotlib_script(log_dir="csv_logs", run_name=run_name)
    
    # Debug configuration
    print(f"Agent config type: {type(config)}")
    print(f"Agent config keys: {list(config.keys()) if hasattr(config, 'keys') else 'No keys method'}")
    print(f"Agent name: {config.get('agent_name', 'Unknown')}")
    print(f"Advantage weighted: {config.get('advantage_weighted', False)}")
    print(f"KL coeff: {config.get('kl_coeff', 0.0)}")
    print(f"Adv weight coeff: {config.get('adv_weight_coeff', 1.0)}")
    print(f"Agent config after creation: {agent.config if hasattr(agent, 'config') else 'No config'}")
    
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)
            
            # Track training metrics for comprehensive evaluation
            td_loss = update_info.get('critic/critic_loss', 0.0)
            
            # Extract policy actions and weights if available
            # Always try to extract policy actions for KL computation
            try:
                sample_obs = batch['observations'][:32]  # Sample subset for efficiency
                rng, sample_rng = jax.random.split(agent.rng)
                policy_actions = agent.sample_actions(sample_obs, seed=sample_rng)
                
                # Get weights from advantage weighting if active
                weights = None
                if config.get('advantage_weighted', False) and 'actor/weights_mean' in update_info:
                    # Approximate weights calculation
                    q_values = agent.network.select('critic')(sample_obs, actions=batch['actions'][:32])
                    if hasattr(agent.config, 'q_agg') and agent.config['q_agg'] == 'min':
                        q = q_values.min(axis=0)
                    else:
                        q = q_values.mean(axis=0)
                    
                    if config['agent_name'] == 'ifql':
                        v = agent.network.select('value')(sample_obs)
                    else:
                        v = q.mean()
                    
                    advantage = q - v
                    beta = config.get('adv_weight_coeff', 1.0) / (jnp.abs(q).mean() + 1e-6)
                    weights = jnp.exp(beta * advantage)
                    weights = jnp.clip(weights, 0.1, 10.0)
            except Exception as e:
                policy_actions = None
                weights = None
            
            metrics_tracker.add_training_metrics(
                td_loss=td_loss,
                policy_actions=policy_actions,
                data_actions=batch['actions'][:32] if policy_actions is not None else None,
                weights=weights
            )
            
            # Log training metrics to CSV
            csv_training_metrics = {
                'td_loss': td_loss,
                'critic_loss': update_info.get('critic/critic_loss', 0.0),
                'policy_loss': update_info.get('actor/actor_loss', 0.0),
            }
            
            # Add KL divergence if available
            if policy_actions is not None and len(policy_actions) > 0:
                kl_div = metrics_tracker.compute_kl_divergence(policy_actions, batch['actions'][:32])
                csv_training_metrics['kl_divergence'] = kl_div
            
            # Add Q values if available
            if 'critic/q_mean' in update_info:
                csv_training_metrics['q_values'] = update_info['critic/q_mean']
            
            # Add advantage weights if available
            if weights is not None and len(weights) > 0:
                csv_training_metrics['advantage_weights'] = np.mean(weights)
            
            csv_logger.log_training_metrics(step, csv_training_metrics)
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent.
            if FLAGS.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)
            
            # Track training metrics for online phase too
            td_loss = update_info.get('critic/critic_loss', 0.0)
            metrics_tracker.add_training_metrics(td_loss=td_loss)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            # Add evaluation results to comprehensive metrics tracker
            returns = [sum(traj['reward']) for traj in trajs]  # 'reward' not 'rewards'
            info_dicts = [traj.get('info', {}) for traj in trajs]  # 'info' not 'infos'
            
            # Debug info structure
            if len(info_dicts) > 0:
                print(f"First info_dict type: {type(info_dicts[0])}")
                if isinstance(info_dicts[0], list) and len(info_dicts[0]) > 0:
                    print(f"First info_dict[0] keys: {list(info_dicts[0][-1].keys()) if isinstance(info_dicts[0][-1], dict) else 'Not a dict'}")
                elif isinstance(info_dicts[0], dict):
                    print(f"First info_dict keys: {list(info_dicts[0].keys())}")
            
            # Flatten info dicts if they are lists
            flat_info_dicts = []
            for info_dict in info_dicts:
                if isinstance(info_dict, list) and len(info_dict) > 0:
                    flat_info_dicts.append(info_dict[-1])  # Take final info
                elif isinstance(info_dict, dict):
                    flat_info_dicts.append(info_dict)
                else:
                    flat_info_dicts.append({})  # Empty dict as fallback
            
            metrics_tracker.add_evaluation(returns, flat_info_dicts, i)
            
            # Log comprehensive metrics
            comprehensive_metrics = metrics_tracker.log_to_wandb(i, prefix="comprehensive")
            
            # Log evaluation metrics to CSV
            csv_logger.log_evaluation_metrics(
                step=i,
                returns=returns,
                success_rate=success_rate,
                d4rl_score=d4rl_score
            )

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()
    
    # Close CSV logger and create summary
    csv_logger.close()
    print(f"📊 CSV files saved in csv_logs/ directory")
    print(f"📈 To visualize: python csv_logs/plot_{run_name}_metrics.py")


if __name__ == '__main__':
    app.run(main)
