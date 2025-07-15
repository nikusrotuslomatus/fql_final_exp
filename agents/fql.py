import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class FQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with KL pessimism."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def compute_kl_penalty(self, batch, grad_params, rng):
        """Compute KL divergence penalty for pessimism."""
        batch_size = batch['observations'].shape[0]
        
        # Sample actions from current policy (one-step flow)
        rng, policy_rng = jax.random.split(rng)
        policy_actions = self._sample_policy_actions(
            batch['observations'], 
            seed=policy_rng, 
            num_samples=self.config['kl_num_samples'],
            grad_params=grad_params
        )
        
        # Compute Q-values for policy actions
        obs_expanded = jnp.repeat(
            jnp.expand_dims(batch['observations'], 1), 
            self.config['kl_num_samples'], 
            axis=1
        )
        obs_flat = obs_expanded.reshape(-1, *batch['observations'].shape[1:])
        policy_actions_flat = policy_actions.reshape(-1, *policy_actions.shape[2:])
        
        q_policy = self.network.select('critic')(obs_flat, actions=policy_actions_flat, params=grad_params)
        if self.config['q_agg'] == 'min':
            q_policy = q_policy.min(axis=0)
        else:
            q_policy = q_policy.mean(axis=0)
        q_policy = q_policy.reshape(batch_size, self.config['kl_num_samples'])
        
        # Compute Q-values for buffer actions
        q_buffer = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        if self.config['q_agg'] == 'min':
            q_buffer = q_buffer.min(axis=0)
        else:
            q_buffer = q_buffer.mean(axis=0)
        
        # KL penalty: penalize high Q-values on policy actions relative to buffer actions
        kl_penalty = jnp.maximum(0, q_policy.mean(axis=1) - q_buffer).mean()
        
        return kl_penalty

    def _sample_policy_actions(self, observations, seed, num_samples, grad_params):
        """Sample actions from the current one-step policy for KL penalty computation."""
        batch_size = observations.shape[0]
        action_seed, noise_seed = jax.random.split(seed)
        
        # Sample noises
        noises = jax.random.normal(
            action_seed,
            (batch_size, num_samples, self.config['action_dim'])
        )
        
        # Expand observations for vectorized computation
        obs_expanded = jnp.repeat(jnp.expand_dims(observations, 1), num_samples, axis=1)
        obs_flat = obs_expanded.reshape(-1, *observations.shape[1:])
        noises_flat = noises.reshape(-1, self.config['action_dim'])
        
        # Sample from one-step flow
        actions_flat = self.network.select('actor_onestep_flow')(obs_flat, noises_flat, params=grad_params)
        actions = actions_flat.reshape(batch_size, num_samples, self.config['action_dim'])
        actions = jnp.clip(actions, -1, 1)
        
        return actions

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss with KL pessimism."""
        rng, sample_rng, kl_rng = jax.random.split(rng, 3)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()
        
        # Add KL pessimism penalty
        kl_penalty = 0.0
        if self.config['kl_coeff'] > 0:
            kl_penalty = self.compute_kl_penalty(batch, grad_params, kl_rng)
            critic_loss = critic_loss + self.config['kl_coeff'] * kl_penalty

        return critic_loss, {
            'critic_loss': critic_loss,
            'kl_penalty': kl_penalty,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss with advantage weighting."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss with advantage weighting.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        flow_mse = (pred - vel) ** 2
        
        # Compute advantage weights
        if self.config['advantage_weighted'] and self.config['adv_weight_coeff'] > 0:
            # Get Q-values for buffer actions
            q_values = self.network.select('critic')(batch['observations'], actions=batch['actions'])
            if self.config['q_agg'] == 'min':
                q = q_values.min(axis=0)
            else:
                q = q_values.mean(axis=0)
            
            # Estimate V(s) as mean Q over the batch (simple baseline)
            # For better results, could use a separate value network
            v_baseline = q.mean()
            advantage = q - v_baseline
            
            # Compute advantage weights: w = exp(β * Adv)
            # β is scaled by inverse of Q magnitude for stability
            beta = self.config['adv_weight_coeff'] / (jnp.abs(q).mean() + 1e-6)
            weights = jnp.exp(beta * advantage)
            
            # Clip weights to prevent extreme values
            weights = jnp.clip(weights, 0.1, 10.0)
            
            # Apply weights to flow loss
            bc_flow_loss = jnp.mean(weights * flow_mse.mean(axis=-1))
        else:
            bc_flow_loss = jnp.mean(flow_mse)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        # Advantage weighting metrics
        adv_metrics = {}
        if self.config['advantage_weighted'] and self.config['adv_weight_coeff'] > 0:
            q_values = self.network.select('critic')(batch['observations'], actions=batch['actions'])
            if self.config['q_agg'] == 'min':
                q = q_values.min(axis=0)
            else:
                q = q_values.mean(axis=0)
            v_baseline = q.mean()
            advantage = q - v_baseline
            beta = self.config['adv_weight_coeff'] / (jnp.abs(q).mean() + 1e-6)
            weights = jnp.exp(beta * advantage)
            weights = jnp.clip(weights, 0.1, 10.0)
            
            adv_metrics.update({
                'advantage_mean': advantage.mean(),
                'advantage_std': advantage.std(),
                'weights_mean': weights.mean(),
                'weights_std': weights.std(),
                'beta_effective': beta,
            })

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
            **adv_metrics,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the midpoint method.
        
        The midpoint method (RK2) provides O(dt²) accuracy compared to O(dt) for Euler,
        resulting in more accurate flow integration and better distillation targets.
        """
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        dt = 1.0 / self.config['flow_steps']
        
        # Midpoint method (RK2).
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i * dt)
            
            # First evaluation at current point
            v1 = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            
            # Midpoint evaluation
            t_mid = t + dt / 2.0
            actions_mid = actions + v1 * dt / 2.0
            v_mid = self.network.select('actor_bc_flow')(observations, actions_mid, t_mid, is_encoded=True)
            
            # Final step using midpoint velocity
            actions = actions + v_mid * dt
            
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Get default configuration for FQL agent with KL pessimism and advantage weighting.
    
    KL Pessimism hyperparameters:
    - kl_coeff: Coefficient for KL penalty (0.0 to disable). Start with 0.1-1.0.
      Higher values make the critic more pessimistic about OOD actions.
    - kl_num_samples: Number of policy action samples for KL penalty computation.
      More samples = more accurate penalty but slower training. 10-20 is typical.
    
    Advantage Weighting hyperparameters:
    - advantage_weighted: Enable advantage-weighted flow matching (False to disable).
    - adv_weight_coeff: Coefficient for advantage weighting. Higher = more focus on good actions.
      Start with 1.0, increase to 2.0+ for noisy datasets.
    
    Usage examples:
    # Conservative + advantage weighting: 
    kl_coeff=0.5, advantage_weighted=True, adv_weight_coeff=1.0
    # High-quality data: 
    kl_coeff=0.1, advantage_weighted=True, adv_weight_coeff=0.5
    # Noisy demonstrations:
    kl_coeff=0.8, advantage_weighted=True, adv_weight_coeff=2.0
    """
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            flow_steps=15,  # Number of flow steps (increased for better accuracy with midpoint method).
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            kl_coeff=0.0,  # KL coefficient for pessimism.
            kl_num_samples=10,  # Number of samples for KL penalty.
            advantage_weighted=True,  # Whether to use advantage-weighted flow matching.
            adv_weight_coeff=1.0,  # Coefficient for advantage weighting (higher = more emphasis on good actions).
        )
    )
    return config
