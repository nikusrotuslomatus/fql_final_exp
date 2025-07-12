import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
import distrax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class IFQLAgent(flax.struct.PyTreeNode):
    """Implicit flow Q-learning (IFQL) agent with KL pessimism.

    IFQL is the flow variant of implicit diffusion Q-learning (IDQL).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def compute_kl_penalty(self, batch, grad_params, rng):
        """Compute KL divergence penalty for pessimism."""
        batch_size = batch['observations'].shape[0]
        
        # Sample actions from current policy
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
        
        q1, q2 = self.network.select('critic')(obs_flat, actions=policy_actions_flat, params=grad_params)
        q_policy = jnp.minimum(q1, q2).reshape(batch_size, self.config['kl_num_samples'])
        
        # Compute Q-values for buffer actions
        q1_buffer, q2_buffer = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        q_buffer = jnp.minimum(q1_buffer, q2_buffer)
        
        # KL penalty: penalize high Q-values on policy actions relative to buffer actions
        kl_penalty = jnp.maximum(0, q_policy.mean(axis=1) - q_buffer).mean()
        
        return kl_penalty

    def _sample_policy_actions(self, observations, seed, num_samples, grad_params):
        """Sample actions from the current policy for KL penalty computation."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        
        batch_size = observations.shape[0]
        action_seed, noise_seed = jax.random.split(seed)
        
        # Sample noises
        actions = jax.random.normal(
            action_seed,
            (batch_size, num_samples, self.config['action_dim'])
        )
        
        # Expand observations for vectorized computation
        n_observations = jnp.repeat(jnp.expand_dims(observations, 1), num_samples, axis=1)
        n_observations = n_observations.reshape(-1, *observations.shape[1:])
        actions_flat = actions.reshape(-1, self.config['action_dim'])
        
        # Integrate through flow using midpoint method
        dt = 1.0 / self.config['flow_steps']
        for i in range(self.config['flow_steps']):
            t = jnp.full((actions_flat.shape[0], 1), i * dt)
            
            # First evaluation at current point
            v1 = self.network.select('actor_flow')(n_observations, actions_flat, t, is_encoded=True, params=grad_params)
            
            # Midpoint evaluation
            t_mid = t + dt / 2.0
            actions_mid = actions_flat + v1 * dt / 2.0
            v_mid = self.network.select('actor_flow')(n_observations, actions_mid, t_mid, is_encoded=True, params=grad_params)
            
            # Final step using midpoint velocity
            actions_flat = actions_flat + v_mid * dt
        
        actions = actions_flat.reshape(batch_size, num_samples, self.config['action_dim'])
        actions = jnp.clip(actions, -1, 1)
        
        return actions

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params, rng=None):
        """Compute the IQL critic loss with KL pessimism."""
        next_v = self.network.select('value')(batch['next_observations'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        
        # Add KL pessimism penalty
        kl_penalty = 0.0
        if self.config['kl_coeff'] > 0 and rng is not None:
            kl_penalty = self.compute_kl_penalty(batch, grad_params, rng)
            critic_loss = critic_loss + self.config['kl_coeff'] * kl_penalty

        return critic_loss, {
            'critic_loss': critic_loss,
            'kl_penalty': kl_penalty,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the behavioral flow-matching actor loss with advantage weighting."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        flow_mse = (pred - vel) ** 2
        
        # Compute advantage weights
        adv_metrics = {}
        if self.config['advantage_weighted'] and self.config['adv_weight_coeff'] > 0:
            # Get Q-values and V-values for buffer actions
            q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'])
            q = jnp.minimum(q1, q2)  # Use min for conservative estimate
            v = self.network.select('value')(batch['observations'])
            
            # Compute advantage: A(s,a) = Q(s,a) - V(s)
            advantage = q - v
            
            # Compute advantage weights: w = exp(β * Adv)
            # β is scaled by inverse of Q magnitude for stability
            beta = self.config['adv_weight_coeff'] / (jnp.abs(q).mean() + 1e-6)
            weights = jnp.exp(beta * advantage)
            
            # Clip weights to prevent extreme values
            weights = jnp.clip(weights, 0.1, 10.0)
            
            # Apply weights to flow loss
            actor_loss = jnp.mean(weights * flow_mse.mean(axis=-1))
            
            # Metrics for monitoring
            adv_metrics.update({
                'advantage_mean': advantage.mean(),
                'advantage_std': advantage.std(),
                'weights_mean': weights.mean(),
                'weights_std': weights.std(),
                'beta_effective': beta,
            })
        else:
            actor_loss = jnp.mean(flow_mse)

        return actor_loss, {
            'actor_loss': actor_loss,
            **adv_metrics,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, value_rng, critic_rng, actor_rng = jax.random.split(rng, 4)

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, rng=critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
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
        """Sample actions from the actor."""
        orig_observations = observations
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        action_seed, noise_seed = jax.random.split(seed)

        # Sample `num_samples` noises and propagate them through the flow.
        actions = jax.random.normal(
            action_seed,
            (
                *observations.shape[:-1],
                self.config['num_samples'],
                self.config['action_dim'],
            ),
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), self.config['num_samples'], axis=0)
        
        # Midpoint method integration
        dt = 1.0 / self.config['flow_steps']
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], self.config['num_samples'], 1), i * dt)
            
            # First evaluation at current point
            v1 = self.network.select('actor_flow')(n_observations, actions, t, is_encoded=True)
            
            # Midpoint evaluation
            t_mid = t + dt / 2.0
            actions_mid = actions + v1 * dt / 2.0
            v_mid = self.network.select('actor_flow')(n_observations, actions_mid, t_mid, is_encoded=True)
            
            # Final step using midpoint velocity
            actions = actions + v_mid * dt
        actions = jnp.clip(actions, -1, 1)

        # Pick the action with the highest Q-value.
        q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=0)
        actions = actions[jnp.argmax(q)]
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
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_flow'] = encoder_module()

        # Define networks.
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        if encoders.get('actor_flow') is not None:
            # Add actor_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_flow_encoder'] = (encoders.get('actor_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='ifql',  # Agent name.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=15,  # Number of flow steps (increased for better accuracy with midpoint method).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            kl_coeff=0.0,  # KL pessimism coefficient.
            kl_num_samples=10,  # Number of samples for KL penalty.
            advantage_weighted=False,  # Whether to use advantage-weighted flow matching.
            adv_weight_coeff=1.0,  # Coefficient for advantage weighting (higher = more emphasis on good actions).
        )
    )
    return config
