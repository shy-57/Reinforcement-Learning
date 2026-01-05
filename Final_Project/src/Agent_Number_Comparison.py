"""
Multi-Agent Reinforcement Learning (PPO) with TorchRL: MAPPO with varying number of agents in Balance Scenario

===============================================================

**Author**: `Matteo Bettini <https://github.com/matteobettini>`_ (modified)

This tutorial demonstrates how to run MAPPO algorithm in the VMAS "balance" scenario
with different numbers of agents: 2, 4, and 6.
"""

######################################################################
# Install dependencies if needed
# Note: This is for Colab, remove if running locally
# !pip3 install torchrl vmas tqdm
######################################################################

# Torch
import torch

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import os

# Disable log-prob aggregation
set_composite_lp_aggregate(False).set()

######################################################################
# Define Hyperparameters
# ----------------------

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device  # The device where the simulator is run (VMAS can run on GPU)

# Sampling
frames_per_batch = 60000  # Number of team frames collected per training iteration
n_iters = 100  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 10  # Number of optimization steps per training iteration
minibatch_size = 4096  # Size of the mini-batches in each optimization step
lr = 1e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.99  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 0.01  # coefficient of the entropy term in the PPO loss

# Disable log-prob aggregation
set_composite_lp_aggregate(False).set()

######################################################################
# Environment settings
# --------------------

max_steps = 100  # Episode steps before done
scenario_name = "balance"

# Define agent configurations to test
AGENT_NUMS = [2, 4, 6]

# To store results
results = {f"MAPPO_{n}_agents": {"episode_reward_mean": []} for n in AGENT_NUMS}
trained_policies = {}  # Store trained policies for evaluation

######################################################################
# Train MAPPO with different numbers of agents

for n_agents in AGENT_NUMS:
    print(f"\n{'='*50}")
    print(f"Training MAPPO with {n_agents} agents")
    print(f"{'='*50}")

    num_vmas_envs = frames_per_batch // max_steps  # Should be divisible

    # Re-create environment for each agent count
    env = VmasEnv(
        scenario=scenario_name,
        num_envs=num_vmas_envs,
        continuous_actions=True,
        max_steps=max_steps,
        device=vmas_device,
        n_agents=n_agents,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    check_env_specs(env)

    # Policy network (decentralized: each agent has own policy, but parameters NOT shared)
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.full_action_spec[env.action_key].shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=False,  # MAPPO: decentralized policy → no parameter sharing
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob=True,
    )

    # Critic network (centralized: uses all agents' observations, shared parameters)
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=True,      # MAPPO: centralized critic
        share_params=True,     # Shared critic across agents
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    # Data collector
    collector = SyncDataCollector(
        env,
        policy,
        device=vmas_device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    # Loss function
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), lr)

    # Training loop
    pbar = tqdm(total=n_iters, desc=f"MAPPO_{n_agents} - episode_reward_mean = 0")

    for tensordict_data in collector:
        # Expand done and terminated
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        # Logging: Use last-step episode rewards
        episode_rewards = tensordict_data.get(("next", "agents", "episode_reward"))  # [num_envs, max_steps, n_agents, 1]
        last_step_rewards = episode_rewards[:, -1, :, :].reshape(-1)

        if last_step_rewards.numel() > 0:
            episode_reward_mean = last_step_rewards.mean().item()
        else:
            episode_reward_mean = 0.0

        results[f"MAPPO_{n_agents}_agents"]["episode_reward_mean"].append(episode_reward_mean)
        pbar.set_description(
            f"MAPPO_{n_agents} - episode_reward_mean = {episode_reward_mean:.2f}",
            refresh=False
        )
        pbar.update()

    print(f"Finished training MAPPO with {n_agents} agents")
    trained_policies[f"MAPPO_{n_agents}_agents"] = policy  # Save trained policy

# Save results
os.makedirs("results", exist_ok=True)
np.save("results/balance_mappo_agent_ablation.npy", results)

######################################################################
# Plot comparison
plt.figure(figsize=(12, 8))
for n_agents in AGENT_NUMS:
    label = f"MAPPO ({n_agents} agents)"
    plt.plot(
        results[f"MAPPO_{n_agents}_agents"]["episode_reward_mean"],
        label=label,
        linewidth=2.5
    )

plt.xlabel("Training Iterations", fontsize=14)
plt.ylabel("Mean Episode Reward", fontsize=14)
plt.title("MAPPO Performance vs Number of Agents in VMAS Balance Scenario", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("mappo_agent_ablation.png", dpi=300)
plt.show()

#######################################################################
# Evaluation: Test each trained policy 10 times in a fresh environment (manual reward sum)
print("\n" + "="*60)
print("EVALUATION RESULTS (10 episodes per agent configuration)")
print("="*60)
eval_episodes = 10
eval_max_steps = max_steps

for n_agents in AGENT_NUMS:
    algo_key = f"MAPPO_{n_agents}_agents"
    policy = trained_policies[algo_key]
    
    # Create a clean evaluation environment WITHOUT RewardSum
    eval_env = VmasEnv(
        scenario=scenario_name,
        num_envs=1,
        continuous_actions=True,
        max_steps=eval_max_steps,
        device=vmas_device,
        n_agents=n_agents,
    )
    
    episode_rewards = []
    for episode_idx in range(eval_episodes):
        td = eval_env.reset()
        total_reward_per_agent = torch.zeros(n_agents, device=device)
        
        for step in range(eval_max_steps):
            with torch.no_grad():
                td = policy(td)
            td = eval_env.step(td)
            
            # Accumulate rewards manually
            step_rewards = td[("next", "agents", "reward")].squeeze(0)  # Shape: [n_agents, 1]
            total_reward_per_agent += step_rewards.squeeze(-1)
            
            if td[("next", "done")].all():
                break
            td = td["next"]  # Roll to next step
        
        final_reward = total_reward_per_agent.mean().item()
        episode_rewards.append(final_reward)
        print(f"\nEpisode {episode_idx+1} (Final reward: {final_reward:.2f})")
        print("Per-agent final rewards:", total_reward_per_agent.cpu().numpy())
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\n{algo_key} (10 episodes): Mean = {mean_reward:.2f}, Std = {std_reward:.2f}")
    print("-" * 60)

print("="*60)