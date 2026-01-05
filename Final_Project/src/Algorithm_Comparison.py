"""
Multi-Agent Reinforcement Learning (PPO) with TorchRL: CPPO vs MAPPO vs IPPO in Balance Scenario

===============================================================

**Author**: `Matteo Bettini <https://github.com/matteobettini>`_

This tutorial demonstrates how to compare CPPO, MAPPO, and IPPO algorithms in the VMAS "balance" scenario.

Key differences:

- CPPO (Centralized Policy PPO): Centralized training with centralized policy (all agents share the same policy)
- MAPPO (Multi-Agent PPO): Centralized training with decentralized policy (each agent has its own policy, but uses centralized critic)
- IPPO (Independent PPO): Decentralized training and execution (each agent has its own policy and critic)

In the "balance" scenario, agents stand on a plank that is suspended by a spring.
They must move to balance the plank horizontally while avoiding falling off.
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
# Environment
# -----------

max_steps = 100  # Episode steps before done
num_vmas_envs = frames_per_batch // max_steps  # Should be divisible
scenario_name = "balance"
n_agents = 4  # Keep 4 agents as in original

######################################################################
# Define the three algorithms to compare

CPPO_CONFIG = {
    "name": "CPPO",
    "share_parameters_policy": True,
    "mappo": True,
    "share_parameters_critic": True
}

MAPPO_CONFIG = {
    "name": "MAPPO",
    "share_parameters_policy": False,
    "mappo": True,
    "share_parameters_critic": True
}

IPPO_CONFIG = {
    "name": "IPPO",
    "share_parameters_policy": True,
    "mappo": False,
    "share_parameters_critic": False
}

ALGORITHMS = [IPPO_CONFIG, CPPO_CONFIG, MAPPO_CONFIG]

# To store results
results = {config["name"]: {"episode_reward_mean": []} for config in ALGORITHMS}
trained_policies = {}  # Store trained policies for evaluation

######################################################################
# Train each algorithm and collect results

for config in ALGORITHMS:
    print(f"\n{'='*50}")
    print(f"Training {config['name']} algorithm")
    print(f"{'='*50}")

    # Re-create environment for each algorithm
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

    # Policy network
    share_parameters_policy = config["share_parameters_policy"]
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.full_action_spec[env.action_key].shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=share_parameters_policy,
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

    # Critic network
    share_parameters_critic = config["share_parameters_critic"]
    mappo = config["mappo"]
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=mappo,
        share_params=share_parameters_critic,
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
    pbar = tqdm(total=n_iters, desc=f"{config['name']} - episode_reward_mean = 0")

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

        # Logging: Use last-step episode rewards to avoid sticky-done duplication
        batch_size = tensordict_data.batch_size[0]  # num_envs = 600
        episode_rewards = tensordict_data.get(("next", "agents", "episode_reward"))  # [600, 100, 4, 1]

        # Take the reward at the last time step for each environment and agent
        # Shape: [600, 4, 1] → flatten to [600 * 4]
        last_step_rewards = episode_rewards[:, -1, :, :].reshape(-1)

        print(f"\n[DEBUG] {config['name']} - Iteration {pbar.n + 1}")
        print(f"  - tensordict batch shape: {tensordict_data.batch_size}")
        print(f"  - episode_reward shape: {episode_rewards.shape}")
        print(f"  - Using last-step rewards only: {last_step_rewards.numel()} episodes")

        if last_step_rewards.numel() > 0:
            episode_reward_mean = last_step_rewards.mean().item()
            print(f"  - Last-step episode rewards (first 10): {last_step_rewards[:10].cpu().numpy()}")
            print(f"  - Mean episode reward: {episode_reward_mean:.4f}")
        else:
            episode_reward_mean = 0.0
            print("  - No data in last step!")

        results[config["name"]]["episode_reward_mean"].append(episode_reward_mean)
        pbar.set_description(
            f"{config['name']} - episode_reward_mean = {episode_reward_mean:.2f}",
            refresh=False
        )
        pbar.update()

    print(f"Finished training {config['name']} algorithm")
    trained_policies[config["name"]] = policy  # Save trained policy

# Save results
os.makedirs("results", exist_ok=True)
np.save("results/balance_results.npy", results)

######################################################################
# Plot comparison
plt.figure(figsize=(12, 8))
for config in ALGORITHMS:
    plt.plot(
        results[config["name"]]["episode_reward_mean"],
        label=config["name"],
        linewidth=2.5
    )

plt.xlabel("Training Iterations", fontsize=14)
plt.ylabel("Mean Episode Reward", fontsize=14)
plt.title("CPPO vs MAPPO vs IPPO in VMAS Balance Scenario", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("balance_comparison.png", dpi=300)
plt.show()

######################################################################
# Evaluation: Test each trained policy 10 times in a fresh environment
print("\n" + "="*60)
print("EVALUATION RESULTS (10 episodes per algorithm)")
print("="*60)

eval_episodes = 10
eval_max_steps = max_steps
n_agents_eval = n_agents  # = 4, as used in training

for config in ALGORITHMS:
    algo_name = config["name"]
    print(f"\nEvaluating {algo_name}...")
    
    policy = trained_policies[algo_name]

    # Create a clean evaluation environment (no RewardSum, manual reward sum)
    eval_env = VmasEnv(
        scenario=scenario_name,
        num_envs=1,
        continuous_actions=True,
        max_steps=eval_max_steps,
        device=vmas_device,
        n_agents=n_agents_eval,
    )

    episode_rewards = []

    for episode_idx in range(eval_episodes):
        td = eval_env.reset()
        total_reward_per_agent = torch.zeros(n_agents_eval, device=device)

        for step in range(eval_max_steps):
            with torch.no_grad():
                td = policy(td)
            td = eval_env.step(td)

            # Accumulate rewards manually
            step_rewards = td[("next", "agents", "reward")].squeeze(0)  # [n_agents, 1]
            total_reward_per_agent += step_rewards.squeeze(-1)

            if td[("next", "done")].all():
                break

            td = td["next"]  # Roll to next step

        final_reward = total_reward_per_agent.mean().item()
        episode_rewards.append(final_reward)

        print(f"  Episode {episode_idx+1}: Total reward = {final_reward:.2f}")
        print(f"    Per-agent: {total_reward_per_agent.cpu().numpy()}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\n✅ {algo_name} (10 episodes): Mean = {mean_reward:.2f}, Std = {std_reward:.2f}")
    print("-" * 50)

print("="*60)