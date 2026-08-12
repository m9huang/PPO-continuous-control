
# Proximal Policy Optimization (PPO) for continuous control.

import gymnasium as gym
import numpy as np
import os
import time
import torch
from copy import deepcopy
import imageio.v2 as imageio
from ppo_agent_opt3 import PPOAgent
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Create the training environment.
scenario = "Pendulum-v1"
env = gym.make(id=scenario)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# Create output directories.
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
ppo_reward = current_path + '/reward/'
gif_dir = current_path + '/gifs/'
figure_dir = current_path + '/figures/'
os.makedirs(model, exist_ok=True)
os.makedirs(ppo_reward, exist_ok=True)
os.makedirs(gif_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d%H%M%S")

# Training hyperparameters
NUM_EPISODE = 1000
NUM_STEP = 200
UPDATE_INTERVAL = 200
BATCH_SIZE = 50

# GIF recording uses a separate environment and a fixed evaluation seed.
GIF_FPS = 30
GIF_SEED = 123
GIF_EPISODES = {300, 600}

# Create the PPO agent.
agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)


def record_policy_gif(agent, output_path, seed=GIF_SEED, fps=GIF_FPS):
    """Record a deterministic evaluation episode as a GIF."""
    eval_env = gym.make(id=scenario, render_mode="rgb_array")
    frames = []
    actor_was_training = agent.actor.training

    try:
        state, _ = eval_env.reset(seed=seed)
        frames.append(eval_env.render())
        action_high = torch.as_tensor(
            eval_env.action_space.high,
            dtype=torch.float32,
            device=next(agent.actor.parameters()).device,
        )

        # Use the mean action for deterministic evaluation without gradients.
        agent.actor.eval()
        with torch.no_grad():
            while True:
                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=action_high.device,
                ).unsqueeze(0)
                action_mean, _ = agent.actor(state_tensor)
                action = (torch.tanh(action_mean) * action_high).cpu().numpy()[0]

                state, _, terminated, truncated, _ = eval_env.step(action)
                frames.append(eval_env.render())

                if terminated or truncated:
                    break
    finally:
        agent.actor.train(actor_was_training)
        eval_env.close()

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Saved GIF: {output_path}")


# Record the untrained policy in a separate environment.
record_policy_gif(agent, os.path.join(gif_dir, "before_training.gif"))

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
best_reward = -2000
best_average_reward = -2000
best_episode_reward = -np.inf
best_actor_state = None

for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    terminated = False
    episode_reward = 0

    for step_i in range(NUM_STEP):
        action, log_prob = agent.get_action(state)
        value = agent.get_value(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        agent.replay_buffer.add_memo(state, action, reward, value, terminated, log_prob)

        state = next_state

        if terminated or truncated:
            break

    # Track the actor that generated the highest-reward rollout before updating.
    if episode_reward > best_episode_reward:
        best_episode_reward = episode_reward
        best_actor_state = deepcopy(agent.actor.state_dict())

    last_value = agent.get_value(state)

    agent.update(last_value)

    # Capture snapshots after 300 and 600 training episodes.
    episode_number = episode_i + 1
    if episode_number in GIF_EPISODES:
        record_policy_gif(
            agent,
            os.path.join(gif_dir, f"episode_{episode_number}.gif"),
        )

    REWARD_BUFFER[episode_i] = episode_reward
    Average_reward = np.mean(REWARD_BUFFER[max(0, episode_i - 100):(episode_i + 1)])
    if Average_reward >= -270 and Average_reward > best_average_reward:
        best_average_reward = Average_reward
        torch.save(agent.actor.state_dict(), model + f'ppo_actor_{timestamp}_ar_{round(best_average_reward)}.pth')
    print(f"Episode: {episode_i}, Reward: {round(episode_reward, 2)}, Average Reward: {round(Average_reward, 2)}")

torch.save(REWARD_BUFFER, ppo_reward + f'/ppo_reward_{timestamp}.pt')
env.close()

# Record the policy from the highest-reward rollout.
if best_actor_state is not None:
    final_actor_state = deepcopy(agent.actor.state_dict())
    try:
        agent.actor.load_state_dict(best_actor_state)
        record_policy_gif(agent, os.path.join(gif_dir, "best_policy.gif"))
    finally:
        agent.actor.load_state_dict(final_actor_state)

plt.figure(figsize=(10, 6))
steps = np.arange(NUM_EPISODE) * NUM_STEP
plt.plot(steps, REWARD_BUFFER, color='purple', alpha=0.5, label='Reward per Episode')
plt.plot(steps, gaussian_filter1d(REWARD_BUFFER, sigma=5), color='purple', linewidth=2, label='Smoothed Reward')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('PPO Training Curve')
plt.legend()
plt.savefig(figure_dir + f"Rewards-{scenario}-{timestamp}.png", format='png', dpi=300)
plt.show()
