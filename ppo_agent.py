# PPO agent with actor, behavior-policy, and critic networks.

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.optim as optim

device = torch.device("cpu")
print("Computing device: ", device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.clamp(self.fc_std(x), -5, 2)
        std = torch.exp(log_std)

        return mean, std

    def select_action(self, s):
        mu, sigma = self.forward(s)
        normal_dist = Normal(mu, sigma)
        raw_action = normal_dist.sample()
        action = 2.0 * torch.tanh(raw_action)  # Scale actions to [-2, 2].
        log_prob = normal_dist.log_prob(raw_action)
        log_prob -= torch.log(2.0 * (1.0 - torch.tanh(raw_action).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


def squashed_log_prob(dist, action):
    y = action / 2.0
    y = torch.clamp(y, -1 + 1e-6, 1 - 1e-6)

    u = torch.atanh(y)

    log_prob = (
        dist.log_prob(u)
        - torch.log(2.0 * (1.0 - y.pow(2)) + 1e-6)
    )

    return log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)

        return value


class ReplayMemory:
    def __init__(self, batch_size):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.log_prob_cap = []
        self.BATCH_SIZE = batch_size

    def add_memo(self, state, action, reward, value, done, log_prob):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)
        self.log_prob_cap.append(log_prob)

    def sample(self):
        return np.array(self.state_cap), \
            np.array(self.action_cap), \
            np.array(self.reward_cap), \
            np.array(self.value_cap), \
            np.array(self.done_cap), \
            np.array(self.log_prob_cap)

    def shuffle_indices(self):
        num_state = len(self.state_cap)
        batch_start_points = np.arange(0, num_state, self.BATCH_SIZE)
        memory_indicies = np.arange(num_state, dtype=np.int32)
        np.random.shuffle(memory_indicies)
        batches = [memory_indicies[i:i + self.BATCH_SIZE] for i in batch_start_points if i + self.BATCH_SIZE <= num_state]
        # Shuffle memory indices and split them into full batches.

        return batches


    def clear_memo(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.log_prob_cap = []


class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size):
        self.LR_ACTOR = 2e-4
        self.LR_CRITIC = 3e-4

        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.EPOCH = 5
        self.EPSILON_CLIP = 0.2

        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        self.replay_buffer = ReplayMemory(batch_size)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():  # Action selection does not require gradients.
            action, log_prob = self.actor.select_action(state)
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def get_value(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            value = self.critic.forward(state)
        return value.cpu().numpy()[0]

    def update(self, last_value):
        self.old_actor.load_state_dict(self.actor.state_dict())  # Synchronize behavior-policy parameters.
        
        memo_states, memo_actions, memo_rewards, memo_values, memo_dones, memo_log_probs = self.replay_buffer.sample()

        T = len(memo_rewards)

        values_ext = np.concatenate(
            [memo_values.squeeze(), np.array(last_value).reshape(-1)]
        )

        memo_advantage = np.zeros(T, dtype=np.float32)
        next_advantage = 0.0

        for t in reversed(range(T)):
            delta = (
                memo_rewards[t]
                + self.GAMMA * values_ext[t + 1] * (1 - int(memo_dones[t]))
                - values_ext[t]
            )

            memo_advantage[t] = (
                delta
                + self.GAMMA * self.LAMBDA
                * (1 - int(memo_dones[t]))
                * next_advantage
            )

            next_advantage = memo_advantage[t]

        with torch.no_grad():
            memo_advantages_tensor = torch.tensor(memo_advantage).unsqueeze(1).to(device)
            normalized_memo_advantages = (
                (memo_advantages_tensor - memo_advantages_tensor.mean())
                / (memo_advantages_tensor.std() + 1e-8)
            )
            memo_values_tensor = torch.tensor(memo_values).to(device)

        memo_states_tensor = torch.FloatTensor(memo_states).to(device)
        memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)
        memo_log_probs_tensor = torch.FloatTensor(memo_log_probs).to(device)

        for _ in range(self.EPOCH):

            batches = self.replay_buffer.shuffle_indices()

            for batch in batches:
                batch_old_log_probs_tensor = memo_log_probs_tensor[batch]

                mu, sigma = self.actor(memo_states_tensor[batch])
                pi = Normal(mu, sigma)
                batch_log_probs_tensor = squashed_log_prob(pi, memo_actions_tensor[batch])

                ratio = torch.exp(batch_log_probs_tensor - batch_old_log_probs_tensor)  # Probability ratio: pi(a|s) / old_pi(a|s).
                surr1 = ratio * normalized_memo_advantages[batch]  # Weight normalized advantages by the ratio.
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * normalized_memo_advantages[batch]
                # Clip the ratio to limit policy changes.

                entropy = pi.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean() - 0.001 * entropy  # Encourage exploration with entropy regularization.
                # Clipping constrains updates that would move the policy too far.

                batch_returns = memo_advantages_tensor[batch] + memo_values_tensor[batch]
                batch_prediction = self.critic(memo_states_tensor[batch])
                critic_loss = nn.MSELoss()(batch_prediction, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.replay_buffer.clear_memo()

    def save_policy(self):
        torch.save(self.actor.state_dict(), 'ppo_policy_pendulum_v1.para')
