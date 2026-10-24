"""
Part D: Hyperparameter comparison for DQN on LunarLander-v3.

Output: batch_comparison.png, learning_rate_comparison.png,
        epsilon_decay_comparison.png
"""

import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.992
BATCH_SIZE = 128
BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 500


def moving_average(data, window: int = 20) -> np.ndarray:
    """
    Compute a simple moving average using a uniform kernel.

    Parameters
    ----------
    data : array-like
        1-D sequence of values (e.g. per-episode rewards or losses).
    window : int
        Number of elements to average over.

    Returns
    -------
    np.ndarray
        Array of length ``len(data) - window + 1`` containing the smoothed
        values. The first element is the mean of ``data[0:window]``.
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


class QNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim=8,
        action_dim=4,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_capacity=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.q_network = self.q_net
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            expected_state_action_values = (
                rewards + (1 - dones) * self.gamma * next_q_values
            )

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item(), q_values.mean().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_experiment(config_name, num_episodes=500, **kwargs):
    print(f"test start [{config_name}] ({num_episodes} Episodes)...", flush=True)
    env = gym.make("LunarLander-v3")
    agent = DQNAgent(state_dim=8, action_dim=4, **kwargs)
    rewards = []

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, float(done))
            agent.update()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)

        if episode % 100 == 0:
            avg_r = np.mean(rewards[-50:])
            print(
                f"  [{config_name}] Ep {episode}/{num_episodes} | 50-Ep Avg Reward: {avg_r:.2f}",
                flush=True,
            )

    env.close()
    return rewards


def main():
    baseline_params = {
        "lr": 1e-4,
        "batch_size": 64,
        "epsilon_decay": 0.992,
    }
    comparisons = {
        "Batch Size": {
            "filename": "batch_comparison.png",
            "variants": {
                "Small Batch (32)": {"batch_size": 32},
                "Large Batch (128)": {"batch_size": 128},
            },
        },
        "Learning Rate": {
            "filename": "learning_rate_comparison.png",
            "variants": {
                "Low LR (1e-5)": {"lr": 1e-5},
                "High LR (1e-3)": {"lr": 1e-3},
            },
        },
        "Epsilon Decay": {
            "filename": "epsilon_decay_comparison.png",
            "variants": {
                "Low Epsilon Decay (0.990)": {"epsilon_decay": 0.99},
                "High Epsilon Decay (0.995)": {"epsilon_decay": 0.995},
            },
        },
    }

    baseline_rewards = train_experiment(
        "Baseline (lr=1e-4, batch=64, epsilon_decay=0.992)",
        num_episodes=500,
        **baseline_params,
    )

    window = 30
    for parameter, comparison in comparisons.items():
        results = {"Baseline": baseline_rewards}
        for name, overrides in comparison["variants"].items():
            params = baseline_params.copy()
            params.update(overrides)
            results[name] = train_experiment(
                f"{parameter}: {name}", num_episodes=500, **params
            )

        plt.figure(figsize=(10, 6))
        for name, rewards in results.items():
            smoothed = moving_average(rewards, window)
            plt.plot(range(window - 1, len(rewards)), smoothed, label=name, linewidth=2)

        plt.axhline(y=200, color="r", linestyle="--", label="Solved Threshold (200)")
        plt.title(f"{parameter} Comparison on LunarLander-v3")
        plt.xlabel("Episode")
        plt.ylabel("Reward (30-Episode Moving Average)")
        plt.legend()
        plt.grid(True)
        plt.savefig(comparison["filename"], dpi=300)
        plt.close()
        print(f"Saved: {comparison['filename']}")


if __name__ == "__main__":
    main()
