"""
Part C: Evaluate the trained DQN agent (epsilon = 0) and record GIFs.

Requires dqn_lunar_lander.pt produced by PartB.py.
"""

import os
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


def print_stats(stats: dict) -> None:
    """
    Print a formatted summary of episode statistics to the console.

    The *stats* dictionary is expected to contain at least:
        mean_reward, std_reward, min_reward, max_reward,
        mean_length, success_rate

    Expected output format (example)::

        ----------------------------------------
        Random Policy Statistics
        ----------------------------------------
          Mean reward  :   -123.45 ± 80.23
          Min / Max    :   -456.78 / 12.34
          Mean length  :    250.0 steps
          Success rate :      0.0%
        ----------------------------------------

    Hints
    -----
    - Use f-strings with format specs such as ``:.2f`` and ``:.1f``.
    - The success rate in *stats* is stored as a fraction (0–1); multiply by 100
      to display as a percentage.
    """
    sep = "-" * 40
    print(sep)
    print("Episode Statistics")
    print(sep)
    print(f"  Mean reward  : {stats['mean_reward']:>8.2f} ± {stats['std_reward']:.2f}")
    print(f"  Min / Max    : {stats['min_reward']:>8.2f} / {stats['max_reward']:.2f}")
    print(f"  Mean length  : {stats['mean_length']:>8.1f} steps")
    print(f"  Success rate : {stats['success_rate'] * 100:>7.1f}%")
    print(sep)


def record_episodes(num_episodes: int, out_dir: str, policy_fn) -> None:
    """
    Run `num_episodes` using `policy_fn` and save each episode as a GIF.

    Parameters
    ----------
    num_episodes : int
        Number of episodes to record.
    out_dir : str
        Directory in which to save the GIF files.
    policy_fn : callable
        A function that accepts a state (np.ndarray) and returns an integer action.
        For the random baseline use: ``policy_fn=lambda s: env.action_space.sample()``

    Hints
    -----
    - Use ``gym.make('LunarLander-v3', render_mode='rgb_array')`` so that
      ``env.render()`` returns an RGB frame (numpy array).
    - Collect frames in a list during the episode loop, then write them with
      ``imageio.mimsave(path, frames, fps=30)``.
    - Name each file ``episode_{ep+1}.gif`` inside *out_dir*.
    """
    import imageio

    os.makedirs(out_dir, exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    for ep in range(num_episodes):
        state, _ = env.reset()
        frames = []
        done = False
        total_reward = 0.0

        while not done:
            frames.append(env.render())
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        gif_path = os.path.join(out_dir, f"episode_{ep + 1}.gif")
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"  Episode {ep + 1:2d}  reward = {total_reward:7.1f}  → {gif_path}")

    env.close()


def load_checkpoint(agent, filename: str):
    """
    Load a checkpoint saved by ``save_checkpoint`` into *agent*.

    Returns
    -------
    (episode, rewards_history) : (int, list)
    """
    checkpoint = torch.load(filename)
    agent.q_network.load_state_dict(checkpoint["model_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["episode"], checkpoint["rewards_history"]


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


def save_stats_table(stats: dict, successes: int, total: int, output_path: str) -> None:
    """Save the 'Episode Statistics' table (same text as print_stats) as an image."""
    sep = "-" * 40
    text = "\n".join(
        [
            sep,
            "Episode Statistics",
            sep,
            f"  Mean reward  : {stats['mean_reward']:>8.2f} ± {stats['std_reward']:.2f}",
            f"  Min / Max    : {stats['min_reward']:>8.2f} / {stats['max_reward']:.2f}",
            f"  Mean length  : {stats['mean_length']:>8.1f} steps",
            f"  Success rate : {stats['success_rate'] * 100:>7.1f}%",
            sep,
            f"Successful episodes: {successes}/{total}",
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.axis("off")
    ax.text(
        0.02,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=13,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    eval_env = gym.make("LunarLander-v3")

    eval_agent = DQNAgent(state_dim=8, action_dim=4)
    load_checkpoint(eval_agent, "dqn_lunar_lander.pt")
    eval_agent.q_net.eval()
    eval_agent.epsilon = 0.0

    eval_episodes = 100
    eval_rewards = []
    eval_lengths = []
    eval_successes = 0

    print(
        f"Part C after Training ({eval_episodes} Episodes, Epsilon=0)...",
        flush=True,
    )

    for episode in range(eval_episodes):
        state, info = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = eval_agent.select_action(state)
            next_state, reward, terminated, truncated, info = eval_env.step(action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            state = next_state

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)

        if episode_reward >= 200:
            eval_successes += 1

    eval_env.close()

    eval_stats = {
        "episode_rewards": eval_rewards,
        "episode_lengths": eval_lengths,
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "min_reward": float(np.min(eval_rewards)),
        "max_reward": float(np.max(eval_rewards)),
        "mean_length": float(np.mean(eval_lengths)),
        "success_rate": eval_successes / eval_episodes,  # fraction (0-1)
    }

    print("Part C results", flush=True)
    print_stats(eval_stats)
    print(f"Successful episodes: {eval_successes}/{eval_episodes}")

    table_path = "./outputs/part_b_c/episode_statistics.png"
    save_stats_table(eval_stats, eval_successes, eval_episodes, table_path)
    print(f"Statistics table saved to {table_path}", flush=True)

    # Record 5 episodes of the trained agent as GIFs (epsilon = 0)
    record_episodes(5, "./videos_trained", policy_fn=eval_agent.select_action)
    print("record successful and saved in ./videos_trained/", flush=True)


if __name__ == "__main__":
    main()
