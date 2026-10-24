"""
Part B: DQN agent (Q-network, replay buffer, agent) + training on LunarLander-v3.

Outputs:
    dqn_lunar_lander.pt                    (final checkpoint, used by PartC.py)
    checkpoints/dqn_ep{N}.pt               (periodic checkpoints)
    checkpoints/dqn_best.pt                (best 100-episode average so far)
    outputs/part_b_c/training_curves.png
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

# ==========================================
# Hyperparameters
# ==========================================
LEARNING_RATE = 1e-4
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.992
BATCH_SIZE = 128
BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 500
NUM_EPISODES = 500
CHECKPOINT_EVERY = 100  # 此處修改為 100
CHECKPOINT_DIR = "checkpoints"
FINAL_MODEL_PATH = "dqn_lunar_lander.pt"


# ==========================================
# Helper functions
# ==========================================
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
        values.  The first element is the mean of ``data[0:window]``.

    Hints
    -----
    - ``np.convolve(data, kernel, mode='valid')`` with a uniform kernel of
      length *window* is a one-liner solution.
    - Remember to normalise the kernel so its values sum to 1.
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_training_curves(metrics: dict, out_dir: str = "outputs/part_b_c") -> None:
    """
    Generate a 2×2 figure of DQN training diagnostics and save it.

    The four sub-plots should be:
        [0,0] Episode reward + moving average + solved threshold line
        [0,1] Training loss (Huber) + moving average  (skip NaN episodes)
        [1,0] Epsilon decay over episodes
        [1,1] Mean max Q-value + moving average       (skip NaN episodes)

    Parameters
    ----------
    metrics : dict
        Dictionary returned by the ``train()`` function; must contain
        ``episode_rewards``, ``avg_losses``, ``epsilons``, ``mean_q_values``,
        and ``solved_at`` (int or None).
    out_dir : str
        Directory in which to save ``training_curves.png``.

    Hints
    -----
    - Use your ``moving_average()`` helper for smoothing.
    - Loss and Q-value lists may contain ``float('nan')`` for early episodes
      before the buffer is warm; filter these out before plotting.
    - If ``solved_at`` is not None, draw a vertical dashed line on the reward
      sub-plot to mark the episode where the environment was solved.
    - Save to ``os.path.join(out_dir, 'training_curves.png')``.
    """
    SOLVED_THRESHOLD = 200.0
    window = 20
    rewards = metrics["episode_rewards"]
    losses = metrics["avg_losses"]
    epsilons = metrics["epsilons"]
    q_values = metrics["mean_q_values"]
    solved_at = metrics.get("solved_at")
    N = len(rewards)
    episodes = np.arange(1, N + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("DQN Training – LunarLander-v3", fontsize=14)

    # [0,0] Reward curve
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="per episode")
    if len(rewards) >= window:
        ma = moving_average(rewards, window)
        ax.plot(
            episodes[window - 1 :],
            ma,
            color="steelblue",
            linewidth=2,
            label=f"MA-{window}",
        )
    ax.axhline(
        SOLVED_THRESHOLD,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Solved ({SOLVED_THRESHOLD})",
    )
    if solved_at:
        ax.axvline(
            solved_at,
            color="red",
            linestyle=":",
            alpha=0.8,
            label=f"Solved @ ep {solved_at}",
        )
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,1] Loss curve (filter NaNs)
    ax = axes[0, 1]
    valid_loss = [(i + 1, l) for i, l in enumerate(losses) if not np.isnan(l)]
    if valid_loss:
        ep_l, l_vals = zip(*valid_loss)
        ax.plot(ep_l, l_vals, alpha=0.3, color="darkorange", label="per episode")
        if len(l_vals) >= window:
            ma_l = moving_average(l_vals, window)
            ax.plot(
                list(ep_l)[window - 1 :],
                ma_l,
                color="darkorange",
                linewidth=2,
                label=f"MA-{window}",
            )
    ax.set_title("Training Loss (Huber)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,0] Epsilon decay
    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color="purple", linewidth=2)
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε (exploration prob.)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # [1,1] Mean Q-values (filter NaNs)
    ax = axes[1, 1]
    valid_q = [(i + 1, q) for i, q in enumerate(q_values) if not np.isnan(q)]
    if valid_q:
        ep_q, q_vals = zip(*valid_q)
        ax.plot(ep_q, q_vals, alpha=0.3, color="crimson", label="per episode")
        if len(q_vals) >= window:
            ma_q = moving_average(q_vals, window)
            ax.plot(
                list(ep_q)[window - 1 :],
                ma_q,
                color="crimson",
                linewidth=2,
                label=f"MA-{window}",
            )
    ax.set_title("Mean Max Q-Value")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q̄")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    print(f"Saved training curves → {path}")
    plt.close()


def save_checkpoint(agent, episode: int, rewards: list, filename: str) -> None:
    """Save agent weights, optimiser state, and reward history to a .pt file."""
    torch.save(
        {
            "episode": episode,
            "model_state_dict": agent.q_network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "rewards_history": rewards,
        },
        filename,
    )


# ==========================================
# Part B: agent classes
# ==========================================
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

        # alias so the save_checkpoint / load_checkpoint helpers (agent.q_network) work
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


def train(env, agent, num_episodes=NUM_EPISODES, checkpoint_every=CHECKPOINT_EVERY):
    """Training loop with episode management, metric logging and checkpointing.

    Returns a metrics dict in the format expected by plot_training_curves().
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    episode_rewards = []
    episode_losses = []
    episode_epsilons = []
    episode_mean_q = []
    solved_at = None  # first episode where the 100-ep average reaches 200
    best_avg = -np.inf

    print("Training Start...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        losses = []
        q_vals = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, float(done))

            loss, mean_q = agent.update()
            if loss is not None:
                losses.append(loss)
                q_vals.append(mean_q)

            state = next_state
            episode_reward += reward

        agent.decay_epsilon()

        episode_rewards.append(episode_reward)
        episode_epsilons.append(agent.epsilon)
        # NaN when the buffer is not warm yet (plot_training_curves skips NaNs)
        episode_losses.append(np.mean(losses) if losses else float("nan"))
        episode_mean_q.append(np.mean(q_vals) if q_vals else float("nan"))

        recent_avg_reward = np.mean(episode_rewards[-100:])
        if (
            solved_at is None
            and len(episode_rewards) >= 100
            and recent_avg_reward >= 200
        ):
            solved_at = episode

        history = [float(r) for r in episode_rewards]

        # periodic checkpoint
        if episode % checkpoint_every == 0:
            path = os.path.join(CHECKPOINT_DIR, f"dqn_ep{episode}.pt")
            save_checkpoint(agent, episode, history, path)
            print(f"  Checkpoint saved -> {path}")

        # best-model checkpoint (needs a full 100-episode window)
        if len(episode_rewards) >= 100 and recent_avg_reward > best_avg:
            best_avg = recent_avg_reward
            save_checkpoint(
                agent, episode, history, os.path.join(CHECKPOINT_DIR, "dqn_best.pt")
            )

        if episode % 20 == 0:
            print(
                f"Episode {episode:3d}/{num_episodes} | Reward: {episode_reward:7.2f} |"
                f" 100-Ep Avg: {recent_avg_reward:7.2f} | Epsilon: {agent.epsilon:.3f}"
            )

    return {
        "episode_rewards": episode_rewards,
        "avg_losses": episode_losses,
        "epsilons": episode_epsilons,
        "mean_q_values": episode_mean_q,
        "solved_at": solved_at,
    }


def main():
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    metrics = train(env, agent, num_episodes=NUM_EPISODES)
    env.close()

    save_checkpoint(
        agent,
        NUM_EPISODES,
        [float(r) for r in metrics["episode_rewards"]],
        FINAL_MODEL_PATH,
    )
    print(f"Model saved to {FINAL_MODEL_PATH}")

    plot_training_curves(metrics)  # saves outputs/part_b_c/training_curves.png


if __name__ == "__main__":
    main()
