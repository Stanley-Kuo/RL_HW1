"""
Part A: Random-policy baseline on LunarLander-v3.
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


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


def plot_baseline(
    stats: dict, out_path: str = "outputs/part_a/baseline_stats.png"
) -> None:
    """
    Generate a 2×2 figure summarising the random-policy baseline and save it.

    The four sub-plots should be:
        [0,0] Episode reward over time (line plot + mean horizontal line)
        [0,1] Reward distribution (histogram + mean vertical line)
        [1,0] Episode length over time (line plot + mean horizontal line)
        [1,1] Text summary box (mean, std, min, max, mean length, success rate)

    Parameters
    ----------
    stats : dict
        Dictionary returned by ``run_random_baseline()``; must contain
        ``episode_rewards``, ``episode_lengths``, ``mean_reward``,
        ``std_reward``, ``min_reward``, ``max_reward``, ``mean_length``,
        ``success_rate``.
    out_path : str
        File path (PNG) where the figure is saved.

    Hints
    -----
    - Use ``plt.subplots(2, 2, figsize=(12, 8))``.
    - For the text panel use ``axes[1,1].axis('off')`` then
      ``axes[1,1].text(...)``.  A monospace font and a ``bbox`` with
      ``boxstyle='round'`` look clean.
    - Call ``plt.tight_layout()`` before saving.
    - Use ``plt.close()`` after saving to free memory.
    """
    rewards = stats["episode_rewards"]
    lengths = stats["episode_lengths"]
    episodes = range(1, len(rewards) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Part A – Random Policy Baseline", fontsize=14)

    # [0,0] Reward over time
    axes[0, 0].plot(episodes, rewards, alpha=0.6, color="steelblue")
    axes[0, 0].axhline(
        stats["mean_reward"],
        color="red",
        linestyle="--",
        label=f"Mean = {stats['mean_reward']:.1f}",
    )
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # [0,1] Reward distribution
    axes[0, 1].hist(rewards, bins=20, color="steelblue", edgecolor="white")
    axes[0, 1].axvline(
        stats["mean_reward"],
        color="red",
        linestyle="--",
        label=f"Mean = {stats['mean_reward']:.1f}",
    )
    axes[0, 1].set_title("Reward Distribution")
    axes[0, 1].set_xlabel("Total Reward")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # [1,0] Episode lengths
    axes[1, 0].plot(episodes, lengths, alpha=0.6, color="darkorange")
    axes[1, 0].axhline(
        stats["mean_length"],
        color="red",
        linestyle="--",
        label=f"Mean = {stats['mean_length']:.1f}",
    )
    axes[1, 0].set_title("Episode Length")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # [1,1] Summary text box
    summary = (
        f"Random Policy Statistics\n"
        f"{'─' * 30}\n"
        f"  Mean reward  : {stats['mean_reward']:>8.2f}\n"
        f"  Std  reward  : {stats['std_reward']:>8.2f}\n"
        f"  Min  reward  : {stats['min_reward']:>8.2f}\n"
        f"  Max  reward  : {stats['max_reward']:>8.2f}\n"
        f"  Mean length  : {stats['mean_length']:>8.1f} steps\n"
        f"  Success rate : {stats['success_rate'] * 100:>7.1f}%"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.95,
        summary,
        transform=axes[1, 1].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved baseline plot → {out_path}")
    plt.close()


def run_part_a_baseline(num_episodes=100, gif_folder="./videos_random"):
    env = gym.make("LunarLander-v3")

    rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_reward >= 200:
            successes += 1

        if (episode + 1) % 20 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}"
            )

    env.close()

    stats = {
        "episode_rewards": rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": successes / num_episodes,
    }

    print_stats(stats)
    plot_baseline(stats)

    sample_env = gym.make("LunarLander-v3")
    record_episodes(5, gif_folder, policy_fn=lambda s: sample_env.action_space.sample())
    sample_env.close()

    return stats


if __name__ == "__main__":
    raw_env = gym.make("LunarLander-v3")
    raw_env.reset()
    print(f"env successfully created, gymnasium version: {gym.__version__}")
    print(f"Observation Space dimensions: {raw_env.observation_space.shape}")
    print(f"Action Space size: {raw_env.action_space.n}")
    raw_env.close()

    baseline_stats = run_part_a_baseline(num_episodes=100)
