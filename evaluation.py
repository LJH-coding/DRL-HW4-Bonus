"""Evaluation script for Dueling + Double DQN baseline."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from DolphinEnv import DolphinEnv
from training import load_agent_for_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--frames", type=int, default=100_000)
    parser.add_argument("--framestack", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def select_eval_action(model, obs_batch: np.ndarray, device: torch.device) -> np.ndarray:
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
    q_values = model(obs_tensor)
    actions = torch.argmax(q_values, dim=1)
    return actions.cpu().numpy()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = DolphinEnv(args.envs)
    obs, _ = env.reset()
    model = load_agent_for_eval(args.model_path, args.framestack, device)

    steps = 0
    episode_returns = [0.0 for _ in range(args.envs)]
    completed_returns = []
    last_log_t = time.time()
    last_log_steps = 0

    while steps < args.frames:
        actions = select_eval_action(model, obs, device)
        env.step_async(actions)
        next_obs, rewards, dones, truns, infos = env.step_wait()
        steps += args.envs

        for env_id in range(args.envs):
            if infos["Ignore"][env_id]:
                continue
            episode_returns[env_id] += float(rewards[env_id])
            if dones[env_id] or truns[env_id]:
                completed_returns.append(episode_returns[env_id])
                print(f"[Eval Episode] env={env_id} return={episode_returns[env_id]:.2f}")
                episode_returns[env_id] = 0.0

        obs = next_obs

        if steps % 5000 == 0:
            now = time.time()
            fps = (steps - last_log_steps) / max(now - last_log_t, 1e-6)
            print(f"[Eval Progress] steps={steps}/{args.frames} fps={fps:.1f}")
            last_log_t = now
            last_log_steps = steps

    if len(completed_returns) == 0:
        print("No episode finished during evaluation.")
        return

    avg_return = float(np.mean(completed_returns))
    std_return = float(np.std(completed_returns))
    print("=== Evaluation Done ===")
    print(f"Episodes: {len(completed_returns)}")
    print(f"Average return: {avg_return:.2f}")
    print(f"Std return: {std_return:.2f}")


if __name__ == "__main__":
    main()
