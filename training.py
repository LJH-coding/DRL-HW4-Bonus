"""Basic Dueling + Double DQN training script."""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam

from DolphinEnv import DolphinEnv


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


@dataclass
class TrainConfig:
    game: str = "MarioKart"
    envs: int = 4
    frames: int = 1_000_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-4
    buffer_size: int = 100_000
    min_buffer_size: int = 20_000
    train_interval: int = 4
    target_update_interval: int = 5_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 300_000
    grad_clip: float = 10.0
    model_dir: str = "checkpoints"
    model_name: str = "dueling_double_dqn.pt"
    device: str | None = None
    framestack: int = 4


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="MarioKart")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--frames", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--min_buffer_size", type=int, default=20_000)
    parser.add_argument("--train_interval", type=int, default=4)
    parser.add_argument("--target_update_interval", type=int, default=5_000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=300_000)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--model_dir", type=str, default="checkpoints")
    parser.add_argument("--model_name", type=str, default="dueling_double_dqn.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--framestack", type=int, default=4)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.storage: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.storage.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.storage, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
        return states_t, actions_t, rewards_t, next_states_t, dones_t


class DuelingQNet(nn.Module):
    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            feature_dim = self.encoder(torch.zeros(1, in_channels, 75, 140)).shape[1]
        self.value_stream = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, 1))
        self.adv_stream = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, num_actions))

    def forward(self, obs: Tensor) -> Tensor:
        x = obs / 255.0
        features = self.encoder(x)
        value = self.value_stream(features)
        advantage = self.adv_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingDoubleDQNAgent:
    def __init__(self, num_actions: int, cfg: TrainConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.num_actions = num_actions
        self.online = DuelingQNet(cfg.framestack, num_actions).to(device)
        self.target = DuelingQNet(cfg.framestack, num_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = Adam(self.online.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.buffer_size)
        self.train_steps = 0
        self.epsilon = cfg.epsilon_start

    def update_epsilon(self, env_steps: int) -> None:
        frac = min(env_steps / max(self.cfg.epsilon_decay_steps, 1), 1.0)
        self.epsilon = self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def select_action(self, obs_batch: np.ndarray) -> np.ndarray:
        if random.random() < self.epsilon:
            return np.array([random.randrange(self.num_actions) for _ in range(obs_batch.shape[0])], dtype=np.int64)
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        q_values = self.online(obs_tensor)
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def train_step(self) -> float | None:
        if len(self.replay) < self.cfg.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.batch_size, self.device)
        q_values = self.online(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = torch.argmax(self.online(next_states), dim=1, keepdim=True)
            next_target_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.cfg.gamma * (1.0 - dones) * next_target_q

        loss = nn.functional.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.online.state_dict(),
            "num_actions": self.num_actions,
            "framestack": self.cfg.framestack,
            "algo": "dueling_double_dqn",
        }
        torch.save(payload, save_path)


def train(cfg: TrainConfig) -> None:
    print("=== Dueling + Double DQN Baseline ===")
    device = torch.device(cfg.device) if cfg.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = DolphinEnv(cfg.envs)
    obs, info = env.reset()
    num_actions = env.action_space[0].n
    agent = DuelingDoubleDQNAgent(num_actions=num_actions, cfg=cfg, device=device)

    episode_returns = [0.0 for _ in range(cfg.envs)]
    total_steps = 0
    last_log_t = time.time()
    last_log_steps = 0

    while total_steps < cfg.frames:
        agent.update_epsilon(total_steps)
        actions = agent.select_action(obs)

        env.step_async(actions)
        next_obs, rewards, dones, truns, infos = env.step_wait()

        for env_id in range(cfg.envs):
            if infos["Ignore"][env_id]:
                continue

            if infos["First"][env_id]:
                obs[env_id] = next_obs[env_id]

            final_next = next_obs[env_id] if not truns[env_id] else np.array(infos["final_observation"][env_id])
            done = bool(dones[env_id] or truns[env_id])

            agent.replay.add(
                state=obs[env_id],
                action=int(actions[env_id]),
                reward=float(rewards[env_id]),
                next_state=final_next,
                done=done,
            )

            episode_returns[env_id] += float(rewards[env_id])
            if done:
                print(f"[Episode End] env={env_id} return={episode_returns[env_id]:.2f}")
                episode_returns[env_id] = 0.0

        total_steps += cfg.envs
        obs = next_obs

        if total_steps % cfg.train_interval == 0:
            loss = agent.train_step()
            if loss is not None and total_steps % 1000 == 0:
                print(
                    f"step={total_steps} loss={loss:.5f} epsilon={agent.epsilon:.3f} "
                    f"buffer={len(agent.replay)}"
                )

        if total_steps % 5000 == 0:
            now = time.time()
            fps = (total_steps - last_log_steps) / max(now - last_log_t, 1e-6)
            print(f"[Progress] steps={total_steps}/{cfg.frames} fps={fps:.1f}")
            last_log_t = now
            last_log_steps = total_steps

    save_path = Path(cfg.model_dir) / cfg.model_name
    agent.save(save_path)
    print(f"Training finished. Model saved to: {save_path}")
    print(f"Run evaluation with: python evaluation.py --model_path {save_path}")


def load_agent_for_eval(model_path: str, framestack: int, device: torch.device) -> DuelingQNet:
    payload = torch.load(model_path, map_location=device)
    saved_framestack = int(payload.get("framestack", framestack))
    model = DuelingQNet(saved_framestack, payload["num_actions"])
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    train(parse_args())
