"""Train a curriculum RL policy for clutter-aware cube sorting.

This is a high-level sorting policy: the perception module proposes visible
objects, and the RL agent learns which object to pick next while avoiding
distractors and low-confidence/occluded candidates.

Run:
    python 08_train_rl_sorting_policy.py

TensorBoard:
    tensorboard --logdir runs/rl_sorting
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env


MAX_OBJECTS = 10
FEATURES_PER_OBJECT = 6


@dataclass(frozen=True)
class RLCurriculumStage:
    name: str
    max_cubes: int
    distractors: int
    false_positive_rate: float
    occlusion_rate: float
    perception_noise: float


CURRICULUM = (
    RLCurriculumStage("single_cube", 1, 0, 0.00, 0.00, 0.01),
    RLCurriculumStage("three_cubes", 3, 1, 0.03, 0.05, 0.02),
    RLCurriculumStage("light_clutter", 3, 4, 0.08, 0.12, 0.035),
    RLCurriculumStage("dense_clutter", 5, 8, 0.14, 0.22, 0.055),
)


class ClutteredCubeSortingEnv(gym.Env):
    """Feature-level RL environment for cube sorting decisions."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.stage_index = 0
        self.action_space = spaces.Discrete(MAX_OBJECTS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(MAX_OBJECTS * FEATURES_PER_OBJECT + 1,),
            dtype=np.float32,
        )
        self.objects = []
        self.steps = 0
        self.max_steps = 18
        self.sorted_count = 0
        self.total_cubes = 0

    def set_stage(self, stage_index: int):
        self.stage_index = int(np.clip(stage_index, 0, len(CURRICULUM) - 1))

    @property
    def stage(self) -> RLCurriculumStage:
        return CURRICULUM[self.stage_index]

    def _sample_object(self, is_cube: bool):
        x = float(self.rng.uniform(0.30, 0.70))
        y = float(self.rng.uniform(-0.25, 0.25))
        distance_to_zone = float(np.linalg.norm(np.array([x - 0.5, y - 0.35])))
        occluded = self.rng.random() < self.stage.occlusion_rate
        visible = not occluded

        if is_cube:
            cube_score = float(self.rng.uniform(0.70, 1.0))
        else:
            false_positive = self.rng.random() < self.stage.false_positive_rate
            cube_score = float(self.rng.uniform(0.45, 0.78) if false_positive else self.rng.uniform(0.0, 0.35))

        cube_score += float(self.rng.normal(0, self.stage.perception_noise))
        cube_score = float(np.clip(cube_score, 0.0, 1.0))
        return {
            "x": x,
            "y": y,
            "distance": distance_to_zone,
            "is_cube": is_cube,
            "cube_score": cube_score,
            "visible": visible,
            "sorted": False,
        }

    def _get_obs(self):
        obs = np.zeros((MAX_OBJECTS, FEATURES_PER_OBJECT), dtype=np.float32)
        for i, obj in enumerate(self.objects[:MAX_OBJECTS]):
            obs[i, 0] = np.interp(obj["x"], [0.30, 0.70], [-1.0, 1.0])
            obs[i, 1] = np.interp(obj["y"], [-0.25, 0.25], [-1.0, 1.0])
            obs[i, 2] = np.interp(obj["cube_score"], [0.0, 1.0], [-1.0, 1.0])
            obs[i, 3] = np.interp(obj["distance"], [0.0, 0.75], [1.0, -1.0])
            obs[i, 4] = 1.0 if obj["visible"] else -1.0
            obs[i, 5] = 1.0 if obj["sorted"] else -1.0
        stage_value = np.array([np.interp(self.stage_index, [0, len(CURRICULUM) - 1], [-1.0, 1.0])], dtype=np.float32)
        return np.concatenate([obs.reshape(-1), stage_value]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        stage = self.stage
        self.steps = 0
        self.sorted_count = 0
        self.total_cubes = int(self.rng.integers(1, stage.max_cubes + 1))

        self.objects = [self._sample_object(is_cube=True) for _ in range(self.total_cubes)]
        self.objects.extend(self._sample_object(is_cube=False) for _ in range(stage.distractors))
        self.rng.shuffle(self.objects)
        return self._get_obs(), {"stage": stage.name, "total_cubes": self.total_cubes}

    def step(self, action):
        self.steps += 1
        reward = -0.03
        terminated = False
        truncated = self.steps >= self.max_steps
        info = {"stage": self.stage.name}

        if action >= len(self.objects):
            reward -= 0.35
            return self._get_obs(), reward, terminated, truncated, info

        obj = self.objects[int(action)]
        if obj["sorted"]:
            reward -= 0.45
        elif not obj["visible"]:
            reward -= 0.65
        elif not obj["is_cube"]:
            reward -= 1.25
            info["collision_with_distractor"] = True
        else:
            obj["sorted"] = True
            self.sorted_count += 1
            lift_reward = 1.0
            place_reward = 4.0
            distance_bonus = max(0.0, 0.8 - obj["distance"])
            reward += lift_reward + place_reward + distance_bonus
            info["placed_cube"] = True

        if self.sorted_count >= self.total_cubes:
            reward += 3.0
            terminated = True
            info["success"] = True

        return self._get_obs(), reward, terminated, truncated, info


class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / max(self.total_timesteps, 1)
        stage_index = min(int(progress * len(CURRICULUM)), len(CURRICULUM) - 1)
        self.training_env.env_method("set_stage", stage_index)
        self.logger.record("curriculum/stage_index", stage_index)
        return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensorboard-log", type=str, default="runs/rl_sorting")
    parser.add_argument("--output", type=Path, default=Path("models") / "sorting_policy")
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_vec_env(
        ClutteredCubeSortingEnv,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=512,
        gamma=0.97,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
    )
    callback = CurriculumCallback(args.timesteps)
    model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name="ppo_cube_sorting")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"Saved RL sorting policy to {args.output}.zip")
    env.close()


if __name__ == "__main__":
    main()
