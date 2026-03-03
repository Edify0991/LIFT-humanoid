from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferenceClip:
    """A redirected mocap clip used by humanoid mimic tasks.

    All arrays are expected to have leading shape ``[T, ...]``.
    """

    dt: float
    qpos: np.ndarray
    qvel: np.ndarray
    key_body_pos: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])

    def frame(self, idx: int) -> dict[str, np.ndarray]:
        i = int(idx % self.num_frames)
        return {
            "qpos": self.qpos[i],
            "qvel": self.qvel[i],
            "key_body_pos": self.key_body_pos[i],
        }


def load_reference_clip(path: str | Path) -> ReferenceClip:
    """Loads redirected mocap from a .npz file.

    Expected keys:
      - dt: scalar float
      - qpos: [T, nq]
      - qvel: [T, nv]
      - key_body_pos: [T, K, 3]
    """

    data = np.load(Path(path), allow_pickle=False)
    required = {"dt", "qpos", "qvel", "key_body_pos"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Missing keys in {path}: {sorted(missing)}")

    qpos = np.asarray(data["qpos"], dtype=np.float32)
    qvel = np.asarray(data["qvel"], dtype=np.float32)
    key_body_pos = np.asarray(data["key_body_pos"], dtype=np.float32)
    dt = float(data["dt"])

    if qpos.ndim != 2 or qvel.ndim != 2 or key_body_pos.ndim != 3:
        raise ValueError("Invalid redirected mocap shapes. Expected qpos[T,nq], qvel[T,nv], key_body_pos[T,K,3].")
    if not (qpos.shape[0] == qvel.shape[0] == key_body_pos.shape[0]):
        raise ValueError("qpos/qvel/key_body_pos must have identical frame count.")

    return ReferenceClip(dt=dt, qpos=qpos, qvel=qvel, key_body_pos=key_body_pos)


class MotionLibrary:
    """A simple cyclic motion library for SAC mimic training."""

    def __init__(self, clips: list[ReferenceClip]):
        if not clips:
            raise ValueError("At least one redirected mocap clip is required.")
        self._clips = clips

    @property
    def clips(self) -> list[ReferenceClip]:
        return self._clips

    def sample_clip(self, rng: np.random.Generator) -> ReferenceClip:
        return self._clips[int(rng.integers(0, len(self._clips)))]

    def sample_start_frame(self, clip: ReferenceClip, rng: np.random.Generator) -> int:
        return int(rng.integers(0, clip.num_frames))

    def ref_frame_at_step(self, clip: ReferenceClip, start_frame: int, env_step: int, repeat: int = 1) -> dict[str, np.ndarray]:
        frame_idx = start_frame + (env_step // max(repeat, 1))
        return clip.frame(frame_idx)
