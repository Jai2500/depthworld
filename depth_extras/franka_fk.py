"""Analytical forward kinematics for the Franka Panda / FR3 arm.

Pure-PyTorch (autograd-compatible), no external kinematics deps. Used to
reconstruct per-frame wrist-camera pose from `observation.state.joint_position`
plus the JFG `dq` correction.

Ported from `~/PointWorld_Data/real/jfg/franka_fk.py` (PointWorld JFG codebase),
trimmed to only the top-level `franka_fk(q)` entry point — we don't need the
joint-axis Jacobian helper or the pytorch_kinematics-compatible wrapper class
that the JFG optimization loop uses.

DH note: this is **modified DH** (Craig 2005). The standard-DH chain matches
panda_link0 → panda_link8 (the flange before the gripper). Accuracy vs the
URDF-based pytorch_kinematics build is below 1 mm for joint corrections at
the 0.1°-per-joint scale, which is well within other modelling noise for
our purposes.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# DH parameters for Panda joints 1..7, then the link8 fixed flange offset.
# Each entry: (a, d, alpha). theta = q (variable) for joints 1..7; theta = 0
# for the link8 fixed transform.
_DH = [
    (0.0,    0.333,  0.0),
    (0.0,    0.0,   -math.pi / 2),
    (0.0,    0.316,  math.pi / 2),
    (0.0825, 0.0,    math.pi / 2),
    (-0.0825, 0.384, -math.pi / 2),
    (0.0,    0.0,    math.pi / 2),
    (0.088,  0.0,    math.pi / 2),
    (0.0,    0.107,  0.0),     # link8: fixed flange offset (theta=0)
]


def _dh_matrix(a: float, d: float, alpha: float, theta: Tensor) -> Tensor:
    """Build a (..., 4, 4) modified-DH transform with variable theta.

    Convention: `_dh_matrix(a, d, α, θ) = Trans_x(a) · Rot_x(α) · Rot_z(θ) · Trans_z(d)`.

    Args:
        a, d, alpha: DH link parameters (scalars).
        theta: (...) tensor of joint angles in radians (or zero for fixed links).
    Returns:
        (..., 4, 4) homogeneous transform tensor.
    """
    c_t = torch.cos(theta)
    s_t = torch.sin(theta)
    c_a = math.cos(alpha)
    s_a = math.sin(alpha)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)

    row0 = torch.stack([c_t,        -s_t,        zero,          a * one    ], dim=-1)
    row1 = torch.stack([s_t * c_a,  c_t * c_a,   -s_a * one,    -s_a * d * one], dim=-1)
    row2 = torch.stack([s_t * s_a,  c_t * s_a,   c_a * one,     c_a * d * one ], dim=-1)
    row3 = torch.stack([zero,       zero,        zero,          one          ], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


def franka_fk(q: Tensor) -> Tensor:
    """Forward kinematics for the 7-DoF Franka Panda arm.

    Args:
        q: (..., 7) joint angles in radians. Apply any JFG `dq` correction
           BEFORE passing — this function just runs the DH chain on whatever
           you give it.
    Returns:
        T_base_to_link8: (..., 4, 4) pose of panda_link8 (flange) in the
        robot base frame.
    """
    if q.shape[-1] != 7:
        raise ValueError(f"expected q of shape (..., 7), got {q.shape}")

    leading = q.shape[:-1]
    T = torch.eye(4, dtype=q.dtype, device=q.device).expand(*leading, 4, 4).contiguous()
    theta_zero = torch.zeros(leading, dtype=q.dtype, device=q.device)

    for i, (a, d, alpha) in enumerate(_DH[:7]):
        T = T @ _dh_matrix(a, d, alpha, q[..., i])
    # link8 flange: theta=0, fixed offset.
    a, d, alpha = _DH[7]
    T = T @ _dh_matrix(a, d, alpha, theta_zero)
    return T
