# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Free-base center-of-mass and ground-plane ZMP diagnostic calculations."""

from __future__ import annotations

import torch


def compute_center_of_mass(
    body_com_positions_w_m: torch.Tensor, body_masses_kg: torch.Tensor
) -> torch.Tensor:
    """Compute the mass-weighted articulation center of mass in the world frame.

    Args:
        body_com_positions_w_m: Per-body center-of-mass positions [m], shape
            (num_envs, num_bodies, 3).
        body_masses_kg: Per-body masses [kg], shape (num_envs, num_bodies).

    Returns:
        Articulation center-of-mass position [m], shape (num_envs, 3).
    """
    if body_com_positions_w_m.ndim != 3 or body_com_positions_w_m.shape[-1] != 3:
        raise ValueError("Body CoM positions must have shape (num_envs, num_bodies, 3).")
    if body_masses_kg.shape != body_com_positions_w_m.shape[:2]:
        raise ValueError("Body masses must have shape (num_envs, num_bodies).")
    total_mass_kg = body_masses_kg.sum(dim=-1, keepdim=True)
    if torch.any(~torch.isfinite(body_com_positions_w_m)) or torch.any(~torch.isfinite(body_masses_kg)):
        raise ValueError("Body CoM positions and masses must be finite.")
    if torch.any(total_mass_kg <= 0.0):
        raise ValueError("Each articulation must have positive total mass.")
    return (body_com_positions_w_m * body_masses_kg.unsqueeze(-1)).sum(dim=1) / total_mass_kg


def compute_ground_zmp(
    contact_positions_w_m: torch.Tensor,
    contact_forces_w_n: torch.Tensor,
    *,
    ground_height_m: float = 0.0,
    minimum_normal_force_n: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate ground-plane ZMP from filtered contact points and contact forces.

    The moment is sum(contact_position x contact_force) about the world origin.
    The result projects that wrench onto z=ground_height_m. PhysX reports one
    average point and one filtered force per sensor body, so this is an
    average-contact-point estimate rather than a shape-resolved contact wrench.

    Args:
        contact_positions_w_m: One filtered ground-contact position per foot
            [m], shape (num_envs, num_feet, 3).
        contact_forces_w_n: Matching filtered ground-contact forces [N], shape
            (num_envs, num_feet, 3).
        ground_height_m: Ground-plane world z coordinate [m].
        minimum_normal_force_n: Minimum total upward force for a valid estimate
            [N].

    Returns:
        Tuple of ground-plane ZMP positions [m], validity mask, net force [N],
        and net moment about the world origin [N m].
    """
    if contact_positions_w_m.shape != contact_forces_w_n.shape or contact_positions_w_m.ndim != 3:
        raise ValueError("Contact positions and forces must share shape (num_envs, num_feet, 3).")
    if contact_positions_w_m.shape[-1] != 3:
        raise ValueError("Contact vectors must have three components.")
    if minimum_normal_force_n <= 0.0:
        raise ValueError("Minimum normal force must be positive.")

    normal_force_n = contact_forces_w_n[..., 2]
    active_contact = normal_force_n > 0.0
    finite_force = torch.isfinite(contact_forces_w_n).all(dim=-1)
    finite_position = torch.isfinite(contact_positions_w_m).all(dim=-1)
    usable_contact = active_contact & finite_force & finite_position
    usable_positions = torch.where(usable_contact.unsqueeze(-1), contact_positions_w_m, 0.0)
    usable_forces = torch.where(usable_contact.unsqueeze(-1), contact_forces_w_n, 0.0)
    net_force_w_n = usable_forces.sum(dim=1)
    net_moment_w_nm = torch.linalg.cross(usable_positions, usable_forces, dim=-1).sum(dim=1)
    valid = net_force_w_n[:, 2] >= minimum_normal_force_n

    zmp_w_m = torch.zeros_like(net_force_w_n)
    force_z_n = net_force_w_n[:, 2].clamp_min(torch.finfo(net_force_w_n.dtype).eps)
    zmp_w_m[:, 0] = (ground_height_m * net_force_w_n[:, 0] - net_moment_w_nm[:, 1]) / force_z_n
    zmp_w_m[:, 1] = (net_moment_w_nm[:, 0] + ground_height_m * net_force_w_n[:, 1]) / force_z_n
    zmp_w_m[:, 2] = ground_height_m
    return zmp_w_m, valid, net_force_w_n, net_moment_w_nm


def compute_zmp_support_margin(
    zmp_w_m: torch.Tensor,
    zmp_valid: torch.Tensor,
    contact_positions_w_m: torch.Tensor,
    contact_normal_forces_n: torch.Tensor,
    *,
    minimum_contact_force_n: float = 20.0,
    foot_half_length_m: float = 0.130,
    foot_half_width_m: float = 0.043,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure the signed ZMP margin to a conservative parallel-foot support polygon.

    Each valid ankle-roll ground contact is represented by an axis-aligned
    rectangle centered on its filtered average contact point. For two feet,
    the support polygon is the conservative rectangle spanning their lateral
    outer edges and their common longitudinal overlap. This excludes a
    non-overlapping fore-aft gap and is valid only for near-zero foot yaw.

    Args:
        zmp_w_m: Ground-plane ZMP positions [m], shape (num_envs, 3).
        zmp_valid: Whether each ZMP estimate is valid, shape (num_envs,).
        contact_positions_w_m: Filtered foot contact positions [m], shape
            (num_envs, num_feet, 3).
        contact_normal_forces_n: Per-foot upward normal forces [N], shape
            (num_envs, num_feet).
        minimum_contact_force_n: Minimum per-foot normal force used to
            include a foot in the support polygon [N].
        foot_half_length_m: Contact-patch half length along world x [m].
        foot_half_width_m: Contact-patch half width along world y [m].

    Returns:
        Tuple of signed ZMP support margins [m], support-polygon validity,
        and ZMP-outside-support flags. Positive margin is inside, zero is on
        the boundary, and negative margin is the Euclidean distance outside.
        Invalid estimates return zero margin and false validity/violation.
    """
    if zmp_w_m.ndim != 2 or zmp_w_m.shape[-1] != 3:
        raise ValueError("ZMP positions must have shape (num_envs, 3).")
    if zmp_valid.shape != zmp_w_m.shape[:1]:
        raise ValueError("ZMP validity must have shape (num_envs,).")
    if contact_positions_w_m.ndim != 3 or contact_positions_w_m.shape[-1] != 3:
        raise ValueError("Contact positions must have shape (num_envs, num_feet, 3).")
    if contact_normal_forces_n.shape != contact_positions_w_m.shape[:2]:
        raise ValueError("Contact normal forces must have shape (num_envs, num_feet).")
    if minimum_contact_force_n <= 0.0 or foot_half_length_m <= 0.0 or foot_half_width_m <= 0.0:
        raise ValueError("Contact threshold and foot half dimensions must be positive.")

    finite_position = torch.isfinite(contact_positions_w_m).all(dim=-1)
    finite_force = torch.isfinite(contact_normal_forces_n)
    active_foot = finite_position & finite_force & (contact_normal_forces_n >= minimum_contact_force_n)
    x = contact_positions_w_m[..., 0]
    y = contact_positions_w_m[..., 1]
    inf = torch.tensor(float("inf"), dtype=x.dtype, device=x.device)
    negative_inf = torch.tensor(float("-inf"), dtype=x.dtype, device=x.device)
    x_min = torch.where(active_foot, x - foot_half_length_m, negative_inf).max(dim=1).values
    x_max = torch.where(active_foot, x + foot_half_length_m, inf).min(dim=1).values
    y_min = torch.where(active_foot, y - foot_half_width_m, inf).min(dim=1).values
    y_max = torch.where(active_foot, y + foot_half_width_m, negative_inf).max(dim=1).values
    support_valid = zmp_valid & active_foot.any(dim=1) & (x_min <= x_max) & (y_min <= y_max)

    dx_low = x_min - zmp_w_m[:, 0]
    dx_high = zmp_w_m[:, 0] - x_max
    dy_low = y_min - zmp_w_m[:, 1]
    dy_high = zmp_w_m[:, 1] - y_max
    outside_x = torch.maximum(dx_low, dx_high).clamp_min(0.0)
    outside_y = torch.maximum(dy_low, dy_high).clamp_min(0.0)
    outside_distance = torch.sqrt(outside_x.square() + outside_y.square())
    inside_margin = torch.stack((-dx_low, -dx_high, -dy_low, -dy_high), dim=-1).min(dim=-1).values
    margin_m = torch.where(outside_distance > 0.0, -outside_distance, inside_margin)
    margin_m = torch.where(support_valid, margin_m, torch.zeros_like(margin_m))
    outside_support = support_valid & (margin_m < 0.0)
    return margin_m, support_valid, outside_support
