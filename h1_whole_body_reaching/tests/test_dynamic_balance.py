# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for free-base center-of-mass and ground-plane ZMP diagnostics."""

from __future__ import annotations

import pytest
import torch

from h1_whole_body_reaching.dynamic_balance import (
    compute_center_of_mass,
    compute_ground_zmp,
    compute_zmp_support_margin,
)


def test_center_of_mass_is_mass_weighted() -> None:
    """The articulation CoM follows the independently calculable weighted mean."""
    positions = torch.tensor([[[0.0, 0.0, 1.0], [2.0, 0.0, 2.0]]])
    masses = torch.tensor([[1.0, 3.0]])
    torch.testing.assert_close(compute_center_of_mass(positions, masses), torch.tensor([[1.5, 0.0, 1.75]]))


def test_ground_zmp_cancels_the_horizontal_contact_moment() -> None:
    """The returned point zeroes the horizontal moment on the configured plane."""
    positions = torch.tensor([[[0.20, -0.10, 0.0], [-0.10, 0.30, 0.0]]])
    forces = torch.tensor([[[10.0, 5.0, 100.0], [-5.0, -2.0, 200.0]]])
    zmp, valid, net_force, moment = compute_ground_zmp(positions, forces, minimum_normal_force_n=20.0)

    residual_moment = moment - torch.linalg.cross(zmp, net_force, dim=-1)
    assert valid.item()
    torch.testing.assert_close(residual_moment[:, :2], torch.zeros(1, 2), atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(zmp[:, 2], torch.zeros(1))


def test_ground_zmp_supports_an_elevated_reference_plane() -> None:
    """Tangential force shifts the ZMP consistently on a nonzero-height plane."""
    positions = torch.tensor([[[0.4, 0.2, 0.3]]])
    forces = torch.tensor([[[30.0, -20.0, 100.0]]])
    zmp, valid, _, _ = compute_ground_zmp(
        positions, forces, ground_height_m=0.3, minimum_normal_force_n=20.0
    )
    assert valid.item()
    torch.testing.assert_close(zmp, positions[:, 0])


def test_ground_zmp_returns_invalid_for_lost_contact_without_nan() -> None:
    """A foot-lift case is invalid but remains numerically safe for vectorized use."""
    positions = torch.full((1, 2, 3), float("nan"))
    forces = torch.zeros(1, 2, 3)
    zmp, valid, net_force, moment = compute_ground_zmp(positions, forces)
    assert not valid.item()
    assert torch.isfinite(zmp).all()
    assert torch.isfinite(net_force).all()
    assert torch.isfinite(moment).all()


def test_center_of_mass_rejects_nonpositive_total_mass() -> None:
    """Invalid mass data is rejected before it can contaminate diagnostics."""
    with pytest.raises(ValueError, match="positive total mass"):
        compute_center_of_mass(torch.zeros(1, 1, 3), torch.zeros(1, 1))


def test_zmp_support_margin_uses_the_conservative_common_longitudinal_overlap() -> None:
    """The two-foot polygon admits its common fore-aft region and lateral span."""
    zmp = torch.tensor([[0.0, 0.0, 0.0], [0.14, 0.0, 0.0], [0.0, 0.15, 0.0]])
    contacts = torch.tensor([[[0.0, 0.10, 0.0], [0.0, -0.10, 0.0]]] * 3)
    forces = torch.full((3, 2), 100.0)
    margin, valid, violation = compute_zmp_support_margin(
        zmp, torch.ones(3, dtype=torch.bool), contacts, forces, foot_half_length_m=0.13, foot_half_width_m=0.04
    )

    torch.testing.assert_close(margin[0], torch.tensor(0.13))
    assert valid.tolist() == [True, True, True]
    assert violation.tolist() == [False, True, True]
    assert margin[1] < 0.0
    assert margin[2] < 0.0


def test_zmp_support_margin_is_invalid_when_no_foot_meets_the_force_threshold() -> None:
    """Missing foot support does not create a fabricated ZMP violation."""
    margin, valid, violation = compute_zmp_support_margin(
        torch.zeros(1, 3),
        torch.ones(1, dtype=torch.bool),
        torch.full((1, 2, 3), float("nan")),
        torch.zeros(1, 2),
    )
    torch.testing.assert_close(margin, torch.zeros(1))
    assert not valid.item()
    assert not violation.item()
