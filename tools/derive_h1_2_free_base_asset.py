# Copyright (c) 2026, H1-2 Whole-Body Reaching Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Create the project-owned free-base H1-2 URDF from the pinned official asset."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import xml.etree.ElementTree as xml_etree


COLLISION_REPAIRS = (
    "left_hip_yaw_link",
    "left_ankle_pitch_link",
    "right_hip_yaw_link",
    "right_ankle_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
LOGO_LINK_NAME = "logo_link"


def _make_mesh_collision(link: xml_etree.Element) -> xml_etree.Element:
    """Create a collision element that matches a link's first visual mesh."""
    visual = link.find("visual")
    if visual is None:
        raise ValueError(f"Link {link.attrib['name']!r} has no visual mesh to use for collision geometry.")
    collision = xml_etree.Element("collision")
    origin = visual.find("origin")
    if origin is not None:
        collision.append(copy.deepcopy(origin))
    geometry = visual.find("geometry")
    if geometry is None or geometry.find("mesh") is None:
        raise ValueError(f"Link {link.attrib['name']!r} has no visual mesh to use for collision geometry.")
    collision.append(copy.deepcopy(geometry))
    return collision


def _add_logo_inertia(link: xml_etree.Element) -> None:
    """Add a small positive-definite inertial to the visual-only fixed logo link."""
    if link.find("inertial") is not None:
        raise ValueError(f"Link {LOGO_LINK_NAME!r} unexpectedly already has inertial properties.")
    inertial = xml_etree.Element("inertial")
    xml_etree.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    xml_etree.SubElement(inertial, "mass", {"value": "0.001"})
    xml_etree.SubElement(inertial, "inertia", {"ixx": "1e-8", "ixy": "0", "ixz": "0", "iyy": "1e-8", "iyz": "0", "izz": "1e-8"})
    link.insert(0, inertial)


def derive_asset(source_path: Path, output_path: Path) -> None:
    """Derive a free-base-safe URDF without changing the pinned official source.

    Args:
        source_path: Pinned official H1-2 handless URDF path.
        output_path: Derived URDF output path.
    """
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    root = xml_etree.parse(source_path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    missing_links = [name for name in COLLISION_REPAIRS if name not in links]
    if missing_links:
        raise ValueError(f"Official URDF is missing expected links: {missing_links}")
    for link_name in COLLISION_REPAIRS:
        link = links[link_name]
        if link.findall("collision"):
            raise ValueError(f"Link {link_name!r} unexpectedly already has collision geometry.")
        link.append(_make_mesh_collision(link))
    logo_link = links.get(LOGO_LINK_NAME)
    if logo_link is None:
        raise ValueError(f"Official URDF is missing expected link {LOGO_LINK_NAME!r}.")
    _add_logo_inertia(logo_link)

    mesh_directory = source_path.parent / "meshes"
    relative_mesh_directory = Path(os.path.relpath(mesh_directory, output_path.parent))
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename is None or not filename.startswith("meshes/"):
            raise ValueError(f"Unexpected mesh path {filename!r} in official URDF.")
        mesh.attrib["filename"] = (relative_mesh_directory / filename.removeprefix("meshes/")).as_posix()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n".join(line.rstrip() for line in xml_etree.tostring(root, encoding="unicode").splitlines())
    output_path.write_text(
        "<?xml version=\"1.0\"?>\n"
        "<!-- Derived from Unitree h1_2_handless.urdf at the pinned BSD-3-Clause source commit. -->\n"
        "<!-- Adds collisions only for the listed omitted movable-link meshes. -->\n"
        f"{serialized}\n",
        encoding="utf-8",
    )


def main() -> None:
    """Create the derived asset from command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pinned official h1_2_handless.urdf path.")
    parser.add_argument("output", type=Path, help="Derived free-base URDF path.")
    arguments = parser.parse_args()
    derive_asset(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
