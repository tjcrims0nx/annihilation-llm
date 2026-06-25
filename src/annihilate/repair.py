# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Install repair and uninstall helpers for Annihilate.

This module intentionally uses only the Python standard library so it can run
even when heavyweight ML dependencies are missing or broken.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

PACKAGE_NAME = "annihilate-llm"
OLD_PACKAGE_NAMES = ("heretic-llm", "heretic")
SCRIPT_NAMES = (
    "annihilate",
    "annihilation",
    "annihilate-doctor",
    "annihilate-repair",
    "annihilate-uninstall",
    "heretic",
)


@dataclass(frozen=True)
class DistributionInfo:
    name: str
    installed: bool
    version: str | None = None
    location: str | None = None
    entry_points: tuple[str, ...] = ()


def _distribution_info(name: str) -> DistributionInfo:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return DistributionInfo(name=name, installed=False)

    try:
        location = str(dist.locate_file(""))
    except Exception:
        location = None

    entry_points = tuple(
        f"{ep.name} = {ep.value}"
        for ep in dist.entry_points
        if ep.group == "console_scripts"
    )

    return DistributionInfo(
        name=name,
        installed=True,
        version=dist.version,
        location=location,
        entry_points=entry_points,
    )


def _script_candidates(name: str) -> list[str]:
    names = [name]
    if sys.platform == "win32":
        names.extend([f"{name}.exe", f"{name}.cmd", f"{name}.bat", f"{name}.ps1"])
    return names


def _which_all(name: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for candidate in _script_candidates(name):
        path = shutil.which(candidate)
        if not path:
            continue

        key = path.lower() if sys.platform == "win32" else path
        if key not in seen:
            seen.add(key)
            found.append(path)
    return found


def _run_pip(args: Iterable[str]) -> int:
    command = [sys.executable, "-m", "pip", *args]
    print("+ " + " ".join(command))
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _print_distributions() -> None:
    print("Python:", sys.executable)
    print("Version:", sys.version.split()[0])
    print()

    for name in (PACKAGE_NAME, *OLD_PACKAGE_NAMES):
        info = _distribution_info(name)
        if not info.installed:
            print(f"{name}: not installed")
            continue

        print(f"{name}: {info.version}")
        if info.location:
            print(f"  location: {info.location}")
        if info.entry_points:
            print("  console scripts:")
            for entry_point in info.entry_points:
                print(f"    {entry_point}")
        else:
            print("  console scripts: none")
    print()


def _print_scripts() -> None:
    print("Command shims on PATH:")
    for name in SCRIPT_NAMES:
        paths = _which_all(name)
        if paths:
            for path in paths:
                print(f"  {name}: {path}")
        else:
            print(f"  {name}: not found")
    print()


def doctor(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="annihilate-doctor",
        description="Inspect Annihilate installation, entrypoints, and stale command shims.",
    )
    parser.parse_args(argv)

    _print_distributions()
    _print_scripts()

    print("Recommended repair command:")
    print(f"  {sys.executable} -m annihilate.repair repair --yes")
    print()
    print("Recommended full uninstall command:")
    print(f"  {sys.executable} -m annihilate.repair uninstall --yes")
    print()
    print(
        "Use the python -m commands above if the annihilate executable itself is broken."
    )
    return 0


def uninstall(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="annihilate-uninstall",
        description="Uninstall Annihilate and old Heretic package names from this Python environment.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually run pip uninstall. Without this, only print the command.",
    )
    parser.add_argument(
        "--keep-heretic",
        action="store_true",
        help="Only uninstall annihilate-llm, leaving old heretic packages alone.",
    )
    args = parser.parse_args(argv)

    packages = [PACKAGE_NAME]
    if not args.keep_heretic:
        packages.extend(OLD_PACKAGE_NAMES)

    print("This affects only the active Python environment:")
    print(f"  {sys.executable}")
    print()

    command = [sys.executable, "-m", "pip", "uninstall", "-y", *packages]
    if not args.yes:
        print("Dry run. Re-run with --yes to execute:")
        print("+ " + " ".join(command))
        return 0

    code = _run_pip(["uninstall", "-y", *packages])
    if code != 0:
        return code

    print()
    print("Uninstall command finished. Checking remaining shims:")
    _print_scripts()
    print(
        "If a command shim remains on Windows, close terminals that may be using it "
        "and run this module again with python -m annihilate.repair uninstall --yes."
    )
    return 0


def repair(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="annihilate-repair",
        description="Clean reinstall Annihilate in this Python environment.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually run uninstall and reinstall. Without this, only print commands.",
    )
    args = parser.parse_args(argv)

    packages = [PACKAGE_NAME, *OLD_PACKAGE_NAMES]
    uninstall_command = [sys.executable, "-m", "pip", "uninstall", "-y", *packages]
    install_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "-U",
        PACKAGE_NAME,
    ]

    print("This repairs only the active Python environment:")
    print(f"  {sys.executable}")
    print()

    if not args.yes:
        print("Dry run. Re-run with --yes to execute:")
        print("+ " + " ".join(uninstall_command))
        print("+ " + " ".join(install_command))
        return 0

    code = _run_pip(["uninstall", "-y", *packages])
    if code != 0:
        return code

    code = _run_pip(["install", "--no-cache-dir", "-U", PACKAGE_NAME])
    if code != 0:
        return code

    print()
    print("Repair command finished. Run:")
    print("  annihilate-doctor")
    print("  annihilate --help")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return doctor([])

    command = argv.pop(0)
    if command == "doctor":
        return doctor(argv)
    if command == "uninstall":
        return uninstall(argv)
    if command == "repair":
        return repair(argv)

    print(f"Unknown repair command: {command}", file=sys.stderr)
    print("Expected one of: doctor, uninstall, repair", file=sys.stderr)
    return 2


def doctor_main() -> int:
    return doctor()


def uninstall_main() -> int:
    return uninstall()


def repair_main() -> int:
    return repair()


if __name__ == "__main__":
    raise SystemExit(main())
