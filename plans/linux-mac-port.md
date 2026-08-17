# Plan: Port Annihilation TUI/glue to Linux & macOS

See `research/annihilation-linux-mac-port.md` for full recon evidence.

## Scope decision
The engine (`src/annihilate/`) is upstream-synced and already cross-platform — **do not touch**.
The port targets the fork-added glue: TUI launcher/venv resolution, env bootstrap, GGUF toolchain download, docs, CI.

## Branch & PR
- Branch: `feat/linux-macos-support` off `origin/main` (collaborator push rights, no fork needed).
- One PR titled `feat: support Linux and macOS` (the port is one semantic change);
  atomic conventional commits inside. If owner wants splits, cut along phases A/B/C below.

## Phases (commits)
- A. `feat(tui): resolve venv interpreter and bootstrap env on Linux and macOS`
  - `tui/src/subprocess.rs`: `python_exe()` → `bin/python` on POSIX, keep venv priority order.
  - `tui/src/subprocess.rs`: `spawn_setup()` → native venv check/create + direct script spawn (no PowerShell).
  - `tui/src/sysinfo.rs`: real `target_os = "macos"` branches (`sysctl`) alongside the existing Linux ones.
- B. `feat(env): make environment verification cross-platform`
  - Rename `verify_env_windows.py` → `verify_env.py`; platform-aware accelerator reinstall
    (Linux: PyPI torch ships CUDA; macOS: no CUDA branch; Windows: keep cu126 path).
- C. `feat(gguf): fetch llama.cpp binaries on Linux and macOS`
  - `scripts/gguf_converter.py`: asset selection by system/arch, tar.gz extraction with the
    existing escape-guard pattern, `chmod +x`, keep sha256 pin env vars.
- D. `feat: add start.sh launcher for POSIX` + `docs: add Linux and macOS quick-start` (README).
- E. `chore(ci): build the TUI on Linux and macOS in CI` (matrix tui job).

## Verification gates
- Linux (local): `cargo fmt/clippy/build/test --locked` in tui/; `uv run ruff format --check`,
  `ruff check`, `ty check --error-on-warning`; live run: `./start.sh` first-run venv bootstrap,
  engine spawn, GGUF toolchain download against latest llama.cpp release.
- macOS: CI `macos-latest` (partner's Mac if available).
- Windows regression: keep `windows-latest` in the matrix; no behavior change for the existing path.

## Out of scope / risks
- No `src/annihilate/**` edits (sync-upstream.yml overwrites them).
- No version bump (build.rs/pyproject↔Cargo.toml lockstep is owner's release call).
- musl/Alpine Linux: document the manual-binary fallback only.
