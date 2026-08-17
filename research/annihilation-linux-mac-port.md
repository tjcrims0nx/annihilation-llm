# Annihilation → Linux/macOS Port: Deep Recon

Repo: `tjcrims0nx/annihilation-llm` @ `main` / `12dc962` (v1.4.8)
Clone at: `/home/salt/Documents/annihilator` · Recon date: 2026-08-17
Our role: collaborator (Blackfrost-AI). Owner merges PRs.

## 1. What the project is

- **Engine** (`src/annihilate/`, ~13.9k LOC Python+Rust total): automatic abliteration
  (directional ablation + TPE search) for transformer LLMs. Fork of `p-e-w/heretic`.
- **TUI** (`tui/`, Rust/ratatui+crossterm): spawns the engine as a subprocess, parses
  JSON line events, renders dashboard. Also drives GGUF export and benchmarks.
- **Scripts** (`scripts/`): gguf_converter, run_benchmarks, chat_server, upload_to_hf.
- Version sync enforced: `tui/build.rs` panics if `tui/Cargo.toml` != `pyproject.toml` version.

## 2. CRITICAL: upstream-sync landmine

`.github/workflows/sync-upstream.yml` (manual dispatch) **wipes and replaces**
`src/annihilate/` + `config*.toml` from `p-e-w/heretic` and renames `heretic` → `annihilate`
via regex. Consequences for our port:

- **Do not modify `src/annihilate/**` or `config*.toml`** unless a bug cannot be fixed elsewhere
  (changes will be overwritten by the next sync, or create merge pain).
- Good news: the heretic-derived engine is already cross-platform (its CI runs on
  `ubuntu-latest`, MPS driver detection via `sw_vers` exists in `system.py:139`,
  ROCm/XPU/MLU/NPU backends handled). The port is almost entirely **fork-added glue**.

## 3. Platform inventory (the actual work)

### Windows-hardcoded (must port)

| # | Location | Problem | Fix shape |
|---|----------|---------|-----------|
| 1 | `tui/src/subprocess.rs:30-49` `python_exe()` | Only probes `<venv>/Scripts/python.exe` | Probe `Scripts/python.exe` on win, `bin/python` elsewhere; keep venv priority `annihilation-env > .venv > venv > env` |
| 2 | `tui/src/subprocess.rs:132-138` `spawn_setup()` | Runs a hardcoded **PowerShell** one-liner that creates the venv and launches `verify_env_windows.py` | Native Rust: check venv dirs, `python -m venv annihilation-env` (or `uv venv`), then spawn the verify script directly. No shell needed |
| 3 | `verify_env_windows.py` | Name is Windows-specific; CUDA reinstall branch uses `--index-url cu126` (Windows-only index pin); content otherwise cross-platform | Rename → `verify_env.py`; make CUDA handling platform-aware (Linux: PyPI torch already ships CUDA; macOS: no CUDA, report MPS); keep OBLITERATUS settings probe intact |
| 4 | `scripts/gguf_converter.py:100-164` `ensure_llama_cpp()` | Binary download only inside `os.name == "nt"` branch; else hard error "Only Windows automatic download is supported currently" | Add Linux/macOS asset selection + tar.gz extraction (see §4) |
| 5 | `start.bat` | Only launcher | Add `start.sh` (3 lines: cd tui && cargo run --release) |
| 6 | `tui/src/sysinfo.rs:62-137` | GPU/RAM: windows branches use PowerShell; **the existing `#[cfg(not(windows))]` branch only works on Linux** (`/proc/cpuinfo`, `free -m` — neither exists on macOS) | Split the not-windows branch into `target_os = "linux"` and `target_os = "macos"` (`sysctl -n hw.memsize`, `vm_stat` or `sysctl -n machdep.cpu.brand_string`) |
| 7 | `README.md` | All commands are `.\annihilation-env\Scripts\python.exe` / PowerShell; GGUF note says "automatic binary download is currently Windows-only" | Add POSIX quick-start blocks; update GGUF note once #4 lands |
| 8 | `.github/workflows/ci.yml` `tui` job | `runs-on: windows-latest` with comment "The TUI is a Windows application" | Matrix `windows-latest + ubuntu-latest + macos-latest`; keep/adjust the comment |

### Already cross-platform (do NOT touch unless required)

- `tui/src/subprocess.rs` `kill()` — `taskkill /T` on win, `pkill -P` + `child.kill()` on POSIX
  (works on both Linux & macOS). Minor: comment says "like python running under sh".
- `tui/src/app.rs:1652` — `NUL` vs `/dev/null` for curl HF model validation (both covered).
- `tui/src/main.rs`, `events.rs`, `parser.rs`, `theme.rs` — no platform code (crossterm is portable).
- `tui/Cargo.toml` / `Cargo.lock` — no target-gated deps; ratatui/crossterm/tokio/arboard all portable.
- `src/annihilate/main.py` — UTF-8 reconfigure + `KMP_DUPLICATE_LIB_OK` harmless on POSIX.
- `src/annihilate/system.py` — accelerator detection covers CUDA/ROCm/XPU/MLU/SDAA/MUSA/NPU/**MPS**.
- `scripts/gguf_converter.py` — everything except §3#4: `_download_verified` (HTTPS+sha256),
  `_safe_extract`, `_install_package` (pip→uv fallback), conversion Popen calls.
- `scripts/run_benchmarks.py`, `chat_server.py`, `upload_to_hf.py` — no platform code found.
- `pyproject.toml` `[tool.uv.sources]` — cu126 index **already gated** `sys_platform == 'win32'`;
  Linux/macOS resolve torch from PyPI (correct: Linux PyPI wheels bundle CUDA, macOS gets MPS).
- `uv.lock` — resolution-markers cover win32/emscripten/other universally; `uv sync` works on Linux today
  (the Python CI job runs ubuntu-latest and passes).
- `.gitattributes` — `* text eol=lf`; no CRLF concerns.

## 4. llama.cpp assets — verified design inputs

Latest release at recon time: **b10472** (`ggml-org/llama.cpp`). Relevant assets (all publish
`sha256:` digests in the API response, so the existing verify pattern transfers 1:1):

- `llama-<tag>-bin-ubuntu-x64.tar.gz`, `llama-<tag>-bin-ubuntu-arm64.tar.gz`
- `llama-<tag>-bin-macos-arm64.tar.gz`, `llama-<tag>-bin-macos-x64.tar.gz`
- `llama-<tag>-bin-win-cpu-x64.zip` (existing Windows path)

Port implications:

- Selection key: `platform.system()` ∈ {Windows, Linux, Darwin} × `platform.machine()`
  (x86_64/AMD64 → x64, arm64/aarch64 → arm64). Linux glibc-based (ubuntu) build is the
  sensible default; musl/Alpine is an edge case → document fallback of placing the binary
  manually under `scripts/llama_cpp_bin/`.
- Format: tar.gz → `tarfile` module with the same escape-path guard as `_safe_extract`,
  then `chmod +x llama-quantize` (tar members from GitHub arrive 0755, but set explicitly).
- Keep env pins `LLAMA_CPP_BIN_SHA256` / `LLAMA_CPP_SRC_SHA256`.

## 5. Repo conventions (must follow in the PR)

- **Commits**: Conventional Commits — `feat(tui):`, `fix(env):`, `chore(ci):`, `docs:`.
- **PR titles**: enforced semantic/conventional (`.github/workflows/semantic-pr.yml`).
- **Style** (`.gemini/styleguide.md`): one change per PR; no unrelated edits (incl. formatting);
  full type annotations; comments start capital / end with period; SPDX headers on new files;
  new Settings mirrored in `config.default.toml` (N/A for us).
- **CI gates**: ruff format/check, `ty check --error-on-warning`; cargo fmt/clippy `-D warnings`/build/test `--locked`.
- **Versions**: bumping anything requires keeping `pyproject.toml` ↔ `tui/Cargo.toml` in lockstep.
  Our port should **not** bump the version — owner decides at merge/release.
- Release pipeline (PyPI trusted publishing + GHCR `linux/amd64` bundle) is already platform-neutral.

## 6. Test assets available on this machine

- Linux x64, zsh, git+gh (Blackfrost-AI collaborator on origin) — can build TUI, run engine venv bootstrap,
  exercise Linux GGUF download path locally.
- macOS: no local box → CI `macos-latest` runner + careful `cfg` review. (Partner may have a Mac.)
