#!/bin/sh
# Linux and macOS equivalent of start.bat. The engine's Python environment is
# created and verified automatically by the TUI on first run.
cd "$(dirname "$0")/tui" && cargo run --release
