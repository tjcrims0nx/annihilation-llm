#!/bin/bash
set -e

echo "==============================================="
echo "   ANNIHILATION LLM - UBUNTU/LINUX LAUNCHER    "
echo "==============================================="

# 1. Check if Rust is installed via rustup (not apt!)
if ! command -v rustup &> /dev/null; then
    echo "⚠️  rustup could not be found."
    echo "Installing the latest secure Rust toolchain (required for Rust 2024 edition)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    
    # Source the cargo env to use it immediately in this session
    source "$HOME/.cargo/env"
    echo "✅ Rust installed successfully!"
else
    echo "✅ rustup is installed. Ensuring you have the latest stable toolchain..."
    rustup update stable
fi

# 2. Verify Python 3.10+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Please run: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

echo "✅ Environment checks passed."
echo "🚀 Compiling and launching the TUI..."
echo "==============================================="

# 3. Launch the app (Cargo will automatically handle the build if needed)
# We go into the tui directory (if we aren't already there) or just run it if the workspace is setup
if [ -d "tui" ]; then
    cd tui
    cargo run --release
else
    cargo run --release
fi
