#!/bin/bash
set -e

INSTALL_DIR="$HOME/annihilation-llm"
echo -e "\033[0;36mInstalling ANNIHILATE to $INSTALL_DIR...\033[0m"

if [ -d "$INSTALL_DIR" ]; then
    echo "Directory already exists. Updating repository..."
    cd "$INSTALL_DIR"
    git pull
else
    git clone https://github.com/tjcrims0nx/annihilation-llm.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo -e "\033[0;36mFetching latest release binary...\033[0m"
OS=$(uname -s)
if [ "$OS" = "Darwin" ]; then
    ASSET_NAME="annihilate-macos"
else
    ASSET_NAME="annihilate-linux"
fi

DOWNLOAD_URL=$(curl -s https://api.github.com/repos/tjcrims0nx/annihilation-llm/releases/latest | grep "browser_download_url.*$ASSET_NAME" | cut -d '"' -f 4)

if [ -n "$DOWNLOAD_URL" ]; then
    mkdir -p tui/target/release
    curl -sSL "$DOWNLOAD_URL" -o tui/target/release/annihilate
    chmod +x tui/target/release/annihilate
    echo -e "\033[0;32mBinary downloaded successfully.\033[0m"
else
    echo -e "\033[0;33mCould not find binary in the latest release. You may need to run 'cargo build' manually.\033[0m"
fi

echo -e "\033[0;32mInstallation complete! Run 'cd ~/annihilation-llm && ./start.sh' to begin.\033[0m"
