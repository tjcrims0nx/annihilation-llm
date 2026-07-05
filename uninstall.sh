#!/bin/bash
set -e

INSTALL_DIR="$HOME/annihilation-llm"

echo -e "\033[0;36mUninstalling ANNIHILATE...\033[0m"

echo "Removing Python environments..."
for env in "annihilation-env" ".venv" "venv" "env"; do
    if [ -d "$INSTALL_DIR/$env" ]; then
        echo "Deleting $INSTALL_DIR/$env..."
        rm -rf "$INSTALL_DIR/$env"
    fi
done

echo -e "\033[0;32mANNIHILATE environments have been safely uninstalled.\033[0m"
