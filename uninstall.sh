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

echo -e "\033[0;33mDo you also want to clear downloaded HuggingFace Models? (This may free up gigabytes of space) [y/N]\033[0m"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    CACHE_PATH="$HOME/.cache/huggingface/hub"
    if [ -d "$CACHE_PATH" ]; then
        echo "Clearing HuggingFace model cache..."
        rm -rf "$CACHE_PATH"
        echo -e "\033[0;32mModel cache cleared.\033[0m"
    else
        echo "No HuggingFace cache found."
    fi
fi

echo -e "\033[0;32mANNIHILATE has been completely uninstalled.\033[0m"
