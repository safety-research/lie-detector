#!/bin/bash
set -e

echo "Setting up repository..."

# Use home directory which should exist
cd ~

# Extract the repository archive
tar -xzf /tmp/repo.tar.gz

# Setup git in the repo
cd ~/repo
if [ ! -d .git ]; then
    git init
    git add -A
    git commit -m "Initial commit from preloaded state"
fi

# Configure git
git config --global user.email "agent@experiment.local"
git config --global user.name "Experiment Agent"
git branch -M main

echo "Repository ready at ~/repo"
pwd  # Show actual path for debugging