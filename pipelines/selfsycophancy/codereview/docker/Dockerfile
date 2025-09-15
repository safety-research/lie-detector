# docker/Dockerfile
# syntax=docker/dockerfile:1
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Enable BuildKit cache mounts
ARG BUILDKIT_INLINE_CACHE=1

# Install system dependencies with cache mount
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    nodejs \
    npm \
    build-essential \
    curl \
    wget \
    vim \
    # Additional tools for SWE-Bench repositories
    libpq-dev \
    libmysqlclient-dev \
    libxml2-dev \
    libxslt1-dev \
    libssl-dev \
    libffi-dev \
    # System utilities
    curl \
    wget \
    vim \
    nano \
    emacs \
    htop \
    tree \
    jq \
    zip \
    unzip \
    tar \
    gzip \
    bzip2 \
    xz-utils \
    # Network tools
    net-tools \
    iputils-ping \
    dnsutils \
    netcat \
    telnet \
    && rm -rf /var/lib/apt/lists/*

# Create and use virtual environment for Python tools
RUN python3 -m venv /opt/python_tools
ENV PATH="/opt/python_tools/bin:$PATH"

# Upgrade pip and install Python tools with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install \
        black \
        flake8 \
        mypy \
        pytest \
        pytest-cov \
        pylint \
        autopep8 \
        # Additional tools for SWE-Bench
        tox \
        coverage \
        bandit \
        isort \
        # SWE-Bench dataset support
        datasets \
        huggingface-hub \
        # Common dependencies for SWE-Bench repos
        django \
        flask \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        sympy \
        requests

# Install Node.js tools
RUN npm install -g \
    eslint \
    prettier \
    jshint \
    typescript \
    @types/node

# Setup Inspect tool support
RUN python3 -m venv /opt/inspect_tool_support && \
    /opt/inspect_tool_support/bin/pip install --upgrade pip && \
    /opt/inspect_tool_support/bin/pip install inspect-tool-support && \
    /opt/inspect_tool_support/bin/inspect-tool-support post-install

ENV PATH="$PATH:/opt/inspect_tool_support/bin"

# Create workspace directories
RUN mkdir -p /workspace /repos /swe-bench

# Create agent user and give ownership of necessary directories
RUN useradd -m -s /bin/bash agent && \
    chown -R agent:agent /workspace /repos /swe-bench && \
    # Also give agent ownership of the virtual environment so they can install packages
    chown -R agent:agent /opt/python_tools

WORKDIR /workspace

# Switch to agent user
USER agent

# Git configuration for PR creation
RUN git config --global user.email "agent@experiment.local" && \
    git config --global user.name "Experiment Agent" && \
    git config --global init.defaultBranch main

# Ensure the virtual environment is activated for the agent user
ENV VIRTUAL_ENV=/opt/python_tools
ENV PATH="/opt/python_tools/bin:$PATH"