#!/bin/bash

# ==============================================================================
# Environment Setup Script
# 用于快速配置 Python 依赖环境
# ==============================================================================

echo "----------------------------------------------------------------"
echo "   🛠️  Setting up Python Environment..."
echo "----------------------------------------------------------------"

# 1. 安装基础依赖
echo "[1/3] Installing base dependencies..."
pip install packaging einops tqdm Pillow

# 2. 安装 PyTorch (CUDA 11.8)
# 如果已安装则跳过，或者可以强制更新
echo "[2/3] Installing PyTorch (2.0.0 + CUDA 11.8)..."
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. 安装 Mamba & Causal Conv1d
echo "[3/3] Installing Mamba SSM & Causal Conv1d..."

# 检查当前目录下是否有预下载的 .whl 文件 (文件名可能根据版本不同)
# Mamba 在 Windows 下编译困难，强烈建议放入对应的 whl 文件
CAUSAL_WHL=$(find . -maxdepth 1 -name "causal_conv1d*.whl" | head -n 1)
MAMBA_WHL=$(find . -maxdepth 1 -name "mamba_ssm*.whl" | head -n 1)

if [ -n "$CAUSAL_WHL" ]; then
    echo "   -> Found local causal_conv1d: $CAUSAL_WHL"
    pip install "$CAUSAL_WHL"
else
    echo "   -> Local causal_conv1d whl not found. Attempting install from PyPI (May fail on Windows)..."
    pip install causal_conv1d==1.1.3 || echo "   ⚠️ Install failed. Please download the .whl file manually."
fi

if [ -n "$MAMBA_WHL" ]; then
    echo "   -> Found local mamba_ssm: $MAMBA_WHL"
    pip install "$MAMBA_WHL"
else
    echo "   -> Local mamba_ssm whl not found. Attempting install from PyPI (May fail on Windows)..."
    pip install mamba_ssm==1.1.3 || echo "   ⚠️ Install failed. Please download the .whl file manually."
fi

echo "----------------------------------------------------------------"
echo "✅ Setup complete (with possible warnings)."
echo "----------------------------------------------------------------"