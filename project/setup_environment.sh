#!/bin/bash
#
# Python环境配置脚本
# 适用于Ubuntu 22.04+ / WSL2
#
# 使用方法:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "VIV预测项目 - Python环境配置"
echo "========================================================================"
echo ""

# 检查是否有sudo权限
if ! sudo -n true 2>/dev/null; then
    echo "需要sudo权限来安装系统包。"
    echo "请输入密码以继续..."
fi

echo ""
echo "步骤1/4: 更新包列表..."
sudo apt update

echo ""
echo "步骤2/4: 安装Python基础工具..."
sudo apt install -y python3-pip python3-venv python3-dev build-essential

echo ""
echo "步骤3/4: 安装科学计算包（系统级，速度快）..."
sudo apt install -y \
    python3-numpy \
    python3-pandas \
    python3-sklearn \
    python3-matplotlib \
    python3-seaborn \
    python3-scipy

echo ""
echo "步骤4/4: 安装额外的Python包（用户级）..."
pip3 install --user imbalanced-learn tabulate

echo ""
echo "========================================================================"
echo "环境配置完成！"
echo "========================================================================"
echo ""
echo "验证安装..."
python3 << 'PYEOF'
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import matplotlib
    import seaborn
    import imblearn

    print("✓ numpy:", np.__version__)
    print("✓ pandas:", pd.__version__)
    print("✓ scikit-learn:", sklearn.__version__)
    print("✓ matplotlib:", matplotlib.__version__)
    print("✓ seaborn:", seaborn.__version__)
    print("✓ imbalanced-learn:", imblearn.__version__)
    print("")
    print("所有依赖包安装成功！")
except ImportError as e:
    print("✗ 安装失败:", e)
    exit(1)
PYEOF

echo ""
echo "========================================================================"
echo "下一步："
echo "  运行SMOTE实验: python3 -m src.imbalance_experiments"
echo "========================================================================"
