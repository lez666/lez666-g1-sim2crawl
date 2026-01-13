#!/bin/bash
# G1 Sim2Crawl 自动安装脚本
# 此脚本将帮助您快速设置项目环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "G1 Sim2Crawl 安装脚本"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在项目根目录
if [ ! -f "README.md" ] || [ ! -d "sim2sim_mj" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    exit 1
fi

# 获取项目根目录
PROJECT_ROOT=$(pwd)
echo -e "${GREEN}项目目录: ${PROJECT_ROOT}${NC}"
echo ""

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误: 未找到 conda，请先安装 Miniconda 或 Anaconda${NC}"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓ 找到 conda${NC}"

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo -e "${GREEN}Python 版本: ${PYTHON_VERSION}${NC}"

# 询问安装类型
echo ""
echo "请选择安装类型:"
echo "1) 仅仿真 (MuJoCo) - 快速安装，无需 Isaac Sim"
echo "2) 完整安装 (训练 + 仿真) - 需要 Isaac Sim"
read -p "请输入选项 [1/2]: " install_type

if [ "$install_type" != "1" ] && [ "$install_type" != "2" ]; then
    echo -e "${RED}无效选项，退出${NC}"
    exit 1
fi

# 步骤 1: 创建/激活 conda 环境
echo ""
echo "=========================================="
echo "步骤 1: 设置 Conda 环境"
echo "=========================================="

ENV_NAME="g1-crawl"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}环境 ${ENV_NAME} 已存在${NC}"
    read -p "是否重新创建? [y/N]: " recreate
    if [ "$recreate" = "y" ] || [ "$recreate" = "Y" ]; then
        echo "删除现有环境..."
        conda env remove -n ${ENV_NAME} -y
        echo "创建新环境..."
        conda create -n ${ENV_NAME} python=3.10 -y
    else
        echo "使用现有环境"
    fi
else
    echo "创建 conda 环境: ${ENV_NAME}"
    conda create -n ${ENV_NAME} python=3.10 -y
fi

echo -e "${GREEN}✓ Conda 环境准备完成${NC}"

# 激活环境并安装依赖
echo ""
echo "=========================================="
echo "步骤 2: 安装仿真依赖"
echo "=========================================="

# 注意: 在脚本中激活conda环境需要使用source
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "安装 MuJoCo 仿真依赖..."
pip install --upgrade pip
pip install -r sim2sim_mj/requirements.txt

echo -e "${GREEN}✓ 仿真依赖安装完成${NC}"

# 如果是完整安装，提供 Isaac Lab 安装指导
if [ "$install_type" = "2" ]; then
    echo ""
    echo "=========================================="
    echo "步骤 3: Isaac Lab 安装指导"
    echo "=========================================="
    echo ""
    echo -e "${YELLOW}完整安装需要手动配置 Isaac Lab:${NC}"
    echo ""
    echo "1. 安装 Isaac Sim:"
    echo "   - 访问: https://developer.nvidia.com/isaac-sim"
    echo "   - 下载并安装 Isaac Sim 2023.1.1+"
    echo ""
    echo "2. 设置环境变量:"
    echo "   export ISAAC_PATH=/path/to/isaac-sim"
    echo ""
    echo "3. 克隆并安装 Isaac Lab:"
    echo "   cd ~/workspace"
    echo "   git clone https://github.com/isaac-sim/IsaacLab.git"
    echo "   cd IsaacLab"
    echo "   ./isaaclab.sh -p -m pip install -e ."
    echo ""
    echo "4. 配置扩展:"
    echo "   cd ~/workspace/IsaacLab"
    echo "   ln -s ${PROJECT_ROOT}/source/g1_crawl source/extensions/g1_crawl"
    echo "   ./isaaclab.sh -p -m pip install -e source/extensions/g1_crawl"
    echo ""
    echo "详细说明请参考: ${PROJECT_ROOT}/INSTALL.md"
    echo ""
fi

# 验证安装
echo ""
echo "=========================================="
echo "步骤 4: 验证安装"
echo "=========================================="

echo "检查 Python 包..."

# 检查关键包
PACKAGES=("mujoco" "torch" "numpy" "pynput")
MISSING_PACKAGES=()

for pkg in "${PACKAGES[@]}"; do
    if python -c "import ${pkg}" 2>/dev/null; then
        echo -e "${GREEN}✓ ${pkg}${NC}"
    else
        echo -e "${RED}✗ ${pkg} (未安装)${NC}"
        MISSING_PACKAGES+=("${pkg}")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "安装完成！"
    echo "==========================================${NC}"
    echo ""
    echo "下一步:"
    echo "1. 激活环境: conda activate ${ENV_NAME}"
    echo "2. 运行仿真: python sim2sim_mj/run_sim2sim_keyboard.py"
    echo ""
    if [ "$install_type" = "2" ]; then
        echo "3. 配置 Isaac Lab (参考上面的指导)"
        echo "4. 运行训练: cd ~/workspace/IsaacLab && ./isaaclab.sh -p scripts/rsl_rl/train.py --task g1-crawl --headless"
    fi
    echo ""
else
    echo ""
    echo -e "${YELLOW}警告: 部分包未正确安装${NC}"
    echo "请手动安装: pip install ${MISSING_PACKAGES[*]}"
    echo ""
fi

echo "安装脚本完成！"