# 🚀 快速开始指南

## 前置条件

- 已安装 Conda (Miniconda 或 Anaconda)
- Git

## 步骤 1: 克隆项目

```bash
git clone <your-repo-url>
cd lez666-g1-sim2crawl
```

## 方式一：自动安装（推荐）

```bash
# 1. 运行自动安装脚本
./setup.sh

# 2. 选择安装类型（1=仅仿真，2=完整安装）

# 3. 激活环境
conda activate g1-crawl

# 4. 运行仿真
python sim2sim_mj/run_sim2sim_keyboard.py
```

## 方式二：手动安装

### 仅仿真（快速）

```bash
# 1. 创建conda环境
conda create -n g1-crawl python=3.10 -y
conda activate g1-crawl

# 2. 安装依赖
pip install -r sim2sim_mj/requirements.txt

# 3. 运行仿真
python sim2sim_mj/run_sim2sim_keyboard.py
```

### 完整安装（训练+仿真）

请参考 `INSTALL.md` 文件中的详细步骤。

## 验证安装

```bash
# 运行检查脚本
python check_setup.py
```

## 下一步

1. **运行仿真**: `python sim2sim_mj/run_sim2sim_keyboard.py`
2. **查看文档**: `INSTALL.md` 或 `sim2sim_mj/README_keyboard_zh.md`
3. **开始训练**: 参考 `INSTALL.md` 中的训练部分

## 需要帮助？

- 查看 `INSTALL.md` 获取详细安装说明
- 查看 `README.md` 了解项目概述
- 运行 `python check_setup.py` 检查安装状态