# G1 Sim2Crawl 安装配置指南

本指南将帮助您安装和配置项目，以便进行训练和仿真。

## 📋 目录

1. [系统要求](#系统要求)
2. [快速开始（仅仿真）](#快速开始仅仿真)
3. [完整安装（训练+仿真）](#完整安装训练仿真)
4. [验证安装](#验证安装)
5. [常见问题](#常见问题)

---

## 系统要求

### 基础要求
- **操作系统**: Linux (Ubuntu 20.04/22.04 推荐) 或 Windows
- **Python**: 3.10 或 3.11
- **Conda**: Miniconda 或 Anaconda
- **GPU**: NVIDIA GPU (推荐，用于训练)
- **内存**: 至少 16GB RAM

### 仅仿真（MuJoCo）
- Python 3.10+
- CUDA (可选，用于PyTorch GPU加速)

### 训练（Isaac Lab）
- Isaac Sim 2023.1.1 或更高版本
- NVIDIA GPU with CUDA support
- 至少 8GB VRAM

---

## 快速开始（仅仿真）

如果您只想运行仿真而不进行训练，可以跳过Isaac Lab的安装。

### 1. 创建Conda环境

```bash
# 创建conda环境
conda create -n g1-crawl python=3.10 -y
conda activate g1-crawl
```

### 2. 安装仿真依赖

```bash
# 进入项目目录（替换为您的实际路径）
cd /path/to/lez666-g1-sim2crawl

# 安装MuJoCo仿真依赖
pip install -r sim2sim_mj/requirements.txt
```

### 3. 验证仿真安装

```bash
# 测试键盘控制仿真
conda activate g1-crawl
python sim2sim_mj/run_sim2sim_keyboard.py
```

如果看到MuJoCo窗口打开，说明安装成功！

---

## 完整安装（训练+仿真）

### 步骤 1: 安装 Isaac Sim

1. **下载 Isaac Sim**
   - 访问: https://developer.nvidia.com/isaac-sim
   - 下载并安装 Isaac Sim 2023.1.1 或更高版本
   - 记录安装路径（例如: `/home/user/isaac-sim`）

2. **设置环境变量**
   
   ```bash
   # 添加到 ~/.bashrc 或 ~/.zshrc
   export ISAAC_PATH=/home/user/isaac-sim  # 替换为您的实际路径
   ```

### 步骤 2: 安装 Isaac Lab

1. **克隆 Isaac Lab 仓库**
   
   ```bash
   # 选择一个工作目录
   cd ~/workspace
   
   # 克隆 Isaac Lab
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   ```

2. **安装 Isaac Lab**
   
   ```bash
   # 使用 Isaac Sim 的 Python 环境安装
   ./isaaclab.sh -p -m pip install -e .
   ```

3. **验证 Isaac Lab 安装**
   
   ```bash
   # 测试导入
   ./isaaclab.sh -p -c "import isaaclab; print('Isaac Lab installed successfully!')"
   ```

### 步骤 3: 配置 G1 Crawl 扩展

1. **创建符号链接或复制扩展**
   
   ```bash
   # 方法1: 创建符号链接（推荐）
   cd ~/workspace/IsaacLab
   ln -s /home/wasabi/lez666-g1-sim2crawl/source/g1_crawl source/extensions/g1_crawl
   
   # 方法2: 或者直接复制
   # cp -r /home/wasabi/lez666-g1-sim2crawl/source/g1_crawl source/extensions/
   ```

2. **安装扩展**
   
   ```bash
   cd ~/workspace/IsaacLab
   ./isaaclab.sh -p -m pip install -e source/extensions/g1_crawl
   ```

### 步骤 4: 创建 Conda 环境（用于仿真）

```bash
# 创建conda环境
conda create -n g1-crawl python=3.10 -y
conda activate g1-crawl

# 安装仿真依赖
cd /home/wasabi/lez666-g1-sim2crawl
pip install -r sim2sim_mj/requirements.txt
```

### 步骤 5: 安装训练依赖

训练依赖会通过 Isaac Lab 自动管理，但您可能需要安装额外的包：

```bash
# 使用 Isaac Sim 的 Python 环境
cd ~/workspace/IsaacLab
./isaaclab.sh -p -m pip install psutil
```

---

## 验证安装

### 验证仿真

```bash
# 激活conda环境
conda activate g1-crawl

# 运行键盘控制仿真
cd /home/wasabi/lez666-g1-sim2crawl
python sim2sim_mj/run_sim2sim_keyboard.py
```

**预期结果**: 
- MuJoCo 窗口打开
- 机器人出现在场景中
- 可以使用键盘控制（方向键、Z/C、IJK等）

### 验证训练环境

```bash
# 使用 Isaac Lab 的 Python 环境
cd ~/workspace/IsaacLab

# 列出可用环境
./isaaclab.sh -p scripts/list_envs.py

# 应该能看到 g1-crawl 相关的任务
```

**预期结果**: 
- 看到 "Isaac-*" 开头的任务列表
- 包含 g1-crawl 相关任务

### 运行训练测试

```bash
cd ~/workspace/IsaacLab

# 运行一个简短的训练测试（headless模式）
./isaaclab.sh -p scripts/rsl_rl/train.py --task g1-crawl --headless --max_iterations 10
```

**预期结果**: 
- 训练开始运行
- 没有错误信息
- 日志文件在 `logs/rsl_rl/` 目录下创建

---

## 运行训练

### 基础训练命令

```bash
cd ~/workspace/IsaacLab

# Headless 训练（无GUI，推荐用于服务器）
./isaaclab.sh -p scripts/rsl_rl/train.py --task g1-crawl --headless

# 带GUI的训练（用于调试）
./isaaclab.sh -p scripts/rsl_rl/train.py --task g1-crawl --gui

# 指定训练迭代次数
./isaaclab.sh -p scripts/rsl_rl/train.py --task g1-crawl --headless --max_iterations 5000

# 指定环境数量
./isaaclab.sh -p scripts/rsl_rl/train.py --task g1-crawl --headless --num_envs 4096
```

### 训练参数

- `--task g1-crawl`: 指定任务名称
- `--headless`: 无GUI模式（推荐）
- `--gui`: 启用GUI（用于调试）
- `--max_iterations N`: 最大训练迭代次数
- `--num_envs N`: 并行环境数量
- `--video`: 录制训练视频
- `--seed N`: 随机种子

### 训练输出

- **日志目录**: `logs/rsl_rl/{experiment_name}/{timestamp}_{run_name}/`
- **检查点**: `logs/rsl_rl/{experiment_name}/{timestamp}_{run_name}/model_{iteration}.pt`
- **TensorBoard**: 在日志目录中运行 `tensorboard --logdir logs/rsl_rl/`

---

## 运行仿真

### 键盘控制仿真

```bash
# 激活conda环境
conda activate g1-crawl

# 进入项目目录（替换为您的实际路径）
cd /path/to/lez666-g1-sim2crawl

# 运行键盘控制
python sim2sim_mj/run_sim2sim_keyboard.py
```

### 手柄控制仿真

```bash
conda activate g1-crawl
cd /path/to/lez666-g1-sim2crawl
python sim2sim_mj/run_sim2sim.py
```

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| ↑/↓ | 前进/后退 |
| ←/→ | 左/右平移 |
| Z/C | 左/右旋转 |
| I | 站立模式 |
| J | 阻尼模式 |
| K | 爬行模式 |
| SPACE | 切换策略 |
| Q/A | 增加/减少增益 |
| H | 显示增益 |
| ESC | 退出 |

---

## 导出策略到 MuJoCo

训练完成后，需要将策略导出为 MuJoCo 可用的格式：

```bash
cd ~/workspace/IsaacLab

# 导出策略（需要根据实际路径调整）
./isaaclab.sh -p scripts/export_mjcf.py --task g1-crawl --checkpoint logs/rsl_rl/.../model_XXXX.pt
```

然后将导出的策略文件复制到 `sim2sim_mj/policies/` 目录。

---

## 常见问题

### 1. Conda 环境问题

**问题**: `conda activate g1-crawl` 失败

**解决**:
```bash
# 初始化conda
conda init bash  # 或 conda init zsh
# 重新打开终端
```

### 2. Isaac Sim 路径问题

**问题**: 找不到 Isaac Sim

**解决**:
```bash
# 检查环境变量
echo $ISAAC_PATH

# 如果没有设置，添加到 ~/.bashrc
export ISAAC_PATH=/path/to/isaac-sim
```

### 3. MuJoCo 依赖问题

**问题**: `mujoco` 安装失败

**解决**:
```bash
# 确保使用正确的Python版本
python --version  # 应该是 3.10 或 3.11

# 升级pip
pip install --upgrade pip

# 重新安装
pip install -r sim2sim_mj/requirements.txt
```

### 4. 训练时内存不足

**问题**: CUDA out of memory

**解决**:
```bash
# 减少并行环境数量
./isaaclab.sh -p scripts/rsl_rl/train.py --task g1-crawl --headless --num_envs 2048
```

### 5. 键盘控制无响应

**问题**: 键盘输入无响应

**解决**:
- 确保 MuJoCo 窗口处于焦点状态
- 检查是否安装了 `pynput`: `pip install pynput`
- 在 Linux 上，确保有 X11/Wayland 会话

### 6. 策略文件找不到

**问题**: `policy not found`

**解决**:
- 检查 `sim2sim_mj/policies/` 目录是否存在策略文件
- 确认策略文件名与配置中的路径匹配
- 如果训练完成，需要先导出策略

---

## 下一步

1. **开始训练**: 运行训练命令，等待策略收敛
2. **导出策略**: 将训练好的策略导出为 `.pt` 文件
3. **测试仿真**: 在 MuJoCo 中测试导出的策略
4. **调整参数**: 根据需要调整速度限制、增益等参数

---

## 获取帮助

- **项目 README**: `README.md`
- **键盘控制文档**: `sim2sim_mj/README_keyboard_zh.md`
- **Isaac Lab 文档**: https://isaac-sim.github.io/IsaacLab/

---

*最后更新: 2025-01*