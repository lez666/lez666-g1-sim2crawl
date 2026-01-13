# ✅ 安装完成！

## 安装状态

✅ **仿真环境已配置完成！**

- ✓ Conda 环境: `g1-crawl` (Python 3.11)
- ✓ MuJoCo 3.4.0
- ✓ PyTorch 2.9.1
- ✓ 所有依赖包已安装
- ✓ 项目文件完整

## 当前状态

### ✅ 已就绪
- 仿真环境完全配置
- 所有Python包已安装
- 项目文件结构完整
- 配置文件正常

### ⚠️ 需要策略文件
- 策略目录 `sim2sim_mj/policies/` 为空
- 需要以下策略文件之一：
  - `policy_shamble.pt` (站立/行走策略)
  - `policy_crawl_start.pt` (爬行启动策略)
  - `policy_crawl.pt` (爬行策略)

## 下一步操作

### 选项 1: 使用已有策略文件（如果有）

如果您有训练好的策略文件，请将其复制到 `sim2sim_mj/policies/` 目录：

```bash
# 复制策略文件
cp /path/to/your/policy.pt sim2sim_mj/policies/policy_shamble.pt
```

### 选项 2: 从训练日志导出策略

如果您有训练日志，可以从检查点导出策略：

```bash
# 在 Isaac Lab 环境中
cd ~/workspace/IsaacLab
./isaaclab.sh -p scripts/export_mjcf.py --task g1-crawl --checkpoint logs/rsl_rl/.../model_XXXX.pt
```

### 选项 3: 开始训练新策略

如果您还没有策略文件，可以开始训练：

```bash
# 需要先配置 Isaac Lab（参考 INSTALL.md）
cd ~/workspace/IsaacLab
./isaaclab.sh -p scripts/rsl_rl/train.py --task g1-crawl --headless
```

## 运行仿真

### 测试运行（即使没有策略文件）

即使没有策略文件，您也可以测试仿真环境是否能正常启动：

```bash
conda activate g1-crawl
cd /home/wasabi/lez666-g1-sim2crawl
python sim2sim_mj/run_sim2sim_keyboard.py
```

**注意**: 如果没有策略文件，程序会报错，但您可以验证：
- MuJoCo 是否能正常加载
- 窗口是否能正常显示
- 键盘输入是否能正常响应

### 完整运行（需要策略文件）

```bash
conda activate g1-crawl
cd /home/wasabi/lez666-g1-sim2crawl
python sim2sim_mj/run_sim2sim_keyboard.py
```

## 键盘控制

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

## 验证安装

随时可以运行检查脚本：

```bash
conda activate g1-crawl
python check_setup.py
```

## 获取帮助

- **详细安装说明**: `INSTALL.md`
- **快速开始**: `QUICKSTART.md`
- **键盘控制说明**: `sim2sim_mj/README_keyboard_zh.md`
- **项目概述**: `README.md`

## 训练环境（可选）

如果您想进行训练，还需要配置 Isaac Lab：

1. 安装 Isaac Sim
2. 克隆并安装 Isaac Lab
3. 配置 g1_crawl 扩展

详细步骤请参考 `INSTALL.md` 中的"完整安装"部分。

---

**安装完成时间**: $(date)
**环境**: g1-crawl (Python 3.11)