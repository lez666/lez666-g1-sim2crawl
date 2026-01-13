# 🚀 运行仿真指南

## ✅ 路径问题已修复

现在可以从项目根目录直接运行仿真，无需切换到 `sim2sim_mj/` 目录。

## 运行方式

### 方式 1: 从项目根目录运行（推荐）

```bash
conda activate g1-crawl
cd ~/lez666-g1-sim2crawl
python sim2sim_mj/run_sim2sim_keyboard.py
```

### 方式 2: 从 sim2sim_mj 目录运行

```bash
conda activate g1-crawl
cd ~/lez666-g1-sim2crawl/sim2sim_mj
python run_sim2sim_keyboard.py
```

## ⚠️ 关于策略文件

如果没有策略文件，程序会报错。您有以下选择：

### 选项 1: 使用已有策略文件

如果您有训练好的策略文件（`.pt` 格式），请复制到 `sim2sim_mj/policies/` 目录：

```bash
cp /path/to/your/policy.pt sim2sim_mj/policies/policy_shamble.pt
```

### 选项 2: 从训练日志导出

如果您有 Isaac Lab 训练日志，可以导出策略：

```bash
cd ~/workspace/IsaacLab
./isaaclab.sh -p scripts/export_mjcf.py --task g1-crawl \
  --checkpoint logs/rsl_rl/.../model_XXXX.pt
```

### 选项 3: 开始新训练

参考 `INSTALL.md` 配置 Isaac Lab 后开始训练。

## 键盘控制

程序启动后，使用以下按键控制：

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

## 故障排除

### 错误: Model XML not found

**已修复！** 现在路径会自动解析，无论从哪个目录运行。

### 错误: Policy file not found

这是正常的，需要策略文件才能运行。请参考上面的"关于策略文件"部分。

### 错误: Keyboard input not working

- 确保 MuJoCo 窗口处于焦点状态
- 检查是否安装了 `pynput`: `pip install pynput`
- 在 Linux 上，确保有 X11/Wayland 会话

## 验证安装

运行检查脚本确认一切正常：

```bash
python check_setup.py
python test_simulation.py
```

---

**提示**: 即使没有策略文件，您也可以测试仿真环境是否能正常启动（虽然会报错，但可以验证 MuJoCo 和窗口是否正常工作）。