#!/usr/bin/env python3
"""快速测试仿真环境是否能正常加载"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "sim2sim_mj"))

try:
    print("=" * 60)
    print("测试仿真环境")
    print("=" * 60)
    print()
    
    # 测试导入
    print("[1] 测试包导入...")
    import mujoco
    import torch
    import numpy as np
    import pynput
    print("  ✓ 所有包导入成功")
    print()
    
    # 测试模型加载
    print("[2] 测试模型加载...")
    scene_xml = Path("sim2sim_mj/scene.xml")
    if scene_xml.exists():
        model = mujoco.MjModel.from_xml_path(str(scene_xml.absolute()))
        print(f"  ✓ 模型加载成功 (nq={model.nq}, nv={model.nv}, nu={model.nu})")
    else:
        print("  ✗ scene.xml 未找到")
    print()
    
    # 测试配置文件
    print("[3] 测试配置文件...")
    from run_sim2sim_keyboard import CONFIG
    print(f"  ✓ 配置文件加载成功")
    print(f"    模型文件: {CONFIG['model_xml']}")
    print(f"    初始策略: {CONFIG['policy_path']}")
    print()
    
    # 检查策略文件
    print("[4] 检查策略文件...")
    policy_path = Path("sim2sim_mj") / CONFIG["policy_path"]
    if policy_path.exists():
        print(f"  ✓ 策略文件存在: {policy_path}")
    else:
        print(f"  ⚠ 策略文件不存在: {policy_path}")
        print("     需要策略文件才能运行完整仿真")
    print()
    
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)
    print()
    print("如果所有测试通过，可以运行:")
    print("  python sim2sim_mj/run_sim2sim_keyboard.py")
    print()
    
except Exception as e:
    print(f"  ✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
