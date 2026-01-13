#!/usr/bin/env python3
"""快速检查项目安装状态"""

import sys
import importlib
from pathlib import Path

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def check_package(name, import_name=None):
    """检查Python包是否安装"""
    if import_name is None:
        import_name = name
    
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

def check_file(path):
    """检查文件是否存在"""
    return Path(path).exists()

def main():
    print(f"{Colors.BLUE}{'='*60}")
    print("G1 Sim2Crawl 安装检查")
    print(f"{'='*60}{Colors.END}\n")
    
    # 检查项目结构
    print(f"{Colors.BLUE}[1] 检查项目结构{Colors.END}")
    project_root = Path(__file__).parent
    required_dirs = [
        "sim2sim_mj",
        "source/g1_crawl",
        "scripts/rsl_rl"
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  {Colors.GREEN}✓{Colors.END} {dir_path}")
        else:
            print(f"  {Colors.RED}✗{Colors.END} {dir_path} (缺失)")
            all_ok = False
    
    print()
    
    # 检查仿真依赖
    print(f"{Colors.BLUE}[2] 检查仿真依赖 (MuJoCo){Colors.END}")
    sim_packages = {
        "mujoco": "mujoco",
        "torch": "torch",
        "numpy": "numpy",
        "pynput": "pynput",
        "glfw": "glfw",
    }
    
    sim_ok = True
    for pkg_name, import_name in sim_packages.items():
        installed, version = check_package(pkg_name, import_name)
        if installed:
            print(f"  {Colors.GREEN}✓{Colors.END} {pkg_name} (版本: {version})")
        else:
            print(f"  {Colors.RED}✗{Colors.END} {pkg_name} (未安装)")
            sim_ok = False
    
    print()
    
    # 检查关键文件
    print(f"{Colors.BLUE}[3] 检查关键文件{Colors.END}")
    key_files = [
        "sim2sim_mj/run_sim2sim_keyboard.py",
        "sim2sim_mj/scene.xml",
        "sim2sim_mj/requirements.txt",
    ]
    
    files_ok = True
    for file_path in key_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  {Colors.GREEN}✓{Colors.END} {file_path}")
        else:
            print(f"  {Colors.RED}✗{Colors.END} {file_path} (缺失)")
            files_ok = False
    
    print()
    
    # 检查策略文件
    print(f"{Colors.BLUE}[4] 检查策略文件{Colors.END}")
    policies_dir = project_root / "sim2sim_mj" / "policies"
    if policies_dir.exists():
        policy_files = list(policies_dir.glob("*.pt"))
        if policy_files:
            print(f"  {Colors.GREEN}✓{Colors.END} 找到 {len(policy_files)} 个策略文件:")
            for pf in policy_files:
                print(f"      - {pf.name}")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.END} 策略目录存在但无策略文件")
            print(f"      (训练完成后需要导出策略)")
    else:
        print(f"  {Colors.YELLOW}⚠{Colors.END} 策略目录不存在 (首次运行会自动创建)")
    
    print()
    
    # 检查Isaac Lab (可选)
    print(f"{Colors.BLUE}[5] 检查训练环境 (Isaac Lab){Colors.END}")
    isaac_path = Path.home() / "workspace" / "IsaacLab"
    if isaac_path.exists():
        print(f"  {Colors.GREEN}✓{Colors.END} Isaac Lab 目录存在: {isaac_path}")
        
        # 检查扩展链接
        ext_link = isaac_path / "source" / "extensions" / "g1_crawl"
        if ext_link.exists() or ext_link.is_symlink():
            print(f"  {Colors.GREEN}✓{Colors.END} g1_crawl 扩展已链接")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.END} g1_crawl 扩展未链接")
            print(f"      运行: ln -s {project_root}/source/g1_crawl {ext_link}")
    else:
        print(f"  {Colors.YELLOW}⚠{Colors.END} Isaac Lab 未找到 (仅仿真模式)")
        print(f"      如需训练，请参考 INSTALL.md")
    
    print()
    
    # 总结
    print(f"{Colors.BLUE}{'='*60}")
    print("检查总结")
    print(f"{'='*60}{Colors.END}\n")
    
    if sim_ok and files_ok:
        print(f"{Colors.GREEN}✓ 仿真环境就绪！{Colors.END}")
        print(f"\n可以运行: {Colors.BLUE}python sim2sim_mj/run_sim2sim_keyboard.py{Colors.END}")
    else:
        print(f"{Colors.RED}✗ 仿真环境未完全配置{Colors.END}")
        print(f"\n请运行: {Colors.BLUE}./setup.sh{Colors.END} 或参考 INSTALL.md")
    
    if isaac_path.exists():
        print(f"\n{Colors.GREEN}✓ 训练环境已配置{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}⚠ 训练环境未配置 (可选){Colors.END}")
    
    print()

if __name__ == "__main__":
    main()