"""
Run all SeismicLab Python demos.

转换自 MATLAB: SeismicLab_demos/
"""

from __future__ import annotations

import sys
from pathlib import Path


def run_demo(demo_file: str):
    """
    Run a single demo script.

    Parameters
    ----------
    demo_file : str
        Path to the demo Python file
    """
    import subprocess
    import sys

    demo_path = Path(__file__).parent / demo_file
    if not demo_path.exists():
        print(f"❌ Demo file not found: {demo_path}")
        return False

    try:
        print(f"\n{'='*70}")
        print(f"Running: {demo_file}")
        print('='*70)
        result = subprocess.run(
            [sys.executable, str(demo_path)],
            capture_output=False,
            cwd=str(demo_path.parent)
        )
        success = result.returncode == 0
        if success:
            print(f"✅ {demo_file} completed successfully")
        return success
    except Exception as e:
        print(f"❌ {demo_file} failed with error:")
        print(f"   {type(e).__name__}: {e}")
        return False


def main():
    """
    Run all demo scripts.
    """
    demos = [
        # 基础演示
        ("fx_decon_demo.py", "FX 反褶积去噪"),
        ("med_demo.py", "中值滤波去噪"),
        ("moveout_demo.py", "动校正演示"),
        ("parabolic_moveout_demo.py", "抛物线时差校正"),
        ("pocs_demo.py", "凸集投影"),

        # Radon 变换
        ("radon_demo_1.py", "Radon 变换去多次波"),
        ("radon_demo_2.py", "Radon 变换重建"),

        # 反褶积
        ("sparse_decon_demo.py", "稀疏反褶积"),
        ("spiking_decon_demo.py", "尖脉冲反褶积"),

        # 其他
        ("spitz_demo.py", "Spitz 插值"),
        ("va_demo.py", "速度分析"),
    ]

    print("=" * 70)
    print("SeismicLab Python 演示脚本集合")
    print("=" * 70)
    print(f"总演示数: {len(demos)}")
    print()

    results = []
    for demo_file, demo_desc in demos:
        print(f"\n📋 {demo_desc} ({demo_file})")
        success = run_demo(demo_file)
        results.append((demo_file, demo_desc, success))

    # Summary
    print("\n" + "=" * 70)
    print("演示脚本运行总结")
    print("=" * 70)

    successful = sum(1 for _, _, success in results if success)
    failed = len(results) - successful

    for demo_file, demo_desc, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{status:10} - {demo_desc} ({demo_file})")

    print()
    print(f"总计: {len(results)} 个演示")
    print(f"成功: {successful} 个")
    print(f"失败: {failed} 个")
    print(f"成功率: {100 * successful / len(results):.1f}%")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    success = main()
    sys.exit(0 if success else 1)
