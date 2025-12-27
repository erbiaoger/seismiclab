"""
Generate all figures from demo scripts and save to docs/figs directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
import subprocess
import shutil

# Demo scripts to run
DEMOS = [
    "fx_decon_demo.py",
    "med_demo.py",
    "moveout_demo.py",
    "parabolic_moveout_demo.py",
    "pocs_demo.py",
    "radon_demo_1.py",
    "radon_demo_2.py",
    "sparse_decon_demo.py",
    "spiking_decon_demo.py",
    "spitz_demo.py",
    "va_demo.py",
]


def run_demo(demo_file: str) -> bool:
    """Run a single demo script and return success status."""
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
            capture_output=True,
            cwd=str(demo_path.parent),
            text=True
        )
        if result.returncode == 0:
            print(f"✅ {demo_file} completed successfully")
            return True
        else:
            print(f"❌ {demo_file} failed with return code {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ {demo_file} failed with error:")
        print(f"   {type(e).__name__}: {e}")
        return False


def main():
    """Run all demos and collect generated figures."""
    examples_dir = Path(__file__).parent
    figs_dir = examples_dir / "figs"
    docs_figs_dir = examples_dir.parent / "docs" / "figs"

    # Create figs directories
    figs_dir.mkdir(exist_ok=True)
    docs_figs_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("SeismicLab Demo Figure Generator")
    print("=" * 70)
    print(f"Total demos: {len(DEMOS)}")
    print()

    # Run all demos
    results = []
    for demo_file in DEMOS:
        success = run_demo(demo_file)
        results.append((demo_file, success))

    # Summary
    print("\n" + "=" * 70)
    print("Demo Execution Summary")
    print("=" * 70)

    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful

    for demo_file, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status:12} - {demo_file}")

    print()
    print(f"Total: {len(results)} demos")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100 * successful / len(results):.1f}%")
    print("=" * 70)

    # Move generated figures to docs/figs
    print("\n" + "=" * 70)
    print("Moving figures to docs/figs...")
    print("=" * 70)

    generated_files = list(figs_dir.glob("*.png"))
    print(f"Found {len(generated_files)} figures in {figs_dir}")

    for fig_file in generated_files:
        dest = docs_figs_dir / fig_file.name
        shutil.copy2(fig_file, dest)
        print(f"✅ Copied {fig_file.name} -> docs/figs/")

    print()
    print(f"✅ All figures copied to {docs_figs_dir}")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    # Use non-interactive backend for matplotlib
    import matplotlib
    matplotlib.use('Agg')

    success = main()
    sys.exit(0 if success else 1)
