"""
稀疏反褶积示例：读取小型叠前数据和子波，利用稀疏约束恢复反射系数，并生成预测数据与原始剖面对比。
"""

from __future__ import annotations  # 支持前向注解

import sys  # 系统路径
from pathlib import Path  # 路径工具

import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图

from seismiclab import read_su, seismic_colormap, sparse_decon, wigb  # 导入处理函数

DATA_DIR = Path("data")  # 数据目录


def main():
    d, Hs = read_su(DATA_DIR / "small_stack.su")  # 读取叠前数据
    dtsec = float(Hs[0]["dt"]) / 1_000_000.0  # 采样间隔秒
    cdp = np.array([h["cdp"] for h in Hs])  # CDP轴

    w, _ = read_su(DATA_DIR / "wavelet_for_small_stack.su")  # 读取子波
    w = w[:, 0]  # 取第一道

    max_iter = 20  # 最大迭代
    mu = 1.1  # 正则权重
    r, dp = sparse_decon(d, w, mu, max_iter, dtsec)  # 稀疏反褶积

    nt, ncdp = d.shape  # 数据维度
    t_axis = np.arange(nt) * dtsec  # 时间轴
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), sharey=True)  # 三图对比
    plt.sca(ax[0]); wigb(d, 2, cdp, t_axis); ax[0].set_title("Input")  # 输入剖面
    plt.sca(ax[1]); wigb(r, 2, cdp, t_axis); ax[1].set_title("Reflectivity")  # 反射系数
    plt.sca(ax[2]); wigb(dp, 2, cdp, t_axis); ax[2].set_title("Predicted data")  # 预测数据
    for a in ax:  # 设置轴标签
        a.set_xlabel("cdp")
        a.set_ylabel("Time (s)")
    fig.tight_layout()  # 紧凑布局
    fig.savefig(Path(__file__).parent / "figs" / "sparse_decon_demo.png", dpi=150, bbox_inches="tight")
    plt.show()  # 显示


if __name__ == "__main__":  # 入口
    main()  # 执行示例
