"""
抛物线残余时差谱示例：多次波道集先做NMO校正，再计算抛物线moveout谱以突出剩余时差能量分布。
"""

from __future__ import annotations  # 前向注解

import sys  # 系统接口
from pathlib import Path  # 路径工具

import numpy as np  # 数值运算
import matplotlib.pyplot as plt  # 绘图

from seismiclab import nmo, parabolic_moveout, read_su, seismic_colormap  # 导入函数


DATA_DIR = Path("data")  # 数据目录


def main():
    d, headers = read_su(DATA_DIR / "syn_cmp_mult.su")  # 读取多次波数据
    h = np.array([hdr["offset"] for hdr in headers], dtype=float)  # 偏移距
    dtsec = float(headers[0]["dt"]) / 1_000_000.0  # 采样间隔秒

    tnmo = []  # NMO时深点（空表示仅残余）
    vnmo = [3000.0]  # 速度
    max_stretch = 40.0  # 最大拉伸
    d_nmo, _, _, _ = nmo(d, dtsec, h, tnmo, vnmo, max_stretch)  # NMO校正

    qmin, qmax, nq = -0.5, 1.2, 120  # 残余时差范围
    R, L = 1, 20  # 正则与窗长
    S, tau, q = parabolic_moveout(d_nmo, dtsec, h, qmin, qmax, nq, R, L)  # 计算谱

    nt, _ = d.shape  # 样本数
    time_axis = np.arange(nt) * dtsec  # 时间轴
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), sharey=False)  # 三幅图
    ax[0].imshow(d, aspect="auto", extent=[h[0], h[-1], time_axis[-1], time_axis[0]], cmap=seismic_colormap())  # 原始
    ax[0].set_title("Original")  # 标题
    ax[0].set_xlabel("Offset [m]")  # x轴
    ax[0].set_ylabel("Time [s]")  # y轴
    ax[1].imshow(
        d_nmo, aspect="auto", extent=[h[0], h[-1], time_axis[-1], time_axis[0]], cmap=seismic_colormap()
    )  # NMO后
    ax[1].set_title("After NMO")  # 标题
    im = ax[2].imshow(
        S,
        aspect="auto",
        extent=[q[0], q[-1], tau[-1], tau[0]],
        cmap="gray_r",
        vmin=0,
        vmax=0.5 * np.max(S),
    )  # 残余谱
    ax[2].set_title("Residual moveout spectrum")  # 标题
    ax[2].set_xlabel("q [s]")  # q轴
    fig.colorbar(im, ax=ax[2])  # 色条
    fig.tight_layout()  # 紧凑布局
    fig.savefig(Path(__file__).parent / "figs" / "parabolic_moveout_demo.png", dpi=150, bbox_inches="tight")
    plt.show()  # 显示


if __name__ == "__main__":  # 入口
    main()  # 运行
