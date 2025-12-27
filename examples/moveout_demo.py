"""
常规NMO/INMO示例：读取合成CMP道集，按给定速度函数做常速校正和逆校正，并输出对比图及SU文件。
"""

from __future__ import annotations  # 支持前向类型注解

import sys  # 系统接口
from pathlib import Path  # 路径工具

import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图

from seismiclab import inmo, nmo, read_su, seismic_colormap, write_su  # 导入处理函数


DATA_DIR = Path("data")  # 数据目录


def main():
    d, headers = read_su(DATA_DIR / "syn_cmp.su")  # 读取合成CMP数据
    h = np.array([hdr["offset"] for hdr in headers], dtype=float)  # 偏移距数组
    dtsec = float(headers[0]["dt"]) / 1_000_000.0  # 采样间隔秒

    tnmo = np.array([0.5, 1.22, 1.65])  # NMO时深点
    vnmo = np.array([2000.0, 5000.0, 2500.0])  # 对应速度
    max_stretch = 400.0  # 最大拉伸百分比

    d_nmo, _, _, _ = nmo(d, dtsec, h, tnmo, vnmo, max_stretch)  # 执行NMO校正
    d_inmo, _, _, _ = inmo(d_nmo, dtsec, h, tnmo, vnmo, max_stretch)  # 反NMO校正

    # write_su("examples", "data", "syn_cmp_nmo_py.su", d_nmo, headers, dt=dtsec)  # 保存NMO结果
    # write_su("examples", "data", "syn_cmp_inmo_py.su", d_inmo, headers, dt=dtsec)  # 保存INMO结果

    nt, nx = d.shape  # 数据维度
    taxis = np.arange(nt) * dtsec  # 时间轴
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)  # 两幅对比图
    im0 = ax[0].imshow(d, aspect="auto", extent=[h[0], h[-1], taxis[-1], taxis[0]], cmap=seismic_colormap())  # 原始
    im1 = ax[1].imshow(d_nmo, aspect="auto", extent=[h[0], h[-1], taxis[-1], taxis[0]], cmap=seismic_colormap())  # NMO后
    ax[0].set_title("Original CMP")  # 左图标题
    ax[1].set_title("After NMO")  # 右图标题
    ax[0].set_xlabel("Offset [m]")  # 左图x轴
    ax[1].set_xlabel("Offset [m]")  # 右图x轴
    ax[0].set_ylabel("Time [s]")  # y轴
    fig.colorbar(im1, ax=ax[1], shrink=0.7)  # 色条
    fig.tight_layout()  # 紧凑布局
    fig.savefig(Path(__file__).parent / "figs" / "moveout_demo.png", dpi=150, bbox_inches="tight")
    plt.show()  # 显示


if __name__ == "__main__":  # 入口
    main()  # 执行示例
