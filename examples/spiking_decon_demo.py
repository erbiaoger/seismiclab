"""
尖脉冲反褶积示例：生成或加载测试子波与地震记录，运行spiking算法压制子波，比较处理前后波形和频谱。
"""

from __future__ import annotations  # 支持前向注解

import sys  # 系统路径
from pathlib import Path  # 路径工具

import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图
import scipy.io as sio  # 读MAT文件

from seismiclab import (  # 导入函数
    laplace_mixture,
    plot_wb,
    read_su,
    spiking,
    taper,
)

DATA_DIR = Path("data")  # 数据目录


def smooth_spectrum_matlab(d, dt, L, io="li"):
    """
    Reproduce MATLAB smooth_spectrum.m behaviour for comparability.
    """

    d = np.asarray(d, dtype=float)  # 转为float
    nt = d.shape[0]  # 样本数
    aux = d.reshape(nt, -1)  # 展平道集
    wind = np.hamming(2 * L + 1)  # 汉明窗
    nf = max(2 * 2 ** int(np.ceil(np.log2(nt))), 2048)  # FFT长度
    f = np.arange(nf // 2 + 1) / (dt * nf)  # 频率轴

    D = np.fft.fft(aux, n=nf, axis=0)  # FFT
    Dpow = np.sum(np.abs(D) ** 2, axis=1)  # 功率
    Dconv = np.convolve(Dpow, wind, mode="full")  # 平滑
    Dconv = Dconv[L : L + nf]  # trim to nf samples (same as MATLAB conv then slice)
    Dconv = Dconv[: nf // 2 + 1]  # 只取正频
    Dconv = Dconv / np.max(Dconv)  # 归一化

    if io == "db":
        P = 10 * np.log10(Dconv)  # 转dB
        P[P < -40] = -40  # 底噪截断
    else:
        P = Dconv  # 线性尺度

    return P, f  # 返回谱与频率


def main():
    nr = 400  # 样本数
    dt = 0.004  # 采样间隔
    matlab_mat = DATA_DIR.parent / "SeismicLab_demos" / "matlab_spiking_decon_demo.mat"  # MATLAB参考数据

    if matlab_mat.exists():  # 若有MAT文件则直接加载
        m = sio.loadmat(matlab_mat)  # 读取MAT
        s = m["s"].ravel()  # 输入
        o = m["o"].ravel()  # 输出
    else:  # 否则生成合成数据
        np.random.seed(0)  # 固定随机种子
        w, _ = read_su(DATA_DIR / "min_phase_wavelet.su")  # 读取最小相位子波
        w = w[:, 0]  # 取第一道
        r = laplace_mixture(nr, [0.001, 0.1, 0.7])  # 生成拉普拉斯反射系数
        s = np.convolve(r, w)[:nr]  # 卷积生成地震记录
        s, _ = taper(s[:, None], 10, 10)  # 首尾加窗
        s = s.flatten()  # 展平
        s = s + 0 * np.max(s) * np.random.randn(*s.shape) / 12.0  # 可选噪声
        prewhitening = 0.1  # 预白化
        lf = 50  # 滤波长度
        _, o = spiking(s[:, None], lf, prewhitening)  # 尖脉冲反褶积
        o = o.flatten()  # 展平输出

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 两行图
    time = np.arange(nr) * dt  # 时间轴
    plt.sca(ax[0])  # 切换轴
    plot_wb(time, np.vstack([o, s]).T, 0)  # top: after decon, bottom: input
    ax[0].set_title("Seismogram (top: After decon, bottom: Input)")  # 标题
    ax[0].set_xlabel("Time (s)")  # x轴

    Ps, faxis = smooth_spectrum_matlab(s, dt, 30, "li")  # 输入谱
    Po, _ = smooth_spectrum_matlab(o, dt, 30, "li")  # 输出谱

    ax[1].plot(faxis, Po, label="After")  # 输出谱线
    ax[1].plot(faxis, Ps, label="Before")  # 输入谱线
    ax[1].set_title("Power spectrum")  # 标题
    ax[1].set_xlabel("Frequency (Hz)")  # x轴
    ax[1].set_ylabel("Normalized PSD")  # y轴
    ax[1].legend()  # 图例
    ax[1].grid()  # 网格
    plt.tight_layout()  # 紧凑布局
    fig.savefig(Path(__file__).parent / "figs" / "spiking_decon_demo.png", dpi=150, bbox_inches="tight")
    plt.show()  # 显示


if __name__ == "__main__":  # 入口
    main()  # 运行示例
