"""
Lightweight Python translation of SeismicLab helpers used by the demo scripts.

Only the routines required by the MATLAB demos under ``SeismicLab_demos`` are
exposed. The implementations follow the original algorithms closely but use
NumPy/SciPy tooling to keep the code simple to read and run.

New modules added (December 2025):
- solvers: Optimization algorithms (FISTA, CGLS, IRLS, etc.)
- rank_reduction: Low-rank matrix approximations (CUR, randomized SVD)
- linear_operators: Linear operators for inverse problems
- bp_filter: Band-pass filtering
- kl_transform: Karhunen-Loève transform
- fxy_eigen: FXY eigenimage decomposition
- mssa: Multichannel Singular Spectrum Analysis
- dephasing: Phase correction and kurtosis
- scaling_tapering: Gain and taper functions
- pmf: Parallel matrix factorization
- mwni: Multi-window noise inversion
"""

# Original modules
from .io import (
    read_su,
    write_su,
    extract_header_word,
    sort_su,
)
from .velan_nmo import (
    nmo,
    inmo,
    velan,
    parabolic_moveout,
    lmo,
    stackgather,
)
from .radon import (
    pradon_demultiple,
    forward_radon_freq,
    inverse_radon_freq,
)
from .fx import (fx_decon,
    spitz_fx_interpolation
)
from .synthetics import (
    data_cube,
    linear_events,
    laplace,
    laplace_mixture,
    gauss_mixture,
    bernoulli_refl,
    bernoulli,
    flat_events,
    hyperbolic_events,
    hyperbolic_apex_shifted_events,
    parabolic_events,
    trapezoidal_wavelet,
    rotated_wavelet,
    make_section,
    make_traces,
    ricker,
)
from .decon import (
    sparse_decon,
    spiking,
    taper,
    ls_inv_filter,
    delay,
    predictive,
    zeros_wav,
    kolmog,
    polar_plot,
)
from .pocs import (
    pocs
)
from .spectra import (
    smooth_spectrum,
    fk_spectra,
)
from .med_alg import (
    med
)
from .plotting import (
    wigb,
    plot_wb,
    pimage,
    clip,
    seismic_colormap,
    plot_spectral_attributes,
    sgray,
)
from .util import (
    add_noise,
    quality,
    chi2,
    perc,
)

# New modules (2025)
from .solvers import (
    fista,
    cgls,
    cglsw,
    irls,
    power_method,
    thresholding,
    cgdot,
)
from .rank_reduction import (
    cur,
    cur_Old,
    rand_svd,
    rqrd,
)
from .bp_filter import (
    bp_filter,
)
from .kl_transform import (
    kl,
)
from .fxy_eigen import (
    fxy_eigen_images,
)
from .mssa import (
    mssa_2d,
    mssa_3d,
    mssa_3d_interp,
)
from .dephasing import (
    phase_correction,
    kurtosis_of_traces,
)
from .scaling_tapering import (
    taper as taper_func,
    gain,
    envelope,
)
from .pmf import (
    pmf,
    completion,
    completion as pmf_completion,  # 别名
)
from .mwni import (
    mwni,
    mwni_irls,
    operator_nfft,
)
from .linear_operators import (
    Matrix_Multiply_operator,
    Mutes_operator,
    NMO_operator,
    radon_fx,
    radon_tx,
    ash_radon_tx,
    radon_general_tx,
    operator_radon_freq,
    Operator_Radon_Freq,  # 别名
    operator_radon_stolt,
    Operator_Radon_Stolt,  # 别名
)

__all__ = [
    # Original modules
    "read_su",                  # 读取SU格式的地震数据
    "write_su",                 # 写入SU格式的地震数据
    "extract_header_word",      # 提取SEGY头字
    "sort_su",                  # 按头字排序
    "nmo",                      # 正常时差校正（Normal Moveout）
    "inmo",                     # 反正常时差校正（Inverse Normal Moveout）
    "velan",                    # 速度分析（Velocity Analysis）
    "parabolic_moveout",        # 抛物线时差校正
    "lmo",                      # 线性时差校正（Linear Moveout）
    "stackgather",              # 叠加（Stack）
    "pradon_demultiple",        # 抛物线Radon变换去多次波
    "forward_radon_freq",       # Radon变换（频率域正向）
    "inverse_radon_freq",       # Radon变换（频率域反向）
    "fx_decon",                 # 频率-空间域反褶积
    "spitz_fx_interpolation",   # Spitz频率-空间域插值
    "data_cube",                # 生成三维数据立方体
    "linear_events",            # 生成线性同相轴事件
    "laplace",                  # Laplace分布
    "laplace_mixture",          # 混合Laplace分布
    "gauss_mixture",            # 混合Gaussian分布
    "bernoulli_refl",           # Bernoulli反射系数序列
    "bernoulli",                # Bernoulli稀疏序列
    "flat_events",              # 平直事件生成
    "hyperbolic_events",        # 双曲线事件生成
    "hyperbolic_apex_shifted_events",  # 顶点偏移双曲线事件
    "parabolic_events",         # 抛物线事件生成
    "trapezoidal_wavelet",      # 梯形子波
    "rotated_wavelet",          # 旋转相位子波
    "make_section",             # 生成合成剖面
    "make_traces",              # 生成地震道集
    "ricker",                   # Ricker子波
    "sparse_decon",             # 稀疏反褶积
    "spiking",                  # 脉冲反褶积
    "taper",                    # 数据边缘衰减
    "ls_inv_filter",            # 最小二乘反滤波器
    "delay",                    # 延迟估计
    "predictive",               # 预测反褶积
    "zeros_wav",                # 子波零点计算
    "kolmog",                   # Kolmogorov谱因子分解
    "polar_plot",               # 极坐标图
    "pocs",                     # 凸集投影算法（Projection Onto Convex Sets）
    "smooth_spectrum",          # 频谱平滑
    "med",                      # 中值滤波（Median Filtering）
    "wigb",                     # 绘制波形图（黑白色）
    "plot_wb",                  # 绘制图像（黑白色）
    "clip",                     # 数据裁剪/限幅
    "seismic_colormap",         # 地震数据配色方案
    "plot_spectral_attributes", # 绘制频谱属性（振幅和相位）
    "sgray",                    # 灰度色彩映射
    "add_noise",                # 添加噪声
    "quality",                  # 数据质量评估
    "pimage",                   # 图像显示
    "fk_spectra",               # FK频谱分析
    "chi2",                     # 卡方检验
    "perc",                     # 百分位数裁剪

    # New modules - Solvers
    "fista",                    # 快速迭代收缩阈值算法
    "cgls",                     # 共轭梯度最小二乘
    "cglsw",                    # 加权共轭梯度最小二乘
    "irls",                     # 迭代重加权最小二乘
    "power_method",             # 幂法特征值计算
    "thresholding",             # 阈值操作
    "cgdot",                    # 共轭梯度点积

    # New modules - Rank reduction
    "cur",                      # CUR分解
    "cur_Old",                  # CUR分解（旧版）
    "rand_svd",                 # 随机SVD
    "rqrd",                     # 随机QR分解

    # New modules - Processing
    "bp_filter",                # 带通滤波器
    "kl",                       # Karhunen-Loève变换
    "fxy_eigen_images",         # FXY特征值图像分解

    # New modules - MSSA
    "mssa_2d",                  # 2D多通道奇异谱分析
    "mssa_3d",                  # 3D多通道奇异谱分析
    "mssa_3d_interp",           # 3D MSSA插值

    # New modules - Dephasing
    "phase_correction",         # 相位校正
    "kurtosis_of_traces",       # 峰度计算

    # New modules - Scaling/Tapering
    "taper_func",               # 三角衰减函数（避免与decon.taper冲突）
    "gain",                     # 增益控制
    "envelope",                 # 包络计算

    # New modules - PMF
    "pmf",                      # 平行矩阵分解
    "completion",               # 张量补全
    "pmf_completion",           # 张量补全（别名）

    # New modules - MWNI
    "mwni",                     # 多窗口噪声反演
    "mwni_irls",                # MWNI via IRLS
    "operator_nfft",            # N-D傅里叶算子

    # New modules - Linear operators
    "Matrix_Multiply_operator", # 矩阵乘法算子
    "Mutes_operator",           # 静音/切除算子
    "NMO_operator",             # NMO算子
    "radon_fx",                 # F-X域Radon算子
    "radon_tx",                 # T-X域Radon算子
    "ash_radon_tx",             # 顶点偏移双曲线Radon算子
    "radon_general_tx",         # 通用Radon算子
    "operator_radon_freq",      # 频率域Radon算子
    "Operator_Radon_Freq",      # 频率域Radon算子（MATLAB命名）
    "operator_radon_stolt",     # Radon-Stolt 变换算子
    "Operator_Radon_Stolt",     # Radon-Stolt 变换算子（MATLAB命名）
]
