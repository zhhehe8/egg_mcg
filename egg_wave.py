"""
使用连续小波变换 (CWT) 对心磁数据 (实验数据和空载数据) 进行时频分析，
并将Bx和By分量的结果分别绘制到2x2的子图中。
"""

import numpy as np # NumPy库，用于高效的数值计算，特别是数组操作
import zhplot
import matplotlib.pyplot as plt # Matplotlib库，用于数据可视化和绘图
import pywt # PyWavelets库，用于小波变换计算
import os # OS库，用于操作系统相关功能，如文件路径处理
import time
start_time = time.time()
# --- 配置参数 ---
# 数据加载配置
experimental_filepath = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d20\egg_d20_B30_t1_待破壳.txt' # 实验数据文件路径
no_load_filepath = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\鸡蛋空载\空载1.txt'      # 空载数据文件路径 (请确保此路径正确)

fs = 1000  # 采样率 (Hz)
skip_header = 2 # 加载数据时跳过文件开头的行数
encoding_val = 'utf-8' # 文件编码

# 小波变换参数
wavelet_name = 'cmor1.5-1.0'  # 使用的母小波名称 (Complex Morlet wavelet，B=1.5, C=1.0)
min_freq_target = 1  # Hz, 目标分析的最低频率
max_freq_target = 10 # Hz, 目标分析的最高频率 (用户更新)
num_scales = 128     # 用于CWT的尺度数量，影响频率轴的分辨率

# 计算CWT所用的尺度数组
dt = 1/fs # 采样周期
# s0: 对应max_freq_target的最小尺度
s0 = (pywt.central_frequency(wavelet_name, precision=8) * fs) / max_freq_target
# s_max: 对应min_freq_target的最大尺度
s_max = (pywt.central_frequency(wavelet_name, precision=8) * fs) / min_freq_target
# scales: 在s0和s_max之间生成对数间隔的尺度序列
scales = np.geomspace(s0, s_max, num_scales)

# 绘图参数
cmap_plot = 'bwr'  # 颜色映射表名称 (Blue-White-Red, 蓝白红，一种发散型色图。如果数据全为正，主要显示白到红的部分)
# 注意: 'bwr' 通常用于有正有负的数据，中心点为白色。对于幅度这种非负数据，
# 如果vmin设为0，则主要使用从白到红的色阶。

# 颜色条的幅度范围 (用户更新)
colorbar_magnitude_min = 0
colorbar_magnitude_max = 5.0
# --- 配置参数结束 ---

# --- 数据加载辅助函数 ---
def load_mcg_data(filepath, skip, encoding):
    """
    从指定路径加载心磁数据文件。
    文件应至少包含两列数据 (Bx, By)。

    参数:
        filepath (str): 数据文件路径。
        skip (int): 加载时跳过的文件头部行数。
        encoding (str): 文件编码。

    返回:
        tuple: (Bx_raw, By_raw) 数据或 (None, None) 如果加载失败。
    """
    if not os.path.exists(filepath): # 检查文件是否存在
        print(f"错误: 文件未找到 {filepath}")
        return None, None
    try:
        data = np.loadtxt(filepath, skiprows=skip, encoding=encoding) # 加载数据
        if data.ndim == 1 or data.shape[1] < 2: # 检查数据是否至少有两列
            print(f"  错误：数据文件 {os.path.basename(filepath)} 需要至少两列 (Bx, By)。")
            return None, None
        Bx = data[:, 0] # 第一列为 Bx
        By = data[:, 1] # 第二列为 By
        print(f"数据已从 {os.path.basename(filepath)} 加载。Bx 长度: {len(Bx)}, By 长度: {len(By)}")
        return Bx, By
    except Exception as e: # 捕获加载过程中的其他异常
        print(f"错误: 加载 {os.path.basename(filepath)} 时出错: {e}")
        return None, None

# 1. 加载实验数据
Bx_exp_raw, By_exp_raw = load_mcg_data(experimental_filepath, skip_header, encoding_val)

# 2. 加载空载数据
Bx_noload_raw, By_noload_raw = load_mcg_data(no_load_filepath, skip_header, encoding_val)

# 为两组数据分别创建时间向量 (如果数据成功加载)
time_vector_exp = None
if Bx_exp_raw is not None:
    time_vector_exp = np.arange(len(Bx_exp_raw)) / fs

time_vector_noload = None
if Bx_noload_raw is not None:
    time_vector_noload = np.arange(len(Bx_noload_raw)) / fs

# --- CWT计算和绘图辅助函数 ---
def plot_cwt_scalogram(ax, signal_data, time_vec, scales_arr, wavelet, fs_sig, title, cmap_name, vmin_cbar, vmax_cbar):
    """
    计算给定信号的连续小波变换(CWT)并绘制尺度图(scalogram)。

    参数:
        ax (matplotlib.axes.Axes): 用于绘图的子图对象。
        signal_data (np.array): 一维输入信号。
        time_vec (np.array): 对应信号的时间向量。
        scales_arr (np.array): 用于CWT的尺度数组。
        wavelet (str): 使用的母小波名称。
        fs_sig (float): 信号的采样率。
        title (str): 子图的标题。
        cmap_name (str): 使用的颜色映射表名称。
        vmin_cbar (float): 颜色条的最小值。
        vmax_cbar (float): 颜色条的最大值。

    返回:
        tuple: (matplotlib.image.AxesImage, np.array) 或 (None, None)
               返回imshow对象和对应的频率数组，如果失败则返回None。
    """
    if signal_data is None or time_vec is None: # 检查输入数据是否有效
        ax.set_title(f'{title}\n(数据不可用)')
        ax.text(0.5, 0.5, 'Data not available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return None, None

    print(f"正在为 {title} 计算CWT...")
    try:
        # 执行连续小波变换
        # sampling_period = 1/fs_sig 确保返回的频率单位是Hz
        coefficients, frequencies = pywt.cwt(signal_data, scales_arr, wavelet, sampling_period=1/fs_sig)
    except Exception as e: # 捕获CWT计算过程中的错误
        print(f"  错误: CWT 计算失败 for {title}: {e}")
        ax.set_title(f'{title}\n(CWT 计算失败)')
        ax.text(0.5, 0.5, 'CWT failed', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return None, None

    print(f"CWT 完成。系数矩阵形状: {coefficients.shape}, 频率范围: {frequencies.min():.2f} Hz - {frequencies.max():.2f} Hz")

    # 绘制尺度图 (CWT系数的幅度)
    # extent 参数将图像的像素坐标映射到实际的时间和频率值
    # 注意频率的顺序是 frequencies[-1], frequencies[0] 来使得低频在底部
    im = ax.imshow(np.abs(coefficients), extent=[time_vec[0], time_vec[-1], frequencies[-1], frequencies[0]],
                   aspect='auto', cmap=cmap_name, interpolation='bilinear',
                   vmin=vmin_cbar, vmax=vmax_cbar) # 使用传入的vmin和vmax固定颜色范围
    
    ax.set_title(title, fontsize=10) # 设置子图标题和字体大小
    ax.set_xlabel('Time (s)', fontsize=9) # 设置X轴标签
    ax.set_ylabel('Frequency (Hz)', fontsize=9) # 设置Y轴标签
    ax.tick_params(axis='both', which='major', labelsize=8) # 设置刻度标签字体大小

    # 设置Y轴（频率轴）的刻度
    num_yticks_plot = 5 # Y轴上刻度的数量
    # 在频率数组中均匀选取一些点作为刻度位置
    yticks_idx = np.linspace(0, len(frequencies)-1, num_yticks_plot, dtype=int)
    ax.set_yticks(frequencies[yticks_idx]) # 设置刻度位置
    ax.set_yticklabels([f"{f:.1f}" for f in frequencies[yticks_idx]]) # 设置刻度标签文本

    return im, frequencies # 返回imshow对象和频率数组

# --- 创建图形和子图 (2x2 布局) ---
fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=False, sharey=False) # 创建2行2列的子图网格
fig.suptitle('连续小波变换 (CWT) 尺度图: 实验数据 vs. 空载数据', fontsize=16) # 设置总标题

# --- 自定义颜色条的辅助函数 ---
def add_custom_colorbar(fig_obj, image_mappable, ax_obj, vmin, vmax, label_text):
    """
    为指定的子图添加自定义刻度的颜色条。

    参数:
        fig_obj (matplotlib.figure.Figure): Figure对象。
        image_mappable (matplotlib.image.AxesImage): imshow返回的对象。
        ax_obj (matplotlib.axes.Axes): 颜色条附加到的子图对象。
        vmin (float): 颜色条的最小值。
        vmax (float): 颜色条的最大值。
        label_text (str): 颜色条的标签文本。
    """
    cbar = fig_obj.colorbar(image_mappable, ax=ax_obj, label=label_text, shrink=0.8) # 创建颜色条，shrink使其略小
    num_cbar_ticks = 6  # 颜色条上刻度的数量 (例如，对于0-10范围，生成0,2,4,6,8,10)
    cbar_ticks = np.linspace(vmin, vmax, num_cbar_ticks) # 计算刻度位置
    cbar.set_ticks(cbar_ticks) # 设置刻度位置
    cbar.set_ticklabels([f"{tick:.0f}" for tick in cbar_ticks]) # 设置刻度标签文本 (整数格式)
    return cbar

# --- 依次绘制四个子图 ---
# 子图[0,0]: 实验数据 Bx
im_bx_exp, freqs_bx_exp = plot_cwt_scalogram(axs[0, 0], Bx_exp_raw, time_vector_exp, scales, wavelet_name, fs,
                                             '实验数据 Bx', cmap_plot,
                                             colorbar_magnitude_min, colorbar_magnitude_max)
if im_bx_exp: # 如果绘图成功，添加颜色条
    add_custom_colorbar(fig, im_bx_exp, axs[0, 0], colorbar_magnitude_min, colorbar_magnitude_max, '幅度')

# 子图[0,1]: 实验数据 By
im_by_exp, freqs_by_exp = plot_cwt_scalogram(axs[0, 1], By_exp_raw, time_vector_exp, scales, wavelet_name, fs,
                                             '实验数据 By', cmap_plot,
                                             colorbar_magnitude_min, colorbar_magnitude_max)
if im_by_exp:
    add_custom_colorbar(fig, im_by_exp, axs[0, 1], colorbar_magnitude_min, colorbar_magnitude_max, '幅度')

# 子图[1,0]: 空载数据 Bx
im_bx_noload, freqs_bx_noload = plot_cwt_scalogram(axs[1, 0], Bx_noload_raw, time_vector_noload, scales, wavelet_name, fs,
                                                   '空载数据 Bx', cmap_plot,
                                                   colorbar_magnitude_min, colorbar_magnitude_max)
if im_bx_noload:
    add_custom_colorbar(fig, im_bx_noload, axs[1, 0], colorbar_magnitude_min, colorbar_magnitude_max, '幅度')

# 子图[1,1]: 空载数据 By
im_by_noload, freqs_by_noload = plot_cwt_scalogram(axs[1, 1], By_noload_raw, time_vector_noload, scales, wavelet_name, fs,
                                                   '空载数据 By', cmap_plot,
                                                   colorbar_magnitude_min, colorbar_magnitude_max)
if im_by_noload:
    add_custom_colorbar(fig, im_by_noload, axs[1, 1], colorbar_magnitude_min, colorbar_magnitude_max, '幅度')


# --- 尝试统一所有子图的Y轴（频率轴）范围以便比较 ---
# 选择一个成功计算出的频率数组作为参考
ref_freqs_for_ylim = None
if freqs_bx_exp is not None: ref_freqs_for_ylim = freqs_bx_exp
elif freqs_by_exp is not None: ref_freqs_for_ylim = freqs_by_exp
elif freqs_bx_noload is not None: ref_freqs_for_ylim = freqs_bx_noload
elif freqs_by_noload is not None: ref_freqs_for_ylim = freqs_by_noload

if ref_freqs_for_ylim is not None: # 如果有有效的参考频率
    min_f, max_f = ref_freqs_for_ylim.min(), ref_freqs_for_ylim.max() # 获取最小和最大频率
    for ax_row in axs: # 遍历所有子图行
        for ax in ax_row: # 遍历当前行中的所有子图
            # 仅对成功绘图的子图调整Y轴
            if not ax.get_title().endswith('(数据不可用)') and not ax.get_title().endswith('(CWT 计算失败)'):
                ax.set_ylim(min_f, max_f) # 设置统一的Y轴范围
                # 根据统一的范围重新应用Y轴刻度
                num_yticks_plot_ax = 5
                yticks_idx = np.linspace(0, len(ref_freqs_for_ylim)-1, num_yticks_plot_ax, dtype=int)
                ax.set_yticks(ref_freqs_for_ylim[yticks_idx])
                ax.set_yticklabels([f"{f:.1f}" for f in ref_freqs_for_ylim[yticks_idx]])


plt.tight_layout(rect=[0, 0.03, 1, 0.94]) # 调整子图布局以适应总标题，避免重叠
plt.show() # 显示图形

print("绘图完成。")

# 耗费的时间

end_time = time.time()
print(f"总耗时: {end_time - start_time:.2f} 秒")
# --- 结束 ---