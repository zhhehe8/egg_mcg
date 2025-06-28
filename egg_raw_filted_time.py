"""
Fig2 x,y轴原始信号、滤波信号及R峰展示图 (修改为1x2子图，仅显示Bx时域信号，美化)
"""

import numpy as np
import zhplot # Keep if you use it for matplotlib styling (ensure it's configured if used)
import matplotlib.pyplot as plt
import os
# Assuming egg_functions.py is in the same directory or Python path
from egg_functions import bandpass_filter, apply_notch_filter

### ---- 主程序开始 ---- ###
# 1. 加载数据

"""输入输出目录"""
input_dir = '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B30_t1_待破壳.txt'  # 输入数据文件路径

try:
    data = np.loadtxt(input_dir, skiprows=2, encoding="utf-8")
    if data.ndim == 1 or data.shape[1] < 1: # Check if at least one column exists
        print(f"错误: 数据文件 {os.path.basename(input_dir)} 需要至少一列数据。")
        exit()
    Bx_raw = data[:, 0]
    # By_raw = data[:, 1] # By is commented out by user
except FileNotFoundError:
    print(f"错误: 文件未找到 {input_dir}")
    exit()
except Exception as e:
    print(f"加载数据时发生错误: {e}")
    exit()

fs = 1000  # 采样率

# 2. 参数设置

"""时间参数"""
time = np.arange(len(Bx_raw)) / fs  # 时间向量

""" 设置滤波器参数 """
filter_order_bandpass = 4
lowcut_freq = 0.5
highcut_freq = 45.0
notch_freq = 50.0
Q_factor_notch = 30.0

"""设置信号反转"""
reverse_Bx_signal = False
if reverse_Bx_signal:
    Bx_raw = -Bx_raw
    print("信息: Bx 信号已反转。")

# --- Store original raw signals for plotting ---
Bx_raw_original_for_plot = Bx_raw.copy()

# 3. 对原始数据进行滤波处理
print("开始滤波 Bx_raw 信号...")
Bx_filtered_bp = bandpass_filter(Bx_raw, fs, lowcut_freq, highcut_freq, filter_order_bandpass)

# 3.1 应用陷波滤波器去除工频干扰
Bx_filtered_final = None # Initialize
if Bx_filtered_bp is not None:
    Bx_filtered_final = apply_notch_filter(Bx_filtered_bp, notch_freq, Q_factor_notch, fs)
    print("滤波完成。")
else:
    print("错误: 带通滤波失败，跳过陷波滤波。")


# 4. 绘制结果 (1x2 画布)
print("开始绘制 1x2 时域信号图...")

# --- 美化参数 ---
plt.style.use('seaborn-v0_8-whitegrid') # 使用一个美观的样式，或者 'seaborn-v0_8-pastel', 'ggplot' 等
# zhplot.set_matplotlib(style='academic') # 如果您使用zhplot并且想用它的风格

fig, axs = plt.subplots(1, 2, figsize=(16, 5.5), sharey=False) # 调整figsize使高度略小

plot_time_limit = 40  # seconds, to display

# --- 子图1: 原始 Bx 数据 ---
raw_color = 'cornflowerblue' # 更清晰的颜色
axs[0].plot(time, Bx_raw_original_for_plot, label='Raw', color=raw_color, linewidth=1.2, alpha=0.9)
axs[0].set_title('Raw Signal', fontsize=14)
axs[0].set_xlabel('Time (s)', fontsize=12)
axs[0].set_ylabel('Amplitude (pT or original units)', fontsize=12) # 更具体的单位
axs[0].legend(loc='upper right', fontsize=10)
# axs[0].grid(True, linestyle=':', alpha=0.6, linewidth=0.7) # grid is part of seaborn-whitegrid
axs[0].set_xlim(0, plot_time_limit)
axs[0].tick_params(axis='both', which='major', labelsize=10)


# --- 子图2: 滤波后 Bx 数据 ---
filtered_color = 'crimson' # 与原始信号对比鲜明的颜色
if Bx_filtered_final is not None:
    axs[1].plot(time, Bx_filtered_final, label='Filtered', color=filtered_color, linewidth=1.5)
    axs[1].set_ylabel('Amplitude (Filtered)', fontsize=12) # Y轴标签可以更具体
else:
    axs[1].text(0.5, 0.5, 'Filtered data not available',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='grey')
    axs[1].set_ylabel('Amplitude', fontsize=12)


axs[1].set_title('Filtered Signal (Time Domain)', fontsize=14)
axs[1].set_xlabel('Time (s)', fontsize=12)
axs[1].legend(loc='upper right', fontsize=10)
# axs[1].grid(True, linestyle=':', alpha=0.6, linewidth=0.7)
axs[1].set_xlim(0, plot_time_limit)
axs[1].tick_params(axis='both', which='major', labelsize=10)
# 如果需要，可以为滤波信号设置特定的Y轴范围，例如：
# axs[1].set_ylim(-2, 2) # 示例


# 整体美化
fig.suptitle('Comparison of Raw and Filtered Bx Signal', fontsize=18, y=1.02) # y调整总标题位置
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应总标题
plt.show()

print("\n进程结束！！！")