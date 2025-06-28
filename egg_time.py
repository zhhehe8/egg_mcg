"""
Fig2 x,y轴原始信号、滤波信号及R峰展示图

"""


import numpy as np
import zhplot
import matplotlib.pyplot as plt
import os
# from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks
from egg_functions import bandpass_filter, apply_notch_filter, find_r_peaks_data, plot_signals_with_r_peaks, averaged_cardias_cycle_plot



### ---- 主程序开始 ---- ###
# 1. 加载数据

"""输入输出目录"""
input_dir = '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B30_t1_待破壳.txt'  # 输入数据文件路径
output_dir =  '/Users/yanchen/Desktop'          # 输出目录

data = np.loadtxt(input_dir, skiprows=2, encoding="utf-8")
Bx_raw = data[:, 0]
By_raw = data[:, 1]
fs = 1000  # 采样率

# 2. 检测参数设置

"""时间参数"""
time = np.arange(len(Bx_raw)) / fs  # 时间向量

""" 设置滤波器参数 """
filter_order_bandpass = 4  # 带通滤波器的阶数 (根据用户最新提供)
lowcut_freq = 0.5          # Hz, 低截止频率
highcut_freq = 45.0        # Hz, 高截止频率
notch_freq = 50.0    # Hz, 工频干扰频率
Q_factor_notch = 30.0      # 陷波滤波器的品质因数

""" 设置R峰检测参数 """
R_peak_min_height_factor = 0.6  # R峰最小高度因子 (相对于数据的最大值) 
R_peak_min_distance_ms = 200     # R峰最小距离 (毫秒)


""" 设置平均心跳周期参数 """
pre_r_ms = 100   # R峰前的时间窗口 (毫秒)
post_r_ms = 100  # R峰后的时间窗口 (毫秒)


# 自动从数据文件路径提取 base_filename
base_filename = os.path.splitext(os.path.basename(input_dir))[0]
output_dir = output_dir  # 保持后续变量一致


"""设置信号反转"""
reverse_Bx_signal = False  
if reverse_Bx_signal:
    Bx_raw = -Bx_raw  # 反转Bx信号
    print("信息: Bx 信号已反转。")

reverse_By_signal = False  
if reverse_By_signal:
    By_raw = -By_raw  # 反转By信号
    print("信息: By 信号已反转。")


# 3. 对原始数据进行滤波处理
print("开始滤波 Bx_raw 信号...")
Bx_filtered = bandpass_filter(Bx_raw, fs, lowcut_freq, highcut_freq, filter_order_bandpass)
print("开始滤波 By_raw 信号...")
By_filtered = bandpass_filter(By_raw, fs, lowcut_freq, highcut_freq, filter_order_bandpass)
print("滤波完成。")


## # 3.1 应用陷波滤波器去除工频干扰
Bx_filtered = apply_notch_filter(Bx_filtered, notch_freq, Q_factor_notch, fs)
By_filtered = apply_notch_filter(By_filtered, notch_freq, Q_factor_notch, fs)

# 4.寻找R峰
print("开始在Bx中寻找 R 峰...")
R_peaks_Bx = find_r_peaks_data(Bx_filtered, fs, R_peak_min_height_factor, R_peak_min_distance_ms, identifier="Bx信号")
print(f"在 Bx_filtered 中找到 {len(R_peaks_Bx)} 个R峰。")
R_peaks_By = find_r_peaks_data(By_filtered, fs, R_peak_min_height_factor, R_peak_min_distance_ms, identifier="By信号")
print(f"在 By_filtered 中找到 {len(R_peaks_By)} 个R峰。")

### 4.1 标记R峰为红色空心圆圈
R_peaks_Bx_y = Bx_filtered[R_peaks_Bx] if len(R_peaks_Bx) > 0 else np.array([])

### 调整Bx，By信号的y轴所在区间
Bx_raw += 5
By_raw += 8
Bx_filtered -= 0.5

# 5. 绘制结果
# 绘制Bx和By原始信号和滤波信号
print("开始绘制原始信号与滤波信号对比图...")
fig1 = plot_signals_with_r_peaks(time, Bx_raw, Bx_filtered, By_raw, By_filtered, R_peaks_Bx, R_peaks_By)
plt.show()
# plt.close(fig1)  # 关闭图形以释放内存

# # 6.绘制平均心跳周期
# print("\n开始处理Bx信号的平均心跳周期...")
# averaged_cardias_cycle_plot(
#     data=Bx_filtered,
#     r_peaks_indices=R_peaks_Bx,
#     fs=fs,
#     pre_r_ms=pre_r_ms,
#     post_r_ms=post_r_ms,
#     output_dir=output_dir, 
#     base_filename=base_filename, 
#     identifier="Bx_Filtered"
# )
# print("\n开始处理By信号的平均心跳周期...")
# averaged_cardias_cycle_plot(
#     data=By_filtered,
#     r_peaks_indices=R_peaks_By,
#     fs=fs,
#     pre_r_ms=pre_r_ms,
#     post_r_ms=post_r_ms,
#     output_dir=output_dir, 
#     base_filename=base_filename, 
#     identifier="By_Filtered"
# )

print("\n进程结束！！！")
