"""
将12-20天的心磁数据进行批量处理，提
取平均心跳周期并绘制到同一张图上。
    
"""

### 尚未完工。。。。。。

import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks
import os
import re
import matplotlib.pyplot as plt
import argparse
from egg_functions import(
    load_cardiac_data, 
    bandpass_filter, 
    apply_notch_filter, 
    find_r_peaks_data, 
    plot_signals_with_r_peaks, 
    averaged_cardias_cycle_plot, 
    extract_day_label_from_folder
)

### ---- 配置参数 ---- ###
# 1.加载数据

"""输入输出目录"""
input_dir = '/Users/yanchen/Desktop/Projects/egg_2025/B_egg'  # 输入数据文件路径
output_fig_dir = '/Users/yanchen/Desktop/app'  # 输出目录

skip_header = 2  # 跳过的行数
file_encoding = "utf-8"  # 文件编码
fs = 1000  # 采样率 (Hz)



# 2. 检测参数设置

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


#  --- 特定日龄的的R峰检测参数 ---
day_specific_r_peak_params = {
    "day 2":  {"height_factor": 0.35, "distance_ms": 400}, 
    "day 3":  {"height_factor": 0.35, "distance_ms": 400},
    "day 4":  {"height_factor": 0.35, "distance_ms": 400},
    "day 5":  {"height_factor": 0.35, "distance_ms": 400},
    "day 6":  {"height_factor": 0.35, "distance_ms": 400},
    "day 7":  {"height_factor": 0.35, "distance_ms": 400},
    "day 8":  {"height_factor": 0.35, "distance_ms": 400},
    "day 9":  {"height_factor": 0.35, "distance_ms": 400},
    "day 10": {"height_factor": 0.35, "distance_ms": 200},
    "day 11": {"height_factor": 0.35, "distance_ms": 300},
    "day 12": {"height_factor": 0.35, "distance_ms": 300},
    "day 13": {"height_factor": 0.40, "distance_ms": 300}, 
    "day 14": {"height_factor": 0.40, "distance_ms": 250},
    "day 15": {"height_factor": 0.40, "distance_ms": 250},
    "day 16": {"height_factor": 0.40, "distance_ms": 250},
    "day 17": {"height_factor": 0.40, "distance_ms": 250},
    "day 18": {"height_factor": 0.40, "distance_ms": 200},
    "day 19": {"height_factor": 0.40, "distance_ms": 200},
    "day 20": {"height_factor": 0.45, "distance_ms": 200}, 
    "day 21": {"height_factor": 0.45, "distance_ms": 200},
}

### ---- 主处理函数 ---- ###
