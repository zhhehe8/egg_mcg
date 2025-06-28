"""
批量处理心跳信号，求出每个心跳信号的平均值和标准差。
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks
import os
import re
import matplotlib.pyplot as plt
from egg_functions import(
    load_cardiac_data, 
    bandpass_filter, 
    apply_notch_filter, 
    find_r_peaks_data, 
    plot_signals_with_r_peaks, averaged_cardias_cycle_plot, extract_day_label_from_folder
)

### ---- 配置参数 ---- ###
# 1.加载数据

"""输入输出目录"""
input_dir = '/Users/yanchen/Desktop/Projects/egg_2025/B_egg'  # 输入数据文件路径
output_fig_dir = '/Users/yanchen/Desktop/Projects/average_cycles_99'  # 输出目录

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




# --- 主程序 ---
if __name__ == "__main__":
    if not os.path.isdir(input_dir):
        print(f"错误：根数据目录 '{input_dir}' 不存在或不是一个目录。请检查路径。")
    else:
        print(f"开始批量处理根目录: {input_dir}")
        print(f"平均周期图片将保存到: {output_fig_dir}")
        os.makedirs(output_fig_dir, exist_ok=True) 
        
        processed_files_count = 0
        saved_avg_cycle_images_count = 0

        day_folders = sorted(
            [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf')
        )

        for day_folder_name in day_folders:
            day_folder_path = os.path.join(input_dir, day_folder_name)
            day_label = extract_day_label_from_folder(day_folder_name) 
            print(f"\n--- 开始处理文件夹: {day_folder_name} (日龄标签: {day_label}) ---")
            
            day_r_peak_config = day_specific_r_peak_params.get(day_label, {})
            current_r_peak_height_factor = day_r_peak_config.get("height_factor", R_peak_min_height_factor)
            current_r_peak_distance_ms = day_r_peak_config.get("distance_ms", R_peak_min_distance_ms)
            
            print(f"  使用R峰检测参数: Height Factor = {current_r_peak_height_factor:.2f}, Distance MS = {current_r_peak_distance_ms}")

            current_output_subfolder = os.path.join(output_fig_dir, day_folder_name)
            os.makedirs(current_output_subfolder, exist_ok=True)

            txt_files_in_day_folder = [f for f in os.listdir(day_folder_path) if f.lower().endswith('.txt')]
            if not txt_files_in_day_folder:
                print(f"  在文件夹 {day_folder_name} 中未找到 .txt 文件。")
                continue

            for txt_filename in txt_files_in_day_folder:
                current_file_path = os.path.join(day_folder_path, txt_filename)
                print(f"\n  处理文件: {txt_filename}")
                processed_files_count += 1
                
                Bx_raw, By_raw = load_cardiac_data(current_file_path, skip_header, file_encoding)

                if Bx_raw is None or By_raw is None:
                    continue
                
                Bx_bandpassed = bandpass_filter(Bx_raw, fs=fs, lowcut=lowcut_freq, highcut=highcut_freq, order=filter_order_bandpass)
                Bx_filtered = apply_notch_filter(Bx_bandpassed, notch_freq=notch_freq, quality_factor=Q_factor_notch, fs=fs)
                
                By_bandpassed = bandpass_filter(By_raw, fs=fs, lowcut=lowcut_freq, highcut=highcut_freq, order=filter_order_bandpass)
                By_filtered = apply_notch_filter(By_bandpassed, notch_freq=notch_freq, quality_factor=Q_factor_notch, fs=fs)

                if Bx_filtered is None or By_filtered is None:
                    print(f"  错误：文件 {txt_filename} 的一个或两个通道滤波失败。")
                    continue
                
                min_len = min(len(Bx_filtered), len(By_filtered))
                combined_data_filtered = np.sqrt(Bx_filtered[:min_len]**2 + By_filtered[:min_len]**2)
                
                r_peaks_indices = find_r_peaks_data(
                    combined_data_filtered, fs,
                    min_height_factor=current_r_peak_height_factor,
                    min_distance_ms=current_r_peak_distance_ms,
                    identifier=f"融合信号 ({txt_filename})"
                )

                if r_peaks_indices is not None and len(r_peaks_indices) > 1:
                    output_filename_base_for_plot = os.path.splitext(txt_filename)[0]
                    
                    success = averaged_cardias_cycle_plot(
                        data=combined_data_filtered, 
                        r_peaks_indices=r_peaks_indices, 
                        fs=fs,
                        pre_r_ms = pre_r_ms, 
                        post_r_ms = post_r_ms, 
                        output_dir=current_output_subfolder, 
                        base_filename=output_filename_base_for_plot,
                        identifier=f"融合信号 ({txt_filename})"
                    )
                    if success:
                        saved_avg_cycle_images_count += 1

        print(f"\n--- 批量处理结束 ---")
        print(f"总共处理了 {processed_files_count} 个文件。")
        print(f"成功保存了 {saved_avg_cycle_images_count} 个平均心跳周期图片。")