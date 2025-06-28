import numpy as np
import zhplot
import os
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks



# 定义函数


"""加载心脏数据"""
def load_cardiac_data(filepath, skip_header, file_encoding = 'utf-8'):
    try:
        data = np.loadtxt(filepath, skiprows=skip_header, encoding=file_encoding)
        if data.ndim == 1 or data.shape[1] < 2:
            print(f"  错误：数据文件 {os.path.basename(filepath)} 需要至少两列 (Bx, By)。")
            return None, None 
        Bx_raw = data[:, 0]
        By_raw = data[:, 1]
        return Bx_raw, By_raw
    except Exception as e:
        print(f"错误: 加载数据时出错: {e}")
        return None, None


"""巴特沃斯滤波器"""
def bandpass_filter(data, fs, lowcut, highcut, order):
    if data is None: return None
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0: high = 0.99
    if low <= 0: low = 0.001
    if low >= high: return data
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sosfiltfilt(sos, data)

"""陷波滤波器"""
def apply_notch_filter(data, notch_freq, quality_factor, fs):
    if data is None: return None
    nyquist = 0.5 * fs
    freq_normalized = notch_freq / nyquist
    if freq_normalized >= 1.0 or freq_normalized <= 0: return data
    b, a = iirnotch(freq_normalized, quality_factor)
    return filtfilt(b, a, data)


"""R峰检测函数"""
def find_r_peaks_data(data, fs, min_height_factor, min_distance_ms, identifier="信号",percentile = 100):
    if data is None or len(data) == 0: return np.array([])
    data_max = np.percentile(data, percentile)
    if data_max <= 1e-9: return np.array([])  
    min_h = min_height_factor * data_max
    min_distance = int(min_distance_ms / 1000 * fs)

    try:
        peaks, _ = find_peaks(data, height=min_h, distance=min_distance)
    except Exception as e:
        print(f"  错误 ({identifier}): 调用 find_peaks 时出错: {e}")
        return np.array([])
    return peaks




### 求出平均心跳周期，绘图展示并保存
    
def averaged_cardias_cycle_plot(data, r_peaks_indices, fs,
                                pre_r_ms, post_r_ms, output_dir,base_filename,
                                identifier="信号"):
    if data is None or r_peaks_indices is None or len(r_peaks_indices) < 2:
        return False         # 如果数据或R峰索引无效，直接返回

    pre_r_samples = int(pre_r_ms / 1000 * fs)
    post_r_samples = int(post_r_ms / 1000 * fs)
    total_cycle_samples = pre_r_samples + post_r_samples

    if total_cycle_samples <= 0:
        print(f"  错误 ({identifier}): pre_r_ms 和 post_r_ms 的总和必须大于0。")
        return False          
    
    all_cycles = []
    valid_cycles_count = 0
    for r_peak in r_peaks_indices:
        start_index = r_peak - pre_r_samples
        end_index = r_peak + post_r_samples
        if start_index >= 0 and end_index <= len(data):  # 确保提取范围不超出原始数据边界
            cycle = data[start_index:end_index]
            all_cycles.append(cycle)
            valid_cycles_count += 1

    if valid_cycles_count < 2:
        print(f"  错误 ({identifier}): 有效的心跳周期少于2个，无法计算平均周期。")
        return False
    
    cycles_array = np.array(all_cycles)  # 将周期列表转换为NumPy数组


    """ --- R峰基线校正--- """
    # 1.计算R峰的原始平均周期和标准差
    averaged_cycle = np.mean(cycles_array, axis=0)   # 提取的所有周期的平均值
    std_cycle = np.std(cycles_array, axis=0)       # 提取的所有周期的标准差

    # 2.从原始平均周期中确定基线偏移量
    """ 使用 pre_r_ms 的前30ms作为基线
        窗口，如果 pre_r_ms 不足30ms，
        则使用整个 pre_r_ms """
    baseline_window_samples = int(min(pre_r_ms, 30) / 1000 * fs)  # 基线窗口大小
    if baseline_window_samples > 0 and baseline_window_samples <= pre_r_samples:
        baseline_offset = np.mean(averaged_cycle[:baseline_window_samples])
    else:
        baseline_offset = 0
        print(f"  错误 ({identifier}): 基线校正窗口无效")
    
    # 3.从平均周期中减去基线偏移量
    averaged_cycle_corrected = averaged_cycle - baseline_offset

    # 4.校正背景中的单个R峰周期
    background_cycles_corrected = []
    for cycle in cycles_array:
        if baseline_window_samples > 0 and baseline_window_samples <= pre_r_samples:
            individual_baseline_offset = np.mean(cycle[:baseline_window_samples])
            background_cycles_corrected.append(cycle - individual_baseline_offset)
        else:
            background_cycles_corrected.append(cycle)   # 如果窗口无效，不校正单个周期
    
    background_cycles_corrected = np.array(background_cycles_corrected)

    """ 基线校正完成 """

    """  绘制平均心跳周期 """
    # 1.设置x轴
    cycle_time_axis_r = np.linspace(-pre_r_ms / 1000, (post_r_ms-1) / 1000, total_cycle_samples)

    # 2.绘制背景周期数据（随机20条）
    fig, ax = plt.subplots(figsize = (10, 6))

    ax.set_title(f'{identifier} Averaged Cardiac Cycle\n(Based on {valid_cycles_count} cycles)', fontsize=16)
    ax.set_xlabel('Time relative to R-peak (s)', fontsize=12)
    ax.set_ylabel('Signal Magnitude (pT)', fontsize=12)
    
    num_bg_cycles_to_plot = min(len(background_cycles_corrected),20)
    
    if len(background_cycles_corrected) > num_bg_cycles_to_plot:
        indices_to_plot = np.random.choice(len(background_cycles_corrected), num_bg_cycles_to_plot, replace=False)
    else:
        indices_to_plot = np.arange(len(background_cycles_corrected))
    for i in indices_to_plot:
        ax.plot(cycle_time_axis_r, background_cycles_corrected[i,:], color='lightgray', alpha=0.35, linewidth=0.5)
    
    # 3.绘制校正后的平均心跳周期
    ax.plot(cycle_time_axis_r, averaged_cycle_corrected, color='tomato', linewidth=2, label='Averaged Cycle')

    """  绘制标准差区域  """
    ax.fill_between(cycle_time_axis_r, 
                    averaged_cycle_corrected - std_cycle, 
                    averaged_cycle_corrected + std_cycle, 
                    color='tomato', alpha=0.2, label='±1 Std Dev')
    # 设置y轴范围,
    ax.set_ylim(0, 2)  # 根据数据范围调整y轴
    # 设置x轴范围，最小间隔0.05秒
    ax.set_xlim(-pre_r_ms / 1000, post_r_ms / 1000)

    # 2. 设置主刻度间隔为 0.05 s
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    
    ax.legend(loc='best')
    plt.tight_layout()

    try:
        output_image_filename = f"{base_filename}_{identifier}_averaged_cycle_corrected.png" ## 输出图片名
        output_dir = os.path.join(output_dir, output_image_filename)
    
        plt.savefig(output_dir, dpi=300, bbox_inches='tight')
        print(f"成功: 平均心跳周期图已保存到 {output_dir}")
        # plt.show()  # 显示图形
        return True   # 成功返回True，便于统计saved_avg_cycle_images_count
    except Exception as e:
        print(f"错误: 保存平均心跳周期图时出错: {e}")
        return False
    finally:
        plt.close(fig)  # 关闭图形以释放内存




"""定义绘图函数"""
## 第一张图：原始信号和滤波信号（包含R峰）
def plot_signals_with_r_peaks(time, Bx_raw, Bx_filtered, By_raw, By_filtered, R_peaks_Bx, R_peaks_By):
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    xlim_0 = 0
    xlim_1 = 30
    # Bx 原始信号
    axs[0, 0].plot(time, Bx_raw, label='Raw_data', color='royalblue', alpha=0.7)
    axs[0, 0].set_title('Raw Signal')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 0].set_xlim(xlim_0, xlim_1)
    axs[0, 0].set_ylim(0, 25)

    # By 原始信号
    axs[0, 1].plot(time, By_raw, label='By_Raw', color='royalblue', alpha=0.7)
    axs[0, 1].set_title('By Raw Signal')
    axs[0, 1].set_xlabel('Time(s)')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[0, 1].set_xlim(xlim_0, xlim_1)

    # Bx 滤波信号及R峰
    axs[1, 0].plot(time, Bx_filtered, label='Filtered_R', color='royalblue')
    if len(R_peaks_Bx) > 0:
        axs[1, 0].scatter(time[R_peaks_Bx], Bx_filtered[R_peaks_Bx], facecolors='none', edgecolors='r', marker='o', label='Bx_R peaks')
    axs[1, 0].set_title('Filtered Signal with R Peaks')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[0, 0].set_xlim(xlim_0, xlim_1)
    axs[1, 0].set_ylim(-3, 3)

    # By 滤波信号及R峰
    axs[1, 1].plot(time, By_filtered, label='By_filtered', color='royalblue')
    if len(R_peaks_By) > 0:
        axs[1, 1].scatter(time[R_peaks_By], By_filtered[R_peaks_By], facecolors='none', edgecolors='r', marker='o', label='By_R peaks')
    axs[1, 1].set_title('By Filtered Signal with R Peaks')
    axs[1, 1].set_xlabel('Time(s)')
    axs[1, 1].set_ylabel('Amplitude')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    axs[1, 1].set_xlim(xlim_0, xlim_1)
    axs[1, 1].set_ylim(-3, 3)

    plt.tight_layout()
    # plt.show()
    return fig  # 返回图形对象以便后续处理或保存


""" 提取天数标签 """
def extract_day_label_from_folder(folder_name):
    match_detailed = re.search(r'[Dd](\d+)', folder_name) 
    if match_detailed:
        return f"day {match_detailed.group(1)}"
    match_simple_day = re.search(r'day(\d+)', folder_name, re.IGNORECASE)
    if match_simple_day:
        return f"day {match_simple_day.group(1)}"
    print(f"警告：无法从文件夹 '{folder_name}' 中提取标准日龄标签。将使用文件夹名作为标签。")
    return folder_name