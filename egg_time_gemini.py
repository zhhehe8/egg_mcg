"""Fig2 x,y轴原始信号、滤波信号、R峰展示"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks
import os

# --- Configuration Parameters  ---
SAMPLING_RATE = 1000 
FILE_PATH = '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B30_t1_待破壳.txt'  # 输入数据文件路径 
SKIP_HEADER_LINES = 2
FILE_ENCODING = 'utf-8'
OUTPUT_FIGURE_DIR = '/Users/yanchen/Desktop'          # 输出目录

# Filter Parameters (General)
FILTER_ORDER_BANDPASS = 4
LOWCUT_FREQ = 1.5  # Hz
HIGHCUT_FREQ = 45.0 # Hz
NOTCH_FREQ_MAINS = 50.0  # Hz
Q_FACTOR_NOTCH = 30.0

# R-Peak Detection Parameters - Per Channel
# To use the same R-peak parameters for both channels, set CH1 and CH2 parameters to be identical.
# For example:
# R_PEAK_MIN_HEIGHT_FACTOR_COMMON = 0.3
# R_PEAK_MIN_DISTANCE_MS_COMMON = 150
# R_PEAK_MIN_HEIGHT_FACTOR_CH1 = R_PEAK_MIN_HEIGHT_FACTOR_COMMON
# R_PEAK_MIN_DISTANCE_MS_CH1 = R_PEAK_MIN_DISTANCE_MS_COMMON
# R_PEAK_MIN_HEIGHT_FACTOR_CH2 = R_PEAK_MIN_HEIGHT_FACTOR_COMMON
# R_PEAK_MIN_DISTANCE_MS_CH2 = R_PEAK_MIN_DISTANCE_MS_COMMON

# Bx Parameters
R_PEAK_MIN_HEIGHT_FACTOR_CH1 = 0.3 # Example: Set this to your desired common value
R_PEAK_MIN_DISTANCE_MS_CH1 = 150  # ms, Example: Set this to your desired common value

# By Parameters
R_PEAK_MIN_HEIGHT_FACTOR_CH2 = 0.3 # Example: Make this same as CH1 for common parameters
R_PEAK_MIN_DISTANCE_MS_CH2 = 150  # ms, Example: Make this same as CH1 for common parameters


# Signal Inversion Flags (Set to True to invert the respective channel's signal)
INVERT_CHANNEL_1 = False # For Bx
INVERT_CHANNEL_2 = False # For By

# --- Function Definitions ---

def load_cardiac_data(filepath, skip_header, file_encoding='utf-8'):
    """
    Loads cardiac data from a text file.
    Expects at least two columns for Bx and By.
    """
    try:
        data = np.loadtxt(filepath, skiprows=skip_header, dtype=float, encoding=file_encoding)
        if data.ndim == 1:
            print(f"警告: 文件 {filepath} 只包含一列数据。此脚本需要至少两列。")
            return None, None
        if data.shape[1] < 2:
            print(f"错误：数据文件 {filepath} 至少需要两列才能处理两个通道。当前列数: {data.shape[1]}")
            return None, None
        channel1 = data[:, 0]
        channel2 = data[:, 1]
        return channel1, channel2
    except UnicodeDecodeError as ude:
        print(f"使用 '{file_encoding}' 编码读取文件 '{filepath}' 时出错: {ude}")
        print("如果编码不正确，请尝试其他编码，例如 'gbk' 或 'latin-1'。")
        return None, None
    except Exception as e:
        print(f"读取文件 '{filepath}' 或解析数据时出错: {e}")
        return None, None

def apply_bandpass_filter(data, lowcut, highcut, fs, order):
    """Applies a bandpass Butterworth filter."""
    if data is None: return None
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0:
        high = 0.99
        print(f"警告: highcut频率({highcut}Hz)过高,已调整为奈奎斯特频率的99%。")
    if low <= 0:
        low = 0.001
        print(f"警告: lowcut频率({lowcut}Hz)过低,已调整为0.001*奈奎斯特频率。")
    if low >= high:
        print(f"错误: 带通滤波器的低截止({lowcut}Hz)必须小于高截止({highcut}Hz)。跳过带通滤波。")
        return data # Return original data if filter is invalid
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def apply_notch_filter(data, notch_freq, quality_factor, fs):
    """Applies a notch filter."""
    if data is None: return None
    nyquist = 0.5 * fs
    freq_normalized = notch_freq / nyquist
    if freq_normalized >= 1.0 or freq_normalized <= 0:
        print(f"警告: 陷波频率 {notch_freq}Hz (归一化后 {freq_normalized}) 无效。跳过陷波滤波。")
        return data # Return original data if filter is invalid
    b, a = iirnotch(freq_normalized, quality_factor)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def find_r_peaks_indices(data, fs, min_height_factor, min_distance_ms, channel_name):
    """
    Finds R-peak indices in the signal.
    Returns: np.array of peak indices or empty array if none found/error.
    """
    if data is None or len(data) == 0:
        print(f"警告 ({channel_name}): R峰检测的输入数据为空。")
        return np.array([])

    data_max_val = np.max(data)
    min_h = data_max_val * min_height_factor

    if min_h <= 0:
        data_std = np.std(data)
        min_h = data_std if data_std > 1e-9 else 1e-3
        print(f"提示 ({channel_name}): 动态计算的R峰高度阈值 ({data_max_val * min_height_factor:.3f} pT) <=0. "
              f"已调整为基于标准差或默认值的阈值: {min_h:.3f} pT")

    min_dist_samples = int((min_distance_ms / 1000.0) * fs)

    print(f"\nR峰检测参数 ({channel_name}):")
    print(f"  数据最大值: {data_max_val:.3f} pT")
    print(f"  最小高度因子: {min_height_factor*100:.1f}% -> 计算得到的最小高度阈值 (height): {min_h:.3f} pT")
    print(f"  最小峰间距: {min_distance_ms} ms -> {min_dist_samples} 个采样点")

    height_param = min_h
    try:
        peaks_indices, _ = find_peaks(data, height=height_param, distance=min_dist_samples)
    except Exception as e:
        print(f"错误 ({channel_name}): 调用 find_peaks 时出错: {e}")
        print(f"  使用的参数: height={height_param}, distance={min_dist_samples}")
        return np.array([])

    if len(peaks_indices) == 0:
        print(f"警告: 在 {channel_name} 未检测到R峰。请检查信号形态或调整该通道的R峰检测参数 (高度因子/距离)。")
    else:
        print(f"在 {channel_name} 检测到 {len(peaks_indices)} 个R峰。")
    return peaks_indices

def plot_raw_and_filtered_2x2(time_axis,
                                ch1_raw_data, ch1_filtered_data, ch1_name,
                                ch2_raw_data, ch2_filtered_data, ch2_name,
                                sampling_rate,
                                output_dir=None, base_filename=""):
    """
    Plots raw signals and filtered signals for two channels on a 2x2 subplot grid,
    limited to the first 30 seconds.
    Row 1: Raw signals (Ch1, Ch2)
    Row 2: Filtered signals (Ch1, Ch2)
    """
    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True) # Changed to 2x2
    fig.suptitle('原始信号与滤波后信号对比 (前30秒)', fontsize=18, y=0.98)

    max_samples_30s = int(30 * sampling_rate)
    current_max_samples = len(time_axis)
    plot_samples = min(max_samples_30s, current_max_samples)
    time_axis_30s = time_axis[:plot_samples]

    # Plotting Ch1 Raw
    ax = axs[0, 0]
    if ch1_raw_data is not None:
        ax.plot(time_axis_30s, ch1_raw_data[:plot_samples], label=f'{ch1_name} - 原始信号', color='blue', alpha=0.8, linewidth=1)
        ax.set_title(f'{ch1_name} - 原始信号')
        ax.set_ylabel('磁场强度 (pT)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, f'{ch1_name} 原始数据不可用', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{ch1_name} - 原始信号')
    ax.grid(True)

    # Plotting Ch2 Raw
    ax = axs[0, 1]
    if ch2_raw_data is not None:
        ax.plot(time_axis_30s, ch2_raw_data[:plot_samples], label=f'{ch2_name} - 原始信号', color='green', alpha=0.8, linewidth=1)
        ax.set_title(f'{ch2_name} - 原始信号')
        ax.set_ylabel('磁场强度 (pT)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, f'{ch2_name} 原始数据不可用', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{ch2_name} - 原始信号')
    ax.grid(True)

    # Plotting Ch1 Filtered
    ax = axs[1, 0]
    if ch1_filtered_data is not None:
        ax.plot(time_axis_30s, ch1_filtered_data[:plot_samples], label=f'{ch1_name} - 完全滤波后', color='red', linewidth=1.2)
        ax.set_title(f'{ch1_name} - 完全滤波后')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('磁场强度 (pT)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, f'{ch1_name} 滤波数据不可用', ha='center', va='center', transform=ax.transAxes, color='red')
        ax.set_title(f'{ch1_name} - 完全滤波后')
        ax.set_xlabel('时间 (秒)')
    ax.grid(True)

    # Plotting Ch2 Filtered
    ax = axs[1, 1]
    if ch2_filtered_data is not None:
        ax.plot(time_axis_30s, ch2_filtered_data[:plot_samples], label=f'{ch2_name} - 完全滤波后', color='purple', linewidth=1.2)
        ax.set_title(f'{ch2_name} - 完全滤波后')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('磁场强度 (pT)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, f'{ch2_name} 滤波数据不可用', ha='center', va='center', transform=ax.transAxes, color='red')
        ax.set_title(f'{ch2_name} - 完全滤波后')
        ax.set_xlabel('时间 (秒)')
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle and bottom labels

    if output_dir and base_filename:
        try:
            filename = f"{base_filename}_F1_RawFiltered_2x2_30s.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图1 (2x2) 已保存到: {save_path}")
        except Exception as e:
            print(f"错误：无法保存图1 (2x2) {filename}。原因: {e}")
    plt.show()
    plt.close(fig)

def plot_r_peaks_1x2_30s(time_axis_full, # Full time axis for reference if needed for R-peak logic
                           ch1_filtered_data_full, ch1_r_peaks_indices, ch1_name,
                           ch2_filtered_data_full, ch2_r_peaks_indices, ch2_name,
                           sampling_rate,
                           output_dir=None, base_filename=""):
    """
    Plots filtered data and R-peaks for two channels on a 1x2 subplot grid,
    limited to the first 30 seconds.
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharex=True) # Changed to 1x2
    fig.suptitle('R峰检测结果 (前30秒)', fontsize=16, y=0.98)

    max_samples_30s = int(30 * sampling_rate)
    current_max_samples_full = len(time_axis_full)
    plot_samples = min(max_samples_30s, current_max_samples_full)
    
    time_axis_30s = time_axis_full[:plot_samples]

    # Channel 1 (Bx) R-peaks for first 30s
    ax = axs[0]
    if ch1_filtered_data_full is not None:
        ch1_filtered_data_30s = ch1_filtered_data_full[:plot_samples]
        ax.plot(time_axis_30s, ch1_filtered_data_30s, label=f'{ch1_name} - 滤波信号', linewidth=0.8, color='blue')
        
        # Filter R-peaks that fall within the first 30 seconds
        if ch1_r_peaks_indices is not None and len(ch1_r_peaks_indices) > 0:
            peaks_in_30s_ch1 = ch1_r_peaks_indices[ch1_r_peaks_indices < plot_samples]
            if len(peaks_in_30s_ch1) > 0:
                ax.plot(time_axis_full[peaks_in_30s_ch1], ch1_filtered_data_full[peaks_in_30s_ch1], 
                        "x", color='red', markersize=8, label=f'检测到的R峰 ({len(peaks_in_30s_ch1)}个)')
            else:
                 ax.text(0.5, 0.4, f'前30s无R峰', ha='center', va='center', transform=ax.transAxes, color='orange')
        else:
            ax.text(0.5, 0.5, '未检测到R峰', ha='center', va='center', transform=ax.transAxes, color='red')
        
        ax.set_title(f'{ch1_name} - R峰检测 (前30s)')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('磁场强度 (pT)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, f'{ch1_name} 滤波数据不可用', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{ch1_name} - R峰检测 (前30s)')
        ax.set_xlabel('时间 (秒)')
    ax.grid(True)

    # Channel 2 (By) R-peaks for first 30s
    ax = axs[1]
    if ch2_filtered_data_full is not None:
        ch2_filtered_data_30s = ch2_filtered_data_full[:plot_samples]
        ax.plot(time_axis_30s, ch2_filtered_data_30s, label=f'{ch2_name} - 滤波信号', linewidth=0.8, color='green')

        if ch2_r_peaks_indices is not None and len(ch2_r_peaks_indices) > 0:
            peaks_in_30s_ch2 = ch2_r_peaks_indices[ch2_r_peaks_indices < plot_samples]
            if len(peaks_in_30s_ch2) > 0:
                ax.plot(time_axis_full[peaks_in_30s_ch2], ch2_filtered_data_full[peaks_in_30s_ch2], 
                        "x", color='purple', markersize=8, label=f'检测到的R峰 ({len(peaks_in_30s_ch2)}个)')
            else:
                ax.text(0.5, 0.4, f'前30s无R峰', ha='center', va='center', transform=ax.transAxes, color='orange')

        else:
            ax.text(0.5, 0.5, '未检测到R峰', ha='center', va='center', transform=ax.transAxes, color='red')

        ax.set_title(f'{ch2_name} - R峰检测 (前30s)')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('磁场强度 (pT)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, f'{ch2_name} 滤波数据不可用', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{ch2_name} - R峰检测 (前30s)')
        ax.set_xlabel('时间 (秒)')
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect for suptitle and bottom labels

    if output_dir and base_filename:
        try:
            filename = f"{base_filename}_F2_RPeaks_1x2_30s.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图2 (1x2 R峰) 已保存到: {save_path}")
        except Exception as e:
            print(f"错误：无法保存图2 (1x2 R峰) {filename}。原因: {e}")
    plt.show()
    plt.close(fig)

# --- Main Program ---
if __name__ == "__main__":
    print(f"开始处理文件: {FILE_PATH}")
    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)
    file_base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]

    ch1_data_raw_orig, ch2_data_raw_orig = load_cardiac_data(FILE_PATH, SKIP_HEADER_LINES, FILE_ENCODING)

    ch1_name = "Bx"
    ch2_name = "By"
    ch1_data_raw, ch2_data_raw = None, None

    if ch1_data_raw_orig is not None:
        ch1_data_raw = -ch1_data_raw_orig.copy() if INVERT_CHANNEL_1 else ch1_data_raw_orig.copy()
        if INVERT_CHANNEL_1: print(f"信息: {ch1_name}的信号已反转。")
    else:
        print(f"警告: {ch1_name} 原始数据加载失败或为空。")

    if ch2_data_raw_orig is not None:
        ch2_data_raw = -ch2_data_raw_orig.copy() if INVERT_CHANNEL_2 else ch2_data_raw_orig.copy()
        if INVERT_CHANNEL_2: print(f"信息: {ch2_name}的信号已反转。")
    else:
        print(f"警告: {ch2_name} 原始数据加载失败或为空。")

    time_vector = None
    data_length = 0
    if ch1_data_raw is not None:
        data_length = len(ch1_data_raw)
    elif ch2_data_raw is not None:
        data_length = len(ch2_data_raw)
    
    if data_length > 0:
        time_vector = np.arange(data_length) / SAMPLING_RATE
    else:
        print("错误: 两个通道的数据均加载失败或长度为零。程序终止。")
        exit()

    ch1_data_filtered = None
    if ch1_data_raw is not None:
        print(f"\n--- 开始滤波 {ch1_name} ---")
        ch1_data_bandpassed = apply_bandpass_filter(ch1_data_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
        if ch1_data_bandpassed is not None:
            ch1_data_filtered = apply_notch_filter(ch1_data_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)
        if ch1_data_filtered is None: print(f"警告: {ch1_name} 滤波失败或返回None。")
    else:
        print(f"{ch1_name} 无原始数据进行滤波。")

    ch2_data_filtered = None
    if ch2_data_raw is not None:
        print(f"\n--- 开始滤波 {ch2_name} ---")
        ch2_data_bandpassed = apply_bandpass_filter(ch2_data_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
        if ch2_data_bandpassed is not None:
            ch2_data_filtered = apply_notch_filter(ch2_data_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)
        if ch2_data_filtered is None: print(f"警告: {ch2_name} 滤波失败或返回None。")
    else:
        print(f"{ch2_name} 无原始数据进行滤波。")

    # --- Plot 1: Raw and Filtered Signals (2x2, first 30s) ---
    print("\n--- 正在生成 图1: 原始与滤波信号对比图 (2x2, 前30秒) ---")
    plot_raw_and_filtered_2x2(time_vector,
                                ch1_data_raw, ch1_data_filtered, ch1_name,
                                ch2_data_raw, ch2_data_filtered, ch2_name,
                                SAMPLING_RATE,
                                OUTPUT_FIGURE_DIR, file_base_name)

    # --- R-Peak Detection (on full filtered signals) ---
    ch1_r_peaks_indices = np.array([])
    if ch1_data_filtered is not None:
        print(f"\n--- 开始R峰检测 {ch1_name} ---")
        ch1_r_peaks_indices = find_r_peaks_indices(ch1_data_filtered, SAMPLING_RATE,
                                                 R_PEAK_MIN_HEIGHT_FACTOR_CH1,
                                                 R_PEAK_MIN_DISTANCE_MS_CH1, ch1_name)
    else:
        print(f"{ch1_name} 无滤波数据进行R峰检测。")

    ch2_r_peaks_indices = np.array([])
    if ch2_data_filtered is not None:
        print(f"\n--- 开始R峰检测 {ch2_name} ---")
        ch2_r_peaks_indices = find_r_peaks_indices(ch2_data_filtered, SAMPLING_RATE,
                                                 R_PEAK_MIN_HEIGHT_FACTOR_CH2,
                                                 R_PEAK_MIN_DISTANCE_MS_CH2, ch2_name)
    else:
        print(f"{ch2_name} 无滤波数据进行R峰检测。")

    # --- Plot 2: R-Peak Detection (1x2, first 30s) ---
    print("\n--- 正在生成 图2: R峰检测组合图 (1x2, 前30秒) ---")
    plot_r_peaks_1x2_30s(time_vector, # Full time vector
                           ch1_data_filtered, ch1_r_peaks_indices, ch1_name, # Full filtered data and all peaks
                           ch2_data_filtered, ch2_r_peaks_indices, ch2_name, # Full filtered data and all peaks
                           SAMPLING_RATE,
                           OUTPUT_FIGURE_DIR, file_base_name)

    print("\n--- 所有指定图表处理完毕 ---")