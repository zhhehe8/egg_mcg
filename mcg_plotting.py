# mcg_plotting.py
"""可视化模块，包含所有绘图函数"""
import zhplot
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置字体显示
try:
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 设置标题字体加粗
    plt.rcParams['axes.titleweight'] = 'bold'  # 子图标题加粗
    plt.rcParams['figure.titleweight'] = 'bold'  # 主标题加粗
    # 设置坐标轴标签字体加粗
    plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
    print("字体设置成功: Arial (标题和坐标轴标签已设置为加粗)")
except Exception as e:
    print(f"警告: 字体设置失败: {e}")
    # 备用字体设置
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

def plot_single_channel_filtered(time, raw_signal, filtered_signal, r_peaks, channel_name, output_path):
    """ 为单个通道绘制 2x1 的滤波前后对比图。"""
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    # fig.suptitle(f'{channel_name} 信号处理前后对比', fontsize=16)
    # RAW DATA
    axs[0].plot(time, raw_signal, 'gray', alpha=0.9, label='Raw Signal')
    axs[0].set_title('Raw Signal', fontsize=16, fontweight='bold')
    axs[0].set_ylabel('Amplitude (pT)', fontsize=14, fontweight='bold')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].tick_params(axis='y', labelsize=12)
    # FILTERED DATA
    filtered_signal += 2
    axs[1].plot(time, filtered_signal, 'blue' if 'Bx' in channel_name else 'green', label='Filtered Signal')
    if len(r_peaks) > 0:
        axs[1].plot(time[r_peaks], filtered_signal[r_peaks], 'rX', markersize=10, label='R Peaks')
    axs[1].set_title('Filtered Signal', fontsize=16, fontweight='bold')
    axs[1].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('Amplitude (pT)', fontsize=14, fontweight='bold')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].tick_params(axis='both', labelsize=14)
    axs[1].set_xlim(3, 12)  # 设置x轴范围
    axs[1].set_ylim(0, 8)  # 设置y轴范围
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.show()

def plot_single_channel_averaging(
        median_beat, dtw_beat, time_axis_ms, channel_name, base_filename, output_path, display_method: str = 'median' 
):
    """  为单个通道根据指定方法绘制叠加平均波形。"""
    title_map = {
        'median': 'Median-averaged cardiac cycle',
        'dtw': 'DTW-aligned averaged cardiac cycle',
        'both': 'Averaging method comparison (Median vs DTW)'
    }

    plt.figure(figsize=(5, 4))
    # plt.title(f'{base_filename} - {channel_name}\n{title_map.get(display_method, "叠加平均波形")}', fontsize=16)
    plt.title('Averaged Waveforms on the 20 day', fontsize=16, fontweight='bold')

    offset = 0.2   # 平均波形y轴的偏移量

    if display_method in ['median', 'both']:
        style = '--' if display_method == 'both' else '-'
        label = 'Averaged Waveforms' if display_method == 'both' else 'Averaged Waveforms'
        plt.plot(time_axis_ms, median_beat + offset, color='darkorange', linestyle=style, lw=2, label=label)
        
    if display_method in ['dtw', 'both']:
        label = 'DTW对齐平均' if display_method == 'both' else 'DTW对齐平均波形'
        plt.plot(time_axis_ms, dtw_beat + offset, color=('blue' if 'Bx' in channel_name else 'green'), lw=2, label=label)

    plt.xlabel('Time relative to R peak (ms)', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude (pT)', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.5)
    plt.grid(True, linestyle='--')
    plt.axvline(x=0, color='red', linestyle='-.', alpha=0.8)
    plt.legend()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

# --- 双通道整合绘图函数 ---

def plot_dual_channel_filtered(time, bx_raw, bx_filtered, r_peaks_bx, by_raw, by_filtered, r_peaks_by, base_filename, output_dir):
    """ 绘制一张2x2图，整合对比Bx和By信号滤波前后的效果。"""
    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    fig.suptitle(f'滤波前后信号对比 (双通道) - {base_filename}', fontsize=18)
    axs[0, 0].plot(time, bx_raw, 'gray', alpha=0.9, label='原始信号')
    axs[0, 0].set_title('Bx 通道 - 原始信号')
    axs[0, 0].set_ylabel('幅度')
    axs[0, 0].grid(True); axs[0, 0].legend()
    axs[1, 0].plot(time, bx_filtered, 'blue', label='滤波后信号')
    if len(r_peaks_bx) > 0:
        axs[1, 0].plot(time[r_peaks_bx], bx_filtered[r_peaks_bx], 'rX', markersize=8, label='R 峰')
    axs[1, 0].set_title('Bx 通道 - 滤波后信号')
    axs[1, 0].set_xlabel('时间 (秒)'); axs[1, 0].set_ylabel('幅度')
    axs[1, 0].grid(True); axs[1, 0].legend()
    axs[0, 1].plot(time, by_raw, 'gray', alpha=0.9, label='原始信号')
    axs[0, 1].set_title('By 通道 - 原始信号')
    axs[0, 1].grid(True); axs[0, 1].legend()
    axs[1, 1].plot(time, by_filtered, 'green', label='滤波后信号')
    if len(r_peaks_by) > 0:
        axs[1, 1].plot(time[r_peaks_by], by_filtered[r_peaks_by], 'rX', markersize=8, label='R 峰')
    axs[1, 1].set_title('By 通道 - 滤波后信号')
    axs[1, 1].set_xlabel('时间 (秒)')
    axs[1, 1].grid(True); axs[1, 1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / f"{base_filename}_filtering_comparison_dual.png"
    plt.savefig(output_path, dpi=300)
    print(f"双通道滤波对比图已保存至: {output_path}")
    plt.show()

def plot_dual_channel_averaging(time_axis_ms, median_beat_bx, dtw_beat_bx, median_beat_by, dtw_beat_by, base_filename, output_dir, display_method: str = 'both'):
    """ 绘制一张2x1图，整合对比Bx和By信号的两种叠加平均方法(中位数和DTW对齐)。"""
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    title_map = {
        'median': 'Median-averaged cardiac cycle',
        'dtw': 'DTW-aligned averaged cardiac cycle',
        'both': 'Averaging method comparison (Median vs DTW)'
    }
    fig.suptitle(f'叠加平均方法对比 (双通道) - {base_filename}', fontsize=18)
    axs[0].set_title('Bx 通道')
    if display_method in ['median', 'both']:
        style = '--' if display_method == 'both' else '-'
        axs[0].plot(time_axis_ms, median_beat_bx, color='darkorange', linestyle=style, lw=2, label='中位数平均')
    if display_method in ['dtw', 'both']:
        axs[0].plot(time_axis_ms, dtw_beat_bx, color='blue', lw=2, label='DTW对齐平均')
    axs[0].set_ylabel('幅度')
    axs[0].grid(True, linestyle='--'); axs[0].axvline(x=0, color='r', linestyle='-.', alpha=0.8); axs[0].legend()

    axs[1].set_title('By 通道')
    if display_method in ['median', 'both']:
        style = '--' if display_method == 'both' else '-'
        axs[1].plot(time_axis_ms, median_beat_by, color='darkorange', linestyle=style, lw=2, label='中位数平均')
    if display_method in ['dtw', 'both']:
        axs[1].plot(time_axis_ms, dtw_beat_by, color='green', lw=2, label='DTW对齐平均')
    axs[1].set_xlabel('相对于R峰的时间 (毫秒)'); axs[1].set_ylabel('幅度')
    axs[1].grid(True, linestyle='--'); axs[1].axvline(x=0, color='r', linestyle='-.', alpha=0.8); axs[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = f"{base_filename}_averaging_{display_method}.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300)
    print(f"叠加平均图 ({display_method} 模式) 已保存至: {output_path}")
    plt.show()
# -----------------------

""" 时频分析绘图函数 """
def plot_single_channel_tf(
    t_axis, f_axis, Z_data, 
    method_name: str, channel_name: str, base_filename: str, output_path: Path
):
    """为单个通道绘制时频图。"""
    plt.figure(figsize=(12, 8))
    
    # 使用对数色阶来更好地显示幅度变化
    if method_name == 'STFT':
        plt.pcolormesh(t_axis, f_axis, np.log1p(Z_data), shading='gouraud', vmin=0, vmax=0.3, cmap='bwr')
    else: # CWT
        plt.pcolormesh(t_axis, f_axis, Z_data, shading='gouraud', cmap='bwr')

    plt.title(f'{base_filename} - {channel_name}\n时频分析 ({method_name})', fontsize=16, fontweight='bold')
    plt.xlabel('时间 (秒)', fontweight='bold')
    plt.ylabel('频率 (Hz)', fontweight='bold')
    plt.ylim(0, 10) 
    plt.colorbar(label='幅度 (对数尺度)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def plot_dual_channel_tf(
    tf_results: dict, 
    time_axis: np.ndarray, 
    base_filename: str, 
    output_dir: Path
):
    """为两个通道整合绘制所有可用的时频图。"""
    for method, results in tf_results.items():
        if not results: continue

        fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f'双通道时频分析 ({method.upper()}) - {base_filename}', fontsize=18)

        # Bx 通道
        f_bx, Z_bx = results['bx']
        im1 = axs[0].pcolormesh(time_axis, f_bx, Z_bx, shading='gouraud', cmap='bwr')
        axs[0].set_title('Bx 通道')
        axs[0].set_ylabel('频率 (Hz)', fontweight='bold')
        axs[0].set_ylim(0, 10)
        
        fig.colorbar(im1, ax=axs[0], label='幅度')
        
        # By 通道
        f_by, Z_by = results['by']
        im2 = axs[1].pcolormesh(time_axis, f_by, Z_by, shading='gouraud', cmap='bwr')
        axs[1].set_title('By 通道')
        axs[1].set_xlabel('时间 (秒)', fontweight='bold')
        axs[1].set_ylabel('频率 (Hz)', fontweight='bold')
        axs[1].set_ylim(0, 10)
        fig.colorbar(im2, ax=axs[1], label='幅度')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = output_dir / f"{base_filename}_tf_comparison_{method.lower()}.png"
        plt.savefig(output_path, dpi=300)
        print(f"双通道 {method.upper()} 时频图已保存至: {output_path}")
        plt.show()