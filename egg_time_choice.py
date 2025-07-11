"""将12-20天的心磁数据进行批量处理，提取平均心跳周期并生成汇总图。"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 设置默认字体
plt.rcParams['font.sans-serif'] = ['Arial'] 

# 尝试导入项目中的模块和配置
try:
    import config
    import mcg_processing as mcg
except ImportError:
    print("错误: 无法导入 'config.py' 或 'mcg_processing.py'。")
    print("请确保此脚本与这些文件在同一目录下，或其路径已添加到 PYTHONPATH。")
    sys.exit(1)


def get_averaged_waveform(filepath: Path):
    """
    对单个文件执行完整的信号处理流程，并返回中位数平均波形。
    处理流程与 main.py 和 main_batch_waveform_plot.py 一致。
    """
    print(f"正在处理文件: {filepath.name}...")
    # 1. 加载数据
    bx_raw, by_raw = mcg.load_cardiac_data(filepath)
    if bx_raw is None:
        return None, None

    # 2. 信号预处理与滤波
    fs = config.PROCESSING_PARAMS['fs']
    bx_filtered = mcg.apply_bandpass_filter(bx_raw, fs, **config.FILTER_PARAMS['bandpass'])
    bx_filtered = mcg.apply_notch_filter(bx_filtered, fs, **config.FILTER_PARAMS['notch'])
    if config.FILTER_PARAMS['wavelet']['enabled']:
        wavelet_args = {
        'wavelet': config.FILTER_PARAMS['wavelet']['wavelet'],
        'level': config.FILTER_PARAMS['wavelet']['level'],
        'denoise_levels': config.FILTER_PARAMS['wavelet']['denoise_levels']
    }
        bx_filtered = mcg.apply_wavelet_denoise(bx_filtered, **wavelet_args)
        bx_filtered = mcg.apply_savgol_filter(bx_filtered)

    # 3. R峰检测
    peaks = mcg.find_r_peaks(bx_filtered, fs, **config.R_PEAK_PARAMS)
    precise_peaks = mcg.interpolate_r_peaks(bx_filtered, peaks)

    if len(precise_peaks) < 5:
        print(f"  -> R峰数量不足，跳过。")
        return None, None

    # 4. 定义提取和校正参数
    pre_r_samples = int(config.AVERAGING_PARAMS['pre_r_ms'] * fs / 1000)
    post_r_samples = int(config.AVERAGING_PARAMS['post_r_ms'] * fs / 1000)
    total_cycle_samples = pre_r_samples + post_r_samples

    # 定义用于计算基线偏移的窗口（取每个心拍前30ms）
    baseline_samples_count = int((min(config.AVERAGING_PARAMS['pre_r_ms'], 30) / 1000.0) * fs)
    
    individually_corrected_cycles = []

    # 5. 循环处理每一个R峰，提取并校正对应的周期
    for r_peak_idx in precise_peaks:
        r_peak_idx = int(r_peak_idx)
        start_idx = r_peak_idx - pre_r_samples
        end_idx = r_peak_idx + post_r_samples
        
        if start_idx >= 0 and end_idx <= len(bx_filtered):
            # 提取单个原始心拍
            cycle_data_raw = bx_filtered[start_idx:end_idx]
            
            if len(cycle_data_raw) == total_cycle_samples:
                # 计算该心拍的基线偏移量
                baseline_offset = np.mean(cycle_data_raw[:baseline_samples_count])
                # 校正该心拍
                cycle_data_corrected = cycle_data_raw - baseline_offset
                individually_corrected_cycles.append(cycle_data_corrected)

    if len(individually_corrected_cycles) < 2:
        print(f"  -> 可用的校正后心拍数量不足，跳过。")
        return None, None

    # 6. 对所有已校正的独立周期进行中位数平均
    cycles_array_corrected = np.array(individually_corrected_cycles)
    final_averaged_cycle = np.median(cycles_array_corrected, axis=0)

    # 7. 创建时间轴并返回最终结果
    time_axis_ms = np.linspace(-config.AVERAGING_PARAMS['pre_r_ms'], config.AVERAGING_PARAMS['post_r_ms'], len(final_averaged_cycle))
    
    return time_axis_ms, final_averaged_cycle


def plot_summary_figure(waveforms_data: list, labels: list, output_dir: Path):
    """
    根据收集到的所有波形数据，绘制并保存最终的汇总图。
    """
    if not waveforms_data:
        print("没有可供绘图的数据。")
        return

    # 创建2行1列的画布
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})

    # --- 绘制上图：错位排列的平均波形 ---
    ax_top = axs[0]
    ax_top.set_title('Averaged Waveforms(day12-20)', fontsize=16)
    
    peak_amplitudes = []
    x_offset_step = 80  # 每个波形在x轴上的偏移量

    # 定义一组颜色以便区分
    colors = plt.get_cmap('tab10', len(waveforms_data))

    for i, (time_axis, waveform, label) in enumerate(waveforms_data):
        x_offset = i * x_offset_step
        
        # 找到R峰
        r_peak_index = np.argmin(np.abs(time_axis))
        r_peak_amplitude = waveform[r_peak_index]
        peak_amplitudes.append(r_peak_amplitude)

        # 绘制波形
        ax_top.plot(time_axis + x_offset, waveform, color=colors(i))
        
        # 在峰值上方添加天数标签
        ax_top.text(x_offset, r_peak_amplitude + 0.1, label, fontsize=12, ha='center', color=colors(i))

    ax_top.set_xlabel('Time (ms)', fontsize=14)
    ax_top.set_ylabel('pT', fontsize=14)
    ax_top.grid(True, linestyle='--', alpha=0.6)
    ax_top.set_xlim(-100, x_offset + 100) # 调整X轴范围以显示所有波形
    ax_top.set_ylim(0, 1.2)
    ax_top.tick_params(axis='both', labelsize=13)
    # --- 绘制下图：R峰振幅的柱状图 ---
    ax_bottom = axs[1]
    ax_bottom.set_title('R-peak of Averaged Cycles', fontsize=16)
    x_positions = np.arange(len(labels))
    bars = ax_bottom.bar(x_positions, peak_amplitudes, color='royalblue', alpha=0.8, width=0.6)
    ax_bottom.set_xticks(x_positions)
    ax_bottom.set_xticklabels(labels)
    ax_bottom.set_ylabel('pT', fontsize=14)
    ax_bottom.grid(axis='y', linestyle='--', alpha=0.6)
    ax_bottom.set_ylim(0, 1.2)  # 设置Y轴范围以留出空间标注
    ax_bottom.tick_params(axis='both', labelsize=13)
    # 在每个柱的顶部标注数值
    for bar in bars:
        yval = bar.get_height()
        ax_bottom.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)

    # --- 显示最终图像 ---
    plt.tight_layout()
    save_path = output_dir / 'summary_waveform_plot.jpg'
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 汇总图已成功保存至: {save_path}")
    except Exception as e:
        print(f"\n❌ 保存图片失败: {e}")
    # ---------------------------
    plt.show()


if __name__ == "__main__":
    # 保存目录
    output_dir = Path('/Users/yanchen/Desktop/Projects/egg/egg_2025/Figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # 【12-20天（共9个）数据文件的绝对路径
    # ======================================================================
    base_dir = Path('/Users/yanchen/Desktop/Projects/egg/egg_2025/time_choice')
    file_paths = [
       base_dir / 'egg_d12_B22_t1.txt',
       base_dir / 'egg_d13_B20_t2.txt',
       base_dir / 'egg_d14_B25_t1.txt',
       base_dir / 'egg_d15_B28_t1.txt',
       base_dir / 'egg_d16_B25_t2.txt',
       base_dir / 'egg_d17_B25_t2.txt',
       base_dir / 'egg_d20_B33_t1.txt',
       base_dir / 'egg_d19_B34_t2.txt',
       base_dir / 'egg_d20_B30_t1_待破壳.txt',
    ]
    # ======================================================================

    day_labels = [f'day{i}' for i in range(12, 21)]
    collected_waveforms = []

    # 循环处理每个文件
    for path, label in zip(file_paths, day_labels):
        if not path.exists():
            print(f"警告: 文件不存在，跳过 -> {path}")
            continue
        
        time_axis, median_beat = get_averaged_waveform(path)
        
        if median_beat is not None:
            collected_waveforms.append((time_axis, median_beat, label))

    # 绘制汇总图
    plot_summary_figure(collected_waveforms, day_labels,  output_dir)
