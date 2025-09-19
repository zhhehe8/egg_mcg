"""

综合心跳周期图绘制脚本

上半部分：三组数据（d13, d19, d20）的平均心跳周期对比图（1行3列）

下半部分：12-20天数据的汇总图

"""



import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config
import mcg_processing as mcg



# 设置matplotlib字体

plt.rcParams['font.sans-serif'] = ['Arial']

plt.rcParams['font.size'] = 12



def process_and_extract_averaged_beat(filepath: Path):

    """

    处理单个文件并提取平均心跳周期（仅处理第一列Bx数据）

    返回: (averaged_beat_bx, time_axis_ms, embryo_info)

    """

    print(f"正在处理: {filepath.name}")

    

    # 1. 加载数据（仅使用第一列Bx数据）

    bx_raw, by_raw = mcg.load_cardiac_data(filepath)

    if bx_raw is None:

        print(f"  数据加载失败: {filepath.name}")

        return None, None, None

    

    # 2. 数据预处理

    fs = config.PROCESSING_PARAMS['fs']

    duration_s = config.PROCESSING_PARAMS.get('analysis_duration_s')

    

    if duration_s is not None and duration_s > 0:

        num_samples = int(duration_s * fs)

        if len(bx_raw) > num_samples:

            bx_raw = bx_raw[:num_samples]

            by_raw = by_raw[:num_samples]

    

    # 信号反转（如果需要，仅处理Bx）

    if config.PROCESSING_PARAMS['reverse_Bx']:

        bx_raw = -bx_raw

    

    # 3. 滤波处理（仅处理Bx通道）

    bx_filtered = mcg.apply_bandpass_filter(bx_raw, fs=fs, **config.FILTER_PARAMS['bandpass'])

    bx_filtered = mcg.apply_notch_filter(bx_filtered, fs=fs, **config.FILTER_PARAMS['notch'])

    

    if config.FILTER_PARAMS['wavelet']['enabled']:

        wavelet_args = {

            'wavelet': config.FILTER_PARAMS['wavelet']['wavelet'],

            'level': config.FILTER_PARAMS['wavelet']['level'],

            'denoise_levels': config.FILTER_PARAMS['wavelet']['denoise_levels']

        }

        bx_filtered = mcg.apply_wavelet_denoise(bx_filtered, **wavelet_args)

    

    bx_filtered = mcg.apply_savgol_filter(bx_filtered)

    

    # 4. R峰检测

    integer_peaks = mcg.find_r_peaks(bx_filtered, fs=fs, **config.R_PEAK_PARAMS)

    if len(integer_peaks) < 5:

        print(f"  R峰数量不足: {len(integer_peaks)}")

        return None, None, None

    

    precise_peaks = mcg.interpolate_r_peaks(bx_filtered, integer_peaks)

    print(f"  找到 {len(integer_peaks)} 个R峰")

    

    # 5. 提取心拍并计算平均

    pre_samples = int(config.AVERAGING_PARAMS['pre_r_ms'] * fs / 1000)

    post_samples = int(config.AVERAGING_PARAMS['post_r_ms'] * fs / 1000)

    

    all_beats = mcg._extract_beats(bx_filtered, precise_peaks, pre_samples, post_samples)

    if not all_beats:

        print(f"  无法提取心拍")

        return None, None, None

    

    # 计算平均心拍（使用中位数方法，仅处理Bx通道）

    averaged_beat_bx = mcg.get_median_beat(all_beats)

    

    # 创建时间轴

    time_axis_ms = np.linspace(-config.AVERAGING_PARAMS['pre_r_ms'], 

                              config.AVERAGING_PARAMS['post_r_ms'], 

                              len(averaged_beat_bx))

    

    # 提取胚胎信息

    embryo_info = {

        'filename': filepath.name,

        'day': extract_day_from_filename(filepath.name),

        'num_beats': len(all_beats)

    }

    

    print(f"  处理完成，平均了 {len(all_beats)} 个心拍")

    return averaged_beat_bx, time_axis_ms, embryo_info



def get_averaged_waveform_for_summary(filepath: Path):

    """

    处理单个文件并返回平均波形（用于12-20天汇总图）

    基于 egg_time_choice.py 的处理逻辑

    """

    print(f"正在处理汇总数据: {filepath.name}...")

    

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

    time_axis_ms = np.linspace(-config.AVERAGING_PARAMS['pre_r_ms'], 

                              config.AVERAGING_PARAMS['post_r_ms'], 

                              len(final_averaged_cycle))

    

    return time_axis_ms, final_averaged_cycle



def extract_day_from_filename(filename: str):

    """从文件名中提取天数信息"""

    import re

    day_match = re.search(r'd(\d+)', filename)

    return int(day_match.group(1)) if day_match else 0



def plot_comprehensive_figure(three_group_data, summary_waveforms_data, output_path=None):

    """

    绘制综合图：

    上半部分：1行3列的三组数据对比图

    下半部分：12-20天汇总图（包含错位排列的波形和R峰振幅柱状图）

    """

    # 创建复杂的子图布局

    fig = plt.figure(figsize=(20, 16))

    

    # 使用 GridSpec 来创建更灵活的布局

    from matplotlib.gridspec import GridSpec

    gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1.5, 0.8], hspace=0.3, wspace=0.2)

    

    # =================================================================

    # 上半部分：1行3列的三组数据对比图

    # =================================================================

    # fig.suptitle('Comprehensive Cardiac Cycle Analysis', fontsize=18, fontweight='bold', y=0.93)

    

    # 添加上半部分的总标题

    # fig.text(0.5, 0.88, 'Three Development Stages Comparison (Bx Channel)', 

    #          fontsize=18, fontweight='bold', ha='center')

    

    # 定义颜色

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 红色、青色、蓝色

    

    # 为了统一y轴范围，先找到所有数据的最大最小值（使用原始数据而非归一化）

    all_original_beats = []

    for averaged_beat_bx, time_axis_ms, embryo_info in three_group_data:

        if averaged_beat_bx is not None:

            all_original_beats.append(averaged_beat_bx)

    

    if all_original_beats:

        global_min = min([np.min(beat) for beat in all_original_beats])

        global_max = max([np.max(beat) for beat in all_original_beats])

        y_margin = (global_max - global_min) * 0.1

        ylim_range = [global_min - y_margin, global_max + y_margin]

    else:

        ylim_range = [-1.2, 1.2]

    

    # 绘制上半部分的三个子图

    # 定义每个子图的垂直偏移量

    y_offsets = [-0.04, 0.09, 0.15]  

    

    for i, (averaged_beat_bx, time_axis_ms, embryo_info) in enumerate(three_group_data):

        ax = fig.add_subplot(gs[0, i])

        

        if averaged_beat_bx is not None:

            # 使用原始数据，不进行归一化处理

            # normalized_beat = averaged_beat_bx / np.max(np.abs(averaged_beat_bx))

            

            # 应用垂直偏移

            offset_beat = averaged_beat_bx + y_offsets[i]

            

            # 绘制主要的心跳波形

            ax.plot(time_axis_ms, offset_beat, 

                   color=colors[i], 

                   linewidth=3, 

                   alpha=0.8,

                   label=f'Averaged waveform')

            

            # 添加R峰标记线

            # ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)

            

            # 添加背景颜色区域

            ax.axvspan(-100, -20, alpha=0.1, color='lightblue')

            ax.axvspan(-20, 20, alpha=0.1, color='lightcoral')

            ax.axvspan(20, 100, alpha=0.1, color='lightgreen')

            

            # 设置子图标题

            ax.set_title(f'Day {embryo_info["day"]}', 

                        fontsize=16, fontweight='bold', pad=10)

        else:

            # 如果数据无效，显示空白子图

            ax.text(0.5, 0.5, 'No Data Available', 

                   transform=ax.transAxes, fontsize=16, 

                   ha='center', va='center', color='red')

            ax.set_title(f'Day ? (No Data)', fontsize=16, color='red')



        # 设置子图属性

        ax.set_xlabel('Time relative to R peak (ms)', fontsize=16, fontweight='bold')

        if i == 0:  # 只在第一个子图显示y轴标签

            ax.set_ylabel('Amplitude (pT)', fontsize=14, fontweight='bold')

        

        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        ax.set_xlim(-100, 100)

        ax.set_ylim(0, 1)  # 设置固定的Y轴范围为0-1

        ax.tick_params(axis='both', labelsize=12)

        

        # # 添加文字注释

        # ax.text(0.02, 0.98, f'Pre-R', transform=ax.transAxes, 

        #        fontsize=10, va='top', ha='left', color='blue', alpha=0.7)

        # ax.text(0.48, 0.98, f'QRS', transform=ax.transAxes, 

        #        fontsize=10, va='top', ha='center', color='red', alpha=0.7)

        # ax.text(0.98, 0.98, f'Post-R', transform=ax.transAxes, 

        #        fontsize=10, va='top', ha='right', color='green', alpha=0.7)

    

    # =================================================================

    # 下半部分：12-20天汇总图

    # =================================================================

    

    # 添加下半部分的标题

    # fig.text(0.5, 0.52, 'Development Timeline Summary (Day 12-20)', 

    #          fontsize=16, fontweight='bold', ha='center')

    

    if summary_waveforms_data:

        # 错位排列的平均波形图（占据下半部分的上方）

        ax_summary_waves = fig.add_subplot(gs[1, :])

        ax_summary_waves.set_title('Averaged Waveforms (Day 12-20)', fontsize=16, fontweight='bold')

        

        peak_amplitudes = []

        labels = []

        x_offset_step = 80  # 每个波形在x轴上的偏移量

        

        # 定义一组颜色以便区分

        summary_colors = plt.get_cmap('tab10', len(summary_waveforms_data))

        

        for i, (time_axis, waveform, label) in enumerate(summary_waveforms_data):

            x_offset = i * x_offset_step

            

            # 找到R峰

            r_peak_index = np.argmin(np.abs(time_axis))

            r_peak_amplitude = waveform[r_peak_index]

            peak_amplitudes.append(r_peak_amplitude)

            labels.append(label)

            

            # 绘制波形

            ax_summary_waves.plot(time_axis + x_offset, waveform, color=summary_colors(i), linewidth=2)

            

            # 在峰值上方添加天数标签

            ax_summary_waves.text(x_offset, r_peak_amplitude + 0.1, label, 

                                 fontsize=12, ha='center', color=summary_colors(i))

        

        ax_summary_waves.set_xlabel('Time (ms)', fontsize=14, fontweight='bold')

        ax_summary_waves.set_ylabel('pT', fontsize=14, fontweight='bold')

        ax_summary_waves.grid(True, linestyle='--', alpha=0.6)

        ax_summary_waves.set_xlim(-100, x_offset + 100)

        ax_summary_waves.set_ylim(0, 1.2)

        ax_summary_waves.tick_params(axis='both', labelsize=13)

        

        # R峰振幅的柱状图（占据下半部分的下方）

        ax_summary_bars = fig.add_subplot(gs[2, :])

        ax_summary_bars.set_title('R-peak Amplitudes of Averaged Cycles', fontsize=16, fontweight='bold')

        

        x_positions = np.arange(len(labels))

        bars = ax_summary_bars.bar(x_positions, peak_amplitudes, 

                                  color='royalblue', alpha=0.8, width=0.6)

        ax_summary_bars.set_xticks(x_positions)

        ax_summary_bars.set_xticklabels(labels)

        ax_summary_bars.set_ylabel('pT', fontsize=14, fontweight='bold')

        ax_summary_bars.grid(axis='y', linestyle='--', alpha=0.6)

        ax_summary_bars.set_ylim(0, 1.2)

        ax_summary_bars.tick_params(axis='both', labelsize=13)

        

        # 在每个柱的顶部标注数值

        for bar in bars:

            yval = bar.get_height()

            ax_summary_bars.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, 

                                f'{yval:.2f}', ha='center', va='bottom', fontsize=12)

    

    # 保存图像

    if output_path:

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"综合图表已保存至: {output_path}")

    

    plt.show()



def main():

    """主函数：处理三组数据和12-20天汇总数据，并绘制综合对比图"""

    print("=== 综合心跳周期分析：三组数据对比 + 12-20天汇总 ===")

    

    # =================================================================

    # 第一部分：处理三组重点数据

    # =================================================================

    print("\n--- 处理三组重点数据 ---")

    three_group_files = [

        config.BASE_DIR / 'time_choice/egg_d13_B20_t2.txt',

        config.BASE_DIR / 'time_choice/d17_e7_t3很不错.txt',

        config.BASE_DIR / 'time_choice/egg_d20_B33_t1.txt'

    ]

    

    three_group_data = []

    for filepath in three_group_files:

        if filepath.exists():

            result = process_and_extract_averaged_beat(filepath)

            three_group_data.append(result)

        else:

            print(f"文件不存在: {filepath}")

            three_group_data.append((None, None, None))

    

    # 过滤掉处理失败的数据

    valid_three_group_data = [data for data in three_group_data if data[0] is not None]

    print(f"三组数据中成功处理了 {len(valid_three_group_data)} 组")

    

    # =================================================================

    # 第二部分：处理12-20天汇总数据

    # =================================================================

    print("\n--- 处理12-20天汇总数据 ---")

    

    # 定义12-20天数据文件路径

    base_dir = config.BASE_DIR / 'time_choice'

    summary_file_paths = [

        base_dir / 'egg_d12_B22_t1.txt',

        base_dir / 'egg_d13_B20_t2.txt',

        base_dir / 'egg_d14_B25_t1.txt',

        base_dir / 'egg_d15_B28_t1.txt',

        base_dir / 'egg_d16_B25_t2.txt',

        base_dir / 'egg_d17_B25_t2.txt',

        base_dir / 'egg_d18_B30_t1.txt',  

        base_dir / 'egg_d19_B34_t2.txt',

        base_dir / 'egg_d20_B30_t1_待破壳.txt',

    ]

    

    day_labels = [f'day{i}' for i in range(12, 21)]

    summary_waveforms_data = []

    

    # 处理每个汇总文件

    for path, label in zip(summary_file_paths, day_labels):

        if path.exists():

            time_axis, median_beat = get_averaged_waveform_for_summary(path)

            if median_beat is not None:

                summary_waveforms_data.append((time_axis, median_beat, label))

        else:

            print(f"汇总数据文件不存在: {path}")

    

    print(f"12-20天数据中成功处理了 {len(summary_waveforms_data)} 组")

    

    # =================================================================

    # 第三部分：绘制综合图表

    # =================================================================

    print("\n--- 绘制综合图表 ---")

    

    if len(valid_three_group_data) == 0 and len(summary_waveforms_data) == 0:

        print("错误: 没有有效的数据可以绘制")

        return

    

    # 生成综合图表

    output_path = config.OUTPUT_DIR / "comprehensive_cardiac_cycle_analysis.jpg"

    plot_comprehensive_figure(valid_three_group_data, summary_waveforms_data, output_path)

    

    # =================================================================

    # 第四部分：打印摘要信息

    # =================================================================

    print("\n=== 处理摘要 ===")

    

    print("三组重点数据:")

    for i, (averaged_beat_bx, time_axis_ms, embryo_info) in enumerate(valid_three_group_data):

        print(f"  组 {i+1}: Day {embryo_info['day']}, "

              f"平均了 {embryo_info['num_beats']} 个心拍 (Bx通道), "

              f"文件: {embryo_info['filename']}")

    

    print(f"\n12-20天汇总数据: 成功处理了 {len(summary_waveforms_data)} 个时间点")

    for time_axis, median_beat, label in summary_waveforms_data:

        r_peak_index = np.argmin(np.abs(time_axis))

        r_peak_amplitude = median_beat[r_peak_index]

        print(f"  {label}: R峰振幅 = {r_peak_amplitude:.3f} pT")

        

    print(f"\n✅ 综合图表已保存: {output_path}")

    print(f"包含 {len(valid_three_group_data)} 个重点对比图 + {len(summary_waveforms_data)} 个时间点的发育轨迹")



if __name__ == '__main__':

    # 确保输出目录存在

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main()

