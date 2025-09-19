"""
两组数据平均心跳周期对比图绘制脚本

处理两组数据，计算它们的平均心跳周期，并绘制在1行2列的对比图上
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import config
import mcg_processing as mcg

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12

def process_and_extract_averaged_beat(filepath: Path):
    """
    处理单个文件并提取平均心跳周期（与fig3_4.py中的方法完全一致）
    基于 get_averaged_waveform_for_summary 的处理逻辑
    返回: (averaged_beat_bx, time_axis_ms, embryo_info)
    """
    print(f"正在处理: {filepath.name}")
    
    # 1. 加载数据
    bx_raw, by_raw = mcg.load_cardiac_data(filepath)
    if bx_raw is None:
        print(f"  数据加载失败: {filepath.name}")
        return None, None, None

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
        print(f"  R峰数量不足，跳过。")
        return None, None, None

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
        print(f"  可用的校正后心拍数量不足，跳过。")
        return None, None, None

    # 6. 对所有已校正的独立周期进行中位数平均
    cycles_array_corrected = np.array(individually_corrected_cycles)
    final_averaged_cycle = np.median(cycles_array_corrected, axis=0)

    # 7. 创建时间轴并返回最终结果
    time_axis_ms = np.linspace(-config.AVERAGING_PARAMS['pre_r_ms'], 
                              config.AVERAGING_PARAMS['post_r_ms'], 
                              len(final_averaged_cycle))
    
    # 提取胚胎信息
    embryo_info = {
        'filename': filepath.name,
        'day': extract_day_from_filename(filepath.name),
        'num_beats': len(individually_corrected_cycles)
    }
    
    print(f"  处理完成，平均了 {len(individually_corrected_cycles)} 个心拍")
    return final_averaged_cycle, time_axis_ms, embryo_info

def extract_day_from_filename(filename: str):
    """从文件名中提取天数信息"""
    import re
    day_match = re.search(r'd(\d+)', filename)
    return int(day_match.group(1)) if day_match else 0

def process_stft_data(filepath: Path):
    """
    处理单个文件的时频分析数据
    返回: (f, t_stft, Zxx, title) 或 None
    """
    try:
        print(f"正在处理时频数据: {filepath.name}")
        
        # 加载数据
        bx_raw, by_raw = mcg.load_cardiac_data(filepath)
        if bx_raw is None:
            print(f"  时频数据加载失败: {filepath.name}")
            return None
        
        # 数据预处理
        fs = config.PROCESSING_PARAMS['fs']
        duration_s = config.PROCESSING_PARAMS.get('analysis_duration_s', 30)
        
        if duration_s is not None and duration_s > 0:
            num_samples = int(duration_s * fs)
            if len(bx_raw) > num_samples:
                bx_raw = bx_raw[:num_samples]
        
        # 计算STFT
        f, t_stft, Zxx = signal.stft(bx_raw, fs=fs, nperseg=1024, noverlap=512)
        
        # 提取标题信息
        day = extract_day_from_filename(filepath.name)
        title = f'Day {day}'
        
        print(f"  时频分析完成: {title}")
        return f, t_stft, Zxx, title
        
    except Exception as e:
        print(f"时频分析出错 {filepath}: {e}")
        return None

def plot_comprehensive_comparison(group1_data, group2_data, stft_data_list, output_path=None):
    """
    绘制综合对比图：
    上半部分：1行2列的心跳周期对比图
    下半部分：1行3列的时频图
    
    Args:
        group1_data: (averaged_beat_bx, time_axis_ms, embryo_info) - 第一组数据
        group2_data: (averaged_beat_bx, time_axis_ms, embryo_info) - 第二组数据
        stft_data_list: 包含3个STFT数据的列表，每个元素为(f, t_stft, Zxx, title)
        output_path: 输出图像路径
    """
    # 创建复杂的子图布局
    fig = plt.figure(figsize=(20, 12))
    
    # 使用 GridSpec 来创建更灵活的布局
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.2)
    
    # =================================================================
    # 上半部分：1行2列的心跳周期对比图
    # =================================================================
    
    # 定义颜色
    colors = ['#FF6B6B', '#4ECDC4']  # 红色、青色
    
    # 处理数据并找到全局y轴范围
    all_data = [group1_data, group2_data]
    valid_data = []
    all_beats = []
    
    for data in all_data:
        if data[0] is not None:
            valid_data.append(data)
            all_beats.append(data[0])
    
    if not all_beats:
        print("错误: 没有有效的心跳数据可以绘制")
        return
    
    # 计算全局y轴范围（不进行归一化）
    global_min = min([np.min(beat) for beat in all_beats])
    global_max = max([np.max(beat) for beat in all_beats])
    y_margin = (global_max - global_min) * 0.1
    ylim_range = [global_min - y_margin, global_max + y_margin]
    
    # 绘制心跳周期对比图 (占用前两列)
    groups = [group1_data, group2_data]
    
    for i, (averaged_beat_bx, time_axis_ms, embryo_info) in enumerate(groups):
        ax = fig.add_subplot(gs[0, i])
        
        if averaged_beat_bx is not None:
            # 不进行归一化处理，直接使用原始数据
            
            # 绘制主要的心跳波形
            ax.plot(time_axis_ms, averaged_beat_bx, 
                   color=colors[i], 
                   linewidth=3, 
                   alpha=0.8,
                   label=f'Averaged waveform (n={embryo_info["num_beats"]})')
            
            # 添加R峰标记线
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='R peak')
            
            # 添加背景颜色区域
            ax.axvspan(-100, -20, alpha=0.1, color='lightblue')
            ax.axvspan(-20, 20, alpha=0.1, color='lightcoral')
            ax.axvspan(20, 100, alpha=0.1, color='lightgreen')
            
            # 设置子图标题
            ax.set_title(f'Day {embryo_info["day"]} - {embryo_info["filename"]}', 
                        fontsize=14, fontweight='bold', pad=15)
        else:
            # 如果数据无效，显示空白子图
            ax.text(0.5, 0.5, 'No Data Available', 
                   transform=ax.transAxes, fontsize=16, 
                   ha='center', va='center', color='red')
            ax.set_title(f'Group {i+1} (No Data)', fontsize=14, color='red')
        
        # 设置子图属性
        ax.set_xlabel('Time relative to R peak (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amplitude (pT)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(-100, 100)
        ax.set_ylim(ylim_range)  # 使用原始数据的范围
        ax.tick_params(axis='both', labelsize=11)
        
        # 添加图例（只在第一个子图显示）
        if i == 0 and averaged_beat_bx is not None:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
        
        # 添加统计信息
        if averaged_beat_bx is not None:
            r_peak_index = np.argmin(np.abs(time_axis_ms))
            r_peak_amplitude = averaged_beat_bx[r_peak_index]
            ax.text(0.02, 0.02, f'R-peak amplitude: {r_peak_amplitude:.3f} pT\nBeats averaged: {embryo_info["num_beats"]}', 
                   transform=ax.transAxes, fontsize=10, va='bottom', ha='left', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =================================================================
    # 下半部分：1行3列的时频图
    # =================================================================
    
    if stft_data_list and len(stft_data_list) > 0:
        # 添加下半部分的标题
        fig.text(0.5, 0.52, 'Time-Frequency Analysis (STFT)', 
                 fontsize=16, fontweight='bold', ha='center')
        
        # 绘制时频图
        im = None  # 用于存储colorbar引用
        
        for i, stft_data in enumerate(stft_data_list[:3]):  # 最多显示3个
            if stft_data is None:
                continue
                
            f, t_stft, Zxx, title = stft_data
            ax_stft = fig.add_subplot(gs[1, i])
            
            # 频率掩码（只显示0-40Hz）
            mask = (f <= 40)
            
            # 绘制时频图
            im = ax_stft.pcolormesh(t_stft, f[mask], np.abs(Zxx[mask]), 
                                  shading='gouraud', cmap='viridis', vmin=0, vmax=0.25)
            
            ax_stft.set_title(f'STFT of {title}', fontsize=14, fontweight='bold')
            ax_stft.set_xlim(0, 30)
            ax_stft.set_ylim(1.5, 10)
            ax_stft.tick_params(axis='both', labelsize=12)
            
            # 设置坐标轴标签
            if i == 0:
                ax_stft.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            ax_stft.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        
        # 添加色标
        if im is not None:
            # 计算色标位置
            fig.canvas.draw()
            pos = fig.add_subplot(gs[1, 2]).get_position()
            
            cbar_width = 0.267
            cbar_height = 0.02
            cbar_bottom = 0.05
            cbar_left = pos.x1 - cbar_width
            
            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Magnitude (pT)', fontsize=12, fontweight='bold')
    
    # 设置整体标题
    fig.suptitle('Comprehensive Cardiac Analysis: Beat Cycles & Time-Frequency', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"综合对比图已保存至: {output_path}")
    
    plt.show()

def main():
    """主函数：处理两组心跳数据和三组时频数据并绘制综合对比图"""
    print("=== 综合分析：心跳周期对比 + 时频分析 ===")
    
    # =================================================================
    # 第一部分：处理两组心跳周期数据
    # =================================================================
    print("\n--- 处理心跳周期数据 ---")
    
    # 定义两个心跳数据文件路径 - 您可以根据需要修改这些路径
    heartbeat_file1 = config.BASE_DIR / 'egg_mcg/time_choice/egg_d13_B20_t2.txt'
    heartbeat_file2 = config.BASE_DIR / 'egg_mcg/time_choice/egg_d19_B30_t1.txt'
    
    print(f"\n处理第一组心跳数据: {heartbeat_file1.name}")
    group1_data = process_and_extract_averaged_beat(heartbeat_file1) if heartbeat_file1.exists() else (None, None, None)
    
    print(f"\n处理第二组心跳数据: {heartbeat_file2.name}")
    group2_data = process_and_extract_averaged_beat(heartbeat_file2) if heartbeat_file2.exists() else (None, None, None)
    
    # =================================================================
    # 第二部分：处理三组时频数据
    # =================================================================
    print("\n--- 处理时频分析数据 ---")
    
    # 定义三个时频分析文件路径 - 您可以根据需要修改这些路径
    stft_files = [
        config.BASE_DIR / '朱鹮_250426/control/朱鹮未受精蛋1_t1.txt',
        config.BASE_DIR / 'Results/waveforms_Ibis/waveform_Ibis_choice/朱鹮day22_t1.txt', 
        config.BASE_DIR / 'Results/waveforms_Ibis/waveform_Ibis_choice/朱鹮day25_t1.txt'
    ]

    stft_data_list = []
    for filepath in stft_files:
        if filepath.exists():
            stft_result = process_stft_data(filepath)
            stft_data_list.append(stft_result)
        else:
            print(f"时频数据文件不存在: {filepath}")
            stft_data_list.append(None)
    
    # 过滤掉无效的时频数据
    valid_stft_data = [data for data in stft_data_list if data is not None]
    
    # =================================================================
    # 第三部分：检查数据有效性并绘制综合图表
    # =================================================================
    print("\n--- 生成综合图表 ---")
    
    if group1_data[0] is None and group2_data[0] is None:
        print("错误: 两组心跳数据都无效，无法绘制心跳对比图")
        return
    
    if len(valid_stft_data) == 0:
        print("警告: 没有有效的时频数据")
    
    # 绘制综合对比图
    output_path = config.OUTPUT_DIR / "comprehensive_cardiac_stft_analysis.jpg"
    plot_comprehensive_comparison(group1_data, group2_data, valid_stft_data, output_path)
    
    # =================================================================
    # 第四部分：打印摘要信息
    # =================================================================
    print("\n=== 处理摘要 ===")
    
    print("心跳周期数据:")
    for i, (averaged_beat_bx, time_axis_ms, embryo_info) in enumerate([group1_data, group2_data], 1):
        if averaged_beat_bx is not None:
            print(f"  组 {i}: Day {embryo_info['day']}, "
                  f"平均了 {embryo_info['num_beats']} 个心拍, "
                  f"文件: {embryo_info['filename']}")
        else:
            print(f"  组 {i}: 数据处理失败")
    
    print(f"\n时频分析数据: 成功处理了 {len(valid_stft_data)} 个文件")
    for i, data in enumerate(valid_stft_data):
        if data:
            f, t_stft, Zxx, title = data
            print(f"  {i+1}. {title} - 频率范围: {f[0]:.1f}-{f[-1]:.1f} Hz, 时间长度: {t_stft[-1]:.1f} s")
    
    print(f"\n✅ 综合分析图已保存: {output_path}")
    print(f"包含 2 个心跳周期对比图 + {len(valid_stft_data)} 个时频分析图")

if __name__ == '__main__':
    # 确保输出目录存在
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
