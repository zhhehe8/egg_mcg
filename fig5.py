"""
Fig5 综合分析脚本

功能1：计算并绘制两组数据的平均心跳周期图（基于fig3_4.py）
功能2：计算并绘制三组数据的时频图（基于egg_9figcwt2.py）

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import config
import mcg_processing as mcg

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 14

def process_and_extract_averaged_beat(filepath: Path):
    """
    处理单个文件并提取平均心跳周期（仅处理第一列Bx数据）
    返回: (averaged_beat_bx, time_axis_ms, embryo_info)
    基于fig3_4.py的处理逻辑
    """
    print(f"正在处理心跳数据: {filepath.name}")
    
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

def extract_day_from_filename(filename: str):
    """从文件名中提取天数信息"""
    import re
    day_match = re.search(r'd(\d+)', filename)
    return int(day_match.group(1)) if day_match else 0

def plot_comprehensive_analysis(beat_data_list, stft_file_paths, stft_titles, output_path=None):
    """
    绘制综合分析图：2行2列布局
    上排：两组心跳周期图
    下排：两组时频图
    """
    # 创建2行2列的子图布局，使用GridSpec进行精确控制
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(16, 12))
    
    # 创建GridSpec，为右侧色标留出空间
    gs = GridSpec(2, 2, figure=fig, 
                  width_ratios=[1, 1], 
                  height_ratios=[1, 1],
                  left=0.08, right=0.92, top=0.95, bottom=0.08,
                  wspace=0.25, hspace=0.3)
    
    # 创建子图
    axes = []
    axes.append([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])])  # 上排
    axes.append([fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])])  # 下排
    
    # =================================================================
    # 上排：心跳周期图
    # =================================================================
    
    # 定义颜色
    beat_colors = ['#FF6B6B', '#4ECDC4']  # 红色、青色
    
    # 为了统一y轴范围，先找到所有数据的最大最小值
    all_original_beats = []
    for averaged_beat_bx, time_axis_ms, embryo_info in beat_data_list:
        if averaged_beat_bx is not None:
            all_original_beats.append(averaged_beat_bx)
    
    if all_original_beats:
        global_min = min([np.min(beat) for beat in all_original_beats]) + 0.5  # 考虑平移
        global_max = max([np.max(beat) for beat in all_original_beats]) + 0.5  # 考虑平移
        y_margin = (global_max - global_min) * 0.1
        ylim_range = [global_min - y_margin, global_max + y_margin]
    else:
        ylim_range = [-0.7, 1.7]  # 原来是[-1.2, 1.2]，向上平移0.5后变为[-0.7, 1.7]
    
    # 绘制上排的两个心跳周期图
    for i, (averaged_beat_bx, time_axis_ms, embryo_info) in enumerate(beat_data_list):
        ax = axes[0][i]  # 第一行
        
        if averaged_beat_bx is not None:
            # 绘制主要的心跳波形（向下平移0.4）
            ax.plot(time_axis_ms, averaged_beat_bx + 0.1, 
                   color=beat_colors[i], 
                   linewidth=3, 
                   alpha=0.8,
                   label=f'Averaged waveform')
            
            # 添加背景颜色区域
            ax.axvspan(-100, -20, alpha=0.1, color='lightblue')
            ax.axvspan(-20, 20, alpha=0.1, color='lightcoral')
            ax.axvspan(20, 100, alpha=0.1, color='lightgreen')
            
            # 设置子图标题
            titles = ['Day 22', 'Day 25']  # 固定标题
            ax.set_title(titles[i], 
                        fontsize=18, fontweight='bold', pad=10)
        else:
            # 如果数据无效，显示空白子图
            ax.text(0.5, 0.5, 'No Data Available', 
                   transform=ax.transAxes, fontsize=18, 
                   ha='center', va='center', color='red')
            ax.set_title(f'Heartbeat Cycle - Day ? (No Data)', fontsize=16, color='red')

        # 设置子图属性
        ax.set_xlabel('Time relative to R peak (ms)', fontsize=16, fontweight='bold')
        if i == 0:  # 只在第一个子图显示y轴标签
            ax.set_ylabel('Amplitude (pT)', fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 1)  # 设置固定的Y轴范围为0-1
        ax.tick_params(axis='both', labelsize=14)

    # =================================================================
    # 下排：时频图
    # =================================================================
    
    fs = 1000  # 采样率
    im = None  # 初始化im用于存储pcolormesh对象，以便后续添加色标
    
    for i, filepath in enumerate(stft_file_paths):
        ax = axes[1][i]  # 第二行
        
        try:
            data = np.loadtxt(filepath, skiprows=2, encoding='utf-8')
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            ax.set_title(f'Error loading {stft_titles[i]}', fontsize=14, color='red')
            continue
        
        Bx = data[:, 0]
        f, t_stft, Zxx = signal.stft(Bx, fs=fs, nperseg=1024, noverlap=512)
        mask = (f <= 40)
        
        # 绘制时频图
        im = ax.pcolormesh(t_stft, f[mask], np.abs(Zxx[mask]), 
                          shading='gouraud', cmap='viridis', vmin=0, vmax=0.25)
        ax.set_title(f'STFT of {stft_titles[i]}', fontsize=18, fontweight='bold')
        
        # 设置坐标轴标签
        if i == 0:
            ax.set_ylabel('Frequency (Hz)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold')

        ax.set_xlim(0, 30)
        ax.set_ylim(1.5, 10)
        ax.tick_params(axis='both', labelsize=12)
    
    # 添加色标（仅针对时频图）
    if im is not None:
        # 获取时频图的位置信息，以便与之对齐
        # 获取下排两个时频图的边界
        pos_left = axes[1][0].get_position()   # 左下时频图
        pos_right = axes[1][1].get_position()  # 右下时频图
        
        # 计算色标的位置和尺寸
        # 高度与时频图一致：从下排时频图底部到顶部
        cbar_bottom = pos_left.y0  # 时频图的底部
        cbar_height = pos_left.height  # 时频图的高度
        
        # 宽度为原来的3/5：原宽度0.02 * 3/5 = 0.012
        cbar_width = 0.02 * 3/5  # 0.012
        
        # 水平位置：在右侧，留出适当间距
        cbar_left = 0.93
        
        # 创建色标，与时频图高度一致，宽度为原来的3/5
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Magnitude (pT)', fontsize=14, fontweight='bold')
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"综合分析图已保存至: {output_path}")
    
    plt.show()

def plot_averaged_beats(beat_data_list, output_path=None):
    """
    绘制两组数据的平均心跳周期对比图（保留原函数以备单独使用）
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 定义颜色
    colors = ['#FF6B6B', '#4ECDC4']  # 红色、青色
    
    # 为了统一y轴范围，先找到所有数据的最大最小值
    all_original_beats = []
    for averaged_beat_bx, time_axis_ms, embryo_info in beat_data_list:
        if averaged_beat_bx is not None:
            all_original_beats.append(averaged_beat_bx)
    
    if all_original_beats:
        global_min = min([np.min(beat) for beat in all_original_beats])
        global_max = max([np.max(beat) for beat in all_original_beats])
        y_margin = (global_max - global_min) * 0.1
        ylim_range = [global_min - y_margin, global_max + y_margin]
    else:
        ylim_range = [-1.2, 1.2]
    
    # 绘制两个子图
    for i, (averaged_beat_bx, time_axis_ms, embryo_info) in enumerate(beat_data_list):
        ax = axes[i]
        
        if averaged_beat_bx is not None:
            # 绘制主要的心跳波形
            ax.plot(time_axis_ms, averaged_beat_bx, 
                   color=colors[i], 
                   linewidth=3, 
                   alpha=0.8,
                   label=f'Averaged waveform')
            
            # 添加背景颜色区域
            ax.axvspan(-100, -20, alpha=0.1, color='lightblue')
            ax.axvspan(-20, 20, alpha=0.1, color='lightcoral')
            ax.axvspan(20, 100, alpha=0.1, color='lightgreen')
            
            # 设置子图标题
            ax.set_title(f'Day {embryo_info["day"]} - {embryo_info["num_beats"]} beats averaged', 
                        fontsize=16, fontweight='bold', pad=10)
        else:
            # 如果数据无效，显示空白子图
            ax.text(0.5, 0.5, 'No Data Available', 
                   transform=ax.transAxes, fontsize=18, 
                   ha='center', va='center', color='red')
            ax.set_title(f'Day ? (No Data)', fontsize=16, color='red')

        # 设置子图属性
        ax.set_xlabel('Time relative to R peak (ms)', fontsize=16, fontweight = 'bold')
        if i == 0:  # 只在第一个子图显示y轴标签
            ax.set_ylabel('Amplitude (pT)', fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(-100, 100)
        ax.set_ylim(ylim_range)
        ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"平均心跳图已保存至: {output_path}")
    
    plt.show()

def plot_stft_analysis(file_paths, titles, output_path=None):
    """
    绘制两组数据的时频图（基于egg_9figcwt2.py）
    """
    fs = 1000  # 采样率
    
    # 创建时频图画布（1行2列）
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    
    im = None  # 初始化im用于存储pcolormesh对象，以便后续添加色标
    
    for i, filepath in enumerate(file_paths):
        try:
            data = np.loadtxt(filepath, skiprows=2, encoding='utf-8')
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            axs[i].set_title(f'Error loading {titles[i]}', fontsize=14, color='red')
            continue
        
        Bx = data[:, 0]
        f, t_stft, Zxx = signal.stft(Bx, fs=fs, nperseg=1024, noverlap=512)
        mask = (f <= 40)
        
        # 绘制时频图
        im = axs[i].pcolormesh(t_stft, f[mask], np.abs(Zxx[mask]), 
                              shading='gouraud', cmap='viridis', vmin=0, vmax=0.25)
        axs[i].set_title(f'STFT of {titles[i]}', fontsize=16, fontweight='bold')
        
        # 设置坐标轴标签
        if i == 0:
            axs[i].set_ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
        axs[i].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        
        axs[i].set_xlim(0, 30)
        axs[i].set_ylim(1.5, 10)
        axs[i].tick_params(axis='both', labelsize=10)
    
    # 调整整体布局，为底部的水平色标留出空间
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    
    # 添加色标
    if im is not None:
        # 强制Matplotlib执行布局计算
        fig.canvas.draw()
        
        # 获取最右侧子图的边界
        pos = axs[1].get_position()
        right_align_edge = pos.x1
        
        # 计算色标的位置
        cbar_width = 0.267
        cbar_height = 0.02
        cbar_bottom = 0.08
        cbar_left = right_align_edge - cbar_width
        
        # 创建色标坐标轴
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        
        # 添加水平方向的共享色标
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Magnitude (pT)', fontsize=14, fontweight='bold')
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"时频图已保存至: {output_path}")
    
    plt.show()

def main():
    """
    主函数：处理两组心跳数据和两组时频数据
    """
    print("=== Fig5 综合分析：平均心跳周期 + 时频分析 ===")
    
    # =================================================================
    # 第一部分：处理两组心跳数据（用户将上传）
    # =================================================================
    print("\n--- 处理两组心跳数据 ---")
    
    # 待用户上传数据后，修改这些文件路径
    beat_analysis_files = [
        config.BASE_DIR / 'Results/waveforms_Ibis/waveform_Ibis_choice/朱鹮day25_t1.txt',  
        config.BASE_DIR / 'Results/waveforms_Ibis/waveform_Ibis_choice/朱鹮day22_t1.txt'   
    ]
    
    beat_data_list = []
    for filepath in beat_analysis_files:
        if filepath.exists():
            result = process_and_extract_averaged_beat(filepath)
            beat_data_list.append(result)
        else:
            print(f"文件不存在: {filepath}")
            beat_data_list.append((None, None, None))
    
    # 过滤掉处理失败的数据
    valid_beat_data = [data for data in beat_data_list if data[0] is not None]
    print(f"心跳数据中成功处理了 {len(valid_beat_data)} 组")
    
    # =================================================================
    # 第二部分：处理两组时频数据（用户将上传）
    # =================================================================
    print("\n--- 处理两组时频数据 ---")
    
    # 待用户上传数据后，修改这些文件路径
    stft_file_paths = [
        config.BASE_DIR / 'Results/STFT_朱鹮/STFT of 朱鹮 choice/朱鹮未受精蛋1_t2.txt',  
        config.BASE_DIR / 'Results/STFT_朱鹮/STFT of 朱鹮 choice/朱鹮day25_2_t2.txt'
    ]
    
    stft_titles = ['Unfertilized Egg', 'Day 25']
    
    # 检查文件是否存在
    existing_stft_files = []
    existing_stft_titles = []
    
    for filepath, title in zip(stft_file_paths, stft_titles):
        if filepath.exists():
            existing_stft_files.append(filepath)
            existing_stft_titles.append(title)
        else:
            print(f"时频数据文件不存在: {filepath}")
    
    # =================================================================
    # 第三部分：绘制综合分析图（2行2列）
    # =================================================================
    print("\n--- 绘制综合分析图 ---")
    
    if valid_beat_data and existing_stft_files:
        output_path_comprehensive = config.OUTPUT_DIR / "fig5_comprehensive_analysis.jpg"
        plot_comprehensive_analysis(valid_beat_data, existing_stft_files, 
                                   existing_stft_titles, output_path_comprehensive)
        print(f"综合分析图已完成，包含 {len(valid_beat_data)} 组心跳数据和 {len(existing_stft_files)} 组时频数据")
    else:
        print("数据不完整，无法绘制综合分析图")
        
        # 如果只有心跳数据，单独绘制
        if valid_beat_data:
            output_path_beats = config.OUTPUT_DIR / "fig5_averaged_beats.jpg"
            plot_averaged_beats(valid_beat_data, output_path_beats)
        
        # 如果只有时频数据，单独绘制
        if existing_stft_files:
            output_path_stft = config.OUTPUT_DIR / "fig5_stft_analysis.jpg"
            plot_stft_analysis(existing_stft_files, existing_stft_titles, output_path_stft)
    
    print("\n=== Fig5 分析完成 ===")

if __name__ == '__main__':
    # 确保输出目录存在
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
