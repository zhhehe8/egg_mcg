
""" 主执行文件 """

import numpy as np
from pathlib import Path

# 从自定义模块中导入所有需要的函数和配置
import config
import mcg_processing as mcg
import mcg_plotting as plot

def main():
    """主分析流程函数"""
    print("--- 开始心磁信号分析流程 ---")
    
    # 1. 加载数据
    print(f"加载数据: {config.INPUT_FILE.name}")
    bx_raw, by_raw = mcg.load_cardiac_data(config.INPUT_FILE)
    if bx_raw is None:
        print("数据加载失败，程序终止。")
        return

    # --- 根据配置截取数据前N秒 ---
    fs = config.PROCESSING_PARAMS['fs']
    duration_s = config.PROCESSING_PARAMS.get('analysis_duration_s')

    if duration_s is not None and duration_s > 0:
        num_samples = int(duration_s * fs)
        if len(bx_raw) > num_samples:
            bx_raw = bx_raw[:num_samples]
            by_raw = by_raw[:num_samples]
            print(f"  信息: 已截取数据前 {duration_s} 秒进行分析。")
        else:
            print(f"  警告: 数据总时长小于 {duration_s} 秒，将分析完整数据。")
    # --- 数据切片结束 ---

    # 准备时间轴
    time_s = np.arange(len(bx_raw)) / fs

    # 信号反转
    if config.PROCESSING_PARAMS['reverse_Bx']:
        bx_raw = -bx_raw
    if config.PROCESSING_PARAMS['reverse_By']:
        by_raw = -by_raw

    # 2. 信号滤波
    print("开始信号滤波...")
    # 应用带通滤波
    bx_filtered = mcg.apply_bandpass_filter(bx_raw, fs=config.PROCESSING_PARAMS['fs'], **config.FILTER_PARAMS['bandpass'])
    by_filtered = mcg.apply_bandpass_filter(by_raw, fs=config.PROCESSING_PARAMS['fs'], **config.FILTER_PARAMS['bandpass'])
    # 应用陷波滤波
    bx_filtered_bandpass_notch = mcg.apply_notch_filter(bx_filtered, fs=config.PROCESSING_PARAMS['fs'], **config.FILTER_PARAMS['notch'])
    by_filtered_bandpass_notch = mcg.apply_notch_filter(by_filtered, fs=config.PROCESSING_PARAMS['fs'], **config.FILTER_PARAMS['notch'])

    # (可选) 应用小波去噪
    # if config.FILTER_PARAMS['wavelet']['enabled']:
    #     print("应用小波去噪...")
    #     # 创建一个只包含函数所需参数的新字典
    #     wavelet_args = {
    #         'wavelet': config.FILTER_PARAMS['wavelet']['wavelet'],
    #         'level': config.FILTER_PARAMS['wavelet']['level']
    #     }
    # 应用小波去噪去除肌电干扰
    
    if config.FILTER_PARAMS['wavelet']['enabled']:
        print("应用小波去噪以去除肌电信号...") # 更新打印信息
        # 创建一个只包含函数所需参数的新字典
        wavelet_args = {
            'wavelet': config.FILTER_PARAMS['wavelet']['wavelet'],
            'level': config.FILTER_PARAMS['wavelet']['level'],
            # 【新增】确保新参数被传递
            'denoise_levels': config.FILTER_PARAMS['wavelet']['denoise_levels']
        }
        # 使用这个干净的字典进行参数传递
        bx_filtered = mcg.apply_wavelet_denoise(bx_filtered_bandpass_notch, **wavelet_args)
        by_filtered = mcg.apply_wavelet_denoise(by_filtered_bandpass_notch, **wavelet_args)
    # 最终平滑
    bx_filtered = mcg.apply_savgol_filter(bx_filtered)
    print("滤波完成。")

    # 3. R峰检测与定位
    print("开始R峰检测与定位...")
    integer_peaks_bx = mcg.find_r_peaks(bx_filtered, fs=config.PROCESSING_PARAMS['fs'], **config.R_PEAK_PARAMS)
    precise_peaks_bx = mcg.interpolate_r_peaks(bx_filtered, integer_peaks_bx)
    print(f"Bx通道: 找到 {len(integer_peaks_bx)} 个R峰。")

    integer_peaks_by = mcg.find_r_peaks(by_filtered, fs=config.PROCESSING_PARAMS['fs'], **config.R_PEAK_PARAMS)
    precise_peaks_by = mcg.interpolate_r_peaks(by_filtered, integer_peaks_by)
    print(f"By通道: 找到 {len(integer_peaks_by)} 个R峰。")

    # 4. 叠加平均
    print("开始叠加平均计算...")
    pre_samples = int(config.AVERAGING_PARAMS['pre_r_ms'] * config.PROCESSING_PARAMS['fs'] / 1000)
    post_samples = int(config.AVERAGING_PARAMS['post_r_ms'] * config.PROCESSING_PARAMS['fs'] / 1000)

    # 提取心拍
    all_beats_bx = mcg._extract_beats(bx_filtered, precise_peaks_bx, pre_samples, post_samples)
    all_beats_by = mcg._extract_beats(by_filtered, precise_peaks_by, pre_samples, post_samples)
    
    # 计算中位数平均
    median_beat_bx = mcg.get_median_beat(all_beats_bx)
    median_beat_by = mcg.get_median_beat(all_beats_by)
    
    # 计算DTW平均
    dtw_beat_bx = mcg.get_dtw_beat(all_beats_bx)
    dtw_beat_by = mcg.get_dtw_beat(all_beats_by)
    print("叠加平均计算完成。")

    

    # ---- 5. 时频分析 ----
    print("开始时频分析...")
    tf_method = config.TIME_FREQUENCY_PARAMS['enabled_method'].lower()
    tf_results = {}

    if tf_method in ['stft', 'both']:
        print("  正在计算 STFT...")
        f_stft_bx, t_stft_bx, Z_stft_bx = mcg.calculate_stft(bx_filtered, config.PROCESSING_PARAMS['fs'], **config.TIME_FREQUENCY_PARAMS['stft'])
        f_stft_by, _, Z_stft_by = mcg.calculate_stft(by_filtered, config.PROCESSING_PARAMS['fs'], **config.TIME_FREQUENCY_PARAMS['stft'])
        tf_results['stft'] = {'bx': (f_stft_bx, Z_stft_bx), 'by': (f_stft_by, Z_stft_by), 't': t_stft_bx}

    if tf_method in ['cwt', 'both']:
        print("  正在计算 CWT...")
        f_cwt_bx, Z_cwt_bx = mcg.calculate_cwt(bx_filtered, config.PROCESSING_PARAMS['fs'], **config.TIME_FREQUENCY_PARAMS['cwt'])
        f_cwt_by, Z_cwt_by = mcg.calculate_cwt(by_filtered, config.PROCESSING_PARAMS['fs'], **config.TIME_FREQUENCY_PARAMS['cwt'])
        tf_results['cwt'] = {'bx': (f_cwt_bx, Z_cwt_bx), 'by': (f_cwt_by, Z_cwt_by)}
    
    print("时频分析完成。")

    # 6. 可视化结果
    print("开始生成图表...")
    base_filename = config.INPUT_FILE.stem # 使用pathlib获取不带后缀的文件名

    # --- 绘制滤波对比图 ---
    plot_style_filter = config.PLOTTING_PARAMS['filtering_plot']['style'].lower()

    if plot_style_filter in ['single', 'both']:
        print("生成单通道滤波对比图...")
        plot.plot_single_channel_filtered(
        time_s, bx_raw, bx_filtered, integer_peaks_bx, f'{base_filename} - Bx',
        config.OUTPUT_DIR / f"{base_filename}_Bx_filtered.png"
        )
        plot.plot_single_channel_filtered(
        time_s, by_raw, by_filtered, integer_peaks_by, f'{base_filename} - By',
        config.OUTPUT_DIR / f"{base_filename}_By_filtered.png"
        )

    if plot_style_filter in ['dual', 'both']:
        print("生成双通道整合滤波对比图...")
        plot.plot_dual_channel_filtered(
        time=time_s,
        bx_raw=bx_raw, bx_filtered=bx_filtered, r_peaks_bx=integer_peaks_bx,
        by_raw=by_raw, by_filtered=by_filtered, r_peaks_by=integer_peaks_by,
        base_filename=base_filename,
        output_dir=config.OUTPUT_DIR
        )

    # --- 绘制叠加平均对比图 ---
    plot_style_avg = config.PLOTTING_PARAMS['averaging_plot']['style'].lower()
    if median_beat_bx is not None and dtw_beat_bx is not None:
        avg_time_ms = np.linspace(-config.AVERAGING_PARAMS['pre_r_ms'], config.AVERAGING_PARAMS['post_r_ms'], len(median_beat_bx))
    
    if plot_style_avg in ['single', 'both']:
        print("生成单通道叠加平均对比图...")
        plot.plot_single_channel_averaging(
            median_beat_bx, dtw_beat_bx, avg_time_ms, 'Bx Channel', base_filename,
            config.OUTPUT_DIR / f"{base_filename}_Bx_avg_comparison.png"
        )
        plot.plot_single_channel_averaging(
            median_beat_by, dtw_beat_by, avg_time_ms, 'By Channel', base_filename,
            config.OUTPUT_DIR / f"{base_filename}_By_avg_comparison.png"
        )
    
    if plot_style_avg in ['dual', 'both']:
        print("生成双通道整合叠加平均对比图...")
        plot.plot_dual_channel_averaging(
            time_axis_ms=avg_time_ms,
            median_beat_bx=median_beat_bx, dtw_beat_bx=dtw_beat_bx,
            median_beat_by=median_beat_by, dtw_beat_by=dtw_beat_by,
            base_filename=base_filename,
            output_dir=config.OUTPUT_DIR,
            display_method=config.PLOTTING_PARAMS['averaging_plot']['display_method']
        )
    # --- 时频分析结果绘图 ---
    plot_style_tf = config.TIME_FREQUENCY_PARAMS['plot']['style'].lower()
    for method, results in tf_results.items():
        if not results: continue
        
        t_axis = results.get('t', time_s) # STFT有自己的时间轴，CWT使用原始时间轴
        
        if plot_style_tf in ['single', 'both']:
            plot.plot_single_channel_tf(
                t_axis, results['bx'][0], results['bx'][1], method.upper(), 'Bx Channel', base_filename,
                config.OUTPUT_DIR / f"{base_filename}_Bx_tf_{method}.png"
            )
            plot.plot_single_channel_tf(
                t_axis, results['by'][0], results['by'][1], method.upper(), 'By Channel', base_filename,
                config.OUTPUT_DIR / f"{base_filename}_By_tf_{method}.png"
            )
        
        if plot_style_tf in ['dual', 'both']:
            if method == 'cwt':
                 plot.plot_dual_channel_tf({'cwt': results}, time_s, base_filename, config.OUTPUT_DIR)
            else: # STFT
                 plot.plot_dual_channel_tf({'stft': results}, t_axis, base_filename, config.OUTPUT_DIR)

print("--- 分析流程结束 ---")


if __name__ == '__main__':
    # 确保输出目录存在
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()