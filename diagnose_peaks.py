""" R峰检测诊断脚本
用于单个文件的R峰检测诊断和可视化。"""

import matplotlib.pyplot as plt
import numpy as np

import config
import mcg_processing as mcg

# --- 【配置要调试的文件】 ---
# 诊断文件路径
DIAGNOSE_FILE_PATH = config.BASE_DIR / 'B_egg' / 'B_egg_d20' / 'egg_d20_B30_t1_待破壳.txt'

def diagnose():
    """对单个文件进行R峰检测诊断并可视化"""
    
    # 1. 加载和滤波 (与main_batch流程一致)
    print(f"诊断文件: {DIAGNOSE_FILE_PATH.name}")
    bx_raw, _ = mcg.load_cardiac_data(DIAGNOSE_FILE_PATH)
    if bx_raw is None:
        return

    fs = config.PROCESSING_PARAMS['fs']
    bx_filtered = mcg.apply_bandpass_filter(bx_raw, fs, **config.FILTER_PARAMS['bandpass'])
    bx_filtered = mcg.apply_notch_filter(bx_filtered, fs, **config.FILTER_PARAMS['notch'])
    if config.FILTER_PARAMS['wavelet']['enabled']:
        wavelet_args = {'wavelet': config.FILTER_PARAMS['wavelet']['wavelet'], 'level': config.FILTER_PARAMS['wavelet']['level']}
        bx_filtered = mcg.apply_wavelet_denoise(bx_filtered, **wavelet_args)

    # 2. R峰检测
    params = config.R_PEAK_PARAMS
    peaks = mcg.find_r_peaks(bx_filtered, fs, **params)
    
    # 3. 计算用于可视化的阈值线
    base_amplitude = np.percentile(bx_filtered, params['height_percentile'])
    height_threshold = params['min_height_factor'] * base_amplitude
    
    print("-" * 30)
    print("诊断结果:")
    print(f"  使用的百分位数: {params['height_percentile']}%")
    print(f"  计算出的基准幅度: {base_amplitude:.4f}")
    print(f"  使用的最小高度因子: {params['min_height_factor']}")
    print(f"  最终高度阈值 (height): {height_threshold:.4f}")
    print(f"  检测到的 R 峰数量: {len(peaks)}")
    print("-" * 30)

    # 4. 绘图
    plt.figure(figsize=(20, 8))
    plt.title(f'R-Peak Detection Diagnosis for {DIAGNOSE_FILE_PATH.name}')
    
    time_s = np.arange(len(bx_filtered)) / fs
    plt.plot(time_s, bx_filtered, label='Filtered Signal', color='blue', zorder=1)
    
    # 绘制阈值线
    plt.axhline(y=height_threshold, color='green', linestyle='--', label=f'Height Threshold ({height_threshold:.2f})')
    
    # 绘制检测到的R峰
    if len(peaks) > 0:
        plt.plot(time_s[peaks], bx_filtered[peaks], 'rX', markersize=10, label=f'Detected Peaks ({len(peaks)})', zorder=2)
        
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 15) # 只显示前15秒以便观察细节
    plt.show()

if __name__ == '__main__':
    diagnose()