# mcg_processing.py
"""包含所有数据处理算法"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks, stft, peak_widths, savgol_filter
from fastdtw import fastdtw
import pywt
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import re

# --- 数据加载 ---
def load_cardiac_data(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """从文本文件加载心脏数据 (Bx, By)。"""
    try:
        data = np.loadtxt(filepath, skiprows=2, encoding="utf-8")
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"错误: 数据文件 {filepath.name} 需要至少两列。")
            return None, None
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"错误: 加载数据 '{filepath}' 时出错: {e}")
        return None, None

# --- 滤波函数 ---
def apply_bandpass_filter(data: np.ndarray, fs: int, order: int, lowcut: float, highcut: float) -> np.ndarray:
    """应用巴特沃斯带通滤波器。"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def apply_notch_filter(data: np.ndarray, fs: int, freq: float, q_factor: float) -> np.ndarray:
    """应用陷波滤波器去除工频干扰。"""
    nyquist = 0.5 * fs
    b, a = iirnotch(freq / nyquist, q_factor)
    return filtfilt(b, a, data)

# def apply_wavelet_denoise(data: np.ndarray, wavelet: str, level: int) -> np.ndarray:
#     """应用小波变换进行去噪。"""
#     coeffs = pywt.wavedec(data, wavelet, mode='per', level=level)
#     sigma = np.median(np.abs(coeffs[-1])) / 0.6745
#     threshold = sigma * np.sqrt(2 * np.log(len(data)))
#     new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
#     reconstructed = pywt.waverec(new_coeffs, wavelet, mode='per')
#     return reconstructed[:len(data)] # 确保长度一致


def apply_wavelet_denoise(data: np.ndarray, wavelet: str, level: int, denoise_levels: int) -> np.ndarray:
    """
    (已升级) 使用分层阈值小波变换去除肌电(EMG)等高频噪声。
    
    :param data: 输入信号
    :param wavelet: 小波基函数
    :param level: 总分解层数
    :param denoise_levels: 需要进行去噪处理的细节层数(从最高频开始, 例如3代表只处理D1,D2,D3)
    """
    # 1. 进行小波分解
    coeffs = pywt.wavedec(data, wavelet, mode='per', level=level)
    
    # 2. 估计噪声标准差 (仅基于最高频的细节系数 D1)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # 3. 计算通用阈值
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    # 4. 【核心修改】只对指定的高频细节层进行阈值处理
    new_coeffs = list(coeffs)
    for i in range(1, denoise_levels + 1):
        # 从列表末尾开始索引，-i 对应于细节层 Di
        if i < len(coeffs):
            new_coeffs[-i] = pywt.threshold(coeffs[-i], threshold, mode='soft')
            
    # 5. 重建信号
    reconstructed = pywt.waverec(new_coeffs, wavelet, mode='per')
    
    # 确保输出长度与输入一致
    return reconstructed[:len(data)]

""""""

# --- Savitzky-Golay 滤波(最终平滑信号) ---
def apply_savgol_filter(data: np.ndarray) -> np.ndarray:
    # 窗口必须是奇数，阶数必须小于窗口
    return savgol_filter(data, window_length=21, polyorder=3)

# --- R峰检测与定位 ---
def find_r_peaks(data: np.ndarray, fs: int, min_height_factor: float, min_distance_ms: int, height_percentile: int) -> np.ndarray:
    """在信号中寻找R峰的整数索引(使用百分位数)。 """
    if data is None or len(data) == 0:
        return np.array([])
    
    base_amplitude = np.percentile(data, height_percentile)
    
    if base_amplitude <= 1e-9: 
        return np.array([])
        
    min_h = min_height_factor * base_amplitude
    min_distance = int(min_distance_ms / 1000 * fs)
    
    peaks, _ = find_peaks(data, height=min_h, distance=min_distance)
    return peaks

def interpolate_r_peaks(data: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    """通过抛物线插值，将R峰定位提升到亚样本精度。"""
    precise_indices = []
    for peak_idx in peak_indices:
        if 1 <= peak_idx < len(data) - 1:
            y = data[peak_idx-1 : peak_idx+2]
            x = np.arange(-1, 2)
            coeffs = np.polyfit(x, y, 2)
            if coeffs[0] == 0: # 避免除以零
                precise_indices.append(float(peak_idx))
                continue
            vertex_x = -coeffs[1] / (2 * coeffs[0])
            precise_indices.append(peak_idx + vertex_x)
        else:
            precise_indices.append(float(peak_idx))
    return np.array(precise_indices)

# --- 叠加平均算法 ---
def _extract_beats(signal_data: np.ndarray, precise_peak_indices: np.ndarray, pre_samples: int, post_samples: int) -> List[np.ndarray]:
    """(内部函数)基于精确R峰索引提取所有心拍。"""
    beats_list = []
    original_x = np.arange(len(signal_data))
    relative_x = np.arange(-pre_samples, post_samples)
    for peak_loc in precise_peak_indices:
        absolute_x = peak_loc + relative_x
        if absolute_x[0] < 0 or absolute_x[-1] >= len(signal_data):
            continue
        interpolated_beat = np.interp(absolute_x, original_x, signal_data)
        beats_list.append(interpolated_beat)
    return beats_list

# --- 心拍质量筛选 ---
def screen_beats_by_correlation(all_beats: List[np.ndarray], threshold: float) -> np.ndarray:
    """
    通过与模板的相关性来筛选高质量的心拍。
    """
    if len(all_beats) < 3: # 样本太少，不进行筛选
        return np.array(all_beats)

    beats_array = np.array(all_beats)
    
    # 1. 使用中位数平均创建一个鲁棒的模板
    template_beat = np.median(beats_array, axis=0)
    
    good_beats_indices = []
    # 2. 遍历每个心拍，计算其与模板的相关性
    for i, beat in enumerate(beats_array):
        # np.corrcoef返回一个2x2的相关矩阵，我们需要的是对角线之外的值
        correlation = np.corrcoef(beat, template_beat)[0, 1]
        
        # 3. 如果相关性高于阈值，则保留该心拍的索引
        if correlation >= threshold:
            good_beats_indices.append(i)
            
    num_original = len(all_beats)
    num_kept = len(good_beats_indices)
    print(f"  心拍质量筛选: {num_original} 个心拍中保留了 {num_kept} 个 (阈值: >{threshold})")
    
    # 4. 返回所有高质量的心拍
    return beats_array[good_beats_indices]

def get_median_beat(all_beats: List[np.ndarray]) -> Optional[np.ndarray]:
    """计算中位数平均心拍。"""
    if not all_beats:
        return None
    return np.median(np.array(all_beats), axis=0)

def get_dtw_beat(all_beats: List[np.ndarray]) -> Optional[np.ndarray]:
    """计算DTW对齐后的平均心拍。"""
    if len(all_beats) < 2:
        return np.array(all_beats[0]) if all_beats else None

    beats_array = np.array(all_beats)
    template_beat = np.median(beats_array, axis=0)
    
    warped_beats = []
    print(f"开始对 {len(beats_array)} 个心拍进行DTW对齐...")
    for i, beat in enumerate(beats_array):
        _, path = fastdtw(template_beat, beat, dist=lambda a, b: (a - b)**2)
        
        warped_beat = np.zeros_like(template_beat)
        warp_counts = np.zeros_like(template_beat, dtype=int)
        
        for template_idx, beat_idx in path:
            warped_beat[template_idx] += beat[beat_idx]
            warp_counts[template_idx] += 1
            
        warp_counts[warp_counts == 0] = 1
        warped_beat /= warp_counts
        warped_beats.append(warped_beat)
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(beats_array)}...")
            
    return np.mean(np.array(warped_beats), axis=0)


""" 计算时频图（stft 和 cwt） """
def calculate_stft(data: np.ndarray, fs: int, window_length: int, overlap_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算短时傅里叶变换 (STFT)。"""
    noverlap = int(window_length * overlap_ratio)
    f, t, Zxx = stft(data, fs, nperseg=window_length, noverlap=noverlap)
    return f, t, np.abs(Zxx)

def calculate_cwt(data: np.ndarray, fs: int, wavelet: str, max_scale: int, num_scales: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算连续小波变换 (CWT)。"""
    # 创建一个尺度数组
    scales = np.arange(1, max_scale)
    # 计算CWT
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=1/fs)
    return frequencies, np.abs(coefficients)

""" 参数计算函数 """
def calculate_hrv_params(precise_peaks: np.ndarray, fs: int) -> Dict[str, float]:
    """计算心率和HRV时域参数。"""
    if len(precise_peaks) < 2:
        return {'mean_hr': np.nan, 'sdnn': np.nan, 'rmssd': np.nan}
    
    rr_intervals_samples = np.diff(precise_peaks)
    rr_intervals_ms = (rr_intervals_samples / fs) * 1000
    
    mean_rr = np.mean(rr_intervals_ms)
    mean_hr = 60000 / mean_rr if mean_rr > 0 else 0
    sdnn = np.std(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2))
    
    return {'mean_hr': mean_hr, 'sdnn': sdnn, 'rmssd': rmssd}

def calculate_morphology_params(averaged_beat: np.ndarray, fs: int, pre_samples: int) -> Dict[str, float]:
    """从平均心拍中计算形态学参数。"""
    if averaged_beat is None or len(averaged_beat) == 0:
        return {'qrs_amplitude': np.nan, 'qrs_width_ms': np.nan}
        
    r_peak_index = pre_samples # R峰位于窗口中心
    r_peak_amplitude = averaged_beat[r_peak_index]
    
    # 计算QRS波宽度 (在半高处)
    try:
        widths, _, _, _ = peak_widths(averaged_beat, [r_peak_index], rel_height=0.5)
        qrs_width_ms = (widths[0] / fs) * 1000 if len(widths) > 0 else np.nan
    except:
        qrs_width_ms = np.nan # 如果计算出错则返回NaN

    return {'qrs_amplitude': r_peak_amplitude, 'qrs_width_ms': qrs_width_ms}

""" 单文件处理总管函数 """
def process_single_file(filepath: Path, config: Dict) -> Optional[Tuple[Dict[str, Any], Optional[np.ndarray]]]:
    """
    对单个数据文件执行完整的处理和参数提取流程。
    """
    print(f"\n--- 正在处理文件: {filepath.name} ---")
    
    # 1. 加载数据等预处理 
    bx_raw, _ = load_cardiac_data(filepath)
    if bx_raw is None: return None, None
    fs = config['PROCESSING_PARAMS']['fs']
    duration_s = config['PROCESSING_PARAMS'].get('analysis_duration_s')
    if duration_s is not None and duration_s > 0:
        num_samples_to_keep = int(duration_s * fs)
        if len(bx_raw) > num_samples_to_keep:
            bx_raw = bx_raw[:num_samples_to_keep]

    day_match = re.search(r'[Dd](\d+)', filepath.name)
    day = int(day_match.group(1)) if day_match else -1
    embryo_id = filepath.stem

    if config['PROCESSING_PARAMS']['reverse_Bx']:
        bx_raw = -bx_raw

    # 2. 滤波
    bx_filtered_bandpass = apply_bandpass_filter(bx_raw, fs, **config['FILTER_PARAMS']['bandpass'])
    bx_filtered_bandpass_notch = apply_notch_filter(bx_filtered_bandpass, fs, **config['FILTER_PARAMS']['notch'])
    if config['FILTER_PARAMS']['wavelet']['enabled']:
        wavelet_args = {'wavelet': config['FILTER_PARAMS']['wavelet']['wavelet'], 'level': config['FILTER_PARAMS']['wavelet']['level'], 'denoise_levels': config['FILTER_PARAMS']['wavelet']['denoise_levels']}
        bx_filtered = apply_wavelet_denoise(bx_filtered_bandpass_notch, **wavelet_args)
    else:
        bx_filtered = bx_filtered_bandpass_notch

    # 3. R峰检测
    integer_peaks = find_r_peaks(bx_filtered, fs, **config['R_PEAK_PARAMS'])
    if len(integer_peaks) < 5: return None, None
    precise_peaks = interpolate_r_peaks(bx_filtered, integer_peaks)

    # 4. 提取心拍 
    pre_samples = int(config['AVERAGING_PARAMS']['pre_r_ms'] * fs / 1000)
    post_samples = int(config['AVERAGING_PARAMS']['post_r_ms'] * fs / 1000)
    all_beats = _extract_beats(bx_filtered, precise_peaks, pre_samples, post_samples)
    if not all_beats: return None, None
    beats_to_average = np.array(all_beats)

    # 5. 根据配置选择性地进行叠加平均计算 
    avg_method = config['AVERAGING_PARAMS'].get('method', 'dtw').lower()
    print(f"  采用 '{avg_method}' 方法进行叠加平均...")
    averaged_beat = None
    if avg_method == 'median':
        averaged_beat = get_median_beat(beats_to_average)
    elif avg_method == 'dtw':
        averaged_beat = get_dtw_beat(list(beats_to_average))
    elif avg_method == 'both':
        averaged_beat = get_dtw_beat(list(beats_to_average)) # 默认使用更高质量的DTW结果
        
    if averaged_beat is None: return None, None
        
    # 6. 计算参数
    hrv_params = calculate_hrv_params(precise_peaks, fs)
    morphology_params = calculate_morphology_params(averaged_beat, fs, pre_samples)

    # 7. 汇总结果
    result_dict = {
        'filepath': str(filepath),
        'embryo_id': embryo_id,
        'day': day,
        'num_r_peaks': len(integer_peaks),
        'avg_method_used': avg_method,
        **hrv_params,
        **morphology_params
    }
    
    return result_dict, averaged_beat