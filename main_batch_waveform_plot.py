import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 尝试导入项目中的模块和配置
try:
    import config
    import mcg_processing as mcg
except ImportError:
    print("错误: 无法导入 'config.py' 或 'mcg_processing.py'。")
    print("请确保此脚本与这些文件在同一目录下，或其路径已添加到 PYTHONPATH。")
    sys.exit(1)


def plot_and_save_single_waveform(
    time_axis_ms: np.ndarray,
    waveform_data: np.ndarray,
    channel_name: str,
    base_filename: str,
    output_path: Path
):
    """
    为单个通道绘制并保存平均心跳波形图（不显示）。
    """
    plt.figure(figsize=(10, 6))
    # 使用一种鲜明的颜色绘制波形
    plt.plot(time_axis_ms, waveform_data, color='deepskyblue', linewidth=2.5)
    plt.title(f'Median Averaged Cardiac Cycle - {channel_name} ({base_filename})', fontsize=16)
    plt.xlabel('Time Relative to R-Peak (ms)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    # 在R峰位置（0ms）画一条垂直红线作为参考
    plt.axvline(x=0, color='crimson', linestyle='-.', linewidth=1.5)
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path)
    # 关闭图形，释放内存（在循环中至关重要）
    plt.close()
    print(f"  ✓ 波形图已保存至: {output_path.name}")


def process_file_and_plot(filepath: Path, output_dir: Path):
    """
    对单个文件执行“滤波->R峰检测->中位数平均->绘图保存”的完整流程。
    """
    print(f"--- 正在处理: {filepath.name} ---")

    # 1. 加载数据
    bx_raw, by_raw = mcg.load_cardiac_data(filepath)
    if bx_raw is None:
        print(f"  ✗ 加载数据失败，跳过此文件。")
        return

    # --- 2. 信号预处理与滤波 (流程与 main.py 完全一致) ---
    fs = config.PROCESSING_PARAMS['fs']

    # 可选的数据截取
    duration_s = config.PROCESSING_PARAMS.get('analysis_duration_s')
    if duration_s and duration_s > 0:
        num_samples = int(duration_s * fs)
        if len(bx_raw) > num_samples:
            bx_raw, by_raw = bx_raw[:num_samples], by_raw[:num_samples]

    # 信号反转
    if config.PROCESSING_PARAMS['reverse_Bx']:
        bx_raw = -bx_raw
    if config.PROCESSING_PARAMS['reverse_By']:
        by_raw = -by_raw

    # 应用各类滤波器
    bx_filtered = mcg.apply_bandpass_filter(bx_raw, fs, **config.FILTER_PARAMS['bandpass'])
    by_filtered = mcg.apply_bandpass_filter(by_raw, fs, **config.FILTER_PARAMS['bandpass'])
    bx_filtered = mcg.apply_notch_filter(bx_filtered, fs, **config.FILTER_PARAMS['notch'])
    by_filtered = mcg.apply_notch_filter(by_filtered, fs, **config.FILTER_PARAMS['notch'])
    if config.FILTER_PARAMS['wavelet']['enabled']:
        wavelet_args = {
            'wavelet': config.FILTER_PARAMS['wavelet']['wavelet'],
            'level': config.FILTER_PARAMS['wavelet']['level'],
            'denoise_levels': config.FILTER_PARAMS['wavelet']['denoise_levels']
        }
        bx_filtered = mcg.apply_wavelet_denoise(bx_filtered, **wavelet_args)
        by_filtered = mcg.apply_wavelet_denoise(by_filtered, **wavelet_args)
    # Savitzky-Golay 最终平滑
    bx_filtered = mcg.apply_savgol_filter(bx_filtered)
    by_filtered = mcg.apply_savgol_filter(by_filtered)

    # --- 3. R峰检测 ---
    peaks_bx = mcg.find_r_peaks(bx_filtered, fs, **config.R_PEAK_PARAMS)
    precise_peaks_bx = mcg.interpolate_r_peaks(bx_filtered, peaks_bx)
    peaks_by = mcg.find_r_peaks(by_filtered, fs, **config.R_PEAK_PARAMS)
    precise_peaks_by = mcg.interpolate_r_peaks(by_filtered, peaks_by)

    if len(precise_peaks_bx) < 5 or len(precise_peaks_by) < 5:
        print(f"  ✗ R峰数量不足，跳过此文件。")
        return

    # --- 4. 提取心拍并计算中位数平均 ---
    pre_samples = int(config.AVERAGING_PARAMS['pre_r_ms'] * fs / 1000)
    post_samples = int(config.AVERAGING_PARAMS['post_r_ms'] * fs / 1000)
    all_beats_bx = mcg._extract_beats(bx_filtered, precise_peaks_bx, pre_samples, post_samples)
    all_beats_by = mcg._extract_beats(by_filtered, precise_peaks_by, pre_samples, post_samples)

    if not all_beats_bx or not all_beats_by:
        print(f"  ✗ 未能成功提取心拍，跳过此文件。")
        return

    median_beat_bx = mcg.get_median_beat(all_beats_bx)
    median_beat_by = mcg.get_median_beat(all_beats_by)

    # --- 5. 【新增】绘制并保存图片 ---
    base_filename = filepath.stem
    time_axis_ms = np.linspace(-config.AVERAGING_PARAMS['pre_r_ms'], config.AVERAGING_PARAMS['post_r_ms'], len(median_beat_bx))
    
    # 绘制 Bx
    plot_and_save_single_waveform(
        time_axis_ms, median_beat_bx, 'Bx', base_filename,
        output_dir / f"{base_filename}_median_waveform_Bx.png"
    )
    # 绘制 By
    plot_and_save_single_waveform(
        time_axis_ms, median_beat_by, 'By', base_filename,
        output_dir / f"{base_filename}_median_waveform_By.png"
    )


def run_batch_waveform_plotting():
    """
    执行批量处理的主函数。
    """
    # 1. 设置输出目录
    output_dir = Path('/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/waveforms_Ibis')
    print(f"所有平均波形图片将保存至: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 从 config.py 获取数据根目录
    data_root_dir = config.DATA_ROOT_DIR
    if not data_root_dir.is_dir():
        print(f"错误: 数据根目录 '{data_root_dir}' 不存在或不是一个文件夹。", file=sys.stderr)
        return

    print("\n=========================================")
    print(f"开始批量生成平均波形图，扫描目录: {data_root_dir}")
    print("=========================================")

    # 3. 查找所有 .txt 文件
    all_files = sorted(data_root_dir.glob('*/*.txt'))
    if not all_files:
        print("警告：在子文件夹中未找到.txt文件，尝试直接扫描根目录...")
        all_files = sorted(data_root_dir.glob('*.txt'))

    if not all_files:
        print("错误：在指定目录及其一级子目录中均未找到任何 .txt 文件。")
        return

    # 4. 循环处理每个文件
    total_files = len(all_files)
    print(f"找到 {total_files} 个文件，开始处理...")
    for i, filepath in enumerate(all_files):
        try:
            process_file_and_plot(filepath, output_dir)
            print(f"  进度: {i + 1}/{total_files}\n")
        except Exception as e:
            print(f"  ✗ 处理文件 {filepath.name} 时发生未知错误: {e}")

    print("\n=========================================")
    print("✅ 所有文件处理完毕！")
    print(f"结果已全部保存在: {output_dir}")
    print("=========================================")


if __name__ == '__main__':
    run_batch_waveform_plotting()