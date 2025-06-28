"""将12-20天的心磁数据进行批量处理，提取平均心跳周期并生成汇总图。"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, find_peaks
import os
import re
import matplotlib.pyplot as plt
import matplotlib # 确保matplotlib被导入
import argparse # 导入argparse用于处理命令行参数

# --- 配置参数 (用户可根据需要调整) ---
SAMPLING_RATE = 1000  # Hz

# *** 指向包含所有数据文件的根目录 ***
ROOT_DATA_DIR = r'C:\Users\Xiaoning Tan\Desktop\time_choice'
OUTPUT_FIGURE_DIR = r'C:\Users\Xiaoning Tan\Desktop\time_choice' # *** 汇总图片也在此显示/曾保存的目录 ***

SKIP_HEADER_LINES = 2
FILE_ENCODING = 'utf-8'

# 滤波参数 (通用)
FILTER_ORDER_BANDPASS = 4
LOWCUT_FREQ = 0.5
HIGHCUT_FREQ = 45.0
NOTCH_FREQ_MAINS = 50.0
Q_FACTOR_NOTCH = 30.0

# R峰检测参数 (通用 - 作为默认值)
DEFAULT_R_PEAK_MIN_HEIGHT_FACTOR = 0.4
DEFAULT_R_PEAK_MIN_DISTANCE_MS = 150

# 周期提取和平均参数 (通用)
PRE_R_PEAK_MS = 100
POST_R_PEAK_MS = 100

# --- 汇总图中特定日龄标签的Y轴对齐位置 ---
ALIGNED_LABEL_Y_POSITION = 0.9 # Day 12-17 标签的Y轴位置，已更新

# --- 日龄特定的R峰检测参数 ---
DAY_SPECIFIC_R_PEAK_PARAMS = {
    "Day 2":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 3":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 4":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 5":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 6":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 7":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 8":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 9":  {"height_factor": 0.35, "distance_ms": 400},
    "Day 10": {"height_factor": 0.35, "distance_ms": 200},
    "Day 11": {"height_factor": 0.35, "distance_ms": 300},
    "Day 12": {"height_factor": 0.35, "distance_ms": 300},
    "Day 13": {"height_factor": 0.40, "distance_ms": 300},
    "Day 14": {"height_factor": 0.40, "distance_ms": 250},
    "Day 15": {"height_factor": 0.40, "distance_ms": 250},
    "Day 16": {"height_factor": 0.40, "distance_ms": 250},
    "Day 17": {"height_factor": 0.40, "distance_ms": 250},
    "Day 18": {"height_factor": 0.40, "distance_ms": 200},
    "Day 19": {"height_factor": 0.40, "distance_ms": 200},
    "Day 20": {"height_factor": 0.45, "distance_ms": 200},
    "Day 21": {"height_factor": 0.45, "distance_ms": 200},
}


# --- 函数定义 ---

def extract_day_label_from_filename(filename):
    match = re.search(r'[Dd](\d+)', filename)
    if match:
        return f"Day {match.group(1)}"
    match_day_word = re.search(r'day(\d+)', filename, re.IGNORECASE)
    if match_day_word:
        return f"Day {match_day_word.group(1)}"
    return None


def load_cardiac_data(filepath, skip_header, file_encoding='utf-8'):
    try:
        data = np.loadtxt(filepath, skiprows=skip_header, dtype=float, encoding=file_encoding)
        if data.ndim == 1 or data.shape[1] < 2:
            print(f"   错误：数据文件 {os.path.basename(filepath)} 需要至少两列 (Bx, By)。")
            return None, None
        channel1 = data[:, 0]
        channel2 = data[:, 1]
        return channel1, channel2
    except Exception as e:
        print(f"   读取文件 '{os.path.basename(filepath)}' 或解析数据时出错: {e}")
        return None, None

def apply_bandpass_filter(data, lowcut, highcut, fs, order):
    if data is None: return None
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0: high = 0.99
    if low <= 0: low = 0.001
    if low >= high: return data
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sosfiltfilt(sos, data)

def apply_notch_filter(data, notch_freq, quality_factor, fs):
    if data is None: return None
    nyquist = 0.5 * fs
    freq_normalized = notch_freq / nyquist
    if freq_normalized >= 1.0 or freq_normalized <= 0: return data
    b, a = iirnotch(freq_normalized, quality_factor)
    return filtfilt(b, a, data)

def find_r_peaks_data(data, fs, min_height_factor, min_distance_ms, identifier="信号",percentile = 95):
    if data is None or len(data) == 0: return np.array([])
    data_max = np.percentile(data, percentile)
    if data_max <= 1e-9: return np.array([])  
    min_h = min_height_factor * data_max
    min_distance = int(min_distance_ms / 1000 * fs)

    try:
        peaks, _ = find_peaks(data, height=min_h, distance=min_distance)
    except Exception as e:
        print(f"  错误 ({identifier}): 调用 find_peaks 时出错: {e}")
        return np.array([])
    return peaks


def calculate_average_cycle_data(data, r_peaks_indices, fs,
                                 pre_r_ms, post_r_ms,
                                 identifier="信号"):
    """
    计算平均心跳周期。先对每个独立周期进行基线校正，然后再平均。
    返回校正后的平均周期数据和以毫秒为单位的时间轴，如果失败则返回 (None, None)。
    """
    if data is None or r_peaks_indices is None or len(r_peaks_indices) < 2:
        return None, None

    pre_r_samples = int((pre_r_ms / 1000.0) * fs)
    post_r_samples = int((post_r_ms / 1000.0) * fs)
    total_cycle_samples = pre_r_samples + post_r_samples

    if total_cycle_samples <= 0:
        print(f"   错误 ({identifier}): 提取的周期长度无效 (pre: {pre_r_ms}ms, post: {post_r_ms}ms)。")
        return None, None

    individually_corrected_cycles = []
    valid_cycles_count = 0

    baseline_window_duration_ms = min(pre_r_ms, 30)
    baseline_samples_count = int((baseline_window_duration_ms / 1000.0) * fs)

    for r_peak_idx in r_peaks_indices:
        start_idx = r_peak_idx - pre_r_samples
        end_idx = r_peak_idx + post_r_samples
        if start_idx >= 0 and end_idx <= len(data):
            cycle_data_raw = data[start_idx:end_idx]
            if len(cycle_data_raw) == total_cycle_samples:
                current_cycle_baseline_offset = 0
                if baseline_samples_count > 0 and baseline_samples_count <= pre_r_samples:
                    current_cycle_baseline_offset = np.mean(cycle_data_raw[:baseline_samples_count])

                cycle_data_corrected = cycle_data_raw - current_cycle_baseline_offset
                individually_corrected_cycles.append(cycle_data_corrected)
                valid_cycles_count += 1

    if valid_cycles_count < 2:
        return None, None

    cycles_array_corrected = np.array(individually_corrected_cycles)
    final_averaged_cycle = np.mean(cycles_array_corrected, axis=0)

    cycle_time_axis_ms = np.arange(-pre_r_samples, post_r_samples) * (1000.0 / SAMPLING_RATE)

    return final_averaged_cycle, cycle_time_axis_ms


def plot_and_save_summary_figure(collected_cycles_data, output_dir,
                                 horizontal_offset_increment,
                                 summary_filename="summary_all_average_cycles.png"):
    if not collected_cycles_data:
        print("没有可供绘制汇总图的平均周期数据。")
        return

    fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [3, 2]})
    ax_summary = axs[0]
    ax_bar = axs[1]

    # --- 绘制平均周期汇总图 (ax_summary) ---
    ax_summary.set_title('Averaged Waveforms(day12-20)', fontsize=15)
    ax_summary.set_xlabel('Time (ms)', fontsize=12, loc='right')
    ax_summary.set_ylabel('pT', fontsize=12)
    ax_summary.set_ylim(0, 2)
    ax_summary.grid(True, which='major', linestyle='-', linewidth='0.7', color='grey')
    ax_summary.grid(True, which='minor', linestyle=':', linewidth='0.4', color='lightgrey')
    ax_summary.minorticks_on()

    total_files = len(collected_cycles_data)
    if total_files <= 10: colors = plt.cm.get_cmap('tab10', total_files if total_files > 0 else 1)
    elif total_files <=20: colors = plt.cm.get_cmap('tab20', total_files if total_files > 0 else 1)
    else: colors = plt.cm.get_cmap('nipy_spectral', total_files if total_files > 0 else 1)

    current_day_group_offset_ms = 0.0
    previous_day_label_for_offset = None

    bar_chart_plot_data = []

    def get_day_num_from_label(day_label_str):
        if day_label_str:
            match = re.search(r'\d+', day_label_str)
            if match:
                return int(match.group())
        return float('inf')

    sorted_collected_cycles = sorted(collected_cycles_data, key=lambda x: get_day_num_from_label(x[1]))

    text_y_base_offset_factor = 0.05 * (ax_summary.get_ylim()[1] - ax_summary.get_ylim()[0])

    for i, (file_label, day_label, time_axis_ms, avg_cycle) in enumerate(sorted_collected_cycles):
        if time_axis_ms is not None and avg_cycle is not None:
            if day_label != previous_day_label_for_offset and previous_day_label_for_offset is not None:
                current_day_group_offset_ms += horizontal_offset_increment

            shifted_time_axis_ms = time_axis_ms + current_day_group_offset_ms
            plot_color = colors(i % colors.N)
            ax_summary.plot(shifted_time_axis_ms, avg_cycle, linewidth=1.5, color=plot_color)

            day_num_for_bar_check = get_day_num_from_label(day_label)
            if 12 <= day_num_for_bar_check <= 20 and len(avg_cycle) > 0:
                r_peak_val = np.max(avg_cycle)
                r_peak_idx_in_cycle = np.argmax(avg_cycle)
                r_peak_x_on_summary_plot = shifted_time_axis_ms[r_peak_idx_in_cycle]

                bar_chart_plot_data.append({
                    'x_pos': r_peak_x_on_summary_plot,
                    'height': r_peak_val,
                    'day_label_short': day_label.lower().replace(" ", "") if day_label else "unknown",
                    # 'color': plot_color # This color is for the waveform, not used for the bar color anymore
                })

            if day_label:
                annotation_text = day_label.lower().replace(" ", "")
                if len(avg_cycle) > 0:
                    peak_idx_in_avg_cycle = np.argmax(avg_cycle)
                    text_x_pos = shifted_time_axis_ms[peak_idx_in_avg_cycle]

                    day_num_for_align_check = get_day_num_from_label(day_label)

                    if 12 <= day_num_for_align_check <= 17:
                        text_y_pos = ALIGNED_LABEL_Y_POSITION
                    else:
                        text_y_pos = avg_cycle[peak_idx_in_avg_cycle] + text_y_base_offset_factor

                    ax_summary.text(text_x_pos, text_y_pos, annotation_text,
                                    ha='center', va='bottom', fontsize=9, color=plot_color)

            if day_label is not None:
                previous_day_label_for_offset = day_label
        else:
            print(f"跳过绘制汇总图中的: {file_label} 因为数据不完整。")

    # --- 绘制R峰最大值柱状图 (ax_bar) ---
    if bar_chart_plot_data:
        bar_x_positions = [item['x_pos'] for item in bar_chart_plot_data]
        bar_heights = [item['height'] for item in bar_chart_plot_data]
        bar_plot_labels = [item['day_label_short'] for item in bar_chart_plot_data]
        consistent_bar_color = 'royalblue' # MODIFICATION: Define a consistent color for all bars

        min_x_diff = np.min(np.diff(sorted(list(set(bar_x_positions))))) if len(set(bar_x_positions)) > 1 else horizontal_offset_increment
        bar_width = min_x_diff * 0.4 if min_x_diff > 0 else horizontal_offset_increment * 0.4

        # MODIFICATION: Use the consistent_bar_color for all bars
        bars = ax_bar.bar(bar_x_positions, bar_heights, width=bar_width, color=consistent_bar_color, alpha=0.8)

        ax_bar.set_xticks(bar_x_positions)
        ax_bar.set_xticklabels(bar_plot_labels, rotation=45, ha="right", fontsize=9)
        ax_bar.set_ylabel('pT', fontsize=12)
        ax_bar.set_title('R-peak of Averaged Cycles', fontsize=16)
        ax_bar.grid(True, axis='y', linestyle='--', alpha=0.7)

        for i, bar_item in enumerate(bars):
            yval = bar_item.get_height()
            text_offset_bar = 0.02 * (ax_bar.get_ylim()[1] - ax_bar.get_ylim()[0]) if ax_bar.get_ylim()[1] > ax_bar.get_ylim()[0] else 0.02
            ax_bar.text(bar_item.get_x() + bar_item.get_width()/2.0, yval + text_offset_bar,
                        f'{yval:.2f}', ha='center', va='bottom', fontsize=8)

        if bar_heights:
            max_bar_height = np.max(bar_heights)
            ax_bar.set_ylim(0, max_bar_height * 1.20 if max_bar_height > 0 else 1)

        if bar_x_positions:
            ax_bar.set_xlim(min(bar_x_positions) - bar_width, max(bar_x_positions) + bar_width)

    plt.tight_layout(pad=2.0, h_pad=3.5)

    # --- 保存图片 ---
    try:
        summary_save_path = os.path.join(output_dir, summary_filename)
        plt.savefig(summary_save_path, dpi=300, bbox_inches='tight')
        print(f"汇总图已保存到: {summary_save_path}")
    except Exception as e:
        print(f"错误：无法保存汇总图。原因: {e}")

    # --- 显示图片 ---
    print(f"汇总图准备显示...")
    plt.show()

    # --- 关闭图片 ---
    plt.close(fig)


# --- 主程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理心磁数据，提取平均心跳周期并生成汇总图。")
    parser.add_argument(
        "-xo", "--x_offset_per_day_group",
        type=float,
        default=50.0,
        help="在汇总图中，每个新的日龄组在X轴上相对于上一个日龄组的水平偏移量 (单位: 毫秒, 默认: 50.0)"
    )
    args = parser.parse_args()

    if not os.path.isdir(ROOT_DATA_DIR):
        print(f"错误：数据目录 '{ROOT_DATA_DIR}' 不存在或不是一个目录。请检查路径。")
    else:
        print(f"开始批量处理目录: {ROOT_DATA_DIR}")
        print(f"使用的X轴日龄组偏移量为: {args.x_offset_per_day_group} ms")
        os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

        processed_files_count = 0
        collected_cycles_for_summary_plot = []

        txt_files_in_folder = sorted([f for f in os.listdir(ROOT_DATA_DIR) if f.lower().endswith('.txt')])

        if not txt_files_in_folder:
            print(f"   在文件夹 {ROOT_DATA_DIR} 中未找到 .txt 文件。")
        else:
            for txt_filename in txt_files_in_folder:
                current_file_path = os.path.join(ROOT_DATA_DIR, txt_filename)
                file_base_name = os.path.splitext(txt_filename)[0]
                print(f"\n--- 开始处理文件: {txt_filename} ---")
                processed_files_count += 1

                day_label = extract_day_label_from_filename(file_base_name)

                day_r_peak_config = DAY_SPECIFIC_R_PEAK_PARAMS.get(day_label, {}) if day_label else {}
                current_r_peak_height_factor = day_r_peak_config.get("height_factor", DEFAULT_R_PEAK_MIN_HEIGHT_FACTOR)
                current_r_peak_distance_ms = day_r_peak_config.get("distance_ms", DEFAULT_R_PEAK_MIN_DISTANCE_MS)

                if day_label:
                    print(f"   检测到日龄标签: {day_label}. 使用R峰检测参数: Height Factor = {current_r_peak_height_factor:.2f}, Distance MS = {current_r_peak_distance_ms}")
                else:
                    print(f"   未从文件名 '{txt_filename}' 检测到日龄标签. 使用默认R峰检测参数: Height Factor = {current_r_peak_height_factor:.2f}, Distance MS = {current_r_peak_distance_ms}")

                ch1_data_raw, ch2_data_raw = load_cardiac_data(current_file_path, SKIP_HEADER_LINES, FILE_ENCODING)

                if ch1_data_raw is None or ch2_data_raw is None:
                    continue

                ch1_bandpassed = apply_bandpass_filter(ch1_data_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
                ch1_filtered = apply_notch_filter(ch1_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)

                ch2_bandpassed = apply_bandpass_filter(ch2_data_raw, LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_RATE, order=FILTER_ORDER_BANDPASS)
                ch2_filtered = apply_notch_filter(ch2_bandpassed, NOTCH_FREQ_MAINS, Q_FACTOR_NOTCH, SAMPLING_RATE)

                if ch1_filtered is None or ch2_filtered is None:
                    print(f"   错误：文件 {txt_filename} 的一个或两个通道滤波失败。")
                    continue

                min_len = min(len(ch1_filtered), len(ch2_filtered))
                combined_data_filtered = np.sqrt(ch1_filtered[:min_len]**2 + ch2_filtered[:min_len]**2)

                r_peaks_indices = find_r_peaks_data(
                    combined_data_filtered, SAMPLING_RATE,
                    min_height_factor=current_r_peak_height_factor,
                    min_distance_ms=current_r_peak_distance_ms,
                    identifier=f"融合信号 ({txt_filename})"
                )

                if r_peaks_indices is not None and len(r_peaks_indices) > 1:
                    avg_cycle_data, time_axis_data_ms = calculate_average_cycle_data(
                        data=combined_data_filtered,
                        r_peaks_indices=r_peaks_indices,
                        fs=SAMPLING_RATE,
                        pre_r_ms=PRE_R_PEAK_MS,
                        post_r_ms=POST_R_PEAK_MS,
                        identifier=f"融合信号 ({txt_filename})"
                    )
                    if avg_cycle_data is not None and time_axis_data_ms is not None:
                        collected_cycles_for_summary_plot.append((file_base_name, day_label, time_axis_data_ms, avg_cycle_data))
                        print(f"   成功计算文件 {txt_filename} 的平均周期。")
                    else:
                        print(f"   未能为文件 {txt_filename} 计算平均周期数据。")
                else:
                    print(f"   信息 ({txt_filename}): 未检测到足够的R峰 ({len(r_peaks_indices) if r_peaks_indices is not None else 0}个) 来提取平均周期。")


        if collected_cycles_for_summary_plot:
            print("\n--- 开始绘制所有文件的平均周期汇总图和R峰柱状图 ---")
            plot_and_save_summary_figure(
                collected_cycles_data=collected_cycles_for_summary_plot,
                output_dir=OUTPUT_FIGURE_DIR,
                horizontal_offset_increment=args.x_offset_per_day_group,
                summary_filename="summary_avg_cycles_and_Rpeaks_final.png"
            )
        else:
            print("\n没有成功处理任何文件以生成平均周期汇总图。")

        print(f"\n--- 批量处理结束 ---")
        print(f"总共处理了 {processed_files_count} 个文件。")
        if collected_cycles_for_summary_plot:
            print(f"汇总图（包含R峰柱状图）已生成、显示并保存，包含 {len(collected_cycles_for_summary_plot)} 个平均周期。")