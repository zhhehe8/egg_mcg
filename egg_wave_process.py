"""
使用连续小波变换 (CWT) 对心磁数据进行时频分析。
该脚本会批量处理指定根目录下的所有子文件夹中的 .txt 数据文件，
并为每个文件生成一张包含Bx和By分量时频图的图像。
输出的图像将保持与输入文件相同的子文件夹结构。
"""

import numpy as np # NumPy库，用于高效的数值计算，特别是数组操作
import matplotlib.pyplot as plt # Matplotlib库，用于数据可视化和绘图
import pywt # PyWavelets库，用于小波变换计算
import os # OS库，用于操作系统相关功能，如文件路径处理
import zhplot # 如果您使用zhplot进行样式配置，请保留

# --- 配置参数 ---
# --- 请修改以下输入和输出目录 ---
root_input_dir = '/Users/yanchen/Desktop/Projects/egg_2025/朱鹮_250426'  # 要批量处理的根文件夹路径
output_dir = '/Users/yanchen/Desktop/Projects/egg_2025/wave_output_朱鹮' # 保存所有输出图像的文件夹

# 信号和文件参数
fs = 1000  # 采样率 (Hz)
skip_header = 2 # 加载数据时跳过文件开头的行数
encoding_val = 'utf-8' # 文件编码

# 小波变换参数
wavelet_name = 'cmor1.5-1.0'
min_freq_target = 1  # Hz
max_freq_target = 10 # Hz
num_scales = 128

# 绘图参数
cmap_plot = 'bwr' # 推荐使用感知均匀的色图，如 'viridis', 'plasma', 'magma'
colorbar_magnitude_min = 0.0
colorbar_magnitude_max = 5.0
# --- 配置参数结束 ---


# --- 辅助函数 (无需修改) ---
def load_mcg_data(filepath, skip, encoding):
    """从指定路径加载心磁数据文件。"""
    if not os.path.exists(filepath):
        print(f"  错误: 文件未找到 {filepath}")
        return None, None
    try:
        data = np.loadtxt(filepath, skiprows=skip, encoding=encoding)
        if data.ndim == 1 or data.shape[1] < 2:
            print(f"  错误：数据文件 {os.path.basename(filepath)} 需要至少两列 (Bx, By)。")
            return None, None
        Bx = data[:, 0]
        By = data[:, 1]
        print(f"  数据已从 {os.path.basename(filepath)} 加载。")
        return Bx, By
    except Exception as e:
        print(f"  错误: 加载 {os.path.basename(filepath)} 时出错: {e}")
        return None, None

def plot_cwt_scalogram(ax, signal_data, fs_sig, scales_arr, wavelet, title, cmap_name, vmin_cbar, vmax_cbar):
    """计算CWT并绘制尺度图。"""
    if signal_data is None:
        ax.set_title(f'{title}\n(数据不可用)')
        ax.text(0.5, 0.5, 'Data not available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return None, None

    time_vec = np.arange(len(signal_data)) / fs_sig
    
    print(f"  正在为 {title} 计算CWT...")
    try:
        coefficients, frequencies = pywt.cwt(signal_data, scales_arr, wavelet, sampling_period=1/fs_sig)
    except Exception as e:
        print(f"    错误: CWT 计算失败 for {title}: {e}")
        ax.set_title(f'{title}\n(CWT 计算失败)')
        return None, None

    im = ax.imshow(np.abs(coefficients), extent=[time_vec[0], time_vec[-1], frequencies[-1], frequencies[0]],
                   aspect='auto', cmap=cmap_name, interpolation='bilinear',
                   vmin=vmin_cbar, vmax=vmax_cbar)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=9)

    num_yticks_plot = 5
    yticks_idx = np.linspace(0, len(frequencies)-1, num_yticks_plot, dtype=int)
    ax.set_yticks(frequencies[yticks_idx])
    ax.set_yticklabels([f"{f:.1f}" for f in frequencies[yticks_idx]])
    return im, frequencies

def add_custom_colorbar(fig_obj, image_mappable, ax_obj, vmin, vmax, label_text):
    """为子图添加自定义刻度的颜色条。"""
    cbar = fig_obj.colorbar(image_mappable, ax=ax_obj, label=label_text, shrink=0.85)
    num_cbar_ticks = 6
    cbar_ticks = np.linspace(vmin, vmax, num_cbar_ticks)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.0f}" for tick in cbar_ticks])
    return cbar


# --- 主程序：批量处理 ---
if __name__ == "__main__":
    # 1. 确保输出目录存在 (此为主输出目录，子文件夹将在循环中创建)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出图像将保存到根目录: {output_dir}")

    # 2. 计算通用的尺度数组
    dt = 1/fs
    s0 = (pywt.central_frequency(wavelet_name, precision=8) * fs) / max_freq_target
    s_max = (pywt.central_frequency(wavelet_name, precision=8) * fs) / min_freq_target
    scales = np.geomspace(s0, s_max, num_scales)

    # 3. 遍历根目录下的所有文件和子文件夹
    file_count = 0
    for dirpath, _, filenames in os.walk(root_input_dir):
        for filename in filenames:
            # 只处理.txt文件
            if filename.endswith('.txt'):
                file_count += 1
                current_filepath = os.path.join(dirpath, filename)
                print(f"\n--- 正在处理第 {file_count} 个文件: {current_filepath} ---")

                # 加载数据
                Bx_raw, By_raw = load_mcg_data(current_filepath, skip_header, encoding_val)
                
                if Bx_raw is None or By_raw is None:
                    continue

                # ================================================================= #
                # --- 关键改动：构建与源文件结构相同的输出路径 ---
                # 1. 获取当前文件相对于根输入目录的相对路径
                relative_path = os.path.relpath(dirpath, root_input_dir)
                
                # 2. 在主输出目录下创建同样的子文件夹结构
                target_output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(target_output_subdir, exist_ok=True)
                # ================================================================= #

                # 创建1x2的画布
                fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
                
                plot_title = os.path.splitext(filename)[0]
                fig.suptitle(f'CWT Scalogram for: {plot_title}', fontsize=16)

                # 在左侧子图绘制 Bx
                im_bx, freqs_bx = plot_cwt_scalogram(axs[0], Bx_raw, fs, scales, wavelet_name,
                                                     'Bx Signal', cmap_plot,
                                                     colorbar_magnitude_min, colorbar_magnitude_max)
                if im_bx:
                    add_custom_colorbar(fig, im_bx, axs[0], colorbar_magnitude_min, colorbar_magnitude_max, 'Magnitude')
                
                # 在右侧子图绘制 By
                im_by, freqs_by = plot_cwt_scalogram(axs[1], By_raw, fs, scales, wavelet_name,
                                                     'By Signal', cmap_plot,
                                                     colorbar_magnitude_min, colorbar_magnitude_max)
                if im_by:
                    add_custom_colorbar(fig, im_by, axs[1], colorbar_magnitude_min, colorbar_magnitude_max, 'Magnitude')
                
                # 调整布局
                plt.tight_layout(rect=[0, 0.03, 1, 0.94])
                
                # 构建最终的输出文件名和完整路径
                output_filename = f"{plot_title}_CWT.png"
                output_path = os.path.join(target_output_subdir, output_filename) # <-- 保存到新的子目录中
                
                try:
                    plt.savefig(output_path, dpi=300)
                    print(f"  图像已保存到: {output_path}")
                except Exception as e:
                    print(f"  错误: 保存图像时出错: {e}")
                
                # 关闭当前图形，释放内存
                plt.close(fig)

    print(f"\n--- 批量处理完成！总共处理了 {file_count} 个文件。---")