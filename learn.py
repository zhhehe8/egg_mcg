


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os 

# ==== 参数设置 ====
fs = 1000  # 采样率
file_paths = [ 
    '/Users/yanchen/Desktop/Projects/egg_2025/空载250218/20250218_空载1.txt',
    '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d2/egg_d2_B3_t2.txt',
    '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B30_t1_待破壳.txt', 
    '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B30_t2.txt',
    '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B30_t3.txt',
    '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B33_t1.txt',
    '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B33_t2.txt',
    '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B33_t3.txt'
]
# 确保 titles 列表的长度和 file_paths 一致，并且正确对应
# 根据你的文件路径，我调整了几个标签，请你确认是否正确
titles = ['NC', 'day2', 'day3', 'day14', 'day15', 'day16', 'day17', 'day20'] 
# 如果 titles 的数量和 file_paths 不一致，请调整，确保一一对应。
# 如果文件路径与实际天数不符，也请修改 titles 列表。

# ==== 创建时域图画布（3行3列） ====
fig_time, axs_time = plt.subplots(3, 3, figsize=(20, 12)) # 稍微调整画布大小以便容纳
axs_time_flat = axs_time.flatten()

# 删除第9个空子图 (如果子图总数大于8)
if len(axs_time_flat) > 8:
    fig_time.delaxes(axs_time_flat[8])
axs_time_plot = axs_time_flat[:8]  # 只保留前8个用于绘图

for i, filepath in enumerate(file_paths):
    if i >= len(axs_time_plot): # 避免超出子图数量
        break
    try:
        data = np.loadtxt(filepath, skiprows=2, encoding='utf-8')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        axs_time_plot[i].set_title(f'Error loading ({titles[i]})', fontsize=10, color='red')
        continue

    if data.ndim == 1 or data.shape[0] == 0 or data.shape[1] < 2:
        print(f"Insufficient data in {filepath}")
        axs_time_plot[i].set_title(f'Insufficient data ({titles[i]})', fontsize=10, color='orange')
        continue
        
    Bx = data[:, 0]
    By = data[:, 1]
    t = np.arange(len(Bx)) / fs

    axs_time_plot[i].plot(t, Bx, label='Bx', color='blue', linewidth=1)
    axs_time_plot[i].plot(t, By, label='By', color='red', linewidth=1, alpha=0.7)
    axs_time_plot[i].set_title(f'Time-Domain ({titles[i]})', fontsize=10)
    axs_time_plot[i].set_xlabel('Time [s]', fontsize=9)
    axs_time_plot[i].set_ylabel('Amplitude', fontsize=9)
    axs_time_plot[i].legend(fontsize=8)

fig_time.tight_layout(pad=2.0) # 给时域图应用tight_layout

# ==== STFT 图画布，同样删除第9个空子图 ====
fig_stft, axs_stft = plt.subplots(3, 3, figsize=(20, 12)) # 稍微调整画布大小
axs_stft_flat = axs_stft.flatten()
if len(axs_stft_flat) > 8:
    fig_stft.delaxes(axs_stft_flat[8])
axs_stft_plot = axs_stft_flat[:8]

# 用于存储从每个STFT中识别出的特征频率
identified_heartbeat_frequencies = []
# 定义心跳频率分析的范围 (你可以根据实际情况调整这个范围)
heartbeat_frequency_band = [1.5, 10]  # Hz

for i, filepath in enumerate(file_paths):
    if i >= len(axs_stft_plot): # 避免超出子图数量
        break
    try:
        data = np.loadtxt(filepath, skiprows=2, encoding='utf-8')
    except Exception as e:
        print(f"Error loading {filepath} for STFT: {e}")
        axs_stft_plot[i].set_title(f'Error loading ({titles[i]})', fontsize=10, color='red')
        identified_heartbeat_frequencies.append(np.nan) # 添加 NaN 表示错误
        continue

    if data.ndim == 1 or data.shape[0] == 0 or data.shape[1] < 1: # STFT只需要一个通道
        print(f"Insufficient data in {filepath} for STFT (Bx channel)")
        axs_stft_plot[i].set_title(f'Insufficient Bx data ({titles[i]})', fontsize=10, color='orange')
        identified_heartbeat_frequencies.append(np.nan)
        continue

    Bx = data[:, 0] # 我们只处理 Bx 通道的 STFT
    
    # STFT 计算
    # nperseg: 每个段的长度。增加它会提高频率分辨率，降低时间分辨率。
    # noverlap: 段之间的重叠点数。通常是 nperseg 的一半左右。
    f_stft, t_stft, Zxx = signal.stft(Bx, fs=fs, nperseg=1024, noverlap=768) # 增加重叠以获得更平滑的时间轴

    # 创建频率掩码，用于绘图和分析 (例如，我们关心 1.5Hz 到 10Hz)
    freq_mask_plot = (f_stft >= heartbeat_frequency_band[0]) & (f_stft <= heartbeat_frequency_band[1])
    
    if not np.any(freq_mask_plot):
        print(f"Warning: No STFT frequencies fall within the plotting band [{heartbeat_frequency_band[0]}, {heartbeat_frequency_band[1]}] Hz for {titles[i]}.")
        axs_stft_plot[i].set_title(f'STFT - No data in band ({titles[i]})', fontsize=10, color='orange')
        identified_heartbeat_frequencies.append(np.nan)
        continue

    # 绘制 STFT 热力图
    im = axs_stft_plot[i].pcolormesh(t_stft, f_stft[freq_mask_plot], np.abs(Zxx[freq_mask_plot, :]), 
                                     shading='gouraud', cmap='bwr', vmin=0, vmax=0.25) # 直接设置颜色范围
    axs_stft_plot[i].set_title(f'STFT of Bx ({titles[i]})', fontsize=10)
    axs_stft_plot[i].set_xlabel('Time [s]', fontsize=9)
    axs_stft_plot[i].set_ylabel('Frequency [Hz]', fontsize=9)
    axs_stft_plot[i].set_ylim(heartbeat_frequency_band[0], heartbeat_frequency_band[1]) # 设置Y轴范围
    fig_stft.colorbar(im, ax=axs_stft_plot[i], label='Magnitude')

    # --- 提取特征心跳频率 ---
    # 在定义的 heartbeat_frequency_band 内寻找平均幅度最大的频率
    # 注意: Zxx 的维度是 (频率点数, 时间点数)
    abs_Zxx_in_band = np.abs(Zxx[freq_mask_plot, :])
    if abs_Zxx_in_band.size > 0:
        avg_magnitude_per_freq_in_band = np.mean(abs_Zxx_in_band, axis=1) #沿时间轴平均
        if avg_magnitude_per_freq_in_band.size > 0:
            peak_freq_index_in_band = np.argmax(avg_magnitude_per_freq_in_band)
            characteristic_freq = f_stft[freq_mask_plot][peak_freq_index_in_band]
            identified_heartbeat_frequencies.append(characteristic_freq)
            print(f"  File: {titles[i]}, Identified Beat Freq (Bx): {characteristic_freq:.2f} Hz")
        else:
            print(f"  File: {titles[i]}, No frequency components in band after masking for STFT analysis.")
            identified_heartbeat_frequencies.append(np.nan)
    else:
        print(f"  File: {titles[i]}, STFT result is empty within the specified band.")
        identified_heartbeat_frequencies.append(np.nan)

fig_stft.tight_layout(pad=2.0) # 给STFT图应用tight_layout

# ==== 创建并绘制心跳频率柱状图 ====
fig_bar, ax_bar = plt.subplots(figsize=(12, 8)) # 为柱状图创建一个新图形

# 过滤掉NaN值（加载失败或频率提取失败的情况）
valid_indices = [idx for idx, freq in enumerate(identified_heartbeat_frequencies) if not np.isnan(freq)]
valid_titles = [titles[idx] for idx in valid_indices]
valid_frequencies = [identified_heartbeat_frequencies[idx] for idx in valid_indices]

if valid_titles: # 只有在有有效数据时才绘制
    bars = ax_bar.bar(valid_titles, valid_frequencies, color='skyblue', edgecolor='black')
    ax_bar.set_xlabel('Signal Source / Day', fontsize=12)
    ax_bar.set_ylabel('Identified Dominant Frequency (Hz) in Bx', fontsize=12)
    ax_bar.set_title('Dominant Frequencies (1.5-10 Hz Band) from STFT of Bx Channel', fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10) # 旋转X轴标签以防重叠
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

    # 在每个柱子上方显示数值
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2f}', 
                    ha='center', va='bottom', fontsize=9)
    
    # 稍微调整Y轴上限，以便容纳柱子上的文本
    if valid_frequencies:
        ax_bar.set_ylim(0, max(valid_frequencies) * 1.15 if max(valid_frequencies) > 0 else 1)

else:
    ax_bar.text(0.5, 0.5, "No valid frequency data to plot for bar chart.", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax_bar.transAxes, fontsize=12)


fig_bar.tight_layout() # 给柱状图应用tight_layout

plt.show() # 显示所有创建的图形