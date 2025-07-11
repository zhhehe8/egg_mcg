"""将9组数据的时频图绘制到同一张图上"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# 设置默认字体
plt.rcParams['font.sans-serif'] = ['Arial'] 

# ==== 参数设置 ====
fs = 1000  # 采样率
file_paths = [ 
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/20250218_空载1.txt',
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d2_B1_t3.txt',
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d9_B15_t1.txt',
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d13_B24_t3.txt',
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d16_B27_t3.txt',
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d17_B29_t3.txt',
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d18_B34_t3.txt',
    # '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d19_B30_t3.txt',
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d19_B34_t2.txt',
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d20_B30_t1_待破壳.txt'
]
titles = ['day0', 'day2', 'day9', 'day13', 'day16', 'day17', 'day18', 'day19', 'day20']

# 输出目录
output_dir = Path('/Users/yanchen/Desktop/Projects/egg/egg_2025/Figures')
output_dir.mkdir(parents=True, exist_ok=True)

# ==== 创建时域图画布（3行3列） ====
fig, axs = plt.subplots(3, 3, figsize=(13, 10))
axs = axs.flatten()



for i, filepath in enumerate(file_paths):
    try:
        data = np.loadtxt(filepath, skiprows=2, encoding='utf-8')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        continue

    Bx = data[:, 0]
    By = data[:, 1]
    t = np.arange(len(Bx)) / fs

    axs[i].plot(t, Bx, label='Bx', color='blue', linewidth=1)
    axs[i].plot(t, By, label='By', color='red', linewidth=1, alpha=0.7)
    axs[i].set_title(f'Time-Domain({titles[i]})', fontsize=14)
    axs[i].set_xlabel('Time (s)', fontsize=12)
    axs[i].set_ylabel('Amplitude (pT)', fontsize=12)
    axs[i].legend(fontsize=12)
    axs[i].tick_params(axis='both', labelsize=10)
    

fig2, axs2 = plt.subplots(3, 3, figsize=(12, 8), sharey=False)
axs2 = axs2.flatten()

im = None # 初始化im

for i, filepath in enumerate(file_paths):
    try:
        data = np.loadtxt(filepath, skiprows=2, encoding='utf-8')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        axs2[i].set_title(f'Error loading {titles[i]}', fontsize=12, color='red')
        continue

    Bx = data[:, 0]
    f, t_stft, Zxx = signal.stft(Bx, fs=fs, nperseg=1024, noverlap=512)
    mask = (f <= 40)

    im = axs2[i].pcolormesh(t_stft, f[mask], np.abs(Zxx[mask]), shading='gouraud', cmap='viridis')
    axs2[i].set_title(f'STFT of {titles[i]}', fontsize=14)
    axs2[i].set_xlabel('Time (s)', fontsize=12)
    # 条件化设置标签
     # 1. Frequency [Hz] 标签：只在第一列显示 (i=0, 3, 6)
    if i % 3 == 0:
        axs2[i].set_ylabel('Frequency (Hz)', fontsize=12)

    # 2. Magnitude 标签：只在第三列显示 (i=2, 5, 8)
    if i % 3 == 2:
        cbar = fig2.colorbar(im, ax=axs2[i])
        cbar.set_label('Magnitude (pT)', fontsize=12)
    else:
        # 其他列只显示颜色条，不显示标签
        fig2.colorbar(im, ax=axs2[i])

    axs2[i].set_xlim(0, 30)
    axs2[i].set_ylim(1.5, 10)
    axs2[i].tick_params(axis='both', labelsize=12)
    im.set_clim(0, 0.25)

# 保存图片
fig.tight_layout()
fig.savefig(output_dir / 'summary_time_domain.jpg', dpi=300)

fig2.tight_layout()
fig2.savefig(output_dir / 'summary_stft.jpg', dpi=300)

print(f"\n图片已成功保存至: {output_dir}")

plt.show()
