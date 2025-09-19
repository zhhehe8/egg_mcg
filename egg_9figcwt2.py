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
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d19_B34_t2.txt',
    '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg/STFT_choice/egg_d20_B30_t1_待破壳.txt'
]
titles = ['day0', 'day2', 'day9', 'day13', 'day16', 'day17', 'day18', 'day19', 'day20']

# 输出目录
output_dir = Path('/Users/yanchen/Desktop/Projects/egg/egg_2025/Figures')
output_dir.mkdir(parents=True, exist_ok=True)

# ==== 创建时频图画布（3行3列）====
fig, axs = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
axs = axs.flatten()

im = None # 初始化im用于存储pcolormesh对象，以便后续添加色标

for i, filepath in enumerate(file_paths):
    try:
        data = np.loadtxt(filepath, skiprows=2, encoding='utf-8')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        axs[i].set_title(f'Error loading {titles[i]}', fontsize=12, color='red')
        continue

    Bx = data[:, 0]
    f, t_stft, Zxx = signal.stft(Bx, fs=fs, nperseg=1024, noverlap=512)
    mask = (f <= 40)

    # 绘制时频图
    im = axs[i].pcolormesh(t_stft, f[mask], np.abs(Zxx[mask]), shading='gouraud', cmap='viridis', vmin=0, vmax=0.25)
    axs[i].set_title(f'STFT of {titles[i]}', fontsize=14, fontweight='bold')

    # 在网格的边缘设置坐标轴标签
    if i % 3 == 0:
        axs[i].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    if i >= 6:
        axs[i].set_xlabel('Time (s)', fontsize=12, fontweight='bold')

    axs[i].set_xlim(0, 30)
    axs[i].set_ylim(1.5, 10)
    axs[i].tick_params(axis='both', labelsize=12)

# 调整整体布局，为底部的水平色标留出空间
fig.tight_layout(rect=[0, 0.1, 1, 1])

# === 主要修改处：动态对齐逻辑 ===

# 1. 强制Matplotlib执行布局计算，以便获取子图的最终位置
fig.canvas.draw()

# 2. 获取最右侧子图的边界。我们以第一行最右边的子图(axs[2])为例，
#    因为同一列的子图右边界是对齐的。
pos = axs[2].get_position()
right_align_edge = pos.x1  # .x1 获取该子图在图表坐标系中的右侧x坐标

# 3. 根据对齐边界计算色标的位置
cbar_width = 0.267   # 色标宽度 (保持上次修改的 2/3 尺寸)
cbar_height = 0.02   # 色标高度
cbar_bottom = 0.08   # 色标的y轴位置

# 新的 left = 对齐位置 - 色标宽度
cbar_left = right_align_edge - cbar_width

# 4. 在计算出的精确位置创建色标坐标轴
cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

# 在新创建的坐标轴上添加水平方向的共享色标
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Magnitude (pT)', fontsize=12, fontweight='bold')


# 保存图片
fig.savefig(output_dir / 'fig6(2).jpg', dpi=300)

print(f"\n图片已成功保存至: {output_dir}")

plt.show()

# 保存图片
