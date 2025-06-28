"""将8组数据的时频图绘制到同一张图上"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==== 参数设置 ====
fs = 1000  # 采样率
file_paths = [ 
    r'C:\Users\Administrator\Desktop\egg_2025\鸡蛋空载\空载1.txt',
    r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg\B_egg_d2\egg_d2_B3_t2.txt',
    r'C:\Users\Administrator\Desktop\fig_choice\egg_d3_B20_t1.txt',
    r'C:\Users\Administrator\Desktop\fig_choice\egg_d14_B28_t3.txt',
    r'C:\Users\Administrator\Desktop\fig_choice\egg_d15_B29_t2.txt',
    r'C:\Users\Administrator\Desktop\fig_choice\egg_d16_B27_t3.txt',
    r'C:\Users\Administrator\Desktop\fig_choice\egg_d17_B33_t2.txt',
    r'C:\Users\Administrator\Desktop\fig_choice\egg_d20_B30_t3.txt'
]
titles = ['NC', 'day2', 'day13', 'day14', 'day15', 'day16', 'day18', 'day20']

# ==== 创建时域图画布（3行3列） ====
fig, axs = plt.subplots(3, 3, figsize=(20, 10))
axs = axs.flatten()

# 删除第9个空子图
fig.delaxes(axs[8])
axs = axs[:8]  # 只保留前8个用于绘图

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
    axs[i].set_title(f'Time-Domain({titles[i]})', fontsize=10)
    axs[i].set_xlabel('Time [s]', fontsize=9)
    axs[i].set_ylabel('Amplitude', fontsize=9)
    axs[i].legend(fontsize=8)

# ==== STFT 图画布，同样删除第9个空子图 ====
fig2, axs2 = plt.subplots(3, 3, figsize=(20, 10))
axs2 = axs2.flatten()
fig2.delaxes(axs2[8])
axs2 = axs2[:8]

for i, filepath in enumerate(file_paths):
    try:
        data = np.loadtxt(filepath, skiprows=2, encoding='utf-8')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        continue

    Bx = data[:, 0]
    f, t_stft, Zxx = signal.stft(Bx, fs=fs, nperseg=1024, noverlap=512)
    mask = (f <= 40)

    im = axs2[i].pcolormesh(t_stft, f[mask], np.abs(Zxx[mask]), shading='gouraud', cmap='bwr')
    axs2[i].set_title(f'STFT of Bx ({titles[i]})', fontsize=10)
    axs2[i].set_xlabel('Time [s]', fontsize=9)
    axs2[i].set_ylabel('Frequency [Hz]', fontsize=9)
    axs2[i].set_ylim(1.5, 10)
    fig2.colorbar(im, ax=axs2[i], label='Magnitude')
    im.set_clim(0, 0.25)

plt.tight_layout()


plt.show()
