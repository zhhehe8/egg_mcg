import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

""" 该脚本用于处理单个 .txt 文件，进行时频分析，并显示图形。"""

def process_file(file_path, empty_Bx, empty_By, fs):
    try:
        # 加载数据
        data = np.loadtxt(file_path, skiprows=2, encoding="utf-8")
        Bx = data[:, 0]
        By = data[:, 1]

        # 创建画布：3 行 2 列
        fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
        ax0, ax5 = axs[0]
        ax1, ax2 = axs[1]
        ax3, ax4 = axs[2]

        # 时域图 - 实验数据
        t = np.arange(len(Bx)) / fs
        ax0.plot(t, Bx, label='Bx', color='blue')
        ax0.plot(t, By, label='By', color='red', alpha=0.7)
        ax0.set_title("Raw Time-Domain Signals (Experiment)")
        ax0.set_xlabel("Time [s]")
        ax0.set_ylabel("Amplitude")
        ax0.legend()

        # 时域图 - 空载数据
        t_empty = np.arange(len(empty_Bx)) / fs
        ax5.plot(t_empty, empty_Bx, label='Empty Bx', color='blue')
        ax5.plot(t_empty, empty_By, label='Empty By', color='red', alpha=0.7)
        ax5.set_title("Raw Time-Domain Signals (Empty)")
        ax5.set_xlabel("Time [s]")
        ax5.set_ylabel("Amplitude")
        ax5.legend()

        # STFT 分析函数
        def plot_stft(ax, signal_data, title):
            f, t_seg, Zxx = signal.stft(signal_data, fs, nperseg=1024, noverlap=512)
            mask = f <= 40
            im = ax.pcolormesh(t_seg, f[mask], np.abs(Zxx[mask]), shading='gouraud', cmap='bwr')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(title)
            ax.set_ylim(1, 10)
            fig.colorbar(im, ax=ax, label='Magnitude')
            im.set_clim(0, 0.25)

        # 绘制 STFT
        plot_stft(ax1, Bx, 'Bx Time-Frequency Analysis')
        plot_stft(ax2, By, 'By Time-Frequency Analysis')
        plot_stft(ax3, empty_Bx, 'Empty Bx Time-Frequency Analysis')
        plot_stft(ax4, empty_By, 'Empty By Time-Frequency Analysis')

        # 布局优化并显示图形
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ 处理失败: {file_path}，错误: {str(e)}")


if __name__ == "__main__":
    # 设置要处理的文件路径和空载文件路径
    file_path = '/Users/yanchen/Desktop/Projects/egg_2025/朱鹮_250426/朱鹮day25_t1.txt'  # 修改为实际文件路径
    empty_file = '/Users/yanchen/Desktop/Projects/egg_2025/朱鹮_250426/空载1.txt'
    fs = 1000  # 采样率

    # 加载空载数据
    empty_data = np.loadtxt(empty_file, skiprows=2, encoding="utf-8")
    empty_Bx = empty_data[:, 0]
    empty_By = empty_data[:, 1]

    # 处理并显示单个文件
    process_file(file_path, empty_Bx, empty_By, fs)
