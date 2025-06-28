import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

""" 该脚本用于批量处理文件夹下的所有.txt文件，进行时频分析，并保存为.png格式的图片。"""

def process_file(file_path, empty_Bx, empty_By, fs, output_dir):
    try:
        data = np.loadtxt(file_path, skiprows=2, encoding="utf-8")
        Bx = data[:, 0]
        By = data[:, 1]

        # 创建画布：3行2列
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

        # STFT - 实验 Bx
        f1, t1, Zxx1 = signal.stft(Bx, fs, nperseg=1024, noverlap=512)
        mask1 = f1 <= 40
        im1 = ax1.pcolormesh(t1, f1[mask1], np.abs(Zxx1[mask1]), shading='gouraud', cmap='bwr')
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_title('Bx Time-Frequency Analysis')
        fig.colorbar(im1, ax=ax1, label='Magnitude')
        im1.set_clim(0, 0.25)
        ax1.set_ylim(1, 10)

        # STFT - 实验 By
        f2, t2, Zxx2 = signal.stft(By, fs, nperseg=1024, noverlap=512)
        mask2 = f2 <= 40
        im2 = ax2.pcolormesh(t2, f2[mask2], np.abs(Zxx2[mask2]), shading='gouraud', cmap='bwr')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_title('By Time-Frequency Analysis')
        fig.colorbar(im2, ax=ax2, label='Magnitude')
        im2.set_clim(0, 0.25)
        ax2.set_ylim(1, 10)

        # STFT - 空载 Bx
        f3, t3, Zxx3 = signal.stft(empty_Bx, fs, nperseg=1024, noverlap=512)
        mask3 = f3 <= 40
        im3 = ax3.pcolormesh(t3, f3[mask3], np.abs(Zxx3[mask3]), shading='gouraud', cmap='bwr')
        ax3.set_ylabel('Frequency [Hz]')
        ax3.set_title('Empty Bx Time-Frequency Analysis')
        fig.colorbar(im3, ax=ax3, label='Magnitude')
        im3.set_clim(0, 0.25)
        ax3.set_ylim(1, 10)

        # STFT - 空载 By
        f4, t4, Zxx4 = signal.stft(empty_By, fs, nperseg=1024, noverlap=512)
        mask4 = f4 <= 40
        im4 = ax4.pcolormesh(t4, f4[mask4], np.abs(Zxx4[mask4]), shading='gouraud', cmap='bwr')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Frequency [Hz]')
        ax4.set_title('Empty By Time-Frequency Analysis')
        fig.colorbar(im4, ax=ax4, label='Magnitude')
        im4.set_clim(0, 0.25)
        ax4.set_ylim(1, 10)

        # 保存图片
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 成功处理: {file_path}")
    except Exception as e:
        print(f"❌ 处理失败: {file_path}，错误: {str(e)}")


def batch_process(input_dir, output_dir, empty_path, fs=1000):
    # 加载空载数据
    empty_data = np.loadtxt(empty_path, skiprows=2, encoding="utf-8")
    empty_Bx = empty_data[:, 0]
    empty_By = empty_data[:, 1]

    # 遍历文件夹下所有 .txt 文件
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                process_file(file_path, empty_Bx, empty_By, fs, output_dir)


if __name__ == "__main__":
    # 设置路径，批量处理B_egg_d2——B_egg_d21
    input_folder = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\B_egg'
    empty_file = r'C:\Users\Xiaoning Tan\Desktop\egg_2025\鸡蛋空载\空载1.txt'
    output_folder = r'C:\Users\Xiaoning Tan\Desktop\egg_STFT_100'


    # 执行批量处理
    batch_process(input_folder, output_folder, empty_file)

    print("🎉 全部处理完成！")
