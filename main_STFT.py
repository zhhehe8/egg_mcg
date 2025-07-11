import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

""" 该脚本用于处理单个 .txt 文件，进行时频分析，并显示和保存图形。"""

# 设置默认字体
plt.rcParams['font.sans-serif'] = ['Arial']  


def plot_single_channel_stft(signal_data, fs, channel_name, base_filename, output_path):
    """
    为单个信号通道计算、绘制、保存并显示STFT时频图。
    """
    print(f"正在为 {channel_name} 通道生成时频图...")

    # 1.进行STFT计算
    f, t_seg, Zxx = signal.stft(signal_data, fs, nperseg=1024, noverlap=512)

    # 2. 创建独立的图表
    fig, ax = plt.subplots(figsize=(4, 3))

    # 3. 使用 pcolormesh 绘制时频图，颜色主题为 'bwr'
    im = ax.pcolormesh(t_seg, f, np.abs(Zxx), shading='gouraud', cmap='bwr')

    # 4. 设置图表标题和坐标轴标签
    ax.set_title(f'STFT of day0', fontsize=14) # 更新标题以包含通道信息
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

    # 5. 设置Y轴范围为 1-10 Hz
    ax.set_ylim(1.5, 10)
    ax.set_xlim(0, 30)  # 设置X轴范围为0-30秒

    # 6. 添加颜色条，并设置其范围为 0-0.25
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (pT)', fontsize=12)
    im.set_clim(0, 0.25)

    # 7. 优化布局
    plt.tight_layout()

    try:
        # 使用 dpi=300 保存高分辨率图片
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图片已成功保存至: {output_path}")
    except Exception as e:
        print(f"❌ 保存图片失败: {e}")
    # ---------------------------------

    # 8. 显示图形
    plt.show()
    # 关闭图形对象，释放内存
    plt.close(fig)


if __name__ == "__main__":
    # 设置要处理的文件路径
    # file_path_str = '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_朱鹮/STFT of 朱鹮 choice/朱鹮day25_2_t2.txt'
    # file_path_str = '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_朱鹮/STFT of 朱鹮 choice/朱鹮day22_t3.txt'
    file_path_str = '/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_朱鹮/STFT of 朱鹮 choice/朱鹮未受精蛋1_t2.txt'
    fs = 1000  # 采样率

    # --- 【修改】定义并创建输出目录 ---
    output_dir = Path('/Users/yanchen/Desktop/Projects/egg/egg_2025/Figures')
    output_dir.mkdir(parents=True, exist_ok=True) # 确保目录存在
    # ---------------------------------

    # 获取文件名用于标题和保存
    file_path_obj = Path(file_path_str)
    base_filename = file_path_obj.stem # 获取不带后缀的文件名

    try:
        # 加载数据，跳过前两行
        data = np.loadtxt(file_path_obj, skiprows=2, encoding="utf-8")
        Bx = data[:, 0]
        By = data[:, 1]
        print(f"成功加载文件: {file_path_obj.name}")

        # --- 【修改】为 Bx 通道构建路径并调用绘图函数 ---
        output_path_bx = output_dir / f"{base_filename}_Bx_STFT.jpg"
        plot_single_channel_stft(Bx, fs, 'Bx', base_filename, output_path_bx)

        # --- 【修改】为 By 通道构建路径并调用绘图函数 ---
        output_path_by = output_dir / f"{base_filename}_By_STFT.jpg"
        plot_single_channel_stft(By, fs, 'By', base_filename, output_path_by)

    except Exception as e:
        print(f"❌ 处理失败: {file_path_obj.name}，错误: {str(e)}")