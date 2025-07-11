import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import sys

"""
该脚本用于批量处理主文件夹下的所有数据文件。
它将参照 egg_STFT.py 中的方法，对每个文件的原始数据进行STFT处理，
并将生成的 Bx 和 By 时频图分别保存到指定目录。
"""


try:
    import config
except ImportError:
    print("错误: 无法导入 'config.py'。")
    print("请确保此脚本与 config.py 在同一目录下，或 config.py 所在的路径已添加到 PYTHONPATH。")
    sys.exit(1)


def process_and_save_stft(filepath: Path, output_dir: Path, fs: int):
    """
    加载单个文件，对其 Bx 和 By 通道进行STFT分析，并分别保存结果图。
    """
    print(f"--- 正在处理: {filepath.name} ---")
    base_filename = filepath.stem

    try:
        # 1. 加载原始数据
        data = np.loadtxt(filepath, skiprows=2, encoding="utf-8")
        Bx = data[:, 0]
        By = data[:, 1]

        # 2. 分别为 Bx 和 By 生成并保存 STFT 图
        for channel_data, channel_name in zip([Bx, By], ['Bx', 'By']):
            # 创建独立的图表
            fig, ax = plt.subplots(figsize=(8, 6))

            # 使用与 egg_STFT.py 相同的参数进行STFT计算
            f, t_seg, Zxx = signal.stft(channel_data, fs, nperseg=1024, noverlap=512)

            # 使用 pcolormesh 绘制时频图
            im = ax.pcolormesh(t_seg, f, np.abs(Zxx), shading='gouraud', cmap='bwr')

            # 设置图表样式
            ax.set_title(f'{channel_name} Time-Frequency Analysis - {base_filename}')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_xlabel('Time [s]')
            ax.set_ylim(1, 10) # Y轴范围与参考脚本一致

            # 添加并设置颜色条
            fig.colorbar(im, ax=ax, label='Magnitude')
            im.set_clim(0, 0.25) # 颜色条范围与参考脚本一致

            plt.tight_layout()

            # 定义输出路径并保存
            output_path = output_dir / f"{base_filename}_{channel_name}_STFT.png"
            plt.savefig(output_path)
            
            # 关闭图形，释放内存（在循环中至关重要）
            plt.close(fig)

        print(f"  ✓ Bx 和 By 的STFT分析图已保存。")

    except Exception as e:
        print(f"  ✗ 处理文件 {filepath.name} 时发生错误: {e}")


def run_batch_processing():
    """
    执行批量处理的主函数。
    """
    # 1. 设置输出目录
    output_dir = Path('/Users/yanchen/Desktop/Projects/egg/egg_2025/Results/STFT_Begg')
    print(f"所有结果将保存至: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 从 config.py 获取数据根目录和采样率
    data_root_dir = config.DATA_ROOT_DIR
    fs = config.PROCESSING_PARAMS.get('fs', 1000)

    if not data_root_dir.is_dir():
        print(f"错误: 数据根目录 '{data_root_dir}' 不存在或不是一个文件夹。", file=sys.stderr)
        return

    print("\n=========================================")
    print(f"开始批量STFT分析，扫描目录: {data_root_dir}")
    print("=========================================")

    # 3. 遍历所有子文件夹中的 .txt 文件
    all_files = list(data_root_dir.glob('*/*.txt'))
    if not all_files:
        print("警告：在子文件夹中未找到.txt文件，尝试直接扫描根目录...")
        all_files = list(data_root_dir.glob('*.txt'))

    if not all_files:
        print("错误：在指定目录及其一级子目录中均未找到任何 .txt 文件。")
        return
        
    total_files = len(all_files)
    print(f"找到 {total_files} 个文件，开始处理...")
    
    # 4. 循环处理每个文件
    for i, filepath in enumerate(all_files):
        process_and_save_stft(filepath, output_dir, fs)
        print(f"  进度: {i + 1}/{total_files}")

    print("\n=========================================")
    print("✅ 所有文件处理完毕！")
    print(f"结果已全部保存在: {output_dir}")
    print("=========================================")


if __name__ == "__main__":
    run_batch_processing()