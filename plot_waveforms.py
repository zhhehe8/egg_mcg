""" 将保存的周期文件(.csv)批量绘制平均心动周期波形图 """

# plot_waveforms.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 导入我们的配置文件
import config

def plot_single_waveform(csv_path: Path, output_dir: Path):
    """
    读取单个平均波形CSV文件，并为其生成和保存一张图表。
    """
    try:
        # 1. 使用 Pandas 加载数据
        df = pd.read_csv(csv_path)
        
        # 检查必需的列是否存在
        if 'Time_ms' not in df.columns or 'Amplitude' not in df.columns:
            print(f"  警告: 文件 {csv_path.name} 缺少 'Time_ms' 或 'Amplitude' 列，跳过。")
            return
            
        time_ms = df['Time_ms']
        amplitude = df['Amplitude']

        # 2. 准备绘图
        plt.figure(figsize=(10, 7))
        
        # 3. 绘制波形
        plt.plot(time_ms, amplitude, color='blue', linewidth=2)
        
        # 4. 添加图表元素，使其更美观、信息更丰富
        plt.title(f'Averaged Cardiac Cycle\n{csv_path.stem}', fontsize=16)
        plt.xlabel('Time relative to R-peak (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 在R峰位置 (t=0) 绘制一条垂直虚线
        plt.axvline(x=0, color='red', linestyle='-.', label='R-Peak (t=0)')
        
        plt.legend()
        plt.tight_layout()
        
        # 5. 定义并保存图像文件
        # 从CSV文件名生成PNG文件名，例如 'file1_avg_waveform.csv' -> 'file1_avg_waveform.png'
        output_plot_path = output_dir / f"{csv_path.stem}.png"
        plt.savefig(output_plot_path, dpi=150) # dpi可以控制图片清晰度
        
        # 6. 关闭当前图形以释放内存 (在循环中绘图时至关重要！)
        plt.close()
        
        print(f"  ✓ 成功绘制并保存: {output_plot_path.name}")

    except Exception as e:
        print(f"  ✗ 处理文件 {csv_path.name} 时出错: {e}")


def main():
    """
    主函数，遍历所有平均波形文件并调用绘图函数。
    """
    print("--- 开始批量绘制平均心动周期图 ---")
    
    # 从配置中获取输入和输出目录
    input_dir = config.WAVEFORM_OUTPUT_DIR
    output_dir = config.WAVEFORM_PLOT_DIR
    
    # 检查输入目录是否存在
    if not input_dir.is_dir():
        print(f"错误: 平均波形数据文件夹 '{input_dir}' 不存在。")
        print("请先运行 main_batch.py 来生成这些文件。")
        sys.exit(1)
        
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 .csv 文件
    waveform_files = sorted(list(input_dir.glob('*.csv')))
    
    if not waveform_files:
        print(f"在 '{input_dir}' 中未找到任何 .csv 文件。")
        return
        
    print(f"找到 {len(waveform_files)} 个平均波形文件，开始绘图...")
    
    # 遍历并处理每个文件
    for csv_path in waveform_files:
        plot_single_waveform(csv_path, output_dir)
        
    print(f"\n✅ 全部完成！所有图表已保存至: {output_dir}")


if __name__ == '__main__':
    # 确保已安装依赖
    try:
        import pandas
        import matplotlib
    except ImportError as e:
        print(f"错误: 缺少必要的库 -> {e}")
        sys.exit(1)
        
    main()