# config.py
"""包含所有可变参数（文件路径、滤波器设置、算法参数等等）"""

from pathlib import Path

# -----------------
# 1. 文件路径配置
# # -----------------
# """ 使用 Path 对象，能更好地处理跨平台路径问题 """
BASE_DIR = Path('/Users/yanchen/Desktop/Projects/egg/egg_2025') # 项目的基础目录
# INPUT_FILE = BASE_DIR/ 'Results/waveforms_Ibis/waveform_Ibis_choice/朱鹮day22_t1.txt'
# INPUT_FILE = BASE_DIR / '/Users/yanchen/Desktop/Projects/egg/egg_2025/朱鹮_250426/day/朱鹮day25_t2.txt' 

# INPUT_FILE = BASE_DIR / '第一批鸡蛋测量数据'/ 'd17_e7_t3很不错.txt'
# INPUT_FILE = BASE_DIR / 'B_egg'/ 'B_egg_d20'/'egg_d20_B30_t1_待破壳.txt'
# INPUT_FILE = BASE_DIR / 'time_choice'/ 'egg_d13_B20_t2.txt'
# INPUT_FILE = BASE_DIR / 'time_choice'/ 'egg_d19_B30_t1.txt'
INPUT_FILE = BASE_DIR / 'time_choice/egg_d20_B33_t1.txt'

# INPUT_FILE = BASE_DIR / '朱鹮_250426'/ '朱鹮day20_t1.txt' 
OUTPUT_DIR = BASE_DIR / 'Figures' # 所有输出（如图片）的保存目录

""" ---- 批量处理数据版本 ---- """
# 指向包含所有天数文件夹下的根目录
# DATA_ROOT_DIR = Path('/Users/yanchen/Desktop/Projects/egg/egg_2025/B_egg')    # B_egg数据的根目录
DATA_ROOT_DIR = Path('/Users/yanchen/Desktop/Projects/egg/egg_2025/朱鹮_250426') # 朱鹮数据的根目录
# 为结果创建输出目录
RESULTS_OUTPUT_DIR = Path('/Users/yanchen/Desktop/Projects/egg_2025/Results/')
# 为平均波形文件(.csv)创建一个专门的子文件夹
WAVEFORM_OUTPUT_DIR = RESULTS_OUTPUT_DIR / 'averaged_waveforms' 
#  为使用平均波形文件(.csv)绘制的图片创建一个专门的输出文件夹
WAVEFORM_PLOT_DIR = RESULTS_OUTPUT_DIR / 'averaged_waveform_plots'

# -----------------
# 2. 信号预处理配置
# -----------------
PROCESSING_PARAMS = {
    'fs': 1000,  # 采样率 (Hz)
    'reverse_Bx': False, # 是否反转Bx信号
    'reverse_By': False, # 是否反转By信号
    'analysis_duration_s': 30,  # 只取每组数据的前30秒进行分析（若要全部分析： 0 或 None）
}

# -----------------
# 3. 滤波器配置
# -----------------
FILTER_PARAMS = {
    'bandpass': {
        'order': 3,
        'lowcut': 0.5,   # Hz, 低截止频率
        'highcut': 45.0, # Hz, 高截止频率
    },
    'notch': {
        'freq': 50.0,    # Hz, 工频干扰频率
        'q_factor': 30.0,
    },
    'wavelet': { # 小波去噪配置
        'enabled': True, # 设置为 True 来启用小波去噪
        'wavelet': 'sym8',
        'level': 7,
        # 新增的小波去噪参数（肌电信号）
        'denoise_levels': 3       # 只对前3层高频细节进行去噪
    }
}

# -----------------
# 4. R峰检测配置
# -----------------
R_PEAK_PARAMS = {
    'min_height_factor': 0.4, # R峰最小高度因子
    'min_distance_ms': 200,   # R峰最小距离 (毫秒)
    'height_percentile': 99,  # 设置用于计算阈值的百分位数
}

# -----------------
# 5. 叠加平均配置
# -----------------
AVERAGING_PARAMS = {
    'pre_r_ms': 100,  # R峰前的时间窗口 (毫秒)
    'post_r_ms': 100, # R峰后的时间窗口 (毫秒)
    'ethod': 'median', # 叠加平均方法 ('median' 或 'dtw')
}


# -----------------
# 6. 时频分析配置
# -----------------
TIME_FREQUENCY_PARAMS = {
    # 'enabled_method' 选项:
    # 'stft' - 只进行短时傅里叶变换
    # 'cwt'  - 只进行连续小波变换
    # 'both' - 两种方法都进行
    # 'none' - 不进行时频分析
    'enabled_method': 'none', # 可选值: 'stft', 'cwt', 'both', 'none'

    'stft': {
        'window_length': 512,       # STFT窗口长度
        'overlap_ratio': 0.5,       # 窗口重叠比例 (例如0.5代表50%)
    },
    'cwt': {
        'wavelet': 'cmor1.5-1.0',    # CWT使用的小波基函数
        'max_scale': 128,           # CWT分析的最大尺度 (值越大，频率越低)
        'num_scales': 100,          # 在1到max_scale之间取多少个尺度
    },
    'plot': {
        # 'style' 选项: 'single', 'dual', 'both', 'none'
        'style': 'single', # 是否生成时频图表
    }
}


# -----------------
# 7. 绘图配置
# -----------------
PLOTTING_PARAMS = {
    # 'style' 选项:
    # 'single' - 每个通道单独生成一张图
    # 'dual'   - Bx和By整合到一张图中显示
    # 'both'   - 以上两种图都生成
    # 'none'   - 不生成任何图表
    'generate_plots_per_file': False, # 是否为每个文件生成图表

    'filtering_plot': {
        'style': 'single', # 'single', 'dual', 'both', 'none'
    },
    'averaging_plot': {
        'style': 'single',
        # 'median' - 只绘制中位数平均波形
        # 'dtw'    - 只绘制DTW对齐平均波形
        # 'both'   - 绘制两者的对比图 (默认)
        'display_method': 'median', 
    }
}

# 8. 保存选项配置
# -----------------
SAVING_PARAMS = {
    # 设置为 True，则会在批量处理时保存每个样本的平均心拍波形
    'save_averaged_waveform': True,
}


