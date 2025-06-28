""" 尝试通过抛物线插值，将R峰的定位提升到亚样本精度。 """
""" 尝试使用求中位数而不是平均数来计算平均心跳周期。  """
""" 尝试使用动态时间规整(DTW)对齐心拍，并进行叠加平均。 """

import numpy as np
import zhplot
import matplotlib.pyplot as plt
import os
from fastdtw import fastdtw
from egg_functions import bandpass_filter, apply_notch_filter, find_r_peaks_data, plot_signals_with_r_peaks

def interpolate_peaks(data, peak_indices):
    """
    通过抛物线插值，提高波峰定位的亚样本精度。

    参数:
    data (np.ndarray): 一维信号数据。
    peak_indices (np.ndarray): 由 find_peaks 等函数找到的波峰的整数索引。

    返回:
    np.ndarray: 包含亚样本精度波峰位置的浮点数索引数组。
    """
    precise_peak_indices = []
    
    for peak_index in peak_indices:
        # 确保波峰不在信号的边缘，以便获取左右两个邻近点
        if 1 <= peak_index < len(data) - 1:
            # 提取波峰及其左右邻近的三个点
            y_coords = data[peak_index-1 : peak_index+2]
            x_coords = np.arange(peak_index-1, peak_index+2)
            
            # 对三个点进行二次多项式（抛物线）拟合
            # y = ax^2 + bx + c
            coeffs = np.polyfit(x_coords, y_coords, 2)
            
            # 计算抛物线的顶点 x = -b / (2a)
            # coeffs[0] 是 a, coeffs[1] 是 b
            # 添加一个微小量到分母以避免除以零
            a, b = coeffs[0], coeffs[1]
            if a == 0:
                 # 如果'a'是0，说明不是抛物线，无法插值，使用原始索引
                precise_peak_indices.append(float(peak_index))
                continue
                
            precise_peak_x = -b / (2 * a)
            precise_peak_indices.append(precise_peak_x)
        else:
            # 如果波峰在边缘，无法插值，直接使用原始整数索引
            precise_peak_indices.append(float(peak_index))
            
    return np.array(precise_peak_indices)


def robust_average_beats_median(signal_data, precise_peak_indices, pre_samples, post_samples):
    """
    通过插值提取心拍，并使用中位数(MEDIAN)进行鲁棒的叠加平均。
    """
    beats_list = []
    original_x_indices = np.arange(len(signal_data))
    relative_x_indices = np.arange(-pre_samples, post_samples)
    
    for peak_loc in precise_peak_indices:
        absolute_x_to_interpolate = peak_loc + relative_x_indices
        if absolute_x_to_interpolate[0] < 0 or absolute_x_to_interpolate[-1] >= len(signal_data):
            continue
        interpolated_beat = np.interp(absolute_x_to_interpolate, original_x_indices, signal_data)
        beats_list.append(interpolated_beat)
        
    if not beats_list:
        print("警告: 未能提取到任何完整的心跳周期。")
        total_samples = pre_samples + post_samples
        return np.zeros(total_samples), np.empty((0, total_samples))

    all_beats = np.array(beats_list)
    
    # --- 核心改动：使用 np.median 代替 np.mean ---
    median_beat = np.median(all_beats, axis=0)
    
    return median_beat, all_beats


def dtw_average_beats(signal_data, precise_peak_indices, pre_samples, post_samples):
    """
    使用动态时间规整(DTW)对齐心拍，并进行叠加平均。
    """
    # 步骤 1: 使用插值提取所有原始心拍
    beats_list = []
    original_x_indices = np.arange(len(signal_data))
    relative_x_indices = np.arange(-pre_samples, post_samples)
    for peak_loc in precise_peak_indices:
        absolute_x_to_interpolate = peak_loc + relative_x_indices
        if absolute_x_to_interpolate[0] < 0 or absolute_x_to_interpolate[-1] >= len(signal_data):
            continue
        interpolated_beat = np.interp(absolute_x_to_interpolate, original_x_indices, signal_data)
        beats_list.append(interpolated_beat)

    if len(beats_list) < 2:
        print("警告: 心拍数量不足，无法进行DTW平均。")
        return np.array(beats_list[0]) if beats_list else np.zeros(pre_samples + post_samples)

    all_beats = np.array(beats_list)
    
    # 步骤 2: 创建一个初始模板 (使用中位数平均法得到一个鲁棒的模板)
    template_beat = np.median(all_beats, axis=0)
    
    # 步骤 3: 对每个心拍进行DTW对齐，并存储规整后的波形
    warped_beats = []
    print(f"开始对 {len(all_beats)} 个心拍进行DTW对齐...")
    
    for i, beat in enumerate(all_beats):
        # --- 【核心修改点】 ---
        # 将 dist=euclidean 替换为一个计算两点间距离的 lambda 函数
        # 使用平方差作为距离度量，这是标准做法。
        distance, path = fastdtw(template_beat, beat, dist=lambda a, b: (a - b)**2)
        
        # 'path' 是一个(模板索引, 当前心拍索引)的元组列表
        # 我们需要根据这个路径来“反向”构建一个规整后(warped)的心拍
        
        # 创建一个与模板等长的、用于存储规整后心拍的空数组
        warped_beat = np.zeros_like(template_beat)
        # 记录每个点被映射了多少次，用于求平均
        warp_counts = np.zeros_like(template_beat)
        
        for template_idx, beat_idx in path:
            warped_beat[template_idx] += beat[beat_idx]
            warp_counts[template_idx] += 1
            
        # 对被多次映射的点求平均值
        # 避免除以零
        warp_counts[warp_counts == 0] = 1
        warped_beat /= warp_counts
        
        warped_beats.append(warped_beat)
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(all_beats)}...")

    # 步骤 4: 对所有规整后的心拍进行最终的平均
    final_averaged_beat = np.mean(np.array(warped_beats), axis=0)
    
    return final_averaged_beat



### ---- 主程序开始 ---- ###
# 1. 加载数据

"""输入输出目录"""
input_dir = '/Users/yanchen/Desktop/Projects/egg_2025/B_egg/B_egg_d20/egg_d20_B30_t1_待破壳.txt'  # 输入数据文件路径
output_dir = '/Users/yanchen/Desktop'          # 输出目录

data = np.loadtxt(input_dir, skiprows=2, encoding="utf-8")
Bx_raw = data[:, 0]
By_raw = data[:, 1]
fs = 1000  # 采样率

# 2. 检测参数设置

"""时间参数"""
time = np.arange(len(Bx_raw)) / fs  # 时间向量

""" 设置滤波器参数 """
filter_order_bandpass = 4  # 带通滤波器的阶数 (根据用户最新提供)
lowcut_freq = 0.5          # Hz, 低截止频率
highcut_freq = 45.0        # Hz, 高截止频率
notch_freq = 50.0    # Hz, 工频干扰频率
Q_factor_notch = 30.0      # 陷波滤波器的品质因数

""" 设置R峰检测参数 """
R_peak_min_height_factor = 0.6  # R峰最小高度因子 (相对于数据的最大值) 
R_peak_min_distance_ms = 200     # R峰最小距离 (毫秒)


""" 设置平均心跳周期参数 """
pre_r_ms = 100   # R峰前的时间窗口 (毫秒)
post_r_ms = 300  # R峰后的时间窗口 (毫秒)


# 自动从数据文件路径提取 base_filename
base_filename = os.path.splitext(os.path.basename(input_dir))[0]
output_dir = output_dir  # 保持后续变量一致


"""设置信号反转"""
reverse_Bx_signal = False  
if reverse_Bx_signal:
    Bx_raw = -Bx_raw  # 反转Bx信号
    print("信息: Bx 信号已反转。")

reverse_By_signal = False  
if reverse_By_signal:
    By_raw = -By_raw  # 反转By信号
    print("信息: By 信号已反转。")


# 3. 对原始数据进行滤波处理
print("开始滤波 Bx_raw 信号...")
Bx_filtered = bandpass_filter(Bx_raw, fs, lowcut_freq, highcut_freq, filter_order_bandpass)
print("开始滤波 By_raw 信号...")
By_filtered = bandpass_filter(By_raw, fs, lowcut_freq, highcut_freq, filter_order_bandpass)
print("滤波完成。")


## # 3.1 应用陷波滤波器去除工频干扰
Bx_filtered = apply_notch_filter(Bx_filtered, notch_freq, Q_factor_notch, fs)
By_filtered = apply_notch_filter(By_filtered, notch_freq, Q_factor_notch, fs)

# 4.寻找R峰
print("开始在Bx中寻找 R 峰 (整数索引)...")
# 步骤1: 先找到常规的整数索引R峰
integer_R_peaks_Bx = find_r_peaks_data(Bx_filtered, fs, R_peak_min_height_factor, R_peak_min_distance_ms, identifier="Bx信号")
print(f"在 Bx_filtered 中找到 {len(integer_R_peaks_Bx)} 个常规R峰。")
# 步骤2: 对找到的R峰进行插值，以获得亚样本精度
print("对R峰进行抛物线插值以提高精度...")
precise_R_peaks_Bx = interpolate_peaks(Bx_filtered, integer_R_peaks_Bx)
print("插值完成。")

print("开始在By中寻找 R 峰 (整数索引)...")
# 步骤1: 先找到常规的整数索引R峰
integer_R_peaks_By = find_r_peaks_data(By_filtered, fs, R_peak_min_height_factor, R_peak_min_distance_ms, identifier="By信号")
print(f"在 By_filtered 中找到 {len(integer_R_peaks_By)} 个常规R峰。")
# 步骤2: 对找到的R峰进行插值，以获得亚样本精度
print("对R峰进行抛物线插值以提高精度...")
precise_R_peaks_By = interpolate_peaks(By_filtered, integer_R_peaks_By)
print("插值完成。")


### 4.1 标记R峰为红色空心圆圈
R_peaks_Bx_y = Bx_filtered[integer_R_peaks_Bx] if len(integer_R_peaks_Bx) > 0 else np.array([])

# ### 调整Bx，By信号的y轴所在区间
# Bx_raw += 5
# By_raw += 8
# Bx_filtered -= 0.5

# 5. 绘制结果
# 绘制Bx和By原始信号和滤波信号
print("开始绘制原始信号与滤波信号对比图...")
fig1 = plot_signals_with_r_peaks(time, Bx_raw, Bx_filtered, By_raw, By_filtered, integer_R_peaks_Bx, integer_R_peaks_By)
plt.show()
# plt.close(fig1)  # 关闭图形以释放内存

# 6.绘制平均心跳周期（中位数平均和 DTW 对比）
pre_samples = int(pre_r_ms * fs / 1000)
post_samples = int(post_r_ms * fs / 1000)

print("\n开始处理Bx信号的平均心跳周期(使用中位数平均)...")
median_beat_Bx, _ = robust_average_beats_median(
    signal_data=Bx_filtered,
    precise_peak_indices=precise_R_peaks_Bx, # 使用高精度索引
    pre_samples=pre_samples,
    post_samples=post_samples
)
print("\n开始处理By信号的平均心跳周期(使用中位数平均)...")
median_beat_By, _ = robust_average_beats_median(
    signal_data=By_filtered,
    precise_peak_indices=precise_R_peaks_By, # 使用高精度索引
    pre_samples=pre_samples,
    post_samples=post_samples
)

# --- DTW对齐平均 ---
print("\n开始处理Bx信号 (方法四：DTW对齐平均)...")
dtw_beat_Bx = dtw_average_beats(
    signal_data=Bx_filtered,
    precise_peak_indices=precise_R_peaks_Bx,
    pre_samples=pre_samples,
    post_samples=post_samples
)
print("开始处理By信号 (方法四：DTW对齐平均)...")
dtw_beat_By = dtw_average_beats(
    signal_data=By_filtered,
    precise_peak_indices=precise_R_peaks_By,
    pre_samples=pre_samples,
    post_samples=post_samples
)

# --- 修改：绘制最终的对比图 ---
if dtw_beat_Bx.size > 0 and dtw_beat_By.size > 0:
    # 设置中文显示
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 未找到SimHei字体，中文可能无法正常显示。")

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    avg_time_axis = np.linspace(-pre_r_ms, post_r_ms, num=len(dtw_beat_Bx))

    fig2.suptitle('方法三 (中位数) vs 方法四 (DTW) 对比图', fontsize=18)

    # --- 绘制Bx通道对比图 ---
    ax1.plot(avg_time_axis, median_beat_Bx, color='darkorange', linestyle='--', linewidth=2, label='方法三：中位数平均')
    ax1.plot(avg_time_axis, dtw_beat_Bx, color='blue', linewidth=2, label='方法四：DTW对齐平均')
    ax1.set_title(f'{base_filename} - Bx通道')
    ax1.set_ylabel('幅度')
    ax1.grid(True, linestyle='--')
    ax1.axvline(x=0, color='red', linestyle='-.', alpha=0.8) # 标记R峰零点
    ax1.legend()

    # --- 绘制By通道对比图 ---
    ax2.plot(avg_time_axis, median_beat_By, color='darkorange', linestyle='--', linewidth=2, label='方法三：中位数平均')
    ax2.plot(avg_time_axis, dtw_beat_By, color='green', linewidth=2, label='方法四：DTW对齐平均')
    ax2.set_title(f'{base_filename} - By通道')
    ax2.set_xlabel('相对于R峰的时间 (毫秒)')
    ax2.set_ylabel('幅度')
    ax2.grid(True, linestyle='--')
    ax2.axvline(x=0, color='red', linestyle='-.', alpha=0.8) # 标记R峰零点
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
print("\n进程结束！！！")
