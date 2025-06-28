""" 用来分析 main_batch.py 生成的统计结果 """

""" 
平均心率(hr),
R-R间期的标准差(sdnn),
相邻R-R间期差值的均方根(rmssd),
QRS波幅(qrs_amplitude),
QRS波宽(qrs_width_ms),
 """

# analyze_results.py (最终版)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import sys
import re

# 导入我们的配置文件以获取路径
import config

def map_day_to_stage(day: int) -> str:
    """根据日龄划分发育阶段。"""
    if 4 <= day <= 7:
        return 'Early (D4-7)'
    elif 8 <= day <= 14:
        return 'Mid (D8-14)'
    elif 15 <= day <= 21:
        return 'Late (D15-21)'
    else:
        return 'Unknown'

def perform_stage_analysis(df: pd.DataFrame, parameter: str):
    """对单个参数在不同发育阶段之间执行ANOVA和Tukey's HSD检验。"""
    print(f"\n{'='*25} 分析参数: {parameter} {'='*25}")
    
    stages = ['Early (D4-7)', 'Mid (D8-14)', 'Late (D15-21)']
    grouped_data = [df[df['stage'] == stage][parameter].dropna() for stage in stages]
    
    if any(len(data) < 2 for data in grouped_data):
        print(f"  警告: '{parameter}' 的数据不足以进行三组比较，跳过统计检验。")
        return

    # 执行ANOVA检验
    f_stat, p_value_anova = stats.f_oneway(*grouped_data)
    print(f"  单因素方差分析 (ANOVA): p-value = {p_value_anova:.4f}")

    # 如果ANOVA结果显著，则执行Tukey's HSD事后检验
    if p_value_anova < 0.05:
        print("  ANOVA 结果显著，执行 Tukey's HSD 事后检验:")
        all_data = pd.concat(grouped_data)
        group_labels = [stage for stage, data in zip(stages, grouped_data) for _ in range(len(data))]
        
        tukey_result = pairwise_tukeyhsd(endog=all_data, groups=group_labels, alpha=0.05)
        print(tukey_result)
    else:
        print("  ANOVA 结果不显著，无需进行事后检验。")

def plot_stage_comparison(df: pd.DataFrame, parameter: str):
    """为单个参数绘制跨发育阶段的箱形图 (Box Plot)。"""
    plt.figure(figsize=(10, 8))
    
    sns.boxplot(data=df, x='stage', y=parameter, 
                order=['Early (D4-7)', 'Mid (D8-14)', 'Late (D15-21)'])
    
    plt.title(f'{parameter} 在不同发育阶段的分布', fontsize=16)
    plt.xlabel('发育阶段', fontsize=12)
    plt.ylabel(parameter, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = config.RESULTS_OUTPUT_DIR / f'analysis_{parameter}_by_stage.png'
    plt.savefig(output_path, dpi=150)
    print(f"  图表已保存至: {output_path.name}")
    plt.show()

def main():
    """主函数，加载结果，执行统计分析和绘图。"""
    print("--- 开始分析最终统计结果 ---")
    
    # 1. 加载由 main_batch.py 生成的原始结果CSV文件
    results_csv_path = config.RESULTS_OUTPUT_DIR / 'chick_embryo_mcg_analysis_results.csv'
    if not results_csv_path.exists():
        print(f"错误: 结果文件 '{results_csv_path}' 不存在。")
        sys.exit(1)
        
    df_raw = pd.read_csv(results_csv_path)
    print(f"成功加载 {len(df_raw)} 条原始测量记录。")

    # --- 【关键预处理步骤】 ---
    # 2. 计算每个鸡胚在每一天的参数平均值
    # 从文件名中提取唯一的胚胎ID (例如 'egg_d10_B13_t1' -> 'egg_d10_B13')
    df_raw['unique_embryo_id'] = df_raw['embryo_id'].str.replace(r'_t\d+$', '', regex=True)
    
    # 按天和唯一ID分组，计算所有参数的平均值
    daily_avg_df = df_raw.groupby(['day', 'unique_embryo_id']).mean(numeric_only=True).reset_index()
    print(f"预处理完成，得到 {len(daily_avg_df)} 条日均值记录。")
    # --- 预处理结束 ---

    # 3. 在日均值数据上进行后续分析：添加“阶段”列
    daily_avg_df['stage'] = daily_avg_df['day'].apply(map_day_to_stage)
    daily_avg_df = daily_avg_df[daily_avg_df['stage'] != 'Unknown']

    # 4. 描述性统计：按阶段计算均值和标准差
    print("\n--- 各阶段参数描述性统计 (Mean ± Std) ---")
    # 首先按阶段分组，然后计算每个参数的均值和标准差
    summary_stats = daily_avg_df.groupby('stage').agg(
        mean_hr_mean=('mean_hr', 'mean'), mean_hr_std=('mean_hr', 'std'),
        sdnn_mean=('sdnn', 'mean'), sdnn_std=('sdnn', 'std'),
        rmssd_mean=('rmssd', 'mean'), rmssd_std=('rmssd', 'std'),
        qrs_amplitude_mean=('qrs_amplitude', 'mean'), qrs_amplitude_std=('qrs_amplitude', 'std'),
        qrs_width_ms_mean=('qrs_width_ms', 'mean'), qrs_width_ms_std=('qrs_width_ms', 'std')
    ).round(2)
    print(summary_stats)
    
    # 5. 循环执行统计检验和绘图
    params_to_analyze = ['mean_hr', 'sdnn', 'rmssd', 'qrs_amplitude', 'qrs_width_ms']
    
    for param in params_to_analyze:
        # 在日均值数据上进行分析和绘图
        perform_stage_analysis(daily_avg_df, param)
        plot_stage_comparison(daily_avg_df, param)
        
    print("\n--- 分析流程结束 ---")

if __name__ == '__main__':
    # 设置matplotlib以支持中文
    try:
        plt.rcParams['font.sans-serif'] = ['Source Han Sans SC']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法设置'PingFang SC'字体，图表中的中文可能无法正常显示。")
    
    # 确保已安装依赖
    try:
        import pandas
        import seaborn
        import statsmodels
    except ImportError as e:
        print(f"错误: 缺少必要的库 -> {e}")
        print("请在您的(venv)虚拟环境中运行: pip install pandas seaborn statsmodels")
        sys.exit(1)
        
    main()