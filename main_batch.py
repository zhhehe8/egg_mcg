""" 批量处理主文件 """

from pathlib import Path
import pandas as pd
import sys
import numpy as np

# 将重构的模块和配置导入
import config
import mcg_processing as mcg

def batch_process_data():
    """
    批量处理所有天数文件夹下的所有数据文件。
    """
    all_results = []
    
    if not config.DATA_ROOT_DIR.is_dir():
        print(f"错误: 数据根目录 '{config.DATA_ROOT_DIR}' 不存在或不是一个文件夹。")
        sys.exit(1)
        
    # 在循环开始前，预先创建好波形保存目录
    if config.SAVING_PARAMS['save_averaged_waveform']:
        config.WAVEFORM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    day_folders = sorted([d for d in config.DATA_ROOT_DIR.iterdir() if d.is_dir()])
    
    for day_folder in day_folders:
        print(f"\n=========================================")
        print(f"进入文件夹: {day_folder.name}")
        print(f"=========================================")
        
        file_paths = sorted(day_folder.glob('*.txt'))
        if not file_paths:
            print("  此文件夹下未找到 .txt 文件。")
            continue

        for filepath in file_paths:
            # 【修改】将所有config都传入，简化了调用
            result_dict, averaged_waveform = mcg.process_single_file(filepath, {
                'PROCESSING_PARAMS': config.PROCESSING_PARAMS,
                'FILTER_PARAMS': config.FILTER_PARAMS,
                'R_PEAK_PARAMS': config.R_PEAK_PARAMS,
                'AVERAGING_PARAMS': config.AVERAGING_PARAMS
                # 注意：我们不再需要传入QUALITY_CONTROL_PARAMS，因为它已合并
            })
            
            if result_dict:
                all_results.append(result_dict)
                
                if config.SAVING_PARAMS['save_averaged_waveform'] and averaged_waveform is not None:
                    time_axis_ms = np.linspace(-config.AVERAGING_PARAMS['pre_r_ms'], config.AVERAGING_PARAMS['post_r_ms'], len(averaged_waveform))
                    waveform_data_to_save = np.vstack((time_axis_ms, averaged_waveform)).T
                    
                   
                    avg_method_name = result_dict['avg_method_used']
                    waveform_filename = f"{filepath.stem}_avg_{avg_method_name}.csv"
                    waveform_output_path = config.WAVEFORM_OUTPUT_DIR / waveform_filename
                    
                    np.savetxt(waveform_output_path, waveform_data_to_save, delimiter=',', header='Time_ms,Amplitude', fmt='%.4f', comments='')
                    print(f"  成功: 平均波形已保存至 -> {waveform_filename}")

    if not all_results:
        print("\n处理完成，但未能成功提取任何数据。")
        return

    print("\n--- 所有文件处理完毕，正在汇总结果... ---")
    results_df = pd.DataFrame(all_results)
    config.RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv_path = config.RESULTS_OUTPUT_DIR / 'chick_embryo_mcg_analysis_results.csv'
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 成功！所有参数汇总结果已保存至: {output_csv_path}")


if __name__ == '__main__':
    try:
        import pandas
    except ImportError:
        print("错误: 需要安装 pandas 库。")
        sys.exit(1)
        
    batch_process_data()