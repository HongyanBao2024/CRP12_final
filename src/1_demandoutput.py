import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import ast  # 安全解析字符串为list
import os
import numpy as np
# 读取CSV文件
df = pd.read_csv('inputoutput/output_demand.csv')  # 替换为你的文件路径
# df = df[['unique_id', 'data_range', 'data_range_ex', 'temperature_ex', 'promo_ex', 'context', 'sales_N', 'DeepNPTS_ex', 'DilatedRNN_ex', 'uni2ts_ex']]
df = df[['unique_id', 'data_range', 'data_range_ex', 'temperature_ex', 'promo_ex', 'context', 'sales_N', 'DilatedRNN_ex']]

model_name = ['DeepNPTS_ex','DilatedRNN_ex','uni2ts_ex']

output_dir = 'inputoutput/DilatedRNN_ex/DilatedRNN_ex_Plots'
os.makedirs(output_dir, exist_ok=True)


# 遍历每一行（每个 unique_id）
for idx, row in df.iterrows():
    unique_id = row['unique_id']
    try:
        date_list = ast.literal_eval(row['data_range_ex'])
        deepnpts = ast.literal_eval(row['DilatedRNN_ex'])
    except Exception as e:
        print(f"❌ Error parsing row for {unique_id}: {e}")
        continue

    # 检查数据长度是否足够
    if len(date_list) < 24 or len(deepnpts) < 12:
        print(f"⚠️ Skipping {unique_id} due to insufficient data.")
        continue

    # 解析未来日期和对应预测值
    future_dates = [datetime.fromisoformat(d.replace("T00:00:00.000", "")) for d in date_list[12:24]]
    future_values = deepnpts[:12]
    month_labels = [dt.strftime('%b') for dt in future_dates]

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(month_labels, future_values, marker='o', linestyle='-', color='teal')
    plt.title(f'ARTC Forecast: {unique_id}')
    plt.xlabel('Date')
    plt.ylabel('Forecasted Sales')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图像
    safe_id = unique_id.replace(" ", "_").replace("/", "_")  # 避免非法文件名
    output_path = os.path.join(output_dir, f'{safe_id}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot for {unique_id} → {output_path}")




