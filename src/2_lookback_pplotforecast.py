import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import ast  # 安全解析字符串为list
import os
import numpy as np

demand_data_path =  'inputoutput/enriched_output_copy.xlsx'

df_demand_long = pd.read_excel(demand_data_path) 
df_demand_long = df_demand_long.rename(columns={"SKU_Country": "unique_id", "Month_Year": "ds", "Sales": "y","Temperature": "temperature_ex", "promo_num": "promo_ex"})
    
import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取数据
df = df_demand_long
df['ds'] = pd.to_datetime(df['ds'])
df['month'] = df['ds'].dt.month
# 提取月号和定义 Oct→Sep 的顺序
month_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

# 按 unique_id 和月份计算平均销量
monthly_avg = df.groupby(['unique_id', 'month'])['y'].mean().reset_index()
# monthly_avg = df.groupby(['unique_id', 'month'])['temperature_ex'].mean().reset_index()

# 图保存目录
output_dir = 'inputoutput/DilatedRNN_ex/OctToSep_MonthlyAvgPlots'
os.makedirs(output_dir, exist_ok=True)

#Todo:  give the specific SKU name and the month and only extract its data

#Todo: get the plot


# 获取所有 unique_id
unique_ids = monthly_avg['unique_id'].unique()
y_values_all = np.empty((0, 12))

# 画图
for uid in unique_ids:
    df_uid = monthly_avg[monthly_avg['unique_id'] == uid].copy()

    # 设置 month 为分类变量，并排序
    df_uid['month'] = pd.Categorical(df_uid['month'], categories=month_order, ordered=True)
    df_uid = df_uid.sort_values('month')

    # 准备画图数据
    x_labels = [month_labels[month_order.index(m)] for m in df_uid['month']]
    y_values = df_uid['y'].values
    # y_values = df_uid['temperature_ex'].values

    y_values_all = np.vstack((y_values_all, y_values))
    # 画图（不连接首尾）
    plt.figure(figsize=(8, 4))
    plt.plot(x_labels, y_values, marker='o', linestyle='-', color='darkorange')
    plt.title(f'Monthly Avg sales (Oct–Sep): {uid}')
    plt.xlabel('Month')
    plt.ylabel('Avg Sales (y)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # 保存图
    safe_name = uid.replace(" ", "_").replace("/", "_")
    plt.savefig(f'{output_dir}/{safe_name}_OctToSep_avg.png')
    plt.close()
    print(f"✅ Saved: {output_dir}/{safe_name}_OctToSep_avg.png")

