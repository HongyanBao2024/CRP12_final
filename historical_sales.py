# historical_sales.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, HTTPException


router = APIRouter()

@router.get("/historical_sales")
def historical_sales(unique_id: str):
    """Return average monthly sales (Oct–Sep) and plot for the given SKU."""
    demand_data_path = "inputoutput/enriched_output.xlsx"
    if not os.path.exists(demand_data_path):
        raise HTTPException(status_code=404, detail="Data file not found")

    df = pd.read_excel(demand_data_path)
    df = df.rename(columns={
        "SKU_Country": "unique_id",
        "Month_Year": "ds",
        "Sales": "y",
        "Temperature": "temperature_ex",
        "promo_num": "promo_ex"
    })

    if unique_id not in df["unique_id"].unique():
        raise HTTPException(status_code=404, detail=f"unique_id not found: {unique_id}")

    df["ds"] = pd.to_datetime(df["ds"])
    df["month"] = df["ds"].dt.month
    month_order = [10,11,12,1,2,3,4,5,6,7,8,9]
    month_labels = ['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']

    monthly_avg = df.groupby(["unique_id","month"])["y"].mean().reset_index()
    df_uid = monthly_avg[monthly_avg["unique_id"] == unique_id].copy()
    df_uid["month"] = pd.Categorical(df_uid["month"], categories=month_order, ordered=True)
    df_uid = df_uid.sort_values("month")

    x_labels = [month_labels[month_order.index(m)] for m in df_uid["month"]]
    y_values = df_uid["y"].values

    os.makedirs("static", exist_ok=True)
    fig_path = f"static/historical_sales_{unique_id}.png"

    plt.figure(figsize=(8,4))
    plt.plot(x_labels, y_values, marker="o", color="darkorange")
    plt.title(f"Monthly Avg Sales (Oct–Sep): {unique_id}")
    plt.xlabel("Month")
    plt.ylabel("Avg Sales (y)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return {
        "unique_id": unique_id,
        "months": x_labels,
        "avg_sales": y_values.tolist(),
        "plot_url": f"/{fig_path}"
    }
