# app.py
import os
import time
from typing import Optional, Dict, List, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends, Query
from pydantic import BaseModel, Field

from neuralforecast import NeuralForecast

# ======== 你的项目依赖 ========
from utils.utils_artc import (
    get_df_context_df_future_shap
)


# ============ 配置 ============
API_TOKEN = os.environ.get("LOCAL_API_TOKEN", "change-me-please")

# 你的训练工件路径（按你贴的变量名）
HORIZON_DEFAULT = int(os.environ.get("HORIZON_DEFAULT", 12))
CONTEXT_SIZE_DEFAULT = int(os.environ.get("CONTEXT_SIZE_DEFAULT", 12))
TESTING_LEN_DEFAULT = int(os.environ.get("TESTING_LEN_DEFAULT", 11))
LAG_DEFAULT = int(os.environ.get("LAG_DEFAULT", 3))

# 你现有的 identifier 目录
IDENTIFIER = os.environ.get(
    "IDENTIFIER",
    "inputoutput"
)

FILE_PATH_S1 = os.path.join(IDENTIFIER, "df_all_mixing_training_s1.json")
NF_PATH = os.path.join(IDENTIFIER, "nf")

# NeuralForecast 预测后的列名通常是模型名，这里设默认名（可由前端传入以防不一致）
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "DilatedRNN")

# ============ 权限校验 ============
def check_auth(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing/invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# ============ 数据模型 ============
class ForecastRequest(BaseModel):
    unique_id: str = Field(..., description="SKU_Country，例如 '438_Malaysia'")
    horizon: int = Field(HORIZON_DEFAULT, ge=1, le=52, description="预测步长")
    context_size: int = Field(CONTEXT_SIZE_DEFAULT, ge=1, description="上下文窗口长度")
    testing_len: int = Field(TESTING_LEN_DEFAULT, ge=1, description="回测/滚动测试长度")
    model_name: str = Field(DEFAULT_MODEL_NAME, description="NeuralForecast 输出列名（模型名）")
    top_n_spikes: int = Field(3, ge=1, le=20, description="返回 spike 的个数")
    spike_threshold: float = Field(0.3, description="MoM 阈值（0.3 表示 +30%）")
    # 你可以按需增加 overrides 字段（服务水平、促销等），在你的 run 逻辑里解释
    overrides: Dict = Field(default_factory=dict, description="额外参数覆盖（可选）")

class ForecastPoint(BaseModel):
    ds: str
    yhat: float

class SpikePoint(BaseModel):
    ds: str
    value: float
    mom_growth: float

class ForecastResponse(BaseModel):
    unique_id: str
    horizon: int
    model_name: str
    predictions_long: List[ForecastPoint]
    predictions_wide: Dict[str, float]  # t1: yhat1, t2: yhat2, ...
    spikes: List[SpikePoint]
    meta: Dict

# ============ 工具函数 ============
def detect_spikes_mom(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "ds",
    group_col: Optional[str] = "unique_id",
    threshold: float = 0.3,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    def _calc_growth(g):
        g = g.sort_values(date_col)
        g["mom_growth"] = g[value_col].pct_change()
        g["is_spike"] = g["mom_growth"] > threshold
        return g

    if group_col and group_col in df.columns:
        out = df.groupby(group_col, group_keys=False).apply(_calc_growth)
    else:
        out = _calc_growth(df)

    return out.sort_values([group_col, date_col] if group_col else [date_col])

# ============ 应用启动：预加载模型和数据 ============
app = FastAPI(title="CRP12 Local Forecast API", version="1.0.0", docs_url="/docs")

from historical_sales import router as sales_router
app.include_router(sales_router, dependencies=[Depends(check_auth)])

from shap_api import router as shap_router
# Share your Bearer auth with this router:
app.include_router(shap_router, dependencies=[Depends(check_auth)], tags=["shap"])

APP_STATE = {
    "nf": None,
    "testing_monthly": None
}

@app.on_event("startup")
def load_artifacts():
    # 1) 加载 NeuralForecast 模型
    nf = NeuralForecast.load(path=NF_PATH)
    # 2) 加载测试数据（用于提取最后一条，生成 context/futr）
    testing_monthly = pd.read_json(FILE_PATH_S1, orient='records', lines=True)

    APP_STATE["nf"] = nf
    APP_STATE["testing_monthly"] = testing_monthly

# ============ 路由 ============
@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": int(time.time())}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest, _=Depends(check_auth)):
    nf: NeuralForecast = APP_STATE["nf"]
    testing_monthly: pd.DataFrame = APP_STATE["testing_monthly"]

    if nf is None or testing_monthly is None:
        raise HTTPException(status_code=500, detail="Artifacts not loaded")

    # 取该 unique_id 的最后一条，做 context 和 future
    df_last = (
        testing_monthly.groupby("unique_id", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    df_target = df_last[df_last["unique_id"] == req.unique_id].copy()
    if df_target.empty:
        raise HTTPException(status_code=404, detail=f"unique_id not found: {req.unique_id}")

    # 生成上下文和未来时间索引
    df_context, df_future = get_df_context_df_future_shap(
        df_target, req.horizon, req.context_size, testing_len=req.testing_len
    )
    df_context["ds"] = pd.to_datetime(df_context["ds"])
    df_future["ds"] = pd.to_datetime(df_future["ds"])

    # 预测
    pred = nf.predict(df=df_context, futr_df=df_future)

    # 列存在性检查
    if req.model_name not in pred.columns:
        # 尝试找一个最像的列名
        candidate_cols = [c for c in pred.columns if c not in ["unique_id", "ds"]]
        msg = f"Column '{req.model_name}' not in predictions. Available: {candidate_cols}"
        raise HTTPException(status_code=400, detail=msg)

    # 计算 MoM spike
    spikes_df = detect_spikes_mom(
        pred, value_col=req.model_name,
        date_col="ds", group_col="unique_id",
        threshold=req.spike_threshold
    )

    # Top-N spikes（按增幅排序）
    top_spikes = (
        spikes_df.sort_values("mom_growth", ascending=False)
        .head(req.top_n_spikes)[["ds", req.model_name, "mom_growth"]]
        .copy()
    )

    # 组织返回（长表）
    preds_long = [
        {"ds": r["ds"].strftime("%Y-%m-%d"), "yhat": float(r[req.model_name])}
        for _, r in pred.sort_values("ds").iterrows()
    ]

    # 组织返回（宽表）t1..tH，便于 Dify 直接填表
    preds_wide = {f"t{i+1}": float(v["yhat"]) for i, v in enumerate(preds_long)}

    spikes_list = [
        {
            "ds": r["ds"].strftime("%Y-%m-%d"),
            "value": float(r[req.model_name]),
            "mom_growth": float(r["mom_growth"]) if pd.notnull(r["mom_growth"]) else None,
        }
        for _, r in top_spikes.iterrows()
    ]

    return ForecastResponse(
        unique_id=req.unique_id,
        horizon=req.horizon,
        model_name=req.model_name,
        predictions_long=preds_long,
        predictions_wide=preds_wide,
        spikes=spikes_list,
        meta={
            "context_size": req.context_size,
            "testing_len": req.testing_len,
            "spike_threshold": req.spike_threshold,
            "overrides": req.overrides,  # 你可以在内部逻辑中真正使用它
        }
    )
