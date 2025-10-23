# shap_api.py
import os
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from neuralforecast import NeuralForecast
from utils.utils_artc import get_df_context_df_future_shap

router = APIRouter()

# -------------------- Config via env (defaults) --------------------
HORIZON_DEFAULT = int(os.environ.get("HORIZON_DEFAULT", 12))
CONTEXT_SIZE_DEFAULT = int(os.environ.get("CONTEXT_SIZE_DEFAULT", 12))
TESTING_LEN_DEFAULT = int(os.environ.get("TESTING_LEN_DEFAULT", 11))
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "DilatedRNN")

IDENTIFIER = os.environ.get("IDENTIFIER", "inputoutput")
FILE_PATH_S1 = os.path.join(IDENTIFIER, "df_all_mixing_training_s1.json")
NF_PATH = os.path.join(IDENTIFIER, "nf")

# -------------------- Lazy artifacts --------------------
_ARTIFACTS = {"nf": None, "testing_monthly": None}

def _load_artifacts_once():
    if _ARTIFACTS["nf"] is None or _ARTIFACTS["testing_monthly"] is None:
        nf = NeuralForecast.load(path=NF_PATH)
        testing_monthly = pd.read_json(FILE_PATH_S1, orient="records", lines=True)
        _ARTIFACTS["nf"] = nf
        _ARTIFACTS["testing_monthly"] = testing_monthly
    return _ARTIFACTS["nf"], _ARTIFACTS["testing_monthly"]

# -------------------- Schemas --------------------
class ShapRequest(BaseModel):
    unique_id: str = Field(..., description="SKU_Country, e.g. '438_Malaysia'")
    horizon: int = Field(HORIZON_DEFAULT, ge=1, le=52)
    context_size: int = Field(CONTEXT_SIZE_DEFAULT, ge=1)
    testing_len: int = Field(TESTING_LEN_DEFAULT, ge=1)
    model_name: str = Field(DEFAULT_MODEL_NAME)
    features: List[str] = Field(default_factory=lambda: ["temperature_ex", "promo_ex"])
    ref_strategy: str = Field("mean", description="Currently supports 'mean'")
    exact_2_features: bool = Field(True)
    topk: int = Field(3, ge=1, le=20, description="Top-K spikes to return")
    selected_dates: Optional[List[str]] = None

class ShapPerDate(BaseModel):
    ds: str
    pred: float
    base: float
    contributions: Dict[str, float]

class TopSpike(BaseModel):
    ds: str
    yhat: float
    mom_growth: Optional[float] = None
    base: float
    contributions: Dict[str, float]

class ShapResponse(BaseModel):
    unique_id: str
    model_name: str
    horizon: int
    features: List[str]
    per_date: List[ShapPerDate]                 # 全部未来日期
    arrays: Dict[str, List[Union[float, str]]]  # ds 为 str，其它为 float
    top_spikes: List[TopSpike]                  # Top-K MoM spikes 的详细贡献
    meta: Dict

# -------------------- Helpers --------------------
def _modify_future(df_future: pd.DataFrame, active_features: List[str], ref_vals: Dict[str, float]) -> pd.DataFrame:
    df_mod = df_future.copy()
    for f, ref_v in ref_vals.items():
        if f not in active_features:
            df_mod[f] = ref_v
    return df_mod

def _ref_values(df_future: pd.DataFrame, features: List[str], strategy: str) -> Dict[str, float]:
    if strategy != "mean":
        raise HTTPException(status_code=400, detail=f"Unsupported ref_strategy '{strategy}'")
    # 确保缺失的特征列存在（用 0 填）
    for f in features:
        if f not in df_future.columns:
            df_future[f] = 0.0
    refs = {f: float(df_future[f].mean()) for f in features}
    return refs

def _detect_spikes_mom(df: pd.DataFrame, value_col: str, date_col: str = "ds", group_col: Optional[str] = "unique_id",
                       threshold: float = 0.3) -> pd.DataFrame:
    g = df.copy()
    g[date_col] = pd.to_datetime(g[date_col])
    def _calc(x):
        x = x.sort_values(date_col)
        x["mom_growth"] = x[value_col].pct_change()
        x["is_spike"] = x["mom_growth"] > threshold
        return x
    if group_col and group_col in g.columns:
        out = g.groupby(group_col, group_keys=False).apply(_calc)
    else:
        out = _calc(g)
    return out.sort_values([group_col, date_col] if group_col else [date_col])

# -------------------- Core endpoint --------------------
@router.post("/shap_contrib", response_model=ShapResponse)
def shap_contrib(req: ShapRequest):
    nf, testing_monthly = _load_artifacts_once()

    # 取目标 SKU 的最后一条记录生成 context/future
    df_last = testing_monthly.groupby("unique_id", sort=False).tail(1).reset_index(drop=True)
    df_target = df_last[df_last["unique_id"] == req.unique_id].copy()
    if df_target.empty:
        raise HTTPException(status_code=404, detail=f"unique_id not found: {req.unique_id}")

    df_context, df_future = get_df_context_df_future_shap(
        df_target, req.horizon, req.context_size, testing_len=req.testing_len
    )
    df_context["ds"] = pd.to_datetime(df_context["ds"])
    df_future["ds"] = pd.to_datetime(df_future["ds"])

    # 参考值 + 主预测
    ref_vals = _ref_values(df_future, req.features, req.ref_strategy)
    pred_df = nf.predict(df=df_context, futr_df=df_future)

    if req.model_name not in pred_df.columns:
        avail = [c for c in pred_df.columns if c not in ["unique_id", "ds"]]
        if not avail:
            raise HTTPException(status_code=400, detail="No model prediction columns found.")
        # 兜底：自动使用第一列
        req.model_name = avail[0]

    # 基线（全部 mask 到 ref）
    fut_base = df_future.copy()
    for f, v in ref_vals.items():
        fut_base[f] = v
    base_df = nf.predict(df=df_context, futr_df=fut_base)

    # ===== 两特征 Shapley（1/2 权重）或 多特征近似 =====
    if req.exact_2_features and len(req.features) == 2:
        f1, f2 = req.features

        def _pred_for(active: List[str]) -> np.ndarray:
            fut = _modify_future(df_future, active_features=active, ref_vals=ref_vals)
            arr = nf.predict(df=df_context, futr_df=fut)[req.model_name].astype(float).values
            return arr

        y_base = base_df[req.model_name].astype(float).values    # S = {}
        y_f1   = _pred_for([f1])                                 # S = {f1}
        y_f2   = _pred_for([f2])                                 # S = {f2}
        y_both = pred_df[req.model_name].astype(float).values    # S = {f1,f2}

        phi_f1 = 0.5 * ((y_f1 - y_base) + (y_both - y_f2))
        phi_f2 = 0.5 * ((y_f2 - y_base) + (y_both - y_f1))
        contrib_arrays = {f1: phi_f1, f2: phi_f2}
    else:
        # 近似：φ_i ≈ pred(S={i}) - base
        contrib_arrays = {}
        for f in req.features:
            fut_only_f = _modify_future(df_future, active_features=[f], ref_vals=ref_vals)
            y_only_f = nf.predict(df=df_context, futr_df=fut_only_f)[req.model_name].astype(float).values
            contrib_arrays[f] = y_only_f - base_df[req.model_name].astype(float).values

    # ===== 组织返回的序列 =====
    ds_vals   = pred_df["ds"].dt.strftime("%Y-%m-%d").tolist()
    yhat_vals = pred_df[req.model_name].astype(float).round(4).tolist()
    base_vals = base_df[req.model_name].astype(float).round(4).tolist()

    arrays: Dict[str, List[Union[float, str]]] = {
        "ds": ds_vals,
        "pred": yhat_vals,
        "base": base_vals,
    }
    for f in req.features:
        arrays[f"{f}_shap"] = np.round(contrib_arrays[f], 4).tolist()

    # ===== per_date（全部未来日期） =====
    per_date: List[ShapPerDate] = []
    for i, ds in enumerate(ds_vals):
        per_date.append(
            ShapPerDate(
                ds=ds,
                pred=float(yhat_vals[i]),
                base=float(base_vals[i]),
                contributions={f: float(np.round(contrib_arrays[f][i], 4)) for f in req.features},
            )
        )

    # ===== 计算 MoM spike 并选 Top-K =====
    tmp = pred_df[["unique_id", "ds", req.model_name]].copy()
    spikes = _detect_spikes_mom(tmp, value_col=req.model_name, date_col="ds", group_col="unique_id", threshold=0.3)
    # 仅本 unique_id
    spikes = spikes[spikes["unique_id"] == req.unique_id]
    spikes_sorted = spikes.sort_values("mom_growth", ascending=False)
    spikes_topk = spikes_sorted.head(req.topk)

    # 从 top-k 日期抽取贡献（对齐 ds 索引）
    ds_to_idx = {ds: idx for idx, ds in enumerate(ds_vals)}
    top_spikes: List[TopSpike] = []
    for _, r in spikes_topk.iterrows():
        ds_str = pd.to_datetime(r["ds"]).strftime("%Y-%m-%d")
        idx = ds_to_idx.get(ds_str, None)
        if idx is None:
            continue
        top_spikes.append(
            TopSpike(
                ds=ds_str,
                yhat=float(yhat_vals[idx]),
                mom_growth=float(r["mom_growth"]) if pd.notnull(r["mom_growth"]) else None,
                base=float(base_vals[idx]),
                contributions={f: float(np.round(contrib_arrays[f][idx], 4)) for f in req.features},
            )
        )

    # 如有 selected_dates，则把 per_date 过滤为仅这些日期（不影响 top_spikes）
    if req.selected_dates:
        sel_set = set(req.selected_dates)
        per_date = [row for row in per_date if row.ds in sel_set]

    return ShapResponse(
        unique_id=req.unique_id,
        model_name=req.model_name,
        horizon=req.horizon,
        features=req.features,
        per_date=per_date,
        arrays=arrays,
        top_spikes=top_spikes,
        meta={
            "context_size": req.context_size,
            "testing_len": req.testing_len,
            "ref_strategy": req.ref_strategy,
            "exact_2_features": req.exact_2_features,
            "topk": req.topk,
            "notes": "Top spikes are computed via MoM pct_change on the predicted series."
        }
    )
