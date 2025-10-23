TASK1: Run app.py
step 1: python -m uvicorn app:app --host 0.0.0.0 --port 8000 --r
Step 2: open another terminal
export LOCAL_API_TOKEN="MY_SUPER_SECRET_TOKEN"
curl -X POST "http://127.0.0.1:8000/forecast" \
  -H "Authorization: Bearer $LOCAL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "unique_id": "438_Malaysia",
        "horizon": 12,
        "context_size": 12,
        "testing_len": 11,
        "model_name": "DilatedRNN",
        "top_n_spikes": 3,
        "spike_threshold": 0.3,
        "overrides": {}
      }'

for Dify Forwarding: https://electrotechnical-unappalled-ruth.ngrok-free.dev 

TASK2: Run historical_sales 
1. Pick a token value and export it before starting the server
    export LOCAL_API_TOKEN="MY_SUPER_SECRET_TOKEN"
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

    Keep that server window open.

2. In a second terminal (same token)
   export LOCAL_API_TOKEN="MY_SUPER_SECRET_TOKEN"
   curl -X GET "http://127.0.0.1:8000/historical_sales?unique_id=438_Malaysia" \
     -H "Authorization: Bearer $LOCAL_API_TOKEN"
 
TASK3: Run shap_api.py
1. Pick a token value and export it before starting the server
    export LOCAL_API_TOKEN="MY_SUPER_SECRET_TOKEN"
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

2. In a second terminal (same token)
    export LOCAL_API_TOKEN="MY_SUPER_SECRET_TOKEN"
    curl -X POST "http://127.0.0.1:8000/shap_contrib" \
  -H "Authorization: Bearer $LOCAL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "unique_id": "438_Malaysia",
        "horizon": 12,
        "context_size": 12,
        "testing_len": 11,
        "model_name": "DilatedRNN",
        "features": ["temperature_ex", "promo_ex"],
        "ref_strategy": "mean",
        "exact_2_features": true,
        "selected_dates": ["2025-09-30","2025-02-28","2025-07-31"]
      }'
