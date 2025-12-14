import time
import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any

from app.metrics import instrument_app
from app.logging_conf import get_logger

MODEL_PATH = "artifacts/model.joblib"

log = get_logger()

app = FastAPI(title="Heart Disease Predictor", version="1.0.0")
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.req_id = req_id
    start = time.time()
    resp = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    resp.headers["x-request-id"] = req_id
    log.info("request_done", extra={"req_id": req_id, "path": request.url.path, "ms": round(duration_ms, 2)})
    return resp

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictRequest, request: Request):
    req_id = request.state.req_id
    df = pd.DataFrame(payload.rows)

    start = time.time()
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    ms = (time.time() - start) * 1000

    # per-sample logging (keep it lightweight; don’t log PHI in real systems)
    for i, (p, pr) in enumerate(zip(pred.tolist(), proba.tolist())):
        log.info("pred", extra={"req_id": req_id, "row": i, "pred": p, "proba": round(pr, 6)})

    return {
        "request_id": req_id,
        "latency_ms": round(ms, 2),
        "predictions": [{"pred": int(p), "proba": float(pr)} for p, pr in zip(pred, proba)]
    }

instrument_app(app)
