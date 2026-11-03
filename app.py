import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
ART = BASE / "artifacts"

app = Flask(__name__)

# ---- Muat data & artefak ringan saat startup ----
series = pd.read_csv(ART / "monthly_series.csv", index_col=0, parse_dates=True)["cases"]
test_df = pd.read_csv(ART / "test_predictions.csv", parse_dates=["date"])
scaler = joblib.load(ART / "scaler.joblib")
with open(ART / "meta.json") as f:
    meta = json.load(f)
TIMESTEPS = int(meta["timesteps"])

_model = None
def get_model():
    # TensorFlow dimuat malas (lazy) agar startup server cepat
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(ART / "lstm_dbd.keras")
    return _model


def forecast_future(n_months: int) -> pd.Series:
    model = get_model()
    window = scaler.transform(series.values[-TIMESTEPS:].reshape(-1, 1)).flatten().tolist()
    preds = []
    for _ in range(n_months):
        x = np.array(window[-TIMESTEPS:]).reshape(1, TIMESTEPS, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        preds.append(p)
        window.append(p)
    out = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten().clip(min=0)
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=n_months, freq="MS")
    return pd.Series(out, index=idx)


@app.route("/")
def index():
    return render_template("index.html", meta=meta)


@app.route("/api/data")
def api_data():
    return jsonify({
        "history": {"dates": [d.strftime("%Y-%m") for d in series.index],
                    "values": series.round(0).tolist()},
        "test": {"dates": test_df["date"].dt.strftime("%Y-%m").tolist(),
                 "actual": test_df["actual"].round(0).tolist(),
                 "predicted": test_df["predicted"].round(0).tolist()},
        "meta": meta,
    })


@app.route("/api/forecast")
def api_forecast():
    try:
        n = int(request.args.get("n", 6))
    except ValueError:
        n = 6
    n = max(1, min(n, 12))
    f = forecast_future(n)
    return jsonify({"n": n,
                    "dates": [d.strftime("%Y-%m") for d in f.index],
                    "values": f.round(0).tolist()})


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
