from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sklearn.metrics import f1_score
import numpy as np

app = FastAPI()

@app.get("/api/predictions")
def predictions():
    return JSONResponse({"y_true": [0,1,1,0], "y_pred": [0,1,0,0]})

@app.get("/api/metrics")
def metrics():
    return JSONResponse({
        "epochs": [1,2,3,4],
        "test_acc": [0.8,0.85,0.9,0.92],
        "test_loss": [0.5,0.4,0.3,0.2],
        "val_acc": [0.75,0.8,0.85,0.88],
        "val_loss": [0.55,0.45,0.35,0.25]
    })

@app.get("/api/polarity")
def polarity():
    # Generate 30 days of polarity data
    from datetime import datetime, timedelta

    base = datetime(2024, 5, 1, 12, 0, 0)
    times = [(base + timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(30)]
    polarity = np.random.uniform(-1, 1, 30).round(2).tolist()
    return JSONResponse({
        "times": times,
        "polarity": polarity
    })

@app.get("/api/f1_score")
def f1_score_api():
    # Simulate F1 scores for 10 epochs
    epochs = list(range(1, 11))
    # Example: random F1 scores between 0.6 and 0.95
    f1_scores = np.random.uniform(0.6, 0.95, 10).round(2).tolist()
    return JSONResponse({
        "epochs": epochs,
        "f1_score": f1_scores
    })

# Use this in your analytics_section or wherever you fetch API data
API_URL = "http://localhost:8000/api"

# To run the server, use the command: uvicorn api_server:app --reload