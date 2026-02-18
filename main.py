from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = joblib.load("churn_model.pkl")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    tenure: int = Form(...),
    monthlycharges: float = Form(...),
    totalcharges: float = Form(...),
    contract: str = Form(...),
    paymentmethod: str = Form(...),
    is_new_customer: int = Form(...),
    avg_monthly_spend: float = Form(...),
    long_contract: int = Form(...),
    monthly_to_total_ratio: float = Form(...)
):
    data = {
        "tenure": tenure,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,
        "contract": contract,
        "paymentmethod": paymentmethod,
        "is_new_customer": is_new_customer,
        "avg_monthly_spend": avg_monthly_spend,
        "long_contract": long_contract,
        "monthly_to_total_ratio": monthly_to_total_ratio
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": int(prediction),
            "probability": round(float(probability), 3)
        }
    )
