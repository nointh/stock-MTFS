from fastapi import FastAPI
from routers import prediction

app = FastAPI(title="Multivariate time series forecasting model serving API")

app.include_router(prediction.router)

@app.get('/')
async def root():
    return {"message": "Welcome to Model Serving API"}