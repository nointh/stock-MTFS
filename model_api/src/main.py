import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv

from routers import prediction, history

load_dotenv(find_dotenv())

MONGO_USERNAME = os.environ.get("MONGO_USERNAME")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD")

mongo_uri = f'mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.7julke9.mongodb.net/?retryWrites=true&w=majority'

app = FastAPI(title="Multivariate time series forecasting model serving API")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router)
# app.include_router(history.router)

@app.on_event("startup")
def start_db_client():
    app.mongo_client = MongoClient(mongo_uri)
    app.database = app.mongo_client['stock_price']

@app.on_event("shutdown")
def shutdown_db_client():
    app.mongo_client.close()

@app.get('/')
async def root():
    return {"message": "Welcome to Model Serving API for Multivariate Time Series Forecasting"}