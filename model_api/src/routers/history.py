import os
from os.path import dirname, abspath, join

from fastapi import APIRouter, Request
from model.stock_model import StockModel
from dateutil import parser
import datetime

router = APIRouter(
    prefix='/history',
    tags=['history']
)

@router.get('')
def get_stock_price_history(request: Request, ticker: str = 'VN30'):
    database = request.app.database
    if ticker not in database.list_collection_names():
        raise HTTPException(status_code=404, detail="Ticker {ticker} not found")
    
    collection = database[ticker]
    data = list(collection.find({}, {'_id': 0}).sort('date'))
    return {
        'ticker': ticker,
        'data': data
    }