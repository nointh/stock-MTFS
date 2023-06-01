from pydantic import BaseModel
from typing import List

import datetime
class PriceModel(BaseModel):
    date: datetime.date
    open: float
    close: float
    high: float
    low: float


class StockModel(BaseModel):
    ticker: str
    
    data: List[PriceModel] = []

