from pydantic import BaseModel
import datetime
class PriceModel(BaseModel):
    date: datetime.date
    open: float
    close: float
    high: float
    low: float


class StockModel(BaseModel):
    ticker: str
    
    data: list[PriceModel] = []
