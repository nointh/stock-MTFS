from dotenv import load_dotenv, find_dotenv
import os
from pymongo import MongoClient
import requests
from dateutil import parser
import datetime
import time
load_dotenv(find_dotenv())

MONGO_USERNAME = os.environ.get("MONGO_USERNAME")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD")
print(MONGO_USERNAME)
print(MONGO_PASSWORD)
mongo_uri = f'mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.7julke9.mongodb.net/?retryWrites=true&w=majority'


mongo_client = MongoClient(mongo_uri)
database = mongo_client['stock_price']

vn30_collection = database['VN30']
print(vn30_collection)

STOCK_LIST = ["BID","BVH","CTG","FPT","GAS","HPG","KDH","MBB","MSN","MWG","NVL","PDR","PNJ","REE","SBT","SSI","STB","TCH","VCB","VIC","VNM"]
print(len(STOCK_LIST))
# for data in vn30_collection.find():
#     print(data)
today_time = int(time.time())
for ticker in STOCK_LIST:
    collection = database[ticker]
    url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={ticker}&type=stock&resolution=D&from=946659600&to={today_time}'
    res = requests.get(url)
    data = res.json().get('data')
    for record in data:
        time = record.pop('tradingDate')
        if time:
            record['date'] = parser.parse(time)
    collection.insert_many(data)

headers = {
        'Connection': 'keep-alive',
        'sec-ch-ua': '"Not A;Brand";v="99", "Chromium";v="98", "Google Chrome";v="98"',
        'DNT': '1',
        'sec-ch-ua-mobile': '?0',
        'X-Fiin-Key': 'KEY',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Fiin-User-ID': 'ID',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
        'X-Fiin-Seed': 'SEED',
        'sec-ch-ua-platform': 'Windows',
        'Origin': 'https://iboard.ssi.com.vn',
        'Sec-Fetch-Site': 'same-site',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': 'https://iboard.ssi.com.vn/',
        'Accept-Language': 'en-US,en;q=0.9,vi-VN;q=0.8,vi;q=0.7'
        }

VN30_URL = "https://fiin-market.ssi.com.vn/MarketInDepth/GetIndexSeries?language=vi&ComGroupCode=VN30&TimeRange=TenYears&id=1"
payload = {}
vn30_res = requests.request("GET", VN30_URL, headers=headers, data=payload)
vn30_data = vn30_res.json().get('items')
vn30_data = list(map(lambda record: {
    'open': record.get('openIndex'),
    'close': record.get('closeIndex'),
    'high': record.get('highestIndex'),
    'low': record.get('lowestIndex'),
    'volume': record.get('matchVolume'),
    'date': parser.parse(record.get('tradingDate')),
    'change': record.get('percentIndexChange')*100
}, vn30_data))
vn30_collection.insert_many(vn30_data)