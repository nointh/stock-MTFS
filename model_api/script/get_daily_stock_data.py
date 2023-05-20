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
# today_time = int(time.mktime(time.strptime("2023-05-1920:59:00", "%Y-%m-%d%H:%M:%S")))
# yesterday_time = today_time - 24*3600
yesterday_time = 0
for ticker in STOCK_LIST:
    collection = database[ticker]
    url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={ticker}&type=stock&resolution=D&from={yesterday_time}&to={today_time}'
    res = requests.get(url)
    data = res.json().get('data')
    record = data[-1]
    time = record.pop('tradingDate')
    if time:
        record['date'] = parser.parse(time)
    print("\n\n------STOCK {} DATA ----------".format(ticker))
    print(record)
    # collection.insert_one(record)

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

VN30_URL = "https://fiin-market.ssi.com.vn/MarketInDepth/GetLatestIndices?language=vi&pageSize=999999&status=1"
payload = {}
indices_res = requests.request("GET", VN30_URL, headers=headers, data=payload)
indices_items = indices_res.json().get('items')
for item in indices_items:
    if item.get('comGroupCode') == 'VN30':
        vn30_data = {
            'open': item.get('openIndex'),
            'close': item.get('closeIndex'),
            'high': item.get('highestIndex'),
            'low': item.get('lowestIndex'),
            'volume': item.get('totalMatchVolume'),
            'date': parser.parse(item.get('tradingDate')),
            'change': item.get('percentIndexChange')*100
        }
        # vn30_collection.insert_one(vn30_data)
        print('\n\n-------VN30 INDEX-----------')
        print(vn30_data)