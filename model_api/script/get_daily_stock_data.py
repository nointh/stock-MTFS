from dotenv import load_dotenv, find_dotenv
import os
from pymongo import MongoClient
import requests
from dateutil import parser
from pymongo import DESCENDING
import time

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Get MongoDB username and password from environment variables
MONGO_USERNAME = os.environ.get("MONGO_USERNAME")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD")
print("MONGO_USERNAME= ",MONGO_USERNAME)
print("MONGO_PASSWORD= ",MONGO_PASSWORD)

# Construct the MongoDB Atlas connection URI using the username and password
mongo_uri = f'mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.7julke9.mongodb.net/?retryWrites=true&w=majority'

# Connect to MongoDB Atlas using the pymongo library
mongo_client = MongoClient(mongo_uri)

# Select the 'stock_price' database and the 'VN30' collection
database = mongo_client['stock_price']   
vn30_collection = database['VN30']

def insert_record(collection, record):
    # Query the collection for a record with the same date
    existing_record = collection.find_one({"date": record["date"]})

    # Check if an existing record with the same date exists
    if existing_record:
        print("Record already exists in collection")
    else:
        # Insert the record into the collection
        collection.insert_one(record)
        print("Record inserted successfully")

# Define a list of stock tickers to retrieve data for
STOCK_LIST = ["BID","BVH","CTG","FPT","GAS","HPG","KDH","MBB","MSN","MWG","NVL","PDR","PNJ","REE","SBT","SSI","STB","TCH","VCB","VIC","VNM"]

# Get the current time in seconds since the epoch
today_time = int(time.time())
yesterday_time = 0

# Loop over each stock ticker in the list
for ticker in STOCK_LIST:
    # Select the collection for this ticker
    collection = database[ticker]
    
    # Construct the API URL to retrieve data for this ticker
    url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={ticker}&type=stock&resolution=D&from={yesterday_time}&to={today_time}'
    
    # Send a GET request to the API and parse the JSON response
    res = requests.get(url)
    data = res.json().get('data')
    
    # Get the last record from the data and remove its 'tradingDate' property
    record = data[-1]
    time = record.pop('tradingDate')
    
    # If a time was found, parse it and add it as a 'date' property to the record
    if time:
        record['date'] = parser.parse(time)
        print("\n\n------STOCK {} DATA ----------".format(ticker))
        print(record)
        # Call our insert_record function to insert this record into the collection if it doesn't already exist
        insert_record(collection, record)

# Define headers for an API request to retrieve VN30 index data
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

# Define the API URL to retrieve VN30 index data
VN30_URL = "https://fiin-market.ssi.com.vn/MarketInDepth/GetLatestIndices?language=vi&pageSize=999999&status=1"
payload = {}

# Send a GET request to the API with our headers and parse the JSON response
indices_res = requests.request("GET", VN30_URL, headers=headers, data=payload)
indices_items = indices_res.json().get('items')

print("\n\n------VN30 DATA ----------")
# Loop over each item in the response data
for item in indices_items:
    # Check if this item has a comGroupCode of VN30
    if item.get('comGroupCode') == 'VN30':
        # Construct a dictionary containing VN30 index data from this item
        vn30_data = {
            'open': item.get('openIndex'),
            'close': item.get('closeIndex'),
            'high': item.get('highestIndex'),
            'low': item.get('lowestIndex'),
            'volume': item.get('totalMatchVolume'),
            'date': parser.parse(item.get('tradingDate')),
            'change': item.get('percentIndexChange')*100
        }
        
        # Check if data for this date already exists in vn30_collection
        existing_data = vn30_collection.find_one({'date': vn30_data['date']})
        if existing_data:
            print(existing_data)
            print("Record already exists in collection")
        else:
            # Insert this VN30 index data into vn30_collection
            print(vn30_data)
            vn30_collection.insert_one(vn30_data)
            print("Record inserted successfully")
        
