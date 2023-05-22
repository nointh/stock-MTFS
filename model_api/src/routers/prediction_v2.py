import os
from os.path import dirname, abspath, join
from fastapi import Request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from common.constant import WORKING_DIR
from common.prepare_data import get_multistock_data, get_vn30_data
from common.time_features import time_features
from common.generate_predict import (
    generate_long_term_predict,
    generate_multistock_lstnet_predict,
    generate_multistock_xgboost_predict,
    generate_multistock_random_forest_predict,
    generate_multistock_mtgnn_predict,
    generate_multistock_var_predict,
    generate_multistock_lstm_predict,
)
from pymongo import MongoClient

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
collection = database['prediction']

def insert_data_into_database(predict_result, algorithm):
    # Insert data into the collection
    for result in predict_result:
        # Check if a record with the same date and algorithm already exists
        existing_record = collection.find_one({"date": result["date"], "algorithm": algorithm})
        if not existing_record:
            # Insert the new record
            result["algorithm"] = algorithm
            collection.insert_one(result)

def get_longterm_forecasting(request: Request, pred_len: int=50):
    data, timestamp = get_vn30_data(request.app.database, seq_len=50, is_close_last=True)
    predict_result = generate_long_term_predict(data, timestamp, pred_len)
    
    # Insert data into the database
    insert_data_into_database(predict_result, "longterm")

def get_lstnet_multistock_prediction(request: Request, pred_len: int=50):
    data, timestamp = get_multistock_data(request.app.database, seq_len=100)
    predict_result = generate_multistock_lstnet_predict(data, timestamp, pred_len)
    
    # Insert data into the database
    insert_data_into_database(predict_result, "lstnet")

def get_mtgnn_multistock_prediction(request: Request, pred_len: int=50):
    data, timestamp = get_multistock_data(request.app.database, seq_len=100)
    predict_result = generate_multistock_mtgnn_predict(data, timestamp, pred_len)
    
    # Insert data into the database
    insert_data_into_database(predict_result, "mtgnn")

def get_xgboost_multistock_prediction(request: Request, pred_len: int=50):
    data, timestamp = get_multistock_data(request.app.database, seq_len=11)
    predict_result = generate_multistock_xgboost_predict(data, timestamp, pred_len)
    
    # Insert data into the database
    insert_data_into_database(predict_result, "xgboost")

def get_randomforest_multistock_prediction(request: Request, pred_len: int=50):
    data, timestamp = get_multistock_data(request.app.database, seq_len=11)
    predict_result = generate_multistock_random_forest_predict(data, timestamp, pred_len)
    
    # Insert data into the database
    insert_data_into_database(predict_result, "randomforest")

def get_var_multistock_prediction(request: Request, pred_len: int=50):
    data, timestamp = get_multistock_data(request.app.database, seq_len=5)
    predict_result = generate_multistock_var_predict(data, timestamp, pred_len)
    
    # Insert data into the database
    insert_data_into_database(predict_result, "var")

def get_lstm_multistock_prediction(request: Request, pred_len: int=50):
    data, timestamp = get_multistock_data(request.app.database, seq_len=5)
    predict_result = generate_multistock_lstm_predict(data, timestamp, pred_len)
    
    # Insert data into the database
    insert_data_into_database(predict_result, "lstm")

# Call the functions directly to insert their return values into the database.
get_longterm_forecasting(request=request,pred_len=pred_len)
get_lstnet_multistock_prediction(request=request,pred_len=pred_len)
get_mtgnn_multistock_prediction(request=request,pred_len=pred_len)
get_xgboost_multistock_prediction(request=request,pred_len=pred_len)
get_randomforest_multistock_prediction(request=request,pred_len=pred_len)
get_var_multistock_prediction(request=request,pred_len=pred_len)
get_lstm_multistock_prediction(request=request,pred_len=pred_len) 
