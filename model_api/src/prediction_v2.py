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
import json
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
from dotenv import load_dotenv, find_dotenv


# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Get MongoDB username and password from environment variables
MONGO_USERNAME = os.environ.get("MONGO_USERNAME")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD")


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
        # Format the date string
        date_string = result["date"].strftime("%Y-%m-%dT%H:%M:%S")
        # Rename the "value" field to "close"
        result["close"] = result.pop("value")
        # Check if a record with the same date and algorithm already exists
        existing_record = collection.find_one({"date": date_string, "algorithm": algorithm})
        if existing_record:
            print("Record already exists for date:", date_string, "and algorithm:", algorithm)
            # Update the "value" field with the value from result
            collection.update_one(
                {"date": date_string, "algorithm": algorithm},
                {"$set": {"close": result["close"]}}
            )
            print("Record updated successfully.")
        else:
            # Insert the new record
            result["algorithm"] = algorithm
            result["date"] = date_string
            collection.insert_one(result)
            print("Record inserted successfully for date:", date_string, "and algorithm:", algorithm)

def get_longterm_forecasting(pred_len: int=50):
    data, timestamp = get_vn30_data(database,seq_len=50, is_close_last=True)
    predict_result = generate_long_term_predict(data, timestamp, pred_len)
    # Insert data into the database
    insert_data_into_database(predict_result, "longterm")

def get_lstnet_multistock_prediction( pred_len: int=50):
    data, timestamp = get_multistock_data(database, seq_len=100)
    predict_result = generate_multistock_lstnet_predict(data, timestamp, pred_len)
    
    VN30_predicted = predict_result["VN30"]
    
    # Insert data into the database
    insert_data_into_database(VN30_predicted, "lstnet")

def get_mtgnn_multistock_prediction( pred_len: int=50):
    data, timestamp = get_multistock_data(database, seq_len=100)
    predict_result = generate_multistock_mtgnn_predict(data, timestamp, pred_len)
    
    VN30_predicted = predict_result["VN30"]

    # Insert data into the database
    insert_data_into_database(VN30_predicted, "mtgnn")

def get_xgboost_multistock_prediction( pred_len: int=50):
    data, timestamp = get_multistock_data(database, seq_len=11)
    predict_result = generate_multistock_xgboost_predict(data, timestamp, pred_len)
    
    VN30_predicted = predict_result["VN30"]

    # Insert data into the database
    insert_data_into_database(VN30_predicted, "xgboost")

def get_randomforest_multistock_prediction( pred_len: int=50):
    data, timestamp = get_multistock_data(database, seq_len=11)
    predict_result = generate_multistock_random_forest_predict(data, timestamp, pred_len)
    
    VN30_predicted = predict_result["VN30"]

    # Insert data into the database
    insert_data_into_database(VN30_predicted, "randomforest")

def get_var_multistock_prediction( pred_len: int=50):
    data, timestamp = get_multistock_data(database, seq_len=5)
    predict_result = generate_multistock_var_predict(data, timestamp, pred_len)
    
    VN30_predicted = predict_result["VN30"]

    # Insert data into the database
    insert_data_into_database(VN30_predicted, "var")

def get_lstm_multistock_prediction( pred_len: int=50):
    data, timestamp = get_multistock_data(database, seq_len=5)
    predict_result = generate_multistock_lstm_predict(data, timestamp, pred_len)
    
    VN30_predicted = predict_result["VN30"]
    
    # Insert data into the database
    insert_data_into_database(VN30_predicted, "lstm")

# Call the functions directly to insert their return values into the database.

get_longterm_forecasting(150)
get_lstnet_multistock_prediction(150)
get_mtgnn_multistock_prediction(150)
get_xgboost_multistock_prediction(150)
get_randomforest_multistock_prediction(150)
get_var_multistock_prediction(150)
get_lstm_multistock_prediction(150) 
