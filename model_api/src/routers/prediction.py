import os
from os.path import dirname, abspath, join

from fastapi import APIRouter, Request
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
    generate_multistock_mtgnn_predict
)

router = APIRouter(
    prefix='/predict',
    tags=['prediction']
)

@router.get('/lstm')
def get_lstm_prediction():
    url = 'https://raw.githubusercontent.com/huy164/datasets/master/VN30_price.csv'
    df = pd.read_csv(url)


    data = df['VN30'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    model = load_model(join(WORKING_DIR, 'static/models/lstm_model.h5'))


    seq_length = 5
    last_sequence = data[-seq_length:]
    last_sequence = np.reshape(last_sequence, (1, seq_length, 1))

    next_day_prediction = model.predict(last_sequence)
    next_day_prediction = scaler.inverse_transform(next_day_prediction)
    prediction_result = float(next_day_prediction[0,0])

    return {'vn30_prediction': prediction_result}

@router.get('/vn30/long-term')
def get_long_term_forecasting(request: Request, pred_len: int=50):
    data, timestamp = get_vn30_data(request.app.database, seq_len=50, is_close_last=True)
    predict_result = generate_long_term_predict(data, timestamp, pred_len)
    return {'data': predict_result}

@router.get('/lstnet')
def get_lstnet_multistock_prediction(request: Request, pred_len: int=50):
    data, timestamp = get_multistock_data(request.app.database, seq_len=100)
    predict_result = generate_multistock_lstnet_predict(data, timestamp, pred_len)
    return {'data': predict_result}

@router.get('/mtgnn')
def get_mtgnn_multistock_prediction(request: Request, pred_len: int=50):
    data, timestamp = get_multistock_data(request.app.database, seq_len=100)
    predict_result = generate_multistock_mtgnn_predict(data, timestamp, pred_len)
    return {'data': predict_result}

@router.get('/xgboost')
def get_xgboost_multistock_prediction(request: Request, pred_len: int=50):
    data, timestamp = get_multistock_data(request.app.database, seq_len=11)
    predict_result = generate_multistock_xgboost_predict(data, timestamp, pred_len)
    return {'data': predict_result}

@router.get('/random-forest')
def get_xgboost_multistock_prediction(request: Request, pred_len: int=50):
    data, timestamp = get_multistock_data(request.app.database, seq_len=11)
    predict_result = generate_multistock_random_forest_predict(data, timestamp, pred_len)
    return {'data': predict_result}