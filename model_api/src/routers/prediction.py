import os
from os.path import dirname, abspath, join

from fastapi import APIRouter
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

url = 'https://raw.githubusercontent.com/huy164/datasets/master/VN30_price.csv'
df = pd.read_csv(url)


data = df['VN30'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

dirname = dirname(dirname(abspath(__file__)))

model = load_model(join(dirname, 'static/models/lstm_model.h5'))

router = APIRouter(
    prefix='/predict',
    tags=['prediction']
)

@router.get('/lstm')
def get_lstm_prediction():
    seq_length = 5
    last_sequence = data[-seq_length:]
    last_sequence = np.reshape(last_sequence, (1, seq_length, 1))

    next_day_prediction = model.predict(last_sequence)
    next_day_prediction = scaler.inverse_transform(next_day_prediction)
    prediction_result = float(next_day_prediction[0,0])

    return {'vn30_prediction': prediction_result}

