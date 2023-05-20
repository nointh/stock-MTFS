
from os.path import join
import pickle

import pandas as pd
import numpy as np
import torch

from algorithm.autoformer import get_long_term_model
from algorithm.lstnet import get_lstnet_multistock_model
from common.constant import WORKING_DIR, STOCK_COLLECTIONS
from common.time_features import time_features


def generate_long_term_predict(data, timestamp, predict_len=50):
    #TODO: get model => load model => predict
    with open(join(WORKING_DIR, 'static/scalers/std_scaler.pkl'), "rb") as input_file:
        scaler = pickle.load(input_file)
    data = scaler.transform(data)
    model = get_long_term_model()
    state_dict = torch.load(join(WORKING_DIR, 'static/models/VN30_Autoformer.pth'))
    model.load_state_dict(state_dict)
    seq_len = model.seq_len
    pred_len = model.pred_len
    label_len = model.label_len
    future_timestamp = list(pd.date_range(start=timestamp[-1], freq='B', periods=pred_len+1)[1:])
    pd_datestamp = pd.to_datetime(
        list(timestamp) + 
        list(pd.date_range(start=timestamp[-1], freq='B', periods=pred_len+1)[1:]))
    data_stamp = time_features(pd_datestamp, freq='b')
    data_stamp = data_stamp.transpose(1, 0)
    mark_data = torch.from_numpy(data_stamp).float()
    mark_data = torch.unsqueeze(mark_data, 0)
    x = torch.from_numpy(data).float()
    x = torch.unsqueeze(x, 0)
    y = x[:, -model.label_len:, :]
    dec_inp = torch.zeros([y.shape[0], pred_len, y.shape[2]])
    dec_inp = torch.cat([y[:, :label_len, :], dec_inp], dim=1)

    x_mark = mark_data[:, :seq_len, :]
    y_mark = mark_data[:, seq_len-label_len:, :]
    model.eval()
    pred_outputs = model(x, x_mark, dec_inp, y_mark)
    pred_outputs = pred_outputs.squeeze().detach().numpy()
    pred_outputs = scaler.inverse_transform(pred_outputs)
    if predict_len <= pred_len:
        pred_outputs = pred_outputs[:predict_len, :]
    
    result = []
    for i, stock_price in enumerate(pred_outputs.tolist()):
        result.append({
            'date': future_timestamp[i],
            'close': stock_price[-1],
            'open': stock_price[0],
            'high': stock_price[1],
            'low': stock_price[2],
            'volume': stock_price[3] * 1e6,
            'change': stock_price[4],
        })
    if predict_len > pred_len:
        remainer = predict_len - pred_len
        result.extend(generate_long_term_predict(data=pred_outputs, 
                                                 timestamp=future_timestamp,
                                                 predict_len=remainer))
    return result


def generate_multistock_lstnet_predict(data, timestamp, predict_len=1):
    model = get_lstnet_multistock_model()
    state_dict = torch.load(join(WORKING_DIR, 'static/models/LSTNet_multistock_state.pt'))
    model.load_state_dict(state_dict)
    predict_result = []
    for i in range(predict_len):
        model.eval()
        input_data = torch.from_numpy(data).float()
        input_data = torch.unsqueeze(input_data, 0)
        output = model(input_data)
        output_as_list = output.squeeze().detach().tolist()
        predict_result.append(output_as_list)
        data = np.roll(data, -1, axis=0)
        data[-1, :] = output_as_list
    future_timestamp = list(pd.date_range(start=timestamp[-1], freq='B', periods=predict_len+1)[1:])
    result = {key: [] for key in STOCK_COLLECTIONS}
    for idx, predict in enumerate(predict_result):
        for stock_idx, data in enumerate(predict):
            result[STOCK_COLLECTIONS[stock_idx]].append({
                'date': future_timestamp[idx],
                'value': data
            })
    return result
    
    