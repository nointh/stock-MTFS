import numpy as np
from common.constant import STOCK_COLLECTIONS


def get_multistock_data(database, seq_len=10):
    data = np.zeros(shape=(seq_len, len(STOCK_COLLECTIONS)))
    timestamp = []
    for index, stock in enumerate(STOCK_COLLECTIONS):
        collection = database[stock]
        stock_data = list(collection.find({}, {'_id': 0, 'date': 1, 'close': 1}).sort('date', -1).limit(seq_len))
        stock_data = sorted(stock_data, key=lambda x: x.get('date'))
        if not timestamp:
            timestamp = list(map(lambda x: x.get('date'), stock_data))
        data[:, index] = list(map(lambda x: x.get('close'), stock_data))
    return data, timestamp

def get_vn30_data(database, seq_len=10, is_close_last=False):
    features = ['close', 'open', 'high', 'low', 'volume', 'change']
    vn30_collection = database['VN30']
    stock_data = list(vn30_collection.find({}, {'_id': 0}).sort('date', -1).limit(seq_len))
    stock_data = sorted(stock_data, key=lambda x: x.get('date'))
    if is_close_last:
        data = list(map(lambda x: [x['open'], x['high'], x['low'], x['volume']/1e6, x['change'], x['close']], stock_data))
    else:
        data = list(map(lambda x: [x['close'], x['open'], x['high'], x['low'], x['volume']/1e6, x['change']], stock_data))
    data_stamp = list(map(lambda x: x.get('date'), stock_data))
    return np.array(data), data_stamp
