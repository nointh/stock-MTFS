# from xgboost import Booster
import pickle
import numpy as np
model = pickle.load(open('exp/save/xgboost.pkl', 'rb'))
# model = Booster()
# model.load_model("exp/save/xgboost.json")
data = np.random.rand(1, 220)
result = model.predict(data)
# model = 
print(result.shape)

print(result)
