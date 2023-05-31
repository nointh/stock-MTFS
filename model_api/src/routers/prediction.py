from fastapi import APIRouter, Request
from datetime import datetime, timedelta

router = APIRouter(
    prefix='/predict',
    tags=['prediction']
)


# @router.get('/vn30/long-term')
# def get_long_term_forecasting(request: Request, pred_len: int=50):

#     return


# @router.get('/lstnet')
# def get_lstnet_multistock_prediction(request: Request, pred_len: int = 50):
#     try:
#         # Get the database connection from the app
#         database = request.app.database

#         # Access the 'prediction' collection
#         collection: Collection = database['prediction']


#         # Calculate the start and end dates
#         start_date = datetime.now().date()
#         end_date = start_date + timedelta(days=pred_len)

#         # Fetch records from the collection based on the specified filters
#         records = collection.find({
#             "algorithm": "lstnet",
#             "date": {"$gte": start_date.isoformat(), "$lt": end_date.isoformat()}
#         }).sort("date", 1).limit(pred_len)

#         # Prepare the JSON response
#         data = {"data": {"VN30": []}}
#         for record in records:
#             data["data"]["VN30"].append({
#                 "date": record["date"],
#                 "value": record["value"]
#             })

#         return data
#     except Exception as e:
#         # Handle the exception and return an error response
#         error_message = {"error": str(e)}
#         return error_message


def fetch_records(request, algorithm, pred_len):
    # Get the database connection from the app
    database = request.app.database

    # Access the 'prediction' collection
    collection: Collection = database['prediction']
    # Calculate the start and end dates
    start_date = datetime.now().date()
    end_date = start_date + timedelta(days=pred_len)

    # Fetch records from the collection based on the specified filters
    records = collection.find({
        "algorithm": algorithm,
        "date": {"$gte": start_date.isoformat(), "$lt": end_date.isoformat()}
    }).sort("date", 1).limit(pred_len)

    # Prepare the JSON response
    data = {"data": {"VN30": []}}
    for record in records:
        data["data"]["VN30"].append({
            "date": record["date"],
            "value": record["value"]
        })

    return data

@router.get('/lstnet')
def get_lstnet_multistock_prediction(request: Request, pred_len: int = 50):
    try:
        # Fetch records for 'lstnet' algorithm
        return fetch_records(request, "lstnet", pred_len)

    except Exception as e:
        # Handle the exception and return an error response
        error_message = {"error": str(e)}
        return error_message

@router.get('/mtgnn')
def get_mtgnn_multistock_prediction(request: Request, pred_len: int = 50):
    try:
        # Fetch records for 'mtgnn' algorithm
        return fetch_records(request,"mtgnn", pred_len)

    except Exception as e:
        # Handle the exception and return an error response
        error_message = {"error": str(e)}
        return error_message

@router.get('/var')
def get_var_multistock_prediction(request: Request, pred_len: int = 50):
    try:
        # Fetch records for 'var' algorithm
        return fetch_records(request, "var", pred_len)

    except Exception as e:
        # Handle the exception and return an error response
        error_message = {"error": str(e)}
        return error_message

# @router.get('/lstm')
# def get_xgboost_multistock_prediction(request: Request, pred_len: int=50):
#     data, timestamp = get_multistock_data(request.app.database, seq_len=5)
#     predict_result = generate_multistock_lstm_predict(data, timestamp, pred_len)
#     return {'data': predict_result}

# @router.get('/xgboost')
# def get_xgboost_multistock_prediction(request: Request, pred_len: int=50):
#     data, timestamp = get_multistock_data(request.app.database, seq_len=11)
#     predict_result = generate_multistock_xgboost_predict(data, timestamp, pred_len)
#     return {'data': predict_result}

# @router.get('/random-forest')
# def get_xgboost_multistock_prediction(request: Request, pred_len: int=50):
#     data, timestamp = get_multistock_data(request.app.database, seq_len=11)
#     predict_result = generate_multistock_random_forest_predict(data, timestamp, pred_len)
#     return {'data': predict_result}
