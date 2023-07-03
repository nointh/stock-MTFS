#!/bin/bash

current_location=$(pwd)

mongorestore --uri "mongodb+srv://user_mongodb_api_sync:user_mongodb_api_sync@cluster0.uwsidsd.mongodb.net" --writeConcern '{w: "majority"}' "$current_location" &&
mongodump --uri "mongodb+srv://user_mongodb_model:user_mongodb_model@cluster0.7julke9.mongodb.net/?retryWrites=true&w=majority/stock_price" --out "$current_location"
