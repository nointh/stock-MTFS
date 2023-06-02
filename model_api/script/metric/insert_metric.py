import os
from pymongo import MongoClient
import json
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Connect to MongoDB Atlas
# Get MongoDB username and password from environment variables
MONGO_USERNAME = os.environ.get("MONGO_USERNAME")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD")

# Construct the MongoDB Atlas connection URI using the username and password
mongo_uri = f'mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.7julke9.mongodb.net/?retryWrites=true&w=majority'

# Connect to MongoDB Atlas using the pymongo library
mongo_client = MongoClient(mongo_uri)

database = mongo_client['stock_price']
collection = database['predictionMetric']

# Read data from JSON file
with open("metric.json") as file:
    data = json.load(file)

for item in data:
    algorithm = item['algorithm']
    # Check if a record with the same algorithm already exists in the collection
    existing_record = collection.find_one({"algorithm": algorithm})
    if existing_record:
        # Update the existing record with the new data
        collection.update_one({"algorithm": algorithm}, {"$set": item})
        print(f"Updated record with algorithm: {algorithm}")
    else:
        # Insert a new record into the collection
        collection.insert_one(item)
        print(f"Inserted new record with algorithm: {algorithm}")

# Close the connection
mongo_client.close()
