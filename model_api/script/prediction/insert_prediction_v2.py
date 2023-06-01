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
print("MONGO_USERNAME =", MONGO_USERNAME)
print("MONGO_PASSWORD =", MONGO_PASSWORD)

# Construct the MongoDB Atlas connection URI using the username and password
mongo_uri = f'mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.7julke9.mongodb.net/?retryWrites=true&w=majority'

# Connect to MongoDB Atlas using the pymongo library
mongo_client = MongoClient(mongo_uri)

database = mongo_client['stock_price']
collection = database['prediction']

algorithms = ["random_forest", "long_term", "mtgnn", "lstm", "var", "lstnet", "xgboost"]

# Loop through each algorithm and insert/update VN30 data in MongoDB collection
for algorithm in algorithms:
    # Load JSON data from file based on algorithm
    with open(f"prediction_{algorithm}.json") as file:
        data = json.load(file)

    # Extract VN30 data from JSON
    vn30_data = data['data']['VN30']

    # Set algorithm value for all entries
    for entry in vn30_data:
        entry['algorithm'] = algorithm

    # Insert or update VN30 data in MongoDB collection, handling duplicates based on date and algorithm
    for entry in vn30_data:
        date = entry['date']
        # Check if record already exists in the collection based on date and algorithm
        existing_entry = collection.find_one({'date': date, 'algorithm': algorithm})
        if existing_entry:
            # Log a message and update the existing record
            print(f"Updating record for date: {date} and algorithm: {algorithm}.")
            collection.update_one({'date': date, 'algorithm': algorithm}, {'$set': entry})
        else:
            # Insert the new entry if it doesn't exist
            collection.insert_one(entry)
            # Print a message for successful record insertion
            print(f"Inserted record for date: {date} and algorithm: {algorithm}.")

# Close the MongoDB connection
mongo_client.close()
