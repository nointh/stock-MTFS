from pymongo import MongoClient

# Connect to the write database
write_client = MongoClient("<WRITE_CONNECTION_STRING>")
write_db = write_client["your_database"]

# Connect to the read database
read_client = MongoClient("<READ_CONNECTION_STRING>")
read_db = read_client["your_database"]

# Get the list of collection names in the write database
collection_names = write_db.list_collection_names()

# Iterate over each collection
for collection_name in collection_names:
    # Retrieve the data from the write database
    data_to_update = write_db[collection_name].find()
    
    # Update the data in the read database
    for document in data_to_update:
        # Modify the document if necessary
        # For example, you can add a new field to the document
        document["new_field"] = "new_value"
        
        # Insert or update the modified document in the read database
        read_db[collection_name].replace_one({"_id": document["_id"]}, document, upsert=True)

# Close the connections
write_client.close()
read_client.close()
