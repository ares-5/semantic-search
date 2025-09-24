from pymongo import MongoClient
import uuid

class MongoDBWriter:
    def __init__(self, mongo_uri="mongodb://localhost:27017", db_name="ecommerce_db", collection_name="products"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_docs(self, df):
        """
        Insert documents from a pandas DataFrame into MongoDb.
        Ensures each row has a unique _id.
        """
        if "_id" not in df.columns:
            df["_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        records = df.to_dict(orient="records")
        self.collection.insert_many(records)
        print(f"Inserted {len(records)} documents into MongoDB collection '{self.collection.name}'.")
        return df
