from pymongo import MongoClient

class MongoDBWriter:
    def __init__(self, mongo_uri="mongodb://localhost:27017", db_name="pasaz_db", collection_name="phd_dissertations"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_docs(self, df):
        docs = []
        for row in df.itertuples():
            doc = {
                "_id": row._id,
                "title": {"en": row.title_en, "sr": row.title_sr},
                "details": {"en": row.abstract_en, "sr": row.abstract_sr}
            }
            docs.append(doc)

        self.collection.insert_many(docs)
        print(f"Inserted {len(docs)} documents into MongoDB collection '{self.collection.name}'.")
        return df