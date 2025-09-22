from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

class ElasticSearchWriter:
    def __init__(self, es_host="http://localhost:9200", embedding_dim_en=None, embedding_dim_sr=None):
        self.es = Elasticsearch(es_host)
        self.create_indices(embedding_dim_en, embedding_dim_sr)

    def create_indices(self, embedding_dim_en, embedding_dim_sr):
        # Dense vector indices
        for lang, dim in [("en", embedding_dim_en), ("sr", embedding_dim_sr)]:
            self.es.indices.create(
                index=f"products_{lang}_vector",
                body={
                    "mappings": {
                        "properties": {
                            "_id": {
                                "type": "keyword"
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": dim,
                                "similarity": "dot_product"
                            }
                        }
                    }
                },
                ignore=400
            )

        # Classical text indices
        for lang in ["en", "sr"]:
            self.es.indices.create(
                index=f"products_{lang}",
                body={
                    "mappings": {
                        "properties": {
                            "_id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "description": {"type": "text"},
                            "details": {"type": "text"},
                            "brand": {"type": "keyword"},
                            "category": {"type": "keyword"},
                            "seller": {"type": "keyword"}
                        }
                    }
                },
                ignore=400
            )

    def insert_docs(self, df, index_name, embedding_col=None, text_cols=None, batch_size=500):
        """
        Insert documents into Elasticsearch using Mongo _id.
        """
        if "_id" not in df.columns:
            raise ValueError("DataFrame must have '_id' column for ES insertion.")

        actions = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Inserting into {index_name}"):
            doc = {"_id": row["_id"]}

            if text_cols:
                for es_field, df_col in text_cols.items():
                    value = row[df_col]
                    # Flatten lists (e.g., product_details) into string
                    if isinstance(value, list):
                        value = ", ".join([str(v) if not isinstance(v, dict) else ", ".join(f"{k}: {v}" for k, v in v.items()) for v in value])
                    doc[es_field] = value

            if embedding_col:
                doc["embedding"] = row[embedding_col]

            actions.append({"_index": index_name, "_id": row["_id"], "_source": doc})

            if len(actions) >= batch_size:
                helpers.bulk(self.es, actions)
                actions = []

        if actions:
            helpers.bulk(self.es, actions)
