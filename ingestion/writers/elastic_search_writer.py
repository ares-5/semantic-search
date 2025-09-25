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
                index=f"phd_dissertations_{lang}_vector",
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

        # BM25 text indices
        for lang in ["en", "sr"]:
            self.es.indices.create(
                index=f"phd_dissertations_{lang}",
                body={
                    "mappings": {
                        "properties": {
                            "_id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "details": {"type": "text"}
                        }
                    }
                },
                ignore=400
            )

    def insert_docs(self, df, lang, embedding_col=None, batch_size=500):
        """
        Insert documents into Elasticsearch using Mongo _id.
        """
        if "_id" not in df.columns:
            raise ValueError("DataFrame must have '_id' column for ES insertion.")

        index_vector = f"phd_dissertations_{lang}_vector" if embedding_col else None
        index_text = f"phd_dissertations_{lang}" if not embedding_col else None

        actions = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Inserting {lang}"):
            doc = {"_id": row["_id"]}
            if embedding_col:
                doc["embedding"] = row[embedding_col]
                actions.append({"_index": index_vector, "_id": row["_id"], "_source": doc})
            else:
                doc["title"] = row[f"title_{lang}"]
                doc["details"] = row[f"abstract_{lang}"]
                actions.append({"_index": index_text, "_id": row["_id"], "_source": doc})

            if len(actions) >= batch_size:
                helpers.bulk(self.es, actions)
                actions = []

        if actions:
            helpers.bulk(self.es, actions)

