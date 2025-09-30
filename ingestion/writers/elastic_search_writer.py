from elasticsearch import Elasticsearch, helpers
import numpy as np
from tqdm import tqdm
import traceback

class ElasticSearchWriter:
    def __init__(self, es_host="http://localhost:9200", embedding_dim_en=None, embedding_dim_sr=None):
        self.es = Elasticsearch(es_host)
        self.embedding_dim_en = embedding_dim_en
        self.embedding_dim_sr = embedding_dim_sr
        self.create_indices()

    def drop_index(self, index_name):
        try:
            if self.es.indices.exists(index=index_name, allow_no_indices=True):
                self.es.indices.delete(index=index_name, ignore=[400, 404])
                print(f"Dropped existing Elasticsearch index '{index_name}'.")
        except Exception as e:
            print(f"Error dropping index '{index_name}': {e}")

    def create_indices(self):
        # Dense vector indices
        for lang, dim in [("en", self.embedding_dim_en), ("sr", self.embedding_dim_sr)]:
            if dim is None:
                continue
            index_name = f"phd_dissertations_{lang}_vector"
            self.drop_index(index_name)
            self.es.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": dim, "similarity": "dot_product"}
                        }
                    }
                }
            )

        # BM25 text indices
        for lang in ["en", "sr"]:
            index_name = f"phd_dissertations_{lang}"
            self.drop_index(index_name)
            self.es.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "title": {"type": "text"},
                            "details": {"type": "text"}
                        }
                    }
                }
            )

    def insert_docs(self, df, lang, embedding_col=None, batch_size=500):
        if "_id" not in df.columns:
            raise ValueError("DataFrame must have '_id' column for ES insertion.")

        index_name = f"phd_dissertations_{lang}_vector" if embedding_col else f"phd_dissertations_{lang}"
        actions = []
        invalid_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Inserting {lang}"):
            doc = {}
            try:
                if embedding_col:
                    emb = np.array(row[embedding_col], dtype=np.float32)

                    if emb.shape[0] != len(df[embedding_col][0]) or np.isnan(emb).any():
                        invalid_rows.append(row["_id"])
                        print(f"Invalid embedding for row {row['_id']}")
                        continue

                    doc["embedding"] = emb.tolist()
                else:
                    doc["title"] = row[f"title_{lang}"]
                    doc["details"] = row[f"abstract_{lang}"]

                actions.append({"_index": index_name, "_id": row["_id"], "_source": doc})

                if len(actions) >= batch_size:
                    success, failed = helpers.bulk(self.es, actions, raise_on_error=False, stats_only=False)
                    for f in failed:
                        print(f"Failed document: {f}")
                    actions = []
            except Exception as e:
                invalid_rows.append(row["_id"])
                print(f"Exception inserting row {row['_id']}: {e}")
                print(traceback.format_exc())
                continue

        if actions:
            success, failed = helpers.bulk(self.es, actions, raise_on_error=False, stats_only=False)
            for f in failed:
                print(f"Failed document: {f}")

        if invalid_rows:
            print(f"Skipped {len(invalid_rows)} invalid rows for {lang}: {invalid_rows[:10]}...")
