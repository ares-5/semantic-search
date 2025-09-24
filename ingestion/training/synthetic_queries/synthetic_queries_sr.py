import json
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ingestion.translator import Translator

translator = Translator(src="en", dest="sr")

input_file = "synthetic_queries_keyword_en_balanced.jsonl"
output_file = "synthetic_queries_keyword_sr_balanced.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for i, line in enumerate(tqdm(fin, desc="Translating synthetic queries to SR")):
        item = json.loads(line.strip())
        sr_query = translator.safe_translate(item["query"], row_index=i)
        sr_positive = translator.safe_translate(item["positive"], row_index=i)
        record = {"query": sr_query, "positive": sr_positive}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved translated synthetic queries to {output_file}")

if translator.failed_rows:
    print("Failed rows:", translator.failed_rows)
    pd.Series(translator.failed_rows).to_csv("failed_queries.csv", index=False)