import csv
import pandas as pd
from datasets import Dataset

import translator

def input_examples_to_dataset(input_examples):
    return Dataset.from_dict({
        "anchor": [ex.texts[0] for ex in input_examples],
        "positive": [ex.texts[1] for ex in input_examples]
    })

# Load original PaSaz dataset (contains both en and sr abstracts)
dataset = pd.read_json("hf://datasets/jerteh/PaSaz/PaSaz.jsonl", lines=True)
df_small = dataset[["abstract_en", "abstract_sr", "full_abstract"]]
df_small = df_small[df_small["full_abstract"] == True]

# File paths
out_file = "generated_queries.tsv"
out_file_sr = "generated_queries_all_sr.tsv"

translator = translator.Translator()

resume_query = "Quality of life assesment in persons with chronic active otitis media"
resume_flag = False  # Will turn True once we hit this query

with open(out_file_sr, "a", encoding="utf-8") as fOut:
    with open(out_file, "r", encoding="utf-8") as fIn:
            reader = csv.reader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            for i, row in enumerate(reader):
                try:
                    if len(row) != 2:
                        continue

                    query_en, abstract_en_trunc = row

                    if not resume_flag:
                        if query_en.strip() == resume_query.strip():
                            resume_flag = True
                        else:
                            continue

                    mask = df_small["abstract_en"].str.contains(abstract_en_trunc, regex=False)
                    matched_row = df_small[mask]
                    if not matched_row.empty:
                        abstract_sr = matched_row.iloc[0]["abstract_sr"]
                        query_sr = translator.safe_translate(query_en, row_index=i)
                        fOut.write(f"{query_sr}\t{abstract_sr}\n")
                except Exception as e:
                    continue

print(f"Translated SR query-abstract pairs saved to {out_file_sr}")
