import json
import pandas as pd
from tqdm import tqdm
from collections import Counter

df = pd.read_json("../../flipkart_fashion_translated.json", orient="records", lines=True)
df = df.sample(2500, random_state=42).reset_index(drop=True)

# Generic queries we want to limit
generic_queries = {"Topwear", "Bottomwear", "Winter Wear", "Blazers", "Ethnic Sets"}

# Max generic repetitons
GENERIC_MAX = 5
generic_counter = Counter()

def generate_keyword_queries(row):
    queries = []
    title = str(row.get("title", "")).strip()
    brand = str(row.get("brand", "")).strip()
    category = str(row.get("sub_category", "")).strip()

    if title:
        queries.append(title)
    if brand and category:
        queries.append(f"{brand} {category}")
    if title and brand:
        queries.append(f"{brand} {title}")
    if category:
        queries.append(category)

    # Filter generic query by limit
    filtered_queries = []
    for q in queries:
        if q in generic_queries:
            if generic_counter[q] < GENERIC_MAX:
                filtered_queries.append(q)
                generic_counter[q] += 1
        else:
            filtered_queries.append(q)
    
    # Limit per product (max 3)
    return filtered_queries[:3]

out_file = "synthetic_queries_keyword_en_balanced.jsonl"
with open(out_file, "w", encoding="utf-8") as f:
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating keyword queries"):
        product_text = (str(row.get("title", "")) + " " + str(row.get("description", ""))).strip()
        if not product_text:
            continue

        queries = generate_keyword_queries(row)
        for q in queries:
            record = {"query": q, "positive": product_text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Keyword synthetic queries saved to {out_file}")
