import os
import pandas as pd
from kagglehub import KaggleDatasetAdapter
import kagglehub

from translator import Translator
from transliterate import translit

# Config
DEBUG = False
translated_file_path = "flipkart_fashion_translated.json"

def to_latin(text: str) -> str:
    if not text:
        return ""
    return translit(text, "sr", reversed=True)

failed_rows = []

if os.path.exists(translated_file_path):
    print("Loading existing translated dataset...")
    df = pd.read_json(translated_file_path, orient="records", lines=True)
else:
    print("Downloading raw dataset from Kaggle...")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "aaditshukla/flipkart-fasion-products-dataset",
        "flipkart_fashion_products_dataset.json",
    )

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Translate
    batch_size = 1000
    title_sr, desc_sr, details_sr = [], [], []
    translator = Translator()

    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        print(f"Translating titles {start_idx}-{end_idx}")
        
        title_sr.extend([
            to_latin(translator.safe_translate(x, row_index=i))
            for i, x in enumerate(df["title"].iloc[start_idx:end_idx], start=start_idx)
        ])
     
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        print(f"Translating descriptions {start_idx}-{end_idx}")

        desc_sr.extend([
            to_latin(translator.safe_translate(x, row_index=i))
            for i, x in enumerate(df["description"].iloc[start_idx:end_idx], start=start_idx)
        ])

    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        print(f"Translating details {start_idx}-{end_idx}")

        details_sr.extend([
            to_latin(translator.safe_translate(
                ", ".join([f"{list(d.keys())[0]}: {list(d.values())[0]}" for d in details]) if details else "",
                row_index=i
            ))
            for i, details in enumerate(df["product_details"].iloc[start_idx:end_idx], start=start_idx)
        ])

    df['title_sr'] = pd.Series(title_sr)
    df['description_sr'] = pd.Series(desc_sr)
    df['details_sr'] = pd.Series(details_sr)

    if translator.failed_rows:
        pd.Series(translator.failed_rows).to_csv("failed_translations.csv", index=False)

    df.to_json(translated_file_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved translated dataset to {translated_file_path}")

if DEBUG:
    df = df.sample(100).reset_index(drop=True)
    print("DEBUG mode active:", len(df))
