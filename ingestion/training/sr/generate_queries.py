# generate_queries_final.py
import os
import torch
import pandas as pd
import classla
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import random

CLASLA_DIR = "/kaggle/input/classla/classla_resources"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
random.seed(42)

MAX_TOKENS = 512

def split_text_to_blocks_sr(text, nlp_pipeline, tokenizer, max_tokens=MAX_TOKENS):
    """
    Split Serbian text into token-limited blocks using Classla + BERTIC tokenizer.
    Returns a list of blocks, each <= max_tokens.
    """
    if not text:
        return []
    
    sentences = [s.text for s in nlp_pipeline(text).sentences]
    blocks = []
    cur_block = ""

    for s in sentences:
        candidate = (cur_block + " " + s).strip() if cur_block else s
        token_count = len(tokenizer.tokenize(candidate))
        if token_count > max_tokens:
            if cur_block:
                blocks.append(cur_block)
            cur_block = s
        else:
            cur_block = candidate

    if cur_block:
        blocks.append(cur_block)

    return blocks

# Load dataset
dataset = load_dataset("jerteh/PaSaz")["train"]
df = pd.DataFrame(dataset)
df = df[df["full_abstract"] == True].reset_index(drop=True)

outfile = "/kaggle/working/generated_queries_sr.tsv"
batch_size = 16

# Initialize NLP pipeline and tokenizer
nlp_sr = classla.Pipeline('sr', processors='tokenize', dir=CLASLA_DIR, use_gpu=(device=="cuda"))
tokenizer_sr = AutoTokenizer.from_pretrained("classla/bcms-bertic")

if not os.path.exists(outfile):
    with open(outfile, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_titles = df["title_sr"].iloc[i:i+batch_size].tolist()
            batch_abstracts = df["abstract_sr"].iloc[i:i+batch_size].tolist()

            for title, abstract in zip(batch_titles, batch_abstracts):
                if not title:
                    continue

                blocks = split_text_to_blocks_sr(abstract, nlp_sr, tokenizer_sr)
                random.shuffle(blocks)  # shuffle

                for block in blocks:
                    block = block.strip()
                    if not block or len(block.split()) < 5:
                        continue
                    fout.write(f"{title}\t{block}\n")

    print("Saved:", outfile)
else:
    print("Already exists:", outfile)
