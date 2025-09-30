# generate_queries_final.py
import os
import torch
import pandas as pd
import nltk
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random

nltk.download("punkt")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
random.seed(42)

MAX_TOKENS = 512

def split_text_to_blocks(text, tokenizer, max_tokens=MAX_TOKENS):
    if not text:
        return []
    sents = nltk.sent_tokenize(text)
    blocks, cur = [], ""
    for sent in sents:
        candidate = (cur + " " + sent).strip() if cur else sent
        if len(tokenizer.tokenize(candidate)) > max_tokens:
            if cur:
                blocks.append(cur)
            cur = sent
        else:
            cur = candidate
    if cur:
        blocks.append(cur)
    return blocks

def is_english(text: str) -> bool:
    if not text:
        return False
    num_ascii = sum(c.isascii() for c in text if c.isalpha())
    num_total = sum(c.isalpha() for c in text)
    if num_total == 0:
        return False
    return (num_ascii / num_total) > 0.8

def paraphrase_t5(text: str, model, tokenizer, num_return_sequences=1):
    prompt = f"Paraphrase the following query concisely: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_p=0.9
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

# Load dataset
df = pd.DataFrame(load_dataset("jerteh/PaSaz")["train"])
df = df[df["full_abstract"] == True].reset_index(drop=True)

# Load T5 model
t5_name = "BeIR/query-gen-msmarco-t5-base-v1"
tok = AutoTokenizer.from_pretrained(t5_name)
t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_name).to(device)

batch_size = 16
num_queries = 3
max_title = 128
max_query = 128
outfile = "/kaggle/working/generated_queries.tsv"

if not os.path.exists(outfile):
    with open(outfile, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(df), batch_size)):
            batch_titles = df["title_en"].iloc[i:i+batch_size].tolist()
            batch_abstracts = df["abstract_en"].iloc[i:i+batch_size].tolist()

            all_blocks = [split_text_to_blocks(abs_text, tok) for abs_text in batch_abstracts]
            for blocks in all_blocks:
                random.shuffle(blocks)

            # Generate queries
            prompts = [
                f"""
                You are a helpful assistant generating search queries.
                Generate a concise, keyword-focused search query that someone would use to find this dissertation.
                Keep it in English.

                Title: {title}
                Abstract: {abstract}

                Search query:
                """
                for title, abstract in zip(batch_titles, batch_abstracts)
            ]

            inputs = tok(
                prompts,
                max_length=max_title + 256,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(device)

            outputs = t5.generate(
                **inputs,
                max_length=max_query,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=num_queries
            )

            for idx, (title, abstract, blocks) in enumerate(zip(batch_titles, batch_abstracts, all_blocks)):
                # Original title (if English)
                if title and is_english(title):
                    for block in blocks:
                        block = block.strip()
                        if not block or len(block.split()) < 5:
                            continue
                        fout.write(f"{title}\t{block}\n")

                # Generated queries
                generated_queries = []
                for q_idx in range(num_queries):
                    gen_query = tok.decode(outputs[idx * num_queries + q_idx], skip_special_tokens=True).strip()
                    if not gen_query or len(gen_query.split()) < 3:
                        continue
                    if gen_query in generated_queries:
                        continue
                    generated_queries.append(gen_query)

                    # Write original generated query
                    for block in blocks:
                        block = block.strip()
                        if not block or len(block.split()) < 5:
                            continue
                        fout.write(f"{gen_query}\t{block}\n")

                    # Generate paraphrases
                    paraphrased_queries = paraphrase_t5(gen_query, t5, tok, num_return_sequences=2)
                    for pq in paraphrased_queries:
                        pq = pq.strip()
                        if not pq or pq in generated_queries:
                            continue
                        generated_queries.append(pq)
                        for block in blocks:
                            block = block.strip()
                            if not block or len(block.split()) < 5:
                                continue
                            fout.write(f"{pq}\t{block}\n")

    print("Saved:", outfile)
else:
    print("Already exists:", outfile)