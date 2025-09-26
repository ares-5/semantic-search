
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import classla
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

nltk.download('punkt')
classla.download('sr')
nlp_sr = classla.Pipeline('sr', processors='tokenize', use_gpu=(device=="cuda"))

dataset = pd.read_json("hf://datasets/jerteh/PaSaz/PaSaz.jsonl", lines=True)
df = dataset[["title_sr", "title_en", "abstract_en", "abstract_sr", "full_abstract"]]
df = df[df["full_abstract"]].reset_index(drop=True)

tokenizer_en = AutoTokenizer.from_pretrained("/kaggle/input/final_en_model/transformers/default/1")
model_en = AutoModel.from_pretrained("/kaggle/input/final_en_model/transformers/default/1").to(device)
tokenizer_sr = AutoTokenizer.from_pretrained("/kaggle/input/search_model_final_sr/transformers/default/1")
model_sr = AutoModel.from_pretrained("/kaggle/input/search_model_final_sr/transformers/default/1").to(device)

def encode_texts_hf(model, tokenizer, texts, lang='en', nlp_sr=None, max_len=512, batch_size=32):
    embeddings = []
    for text in texts:
        # Sentence tokenization
        sentences = sent_tokenize(text) if lang=='en' else [s.text for s in nlp_sr(text).sentences]

        blocks, current_block = [], ""
        for sent in sentences:
            tokens = tokenizer.tokenize((current_block + " " + sent).strip())
            if len(tokens) <= max_len:
                current_block = (current_block + " " + sent).strip()
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = sent
        if current_block:
            blocks.append(current_block)
        if not blocks:
            embeddings.append(np.zeros(model.config.hidden_size, dtype=np.float32))
            continue

        # Encode block batch-wise
        for i in range(0, len(blocks), batch_size):
            batch = blocks[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embs = outputs.last_hidden_state.mean(dim=1)  # mean pooling
            embeddings.extend(batch_embs.cpu().numpy())

    return np.vstack(embeddings)

emb_file_en = "/kaggle/working/embeddings_en.npy"
emb_file_sr = "/kaggle/working/embeddings_sr.npy"

emb_en = np.load(emb_file_en) if os.path.exists(emb_file_en) else np.zeros((0, model_en.config.hidden_size), dtype=np.float32)
emb_sr = np.load(emb_file_sr) if os.path.exists(emb_file_sr) else np.zeros((0, model_sr.config.hidden_size), dtype=np.float32)

batch_size = 64
start_idx = len(emb_en)
print(f"Starting from {start_idx}/{len(df)}")

for i in range(start_idx, len(df), batch_size):
    batch_texts_en = (df["title_en"] + " " + df["abstract_en"]).tolist()[i:i+batch_size]
    batch_texts_sr = (df["title_sr"] + " " + df["abstract_sr"]).tolist()[i:i+batch_size]

    batch_emb_en = encode_texts_hf(model_en, tokenizer_en, batch_texts_en, lang='en')
    batch_emb_sr = encode_texts_hf(model_sr, tokenizer_sr, batch_texts_sr, lang='sr', nlp_sr=nlp_sr)

    batch_emb_en = normalize_embeddings(batch_emb_en)
    batch_emb_sr = normalize_embeddings(batch_emb_sr)

    emb_en = np.vstack([emb_en, batch_emb_en])
    emb_sr = np.vstack([emb_sr, batch_emb_sr])
    np.save(emb_file_en, emb_en)
    np.save(emb_file_sr, emb_sr)

    print(f"Processed up to {i+batch_size}/{len(df)}")

print("Embeddings saved and checkpointed successfully.")
