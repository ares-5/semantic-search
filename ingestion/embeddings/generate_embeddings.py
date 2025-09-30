import os, json, time, multiprocessing
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import nltk
import classla
from nltk.tokenize import sent_tokenize

DATA_JSONL = "hf://datasets/jerteh/PaSaz/PaSaz.jsonl"
MODEL_EN_PATH = "/kaggle/input/search_models/transformers/default/1/search_model_en"
MODEL_SR_PATH = "/kaggle/input/search_models/transformers/default/1/search_model_sr"
CLASLA_DIR = "/kaggle/input/classla/classla_resources"

dst_emb_en = "/kaggle/working/embeddings_en_final.npy"
dst_emb_sr = "/kaggle/working/embeddings_sr_final.npy"

CHECKPOINT_FILE = "/kaggle/working/checkpoint.json"
BATCH_SIZE_DOCS = 64
ENCODE_BATCH_SIZE = 128       # bigger batch for GPU
MAX_TOKENS = 512
MAX_SENTENCES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Sentence splitter
nltk.download('punkt', quiet=True)
nlp_sr = classla.Pipeline('sr', processors='tokenize', dir=CLASLA_DIR, use_gpu=(DEVICE=="cuda"))

def split_text_blocks_for_pool(args):
    text, lang = args
    return split_text_to_blocks(text, lang=lang)

def split_text_to_blocks(text, lang='en'):
    """Split a text into token-limited blocks"""
    if lang=='en':
        sentences = sent_tokenize(text or "")
    else:
        sentences = [s.text for s in nlp_sr(text).sentences]

    blocks = []
    cur = ""
    for s in sentences:
        candidate = (cur + " " + s).strip() if cur else s
        # approximate token count
        token_count = len(candidate.split())
        if token_count > MAX_TOKENS:
            if cur: blocks.append(cur)
            cur = s
        else:
            cur = candidate
    if cur: blocks.append(cur)
    return blocks

def preprocess_documents(texts, lang='en'):
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(split_text_blocks_for_pool, [(t, lang) for t in texts]),
            total=len(texts),
            desc=f"Preprocessing {lang}"
        ))
    return results

def encode_documents(model, doc_blocks, encode_batch_size=ENCODE_BATCH_SIZE):
    """Flatten blocks, encode in large batches, then mean-pool per doc"""
    all_blocks = [b for doc in doc_blocks for b in doc]
    block_counts = [len(doc) for doc in doc_blocks]

    if not all_blocks:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((len(doc_blocks), dim), dtype=np.float32)

    # GPU encoding
    all_embs = model.encode(
        all_blocks, convert_to_numpy=True, normalize_embeddings=True,
        device=DEVICE, batch_size=encode_batch_size, show_progress_bar=True
    )

    # mean-pool
    embeddings = []
    idx = 0
    for cnt in block_counts:
        if cnt==0:
            embeddings.append(np.zeros(all_embs.shape[1], dtype=np.float32))
        else:
            doc_emb = np.mean(all_embs[idx:idx+cnt], axis=0)
            norm = np.linalg.norm(doc_emb)
            if norm > 0:
                doc_emb /= norm
            embeddings.append(doc_emb)
            idx += cnt

    return np.vstack(embeddings)

df = pd.read_json(DATA_JSONL, lines=True)
df = df[df["full_abstract"]==True].reset_index(drop=True)
n_docs = len(df)
print("Total documents:", n_docs)

model_en = SentenceTransformer(MODEL_EN_PATH, device=DEVICE)
model_sr = SentenceTransformer(MODEL_SR_PATH, device=DEVICE)

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f: checkpoint = json.load(f)
    start_idx = checkpoint.get("start_idx", 0)
else:
    start_idx = 0
    checkpoint = {}

emb_en = np.load(dst_emb_en) if os.path.exists(dst_emb_en) else np.zeros((0, model_en.get_sentence_embedding_dimension()), dtype=np.float32)
emb_sr = np.load(dst_emb_sr) if os.path.exists(dst_emb_sr) else np.zeros((0, model_sr.get_sentence_embedding_dimension()), dtype=np.float32)

for i in range(start_idx, n_docs, BATCH_SIZE_DOCS):
    batch_texts_en = (df["title_en"].fillna("") + " " + df["abstract_en"].fillna("")).tolist()[i:i+BATCH_SIZE_DOCS]
    batch_texts_sr = (df["title_sr"] + " " + df["abstract_sr"]).tolist()[i:i+BATCH_SIZE_DOCS]

    t0 = time.perf_counter()
    blocks_en = preprocess_documents(batch_texts_en, lang='en')
    blocks_sr = preprocess_documents(batch_texts_sr, lang='sr')
    batch_emb_en = encode_documents(model_en, blocks_en)
    batch_emb_sr = encode_documents(model_sr, blocks_sr)
    t1 = time.perf_counter()

    emb_en = np.vstack([emb_en, batch_emb_en])
    emb_sr = np.vstack([emb_sr, batch_emb_sr])

    np.save(dst_emb_en, emb_en)
    np.save(dst_emb_sr, emb_sr)
    checkpoint["start_idx"] = i + len(batch_texts_en)
    checkpoint["len_emb_en"] = len(emb_en)
    checkpoint["len_emb_sr"] = len(emb_sr)
    with open(CHECKPOINT_FILE, "w") as f: json.dump(checkpoint, f)

    print(f"Processed {i + len(batch_texts_en)}/{n_docs} docs in {t1-t0:.2f}s")

print("DONE. GPU fully utilized and embeddings checkpointed.")
