from sentence_transformers import SentenceTransformer
import config
import numpy as np

model_en = SentenceTransformer(config.EN_MODEL_PATH, local_files_only=True)
model_sr = SentenceTransformer(config.SR_MODEL_PATH, local_files_only=True)

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def get_embedding(query: str, lang: str = "en") -> list[float]:
    model = model_en if lang == "en" else model_sr
    emb = model.encode([query], normalize_embeddings=False)[0]
    emb = emb / np.linalg.norm(emb)
    print("Norm of embedding:", np.linalg.norm(emb))
    return emb.tolist()

# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# import classla
# from nltk.tokenize import sent_tokenize

# import config

# device = "cuda" if torch.cuda.is_available() else "cpu"

# def normalize_embedding(vec: np.ndarray) -> np.ndarray:
#     norm = np.linalg.norm(vec)
#     if norm == 0:
#         return vec
#     return vec / norm

# # Load models
# tokenizer_en = AutoTokenizer.from_pretrained(config.EN_MODEL_PATH, local_files_only=True)
# model_en = AutoModel.from_pretrained(config.EN_MODEL_PATH, local_files_only=True).to(device)
# tokenizer_sr = AutoTokenizer.from_pretrained(config.SR_MODEL_PATH, local_files_only=True)
# model_sr = AutoModel.from_pretrained(config.SR_MODEL_PATH, local_files_only=True).to(device)
# classla.download('sr')
# nlp_sr = classla.Pipeline('sr', processors='tokenize', use_gpu=(device=="cuda"))

# def encode_text_hf(text: str, lang='en', max_len=512):
#     model, tokenizer = (model_en, tokenizer_en) if lang=='en' else (model_sr, tokenizer_sr)
#     sentences = sent_tokenize(text) if lang=='en' else [s.text for s in nlp_sr(text).sentences]

#     blocks, current_block = [], ""
#     for sent in sentences:
#         tokens = tokenizer.tokenize((current_block + " " + sent).strip())
#         if len(tokens) <= max_len:
#             current_block = (current_block + " " + sent).strip()
#         else:
#             if current_block:
#                 blocks.append(current_block)
#             current_block = sent
#     if current_block:
#         blocks.append(current_block)
#     if not blocks:
#         return np.zeros(model.config.hidden_size, dtype=np.float32)

#     # Encode blocks
#     embeddings = []
#     for block in blocks:
#         inputs = tokenizer([block], padding=True, truncation=True, return_tensors="pt").to(device)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
#             embeddings.append(emb)
#     final_emb = np.vstack(embeddings).mean(axis=0)  # average across blocks
#     return normalize_embedding(final_emb)
