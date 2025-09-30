import torch
import pandas as pd
import nltk
from tqdm import tqdm
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, InputExample, models, losses
)

nltk.download("punkt")

# CONFIG
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
MAX_TOKENS = 512
TRAIN_BATCH_SIZE = 32
EPOCHS = 3

# HELPER: split text into token-limited blocks
def split_text_to_blocks(text, tokenizer, max_tokens=MAX_TOKENS):
    if not text: return []
    sents = nltk.sent_tokenize(text)
    blocks, cur = [], ""
    for sent in sents:
        candidate = (cur + " " + sent).strip() if cur else sent
        if len(tokenizer.tokenize(candidate)) > max_tokens:
            if cur: blocks.append(cur)
            cur = sent
        else:
            cur = candidate
    if cur: blocks.append(cur)
    return blocks

# HELPER: convert InputExamples to HuggingFace Dataset
def to_dataset(examples):
    return Dataset.from_dict({
        "anchor": [e.texts[0] for e in examples],
        "positive": [e.texts[1] for e in examples]
    })

# BUILD fresh model (instead of loading Stage1)
word_emb = models.Transformer(
    "sentence-transformers/msmarco-distilbert-base-tas-b",
    max_seq_length=MAX_TOKENS
)
pooling = models.Pooling(word_emb.get_word_embedding_dimension(), pooling_mode="mean")
model = SentenceTransformer(modules=[word_emb, pooling, models.Normalize()]).to(device)
tokenizer = word_emb.tokenizer

# LOAD synthetic queries
synthetic_file = "/kaggle/input/generated-queries/generated_queries.tsv"
examples = []
seen_pairs = set()  # for removing duplicates

with open(synthetic_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            query, abstract = line.strip().split("\t", 1)
            if len(query.split()) < 3 or len(abstract.split()) < 5:
                continue

            # split abstract into blocks
            blocks = split_text_to_blocks(abstract, tokenizer)
            for block in blocks:
                pair_key = (query.strip().lower(), block.strip().lower())
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                examples.append(InputExample(texts=[query.strip(), block.strip()]))
        except:
            pass

print("Stage pairs after splitting blocks and deduplication:", len(examples))

# Build dataset (all for training, no split)
train_ds = to_dataset(examples)

# LOSS
train_loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=TRAIN_BATCH_SIZE)

# TRAINING ARGS
training_args = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/checkpoint_stage2_blocks",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    logging_steps=100,
    save_strategy="epoch",
    report_to="none"
)

# TRAINER
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    loss=train_loss
)

print("Starting Stage 2 training with block-split abstracts on fresh model...")
trainer.train()

# SAVE FINAL MODEL
model.save("/kaggle/working/model_final_blocks")
print("Final model saved to /kaggle/working/model_final_blocks")
