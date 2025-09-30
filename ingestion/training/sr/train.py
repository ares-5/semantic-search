# Korpus Paralelnih Sažetaka doktorskih disertacija na srpskom i engleskom jeziku
# https://huggingface.co/datasets/jerteh/PaSaz

import os
from tqdm import tqdm
import torch
import pandas as pd
import shutil
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, InputExample, models, losses
)
from datasets import Dataset
import classla
from transformers import AutoTokenizer

# Config
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
MAX_TOKENS = 512
TRAIN_BATCH_SIZE = 32
NUM_EPOCHS = 3
CLASLA_DIR = "/kaggle/input/classla/classla_resources"

print("Using device:", device)

# Helper functions
def input_examples_to_dataset(input_examples):
    return Dataset.from_dict({
        "anchor": [ex.texts[0] for ex in input_examples],
        "positive": [ex.texts[1] for ex in input_examples]
    })

def split_text_to_blocks_sr(text, nlp_pipeline, tokenizer, max_tokens=MAX_TOKENS):
    """Split Serbian text into token-limited blocks using Classla + BERTIC tokenizer"""
    if not text:
        return []
    sentences = [s.text for s in nlp_pipeline(text).sentences]
    blocks, cur_block = [], ""
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

# Load synthetic queries (SR)
out_file = "/kaggle/input/queries-sr/generated_queries_all_sr.tsv"
train_examples = []
seen_pairs = set()  # for removing duplicates

print("Building training examples with SR blocks...")
nlp_sr = classla.Pipeline('sr', processors='tokenize', dir=CLASLA_DIR, use_gpu=(device=="cuda"))
tokenizer_sr = AutoTokenizer.from_pretrained("classla/bcms-bertic")

with open(out_file) as fIn:
    for line in tqdm(fIn, desc="Reading queries"):
        try:
            query, paragraph = line.strip().split('\t', maxsplit=1)
            blocks = split_text_to_blocks_sr(paragraph, nlp_sr, tokenizer_sr)
            for block in blocks:
                pair_key = (query.strip().lower(), block.strip().lower())
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                train_examples.append(InputExample(texts=[query.strip(), block.strip()]))
        except Exception as e:
            continue

print("Total training pairs (with blocks):", len(train_examples))

train_dataset = input_examples_to_dataset(train_examples)

# Build SentenceTransformer model (BERTIC)
normalize = models.Normalize()
transformer = models.Transformer("classla/bcms-bertic", max_seq_length=MAX_TOKENS)
pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
model = SentenceTransformer(modules=[transformer, pooling, normalize]).to(device)

# Loss and Trainer
loss_fn = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=TRAIN_BATCH_SIZE)

training_args = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/bertic-search-checkpoint",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    logging_steps=100,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none"
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    loss=loss_fn
)

# Train
print("Starting BERTIC SR training with query → abstract blocks...")
trainer.train()

# Save final model
output_dir = "/kaggle/working/search_model_sr"
model.save(output_dir)
shutil.make_archive("search_model_sr", 'zip', output_dir)
print(f"Training complete. Model saved to {output_dir}")
