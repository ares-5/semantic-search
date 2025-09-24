import os
from tqdm import tqdm
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer, InputExample, losses, datasets, models
from torch.utils.data import DataLoader
import shutil
from datasets import Dataset

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i) < 128)

def input_examples_to_dataset(input_examples):
    return Dataset.from_dict({
        "anchor": [ex.texts[0] for ex in input_examples],
        "positive": [ex.texts[1] for ex in input_examples]
    })

# CUDA and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
print("Using device:", device)

# PaSaz dataset
dataset = load_dataset("jerteh/PaSaz")
df = pd.DataFrame(dataset["train"])
df_small = df[["title_en", "abstract_en", "full_abstract"]]
df_small = df_small[df_small["full_abstract"] == True]

# File path for synthetic queries
out_file = "/kaggle/input/queries/generated_queries_all (3).tsv"

# Synthetic queries with T5 (generate only if file doesn't exist)
if not os.path.exists(out_file):
    print("Generating synthetic queries...")
    t5_model_name = "BeIR/query-gen-msmarco-t5-base-v1"
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name).to(device)

    batch_size = 16
    num_queries = 5
    max_length_paragraph = 512
    max_length_query = 64
    paragraphs = df_small["abstract_en"].tolist()

    with open(out_file, "w", encoding="utf-8") as fOut:
        for start_idx in tqdm(range(0, len(paragraphs), batch_size)):
            sub_paragraphs = paragraphs[start_idx:start_idx+batch_size]
            inputs = t5_tokenizer(sub_paragraphs, max_length=max_length_paragraph,
                                   truncation=True, padding=True, return_tensors="pt").to(device)

            outputs = t5_model.generate(
                **inputs,
                max_length=max_length_query,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_queries
            )

            for idx, out in enumerate(outputs):
                query = t5_tokenizer.decode(out, skip_special_tokens=True)
                query = _removeNonAscii(query)
                para = sub_paragraphs[int(idx/num_queries)]
                para = _removeNonAscii(para)
                fOut.write("{}\t{}\n".format(query.replace("\t", " ").strip(), para.replace("\t", " ").strip()))

    print(f"Synthetic queries saved to {out_file}")
else:
    print(f"{out_file} already exists, skipping generation.")

# Load SBERT model (MSMARCO)
normalize = models.Normalize()
word_embedding_model = models.Transformer("sentence-transformers/msmarco-distilbert-base-dot-prod-v3", max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean")
sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize]).to(device)

# Load synthetic dataset
train_examples = [] 
with open(out_file) as fIn:
    for line in fIn:
        try:
            query, paragraph = line.strip().split('\t', maxsplit=1)
            train_examples.append(InputExample(texts=[query, paragraph]))
        except:
            pass

print("Loaded training pairs:", len(train_examples))

dataset = input_examples_to_dataset(train_examples)

# Fine-tuning
train_loss = losses.MultipleNegativesRankingLoss(sbert_model)
num_epochs = 3

# Training args
training_args = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/search_model_en_checkpoint",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=32,
    logging_steps=100,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none"
)

# Trainer
trainer = SentenceTransformerTrainer(
    model=sbert_model,
    args=training_args,
    train_dataset=dataset,
    loss=train_loss
)

# Train
print("Starting EN training...")
trainer.train()

# Save final model
sbert_model.save("/kaggle/working/search_model_en")
print("Final model saved to /kaggle/working/search_model_en")
shutil.make_archive("search_model_en", 'zip', "search_model_en")