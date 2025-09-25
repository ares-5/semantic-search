
from sentence_transformers import InputExample

out_file_sr = "generated_queries_all_sr.tsv"

# Load synthetic dataset
train_examples = [] 
with open(out_file_sr) as fIn:
    for line in fIn:
        try:
            query, paragraph = line.strip().split('\t', maxsplit=1)
            train_examples.append(InputExample(texts=[query, paragraph]))
        except:
            pass

print("Loaded training pairs:", len(train_examples))