import torch
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load fine-tuned model
sbert_model = SentenceTransformer("/kaggle/input/final_en_model/transformers/default/1").to(device)

# Load synthetic dataset
out_file = "/kaggle/input/queries/generated_queries_all (3).tsv"
queries, passages, correct_idx = [], [], []

with open(out_file, "r", encoding="utf-8") as fIn:
    for idx, line in enumerate(fIn):
        try:
            query, passage = line.strip().split("\t", maxsplit=1)
            queries.append(query)
            passages.append(passage)
            correct_idx.append(idx)
        except:
            continue

def encode_texts(
        model,
        texts,
        lang='en',
        max_len=512,
        batch_size=32,
        device='cpu', 
        nlp_sr=None):
    """
    Encode a list of texts into embeddings using sentence-level tokenization and pooling.
    - lang: 'en' or 'sr'
    - max_len: max tokens per block (model limit)
    - batch_size: number of texts to process per iteration
    """
    embeddings = []

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx+batch_size]
        batch_embs = []

        for text in batch_texts:
            # Sentence-level tokenization
            if lang == 'en':
                sentences = sent_tokenize(text)
            elif lang == 'sr':
                if nlp_sr is None:
                    raise ValueError("nlp_sr pipeline must be provided for Serbian texts")
                doc = nlp_sr(text)
                sentences = [sent.text for sent in doc.sentences]
            else:
                raise ValueError("lang must be 'en' or 'sr'")

            # Group sentences into blocks <= max_len tokens
            blocks = []
            current_block = ""
            for sentence in sentences:
                # Tokenize current block + new sentence
                tokens = model.tokenize((current_block + " " + sentence).strip())
                if len(tokens) <= max_len:
                    current_block = (current_block + " " + sentence).strip()
                else:
                    if current_block:
                        blocks.append(current_block)
                    current_block = sentence
            if current_block:
                blocks.append(current_block)

            # Encode each block
            block_embs = []
            for block in blocks:
                emb = model.encode([block], convert_to_tensor=True, device=device)
                block_embs.append(emb)

            # Pooling: average block embeddings
            text_emb = torch.mean(torch.stack(block_embs), dim=0)
            batch_embs.append(text_emb)

        # Stack batch embeddings
        batch_embs = torch.stack(batch_embs)
        embeddings.append(batch_embs)

    return torch.cat(embeddings, dim=0)

# Compute Recall@k and MRR
def compute_metrics(model, queries, passages, correct_idx, top_k=[1,5,10], batch_size=32):
    query_embeddings = encode_texts(model,
        texts=queries,
        lang = 'en',
        max_len = 512,
        batch_size = 32,
        device = device,
    )
    passage_embeddings = encode_texts(model,
        texts=passages,
        lang = 'en',
        max_len = 512,
        batch_size = 32,
        device = device,
    )

    cos_sim_matrix = util.cos_sim(query_embeddings, passage_embeddings)
    recalls = {k: 0 for k in top_k}
    mrr_total = 0.0

    for i in range(len(queries)):
        sims = cos_sim_matrix[i]
        sorted_idx = torch.argsort(sims, descending=True)
        correct_rank = (sorted_idx == correct_idx[i]).nonzero(as_tuple=True)[0].item() + 1
        for k in top_k:
            if correct_rank <= k:
                recalls[k] += 1
        mrr_total += 1.0 / correct_rank

    num_queries = len(queries)
    recalls = {k: v / num_queries for k, v in recalls.items()}
    mrr = mrr_total / num_queries
    return recalls, mrr

# Run evaluation
top_k_values = [1,5,10]
recalls, mrr = compute_metrics(sbert_model, queries, passages, correct_idx, top_k=top_k_values)

# Save final results to file
output_file = "semantic_search_metrics.txt"
with open(output_file, "w", encoding="utf-8") as fOut:
    fOut.write("=== Semantic Search Evaluation ===\n")
    fOut.write(f"Recall@1:  {recalls[1]:.4f}\n")
    fOut.write(f"Recall@5:  {recalls[5]:.4f}\n")
    fOut.write(f"Recall@10: {recalls[10]:.4f}\n")
    fOut.write(f"MRR:       {mrr:.4f}\n")

print(f"Final metrics written to {output_file}")
