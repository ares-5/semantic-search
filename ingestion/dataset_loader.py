# https://www.kaggle.com/datasets/aaditshukla/flipkart-fasion-products-dataset?resource=download

import kagglehub
from kagglehub import KaggleDatasetAdapter
from sentence_transformers import models, SentenceTransformer

# Models initialization
transformer_en = models.Transformer("sentence-transformers/all-MiniLM-L6-v2", max_seq_length=256)
pooling_en = models.Pooling(transformer_en.get_word_embedding_dimension(), pooling_mode="mean")

transformer_sr = models.Transformer("DeepPavlov/bert-base-multilingual-cased-sentence", max_seq_length=256)
pooling_sr = models.Pooling(transformer_sr.get_word_embedding_dimension(), pooling_mode="mean")

# normalizer
normalize = models.Normalize()

model_en = SentenceTransformer([transformer_en, pooling_en, normalize])
model_sr = SentenceTransformer([transformer_sr, pooling_sr, normalize])

# Dataset preparation for training

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "aaditshukla/flipkart-fasion-products-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

embeddings = model_en.encode(sentences)

print("First 5 records:", df.head())