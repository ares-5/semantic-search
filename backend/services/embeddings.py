from sentence_transformers import SentenceTransformer
import config

model_en = SentenceTransformer(config.EN_MODEL_PATH)
model_sr = SentenceTransformer(config.SR_MODEL_PATH)

def get_embedding(text: str, lang: str = "en"):
    if lang == "en":
        return model_en.encode([text])[0].tolist()
    else:
        return model_sr.encode([text])[0].tolist()