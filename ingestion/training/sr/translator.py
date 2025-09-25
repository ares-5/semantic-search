import pandas as pd
import time
from deep_translator import GoogleTranslator
import cyrtranslit

class Translator:
    def __init__(self, src="en", dest="sr", retries=3, latin=True):
        self.src = src
        self.dest = dest
        self.retries = retries
        self.failed_rows = []
        self.latin = latin

    def safe_translate(self, text, row_index=None):
        if pd.isna(text) or not str(text).strip():
            return ""

        for i in range(self.retries):
            try:
                translated = GoogleTranslator(source=self.src, target=self.dest).translate(text)
                if self.latin:
                    translated = cyrtranslit.to_latin(translated)
                return translated
            except Exception as e:
                print(f"[Row {row_index}] Translation failed ({i+1}/{self.retries}): {e}")
                time.sleep(2)

        self.failed_rows.append(row_index)
        return text
