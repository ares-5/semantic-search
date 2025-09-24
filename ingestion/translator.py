from deep_translator import GoogleTranslator
import pandas as pd
import time

class Translator:
    def __init__(self, src="en", dest="sr", retries=3):
        self.src = src
        self.dest = dest
        self.retries = retries
        self.failed_rows = []

    def safe_translate(self, text, row_index=None):
        if pd.isna(text) or not str(text).strip():
            return ""

        for i in range(self.retries):
            try:
                return GoogleTranslator(source=self.src, target=self.dest).translate(text)
            except Exception as e:
                print(f"[Row {row_index}] Translation failed ({i+1}/{self.retries}): {e}")
                time.sleep(2)

        self.failed_rows.append(row_index)
        return text
