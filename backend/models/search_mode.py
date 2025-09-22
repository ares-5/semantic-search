from enum import Enum

class SearchMode(str, Enum):
    STANDARD = "standard"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"