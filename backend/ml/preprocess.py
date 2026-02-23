import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)

def preprocess_code(code: str) -> str:
    # remove comments
    code = re.sub(r"//.*|#.*", "", code)

    # lowercase
    code = code.lower()

    # remove symbols
    code = re.sub(r"[^a-z0-9_ ]", " ", code)

    # tokenize
    tokens = code.split()

    # remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS]

    return " ".join(tokens)
