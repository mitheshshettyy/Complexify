import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

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
