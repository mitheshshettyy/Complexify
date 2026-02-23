import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

vectorizer = pickle.load(open(BASE_DIR / "vectorizer.pkl", "rb"))
time_model = pickle.load(open(BASE_DIR / "time_model.pkl", "rb"))
cyclo_model = pickle.load(open(BASE_DIR / "cyclo_model.pkl", "rb"))
read_model = pickle.load(open(BASE_DIR / "read_model.pkl", "rb"))
time_enc = pickle.load(open(BASE_DIR / "time_encoder.pkl", "rb"))
