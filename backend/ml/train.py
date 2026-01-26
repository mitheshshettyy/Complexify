import pandas as pd
import pickle

from preprocess import preprocess_code
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# load dataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dataset" / "code_complexity.csv"

df = pd.read_csv(DATASET_PATH)


# preprocess code
df["clean_code"] = df["code"].apply(preprocess_code)

# vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_code"])

# encode labels
time_enc = LabelEncoder()
space_enc = LabelEncoder()

y_time = time_enc.fit_transform(df["time_complexity"])
y_space = space_enc.fit_transform(df["space_complexity"])
y_cyclo = df["cyclomatic"]
y_read = df["readability"]

# models
time_model = RandomForestClassifier(n_estimators=200)
space_model = RandomForestClassifier(n_estimators=200)
cyclo_model = RandomForestRegressor(n_estimators=200)
read_model = RandomForestRegressor(n_estimators=200)

# train
time_model.fit(X, y_time)
space_model.fit(X, y_space)
cyclo_model.fit(X, y_cyclo)
read_model.fit(X, y_read)

# save everything
# save everything to backend/ directory
MODEL_DIR = BASE_DIR / "backend"

pickle.dump(vectorizer, open(MODEL_DIR / "vectorizer.pkl", "wb"))
pickle.dump(time_model, open(MODEL_DIR / "time_model.pkl", "wb"))
pickle.dump(space_model, open(MODEL_DIR / "space_model.pkl", "wb"))
pickle.dump(cyclo_model, open(MODEL_DIR / "cyclo_model.pkl", "wb"))
pickle.dump(read_model, open(MODEL_DIR / "read_model.pkl", "wb"))
pickle.dump(time_enc, open(MODEL_DIR / "time_encoder.pkl", "wb"))
pickle.dump(space_enc, open(MODEL_DIR / "space_encoder.pkl", "wb"))

print("✅ Training complete. Models saved.")
