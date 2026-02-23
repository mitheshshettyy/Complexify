import pandas as pd
import pickle
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from backend.ml.preprocess import preprocess_code
from backend.ml.features import extract_ast_features
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pathlib import Path

# load dataset
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dataset" / "python_data.jsonl"

print("Loading dataset...")
df = pd.read_json(DATASET_PATH, lines=True)

# Rename columns to match expected format
df.rename(columns={"src": "code", "complexity": "time_complexity"}, inplace=True)

# Generate missing labels using Radon
print("Generating complexity metrics...")
def get_metrics(code):
    try:
        # Cyclomatic Complexity (average of all blocks)
        blocks = cc_visit(code)
        if blocks:
            avg_cc = sum(block.complexity for block in blocks) / len(blocks)
        else:
            avg_cc = 1.0 # Default if no blocks found
            
        # Maintainability Index (Readability)
        mi = mi_visit(code, multi=True)
        
        return avg_cc, mi
    except Exception:
        return 1.0, 50.0 # Default fallback

# Apply metrics generation
metrics = df["code"].apply(lambda x: pd.Series(get_metrics(x)))
df["cyclomatic"] = metrics[0]
df["readability"] = metrics[1]

print("Extracting AST features...")
ast_features = df["code"].apply(lambda x: pd.Series(extract_ast_features(x)))
df["loop_depth"] = ast_features["loop_depth"]
df["recursion_count"] = ast_features["recursion_count"]
df["control_flow_count"] = ast_features["control_flow_count"]

# preprocess code
print("Preprocessing code...")
df["clean_code"] = df["code"].apply(preprocess_code)

# vectorization
print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df["clean_code"])

print("Combining features...")
# AST numerical features
X_num = df[["loop_depth", "recursion_count", "control_flow_count"]].values
X_combined = sp.hstack([X_text, X_num])

# encode labels
time_enc = LabelEncoder()

y_time = time_enc.fit_transform(df["time_complexity"])
y_cyclo = df["cyclomatic"]
y_read = df["readability"]

# models
print("Training models...")
time_model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
cyclo_model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
read_model = RandomForestRegressor(n_estimators=200, n_jobs=-1)

# train
time_model.fit(X_combined, y_time)
cyclo_model.fit(X_combined, y_cyclo)
read_model.fit(X_combined, y_read)

# save everything
print("Saving models...")
MODEL_DIR = BASE_DIR / "backend"

pickle.dump(vectorizer, open(MODEL_DIR / "vectorizer.pkl", "wb"))
pickle.dump(time_model, open(MODEL_DIR / "time_model.pkl", "wb"))
pickle.dump(cyclo_model, open(MODEL_DIR / "cyclo_model.pkl", "wb"))
pickle.dump(read_model, open(MODEL_DIR / "read_model.pkl", "wb"))
pickle.dump(time_enc, open(MODEL_DIR / "time_encoder.pkl", "wb"))

print("✅ Training complete. Models saved.")
