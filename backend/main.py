from fastapi import FastAPI
from pydantic import BaseModel
from backend.ml.preprocess import preprocess_code
from backend.ml.models import (
    vectorizer,
    time_model,
    cyclo_model,
    read_model,
    time_enc
)

app = FastAPI(
    title="Complexify",
    version="0.1.0",
    description="AI-powered Code Complexity Analyzer using ML + NLP"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request Schema ----------
class CodeInput(BaseModel):
    code: str


# ---------- Response Schema ----------
class AnalysisResponse(BaseModel):
    time_complexity: str
    space_complexity: str
    cyclomatic_complexity: float
    readability_score: float
    optimization_suggestions: str


# ---------- Root Route ----------
@app.get("/")
def root():
    return {"status": "Complexify API is running"}


# ---------- Analyze Endpoint ----------
@app.post("/analyze", response_model=AnalysisResponse)
def analyze_code(data: CodeInput):
    import scipy.sparse as sp
    from backend.ml.features import extract_ast_features
    # preprocess input code
    clean_code = preprocess_code(data.code)
    vector = vectorizer.transform([clean_code])
    
    # AST features
    ast_feats = extract_ast_features(data.code)
    import numpy as np
    X_num = np.array([[
        ast_feats["num_for_loops"],
        ast_feats["num_while_loops"],
        ast_feats["num_if_statements"],
        ast_feats["num_functions"],
        ast_feats["num_assignments"],
        ast_feats["num_binary_ops"],
        ast_feats["num_returns"],
        ast_feats["loop_depth"],
        ast_feats["recursion_count"],
        ast_feats["control_flow_count"],
        len(clean_code)
    ]]) * 10.0
    
    # Combine features
    X_combined = sp.hstack([vector, X_num])

    # predictions
    time_pred = time_enc.inverse_transform(time_model.predict(X_combined))[0]
    
    # Heuristic override for obvious control-flow structures (handles ML noise)
    loop_depth = ast_feats["loop_depth"]
    recursion_count = ast_feats["recursion_count"]

    if recursion_count == 0:
        # Non-recursive code should not exceed loop nesting order for common cases.
        if loop_depth >= 3:
            time_pred = "cubic"
        elif loop_depth == 2:
            time_pred = "quadratic"
        elif loop_depth == 1 and time_pred in ["quadratic", "cubic", "np"]:
            time_pred = "linear"
        elif loop_depth == 0:
            time_pred = "constant"
    else:
        # Multiple self-calls in recursion (e.g., naive Fibonacci) grows exponentially.
        if recursion_count >= 2 and loop_depth == 0:
            time_pred = "exponential"
        elif time_pred == "constant":
            time_pred = "linear"

    space_pred = "Unknown" 
    cyclo_pred = float(round(cyclo_model.predict(X_combined)[0], 2))
    read_pred = float(round(read_model.predict(X_combined)[0], 2))

    return {
        "time_complexity": time_pred,
        "space_complexity": space_pred,
        "cyclomatic_complexity": cyclo_pred,
        "readability_score": read_pred,
        "optimization_suggestions": (
            "Reduce nested loops, avoid redundant computations, "
            "and prefer efficient data structures."
        )
    }
