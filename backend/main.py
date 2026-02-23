from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import settings
from backend.ml.models import (
    cyclo_model,
    read_model,
    time_enc,
    time_model,
    vectorizer,
)
from backend.ml.preprocess import preprocess_code

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=settings.app_description,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeInput(BaseModel):
    code: str


class AnalysisResponse(BaseModel):
    time_complexity: str
    space_complexity: str
    cyclomatic_complexity: float
    readability_score: float
    optimization_suggestions: str


@app.get("/")
def root():
    return {"status": "Complexify API is running"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_code(data: CodeInput):
    import numpy as np
    import scipy.sparse as sp

    from backend.ml.features import extract_ast_features

    clean_code = preprocess_code(data.code)
    vector = vectorizer.transform([clean_code])

    ast_feats = extract_ast_features(data.code)
    X_num = np.array(
        [[
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
            len(clean_code),
        ]]
    ) * 10.0

    X_combined = sp.hstack([vector, X_num])
    time_pred = time_enc.inverse_transform(time_model.predict(X_combined))[0]

    loop_depth = ast_feats["loop_depth"]
    recursion_count = ast_feats["recursion_count"]

    if recursion_count == 0:
        if loop_depth >= 3:
            time_pred = "cubic"
        elif loop_depth == 2:
            time_pred = "quadratic"
        elif loop_depth == 1 and time_pred in ["quadratic", "cubic", "np"]:
            time_pred = "linear"
        elif loop_depth == 0:
            time_pred = "constant"
    else:
        if recursion_count >= 2 and loop_depth == 0:
            time_pred = "exponential"
        elif time_pred == "constant":
            time_pred = "linear"

    cyclo_pred = float(round(cyclo_model.predict(X_combined)[0], 2))
    read_pred = float(round(read_model.predict(X_combined)[0], 2))

    return {
        "time_complexity": time_pred,
        "space_complexity": settings.space_complexity_label,
        "cyclomatic_complexity": cyclo_pred,
        "readability_score": read_pred,
        "optimization_suggestions": settings.optimization_suggestions,
    }
