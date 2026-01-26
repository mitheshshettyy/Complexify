from fastapi import FastAPI
from pydantic import BaseModel
from backend.ml.preprocess import preprocess_code
from backend.ml.models import (
    vectorizer,
    time_model,
    space_model,
    cyclo_model,
    read_model,
    time_enc,
    space_enc
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
    # preprocess input code
    clean_code = preprocess_code(data.code)
    vector = vectorizer.transform([clean_code])

    # predictions
    time_pred = time_enc.inverse_transform(time_model.predict(vector))[0]
    space_pred = space_enc.inverse_transform(space_model.predict(vector))[0]
    cyclo_pred = float(round(cyclo_model.predict(vector)[0], 2))
    read_pred = float(round(read_model.predict(vector)[0], 2))

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
