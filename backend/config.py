import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _parse_origins(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return ["*"]
    if value == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_title: str = os.getenv("APP_TITLE", "Complexify")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    app_description: str = os.getenv(
        "APP_DESCRIPTION",
        "AI-powered Code Complexity Analyzer using ML + NLP",
    )
    cors_origins: List[str] = field(default_factory=lambda: _parse_origins(os.getenv("CORS_ORIGINS", "*")))
    space_complexity_label: str = os.getenv("SPACE_COMPLEXITY_LABEL", "Unknown")
    optimization_suggestions: str = os.getenv(
        "OPTIMIZATION_SUGGESTIONS",
        "Reduce nested loops, avoid redundant computations, and prefer efficient data structures.",
    )


settings = Settings()
