from pydantic import BaseModel, field_validator
from typing import Literal


class PredictRequest(BaseModel):
    text: str
    model: Literal["bert", "distilbert"] = "bert"

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_used: str
