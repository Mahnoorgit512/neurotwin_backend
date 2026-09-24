from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_PATH = "./model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "NeuroTwin text API is running"}

@app.post("/predict")
def predict(request: TextRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    prediction = torch.argmax(probabilities).item()

    result = {
        model.config.id2label[i]: round(
            probabilities[i].item() * 100, 2
        )
        for i in range(len(probabilities))
    }

    return {
        "prediction": model.config.id2label[prediction],
        "probabilities": result
    }