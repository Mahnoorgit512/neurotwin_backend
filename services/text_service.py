import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
MODEL_PATH = "text_model"
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)
model.to(device)
model.eval()
def shap_predict(texts):

    results = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        results.append(
            probabilities[0].cpu().numpy()
        )

    import numpy as np

    return np.array(results)

explainer = shap.Explainer(
    shap_predict,
    tokenizer
)
labels = [
    model.config.id2label[i]
    for i in range(model.config.num_labels)
]

def extract_shap_explanation(shap_values, prediction_id):
    tokens = shap_values.data[0]
    values = shap_values.values[0, :, prediction_id]
    explanation = []
    for token, value in zip(tokens, values):
        token = token.strip()
        if not token:
            continue
        explanation.append({
            "token": token,
            "importance": round(float(value), 4)
        })

    explanation.sort(
        key=lambda x: abs(x["importance"]),
        reverse=True
    )

    return explanation[:10]

def analyze_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    prediction_id = torch.argmax(
        probabilities
    ).item()

    shap_values = explainer(
    [text],
    max_evals=100
   )
    shap_explanation = extract_shap_explanation(
    shap_values,
    prediction_id
)
    return {
        "prediction": labels[prediction_id],
        "probabilities": {
            labels[i]: round(
                probabilities[i].item() * 100,
                2
            )
            for i in range(len(labels))
        },
        "shap_explanation": shap_explanation
        }