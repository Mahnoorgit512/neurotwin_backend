def combine_neurotwin_analysis(
    text_result=None,
    audio_result=None,
    image_result=None
):
    return {
        "text": text_result,
        "audio": audio_result,
        "image": image_result
    }

def get_wellbeing_scores(text_result):
    if not text_result:
        return {
            "anxiety": None,
            "depression": None,
            "stress": None,
            "normal": None,
            "available_labels": [],
        }

    probabilities = text_result.get("probabilities") or {}

    labels = {
        str(label).strip().lower(): score
        for label, score in probabilities.items()
    }

    def find_score(*names):
        for name in names:
            if name in labels:
                return labels[name]
        return None

    return {
        "anxiety": find_score("anxiety"),
        "depression": find_score("depression"),
        "stress": find_score("stress"),
        "normal": find_score("normal", "neutral"),
        "available_labels": list(probabilities.keys()),
    }


def get_previous_user_messages(conversation_history):
    messages = []

    for item in conversation_history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role", item.get("sender", ""))

        if str(role).lower() in ("user", "human"):
            message = item.get("text", item.get("message", ""))

            if isinstance(message, str) and message.strip():
                messages.append(message.strip())

    return messages


def create_neurotwin_summary(
    combined_result,
    conversation_history=None
):
    summary = {}
    conversation_history = conversation_history or []
    if combined_result.get("text"):
        text_data = combined_result["text"]
        probabilities = text_data.get(
            "probabilities",
            {}
        )
        summary["text_analysis"] = {
            "prediction": text_data.get(
                "prediction",
                ""
            ),
            "confidence": max(
                probabilities.values()
            ) if probabilities else 0,

            "probabilities": probabilities,

            "shap_explanation": text_data.get(
                "shap_explanation",
                []
            )
        }

    
    if combined_result.get("text"):
        summary["psychological_scores"] = get_wellbeing_scores(
            combined_result["text"]
        )
    if combined_result.get("audio"):
        audio_data = combined_result["audio"]
        probabilities = audio_data.get(
            "probabilities",
            {}
        )
        summary["audio_analysis"] = {
            "transcription": audio_data.get(
                "audio_transcription",
                ""
            ),
            "prediction": audio_data.get(
                "prediction",
                ""
            ),
            "confidence": max(
                probabilities.values()
            ) if probabilities else 0,
            "probabilities": probabilities,
            "shap_explanation": audio_data.get(
                "shap_explanation",
                []
            )
        }

    if combined_result.get("image"):
        image_data = combined_result["image"]
        probabilities = image_data.get(
            "probabilities",
            {}
        )
        summary["image_analysis"] = {
            "emotion": image_data.get(
                "prediction",
                ""
            ),
            "confidence": max(
                probabilities.values()
            ) if probabilities else 0,

            "probabilities": probabilities
        }

    # Dominant facial emotion
    if combined_result.get("image"):

        image_data = combined_result["image"]

        probabilities = image_data.get(
            "probabilities",
            {}
        )

        if probabilities:

            dominant_emotion = max(
                probabilities,
                key=probabilities.get
            )
            summary["dominant_emotion"] = {
                "label": dominant_emotion,
                "confidence": probabilities[
                    dominant_emotion
                ]
            }

    summary["signal_sources"] = {
        "text": combined_result.get("text") is not None,
        "image": combined_result.get("image") is not None,
        "audio": combined_result.get("audio") is not None,
        "conversation_history": bool(
            conversation_history
        )
    }
    observations = []
    if combined_result.get("text"):
        prediction = combined_result["text"].get(
            "prediction",
            ""
        )
        if prediction:
            observations.append(
                f"Text prediction: {prediction}."
            )

    if combined_result.get("image"):
        emotion = combined_result["image"].get(
            "prediction",
            ""
        )
        if emotion:
            observations.append(
                f"Facial emotion: {emotion}."
            )

    if combined_result.get("audio"):

        prediction = combined_result["audio"].get(
            "prediction",
            ""
        )

        if prediction:
            observations.append(
                f"Audio prediction: {prediction}."
            )

    summary["multimodal_summary"] = " ".join(
        observations
    )
    return summary