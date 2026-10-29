import whisper
from services.text_service import analyze_text
whisper_model = whisper.load_model(
    "base",
    download_root="models/whisper"
)

def analyze_audio(audio_path):
    transcription = whisper_model.transcribe(
        audio_path
    )["text"].strip()

    text_result = analyze_text(
        transcription
    )
    return {
        "audio_transcription": transcription,
        "prediction": text_result["prediction"],
        "probabilities": text_result["probabilities"],
        "shap_explanation": text_result.get(
            "shap_explanation",
            []
        )
    }

def transcribe_audio_only(audio_path):
    transcription = whisper_model.transcribe(audio_path)["text"].strip()
    return {"transcription": transcription}    