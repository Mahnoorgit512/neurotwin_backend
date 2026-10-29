import json
import os
import time
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends,BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, firestore,auth
from services.text_service import analyze_text
from services.image_service import analyze_image
from services.audio_service import analyze_audio, transcribe_audio_only
from services.fusion_service import (
    combine_neurotwin_analysis,
    create_neurotwin_summary
)
from services.gemini_service import (
    generate_neurotwin_response,
    build_neurotwin_prompt,
    update_user_memory
)
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import base64
from key_exchange import (
    create_server_keypair,
    derive_session_key,
    decrypt_payload,
    encrypt_payload,
)

# if not firebase_admin._apps:
#     cred = credentials.Certificate(
#         r"C:\secure\firebase-admin.json"
#     )
#     firebase_admin.initialize_app(cred)

if not firebase_admin._apps:
    firebase_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")

    if firebase_json:
        cred = credentials.Certificate(json.loads(firebase_json))
    else:
        cred = credentials.Certificate(
            r"C:\secure\firebase-admin.json"
        )

    firebase_admin.initialize_app(cred)

db = firestore.client()
def save_user_memory(uid: str, memory_text: str):
    db.collection("users").document(uid).collection(
        "memory"
    ).document("profile").set(
        {"memory_text": memory_text},
        merge=True
    )

security = HTTPBearer()
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        decoded_token = auth.verify_id_token(
            credentials.credentials
        )
        return decoded_token["uid"]

    except Exception as e:
        print(
            f"Firebase token verification failed: "
            f"{type(e).__name__}: {e}"
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired Firebase ID token"
        )
from pydantic import BaseModel

class EncryptedRequest(BaseModel):
    client_public_key: str
    encrypted_payload: str

app = FastAPI(
    title="NeuroTwin API"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "NeuroTwin API is running"
    }

@app.post("/predict/text")
def predict_text(text: str,uid: str = Depends(get_current_user)):
    result = analyze_text(text)
    summary = create_neurotwin_summary({
        "text": result,
        "audio": None,
        "image": None
    })
    prompt = build_neurotwin_prompt(
        user_text=text,
        summary=summary
    )
    neurotwin_response = generate_neurotwin_response(
        prompt
    )
    return {
        "analysis": summary,
        "response": neurotwin_response
    }


@app.get("/secure/public-key")
def secure_public_key():
    _, public_key = create_server_keypair()
    return {"server_public_key": public_key}


@app.post("/secure/chat")
async def secure_chat(
    request: EncryptedRequest,
    background_tasks: BackgroundTasks
):
    temp_paths = []
    request_started = time.perf_counter()
    print("\n========== SECURE CHAT STARTED ==========")

    try:
        # 1. Derive the session key and decrypt the incoming request.
        stage_started = time.perf_counter()
        private_key, server_public_key = create_server_keypair()
        session_key = derive_session_key(
            private_key,
            request.client_public_key
        )

        plaintext = decrypt_payload(
            session_key,
            request.encrypted_payload
        )
        payload = json.loads(plaintext.decode("utf-8"))
        print(f"DECRYPTION TIME: {time.perf_counter() - stage_started:.2f} seconds")

        # 2. Verify the Firebase user.
        firebase_token = payload.get("firebase_token")
        if not isinstance(firebase_token, str) or not firebase_token:
            raise HTTPException(status_code=401, detail="Missing Firebase token")

        stage_started = time.perf_counter()
        decoded_token = auth.verify_id_token(firebase_token)
        uid = decoded_token["uid"]
        print(f"AUTH TIME: {time.perf_counter() - stage_started:.2f} seconds")
        

        # 3. Read the decrypted text and conversation history.
        text = payload.get("text")
        if text is not None and not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Invalid text")

        history = payload.get("history", [])
        if isinstance(history, str):
            history = json.loads(history) if history.strip() else []
        if not isinstance(history, list):
            history = []

        user_name = payload.get("user_name")

        text_result = None
        audio_result = None
        image_result = None

        if text and text.strip():
            stage_started = time.perf_counter()
            text_result = analyze_text(text.strip())
            print(f"TEXT ANALYSIS TIME: {time.perf_counter() - stage_started:.2f} seconds")

        # 4. Decode encrypted-in-transit audio bytes into a temporary file.
        audio_b64 = payload.get("audio_base64")
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64, validate=True)
            audio_name = os.path.basename(
                str(payload.get("audio_filename") or "recording.wav")
            )
            audio_suffix = os.path.splitext(audio_name)[1].lower()
            if audio_suffix not in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".webm"}:
                audio_suffix = ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=audio_suffix
            ) as temp_file:
                temp_file.write(audio_bytes)
                audio_path = temp_file.name
                temp_paths.append(audio_path)

            stage_started = time.perf_counter()
            audio_result = analyze_audio(audio_path)
            print(f"AUDIO ANALYSIS TIME: {time.perf_counter() - stage_started:.2f} seconds")
        # 5. Decode image bytes into a temporary file.
        image_b64 = payload.get("image_base64")
        if image_b64:
            image_bytes = base64.b64decode(image_b64, validate=True)
            image_name = os.path.basename(
                str(payload.get("image_filename") or "photo.jpg")
            )
            image_suffix = os.path.splitext(image_name)[1].lower()
            if image_suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                image_suffix = ".jpg"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=image_suffix
            ) as temp_file:
                temp_file.write(image_bytes)
                image_path = temp_file.name
                temp_paths.append(image_path)

            stage_started = time.perf_counter()
            image_result = analyze_image(image_path)
            print(f"IMAGE ANALYSIS TIME: {time.perf_counter() - stage_started:.2f} seconds")

        if not any([text_result, audio_result, image_result]):
            raise HTTPException(
                status_code=400,
                detail="Provide text, audio, or image"
            )

        # 6. Run your existing multimodal fusion.
        combined_result = combine_neurotwin_analysis(
            text_result=text_result,
            audio_result=audio_result,
            image_result=image_result
        )
        
        stage_started = time.perf_counter()
        memory_doc = (
            db.collection("users")
            .document(uid)
            .collection("memory")
            .document("profile")
            .get()
        )
        memory_data = memory_doc.to_dict() or {}
        persistent_memory = memory_data.get("memory_text", "")
        print(f"FIRESTORE MEMORY READ TIME: {time.perf_counter() - stage_started:.2f} seconds")

        summary = create_neurotwin_summary(
            combined_result,
            conversation_history=history
        )

        gemini_text = text
        if (
            not gemini_text
            and audio_result
            and audio_result.get("audio_transcription")
        ):
            gemini_text = audio_result["audio_transcription"]

        prompt = build_neurotwin_prompt(
            user_text=gemini_text,
            summary=summary,
            user_name=user_name,
            history=history,
            persistent_memory=persistent_memory
        )

        neurotwin_response = generate_neurotwin_response(prompt)
        
        
        # 7. Update persistent memory in the background.
        def background_memory_update():
            try:
                updated_memory = update_user_memory(
                    existing_memory=persistent_memory,
                    user_message=gemini_text,
                    assistant_response=neurotwin_response
                )

                if (
                    updated_memory
                    and not updated_memory.startswith("I'm having trouble")
                ):
                    save_user_memory(
                        uid,
                        updated_memory.strip()[:3000]
                    )

            except Exception as memory_error:
                print(
                    f"Secure chat memory update failed: "
                    f"{type(memory_error).__name__}: {memory_error}"
                )

        background_tasks.add_task(background_memory_update)



        # 8. Encrypt the same response fields your Flutter chat expects.
        response_data = {
            "analysis": summary,
            "response": neurotwin_response
        }

        stage_started = time.perf_counter()
        encrypted_response = encrypt_payload(
            session_key,
            json.dumps(response_data).encode("utf-8")
        )
        print(f"ENCRYPTION TIME: {time.perf_counter() - stage_started:.2f} seconds")

        print("SECURITY: Secure multimodal request processed; response encrypted")
        print(f"TOTAL SECURE CHAT TIME: {time.perf_counter() - request_started:.2f} seconds")
        print("========== SECURE CHAT FINISHED ==========")

        return {
            "server_public_key": server_public_key,
            "encrypted_payload": encrypted_response
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Secure chat failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Secure request failed"
        )
    finally:
        for path in temp_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),uid: str = Depends(get_current_user)
):
    image_path = f"temp_{file.filename}"

    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())
    try:
        result = analyze_image(image_path)
        return result
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@app.post("/predict/audio")
async def predict_audio(
    file: UploadFile = File(...), uid: str = Depends(get_current_user)
):

    audio_path = f"temp_{file.filename}"

    with open(audio_path, "wb") as buffer:
        buffer.write(await file.read())
    try:
        result = analyze_audio(audio_path)
        return result
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/transcribe",include_in_schema=False)
async def transcribe(audio: UploadFile = File(...),uid: str = Depends(get_current_user)):
    suffix = os.path.splitext(audio.filename or "recording.wav")[1]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(await audio.read())

        return transcribe_audio_only(temp_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict/fusion")
async def predict_fusion(
    request: Request,
    uid: str = Depends(get_current_user),
    text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    image: UploadFile | None = File(None)
):
    request_started = time.perf_counter()
    print("\n========== FUSION REQUEST STARTED ==========")
    form_data = await request.form()

    user_name = form_data.get("user_name")
    history = form_data.get("history")

    text_result = None
    audio_result = None
    image_result = None
    if text and text.strip():
        stage_started = time.perf_counter()
        text_result = analyze_text(
            text.strip()
        )
        print( f"⏱️ TEXT ANALYSIS TIME: " f"{time.perf_counter() - stage_started:.2f} seconds"
    )
    if audio:
        audio_path = f"temp_{audio.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await audio.read())
        try:
            audio_result = analyze_audio(
                audio_path
            )
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    if image:
        image_path = f"temp_{image.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(await image.read())
        try:
            image_result = analyze_image(
                image_path
            )
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

    combined_result = combine_neurotwin_analysis(
        text_result=text_result,
        audio_result=audio_result,
        image_result=image_result
    )

    conversation_history = []
    if history:
            try:
                conversation_history = json.loads(history)
                print("HISTORY DATA:", conversation_history)
            except json.JSONDecodeError:
                conversation_history = []
    
    print("📚 RECEIVED HISTORY:", conversation_history)
    print("📚 HISTORY COUNT:", len(conversation_history))
        
    memory_doc = (
        db.collection("users")
        .document(uid)
        .collection("memory")
        .document("profile")
        .get()
    )

    memory_data = memory_doc.to_dict() or {}
    persistent_memory = memory_data.get("memory_text", "")

    summary = create_neurotwin_summary(
    combined_result,
    conversation_history=conversation_history
)
    gemini_text = text
    if (
        not gemini_text
        and audio_result
        and audio_result.get("audio_transcription")
    ):
        gemini_text = audio_result[
            "audio_transcription"
        ]
    prompt = build_neurotwin_prompt(
        user_text=gemini_text,
        summary=summary,
        user_name=user_name,
        history=conversation_history,
        persistent_memory=persistent_memory
    )
    stage_started = time.perf_counter()
    stage_started = time.perf_counter()
    neurotwin_response = generate_neurotwin_response(prompt)
    print(f"MAIN GEMINI TIME: {time.perf_counter() - stage_started:.2f} seconds")
    print(
    f"⏱️ GEMINI CHAT RESPONSE TIME: "
    f"{time.perf_counter() - stage_started:.2f} seconds"
    )
    try:
       stage_started = time.perf_counter()
       memory_started = time.perf_counter()
       updated_memory = update_user_memory(
        existing_memory=persistent_memory,
        user_message=gemini_text,
        assistant_response=neurotwin_response
    )
       print(f"BACKGROUND MEMORY GEMINI TIME: " f"{time.perf_counter() - memory_started:.2f} seconds" )
       print(f"⏱️ MEMORY UPDATE TIME: " f"{time.perf_counter() - stage_started:.2f} seconds" )
       if updated_memory and not updated_memory.startswith("I'm having trouble"):
         save_user_memory(uid, updated_memory.strip()[:3000])
    except Exception as e:
     print(f"Memory update failed: {e}")
    print(
    f"⏱️ TOTAL FUSION TIME: "
    f"{time.perf_counter() - request_started:.2f} seconds")
    print("========== FUSION REQUEST FINISHED ==========\n") 
    return {
    "analysis": summary,
    "response": neurotwin_response
    }
