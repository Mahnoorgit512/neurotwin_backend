
import os
import base64
from dotenv import load_dotenv
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

load_dotenv()

def create_server_keypair():
    """Load the persistent server key pair from environment variables."""
    private_text = os.getenv("NEUROTWIN_SERVER_PRIVATE_KEY")
    public_text = os.getenv("NEUROTWIN_SERVER_PUBLIC_KEY")

    if not private_text or not public_text:
        raise RuntimeError(
            "NeuroTwin server keys are not configured."
        )

    private_key = X25519PrivateKey.from_private_bytes(
        base64.b64decode(private_text, validate=True)
    )

    actual_public = base64.b64encode(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode("utf-8")

    if actual_public != public_text:
        raise RuntimeError(
            "NeuroTwin server public and private keys do not match."
        )

    return private_key, actual_public


def derive_session_key(private_key, client_public_key_b64: str):
    """Derive a session key from the client's public key."""
    client_public_bytes = base64.b64decode(
        client_public_key_b64,
        validate=True,
    )

    client_public_key = X25519PublicKey.from_public_bytes(
        client_public_bytes
    )

    shared_secret = private_key.exchange(client_public_key)

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"NeuroTwin-session-v1",
    ).derive(shared_secret)


def encrypt_payload(session_key: bytes, plaintext: bytes) -> str:
    """Encrypt bytes with AES-256-GCM."""
    nonce = os.urandom(12)
    encrypted = AESGCM(session_key).encrypt(
        nonce, plaintext, None
    )

    return base64.b64encode(
        nonce + encrypted
    ).decode("utf-8")


def decrypt_payload(session_key: bytes, encrypted_b64: str) -> bytes:
    """Decrypt bytes with AES-256-GCM."""
    data = base64.b64decode(encrypted_b64, validate=True)

    if len(data) < 28:
        raise ValueError("Invalid encrypted payload.")

    return AESGCM(session_key).decrypt(
        data[:12], data[12:], None
    )
