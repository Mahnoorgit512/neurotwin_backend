
import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def get_aes():
    key_text = os.getenv("NEUROTWIN_AES_KEY")

    if not key_text:
        raise RuntimeError(
            "NEUROTWIN_AES_KEY is not configured."
        )

    try:
        key = base64.b64decode(key_text, validate=True)
    except Exception as exc:
        raise RuntimeError(
            "NEUROTWIN_AES_KEY must be valid Base64."
        ) from exc

    if len(key) != 32:
        raise RuntimeError(
            "AES-256 requires a 32-byte key."
        )

    return AESGCM(key)


def encrypt_text(plaintext: str) -> str:
    aes = get_aes()
    nonce = os.urandom(12)

    ciphertext = aes.encrypt(
        nonce,
        plaintext.encode("utf-8"),
        None,
    )

    # Store nonce and ciphertext together.
    return base64.b64encode(
        nonce + ciphertext
    ).decode("utf-8")


def decrypt_text(encrypted_text: str) -> str:
    aes = get_aes()
    data = base64.b64decode(
        encrypted_text,
        validate=True,
    )

    if len(data) < 28:
        raise ValueError("Invalid encrypted data.")

    nonce = data[:12]
    ciphertext = data[12:]

    return aes.decrypt(
        nonce,
        ciphertext,
        None,
    ).decode("utf-8")
