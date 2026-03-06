"""
encryption.py
=============
AES-256 field-level encryption helpers for CIP compliance.
"""

import os
import base64


def generate_key() -> bytes:
    """Generate a new Fernet encryption key. Store securely — never commit."""
    try:
        from cryptography.fernet import Fernet

        return Fernet.generate_key()
    except ImportError:
        return base64.urlsafe_b64encode(os.urandom(32))


def load_key_from_env(env_var: str = "PAAI_ENCRYPTION_KEY") -> bytes | None:
    """Load encryption key from environment variable."""
    val = os.environ.get(env_var)
    return val.encode() if val else None


def encrypt_field(value: str, key: bytes) -> str:
    from cryptography.fernet import Fernet

    return Fernet(key).encrypt(value.encode()).decode()


def decrypt_field(ciphertext: str, key: bytes) -> str:
    from cryptography.fernet import Fernet

    return Fernet(key).decrypt(ciphertext.encode()).decode()
