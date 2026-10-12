import hashlib
import json


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, normalize_eol: bool = False, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file. With normalize_eol=True, CRLF is converted to LF first
    (so a Windows checkout of a text file hashes the same as on Linux)."""
    if normalize_eol:
        with open(path, "rb") as f:
            return sha256_bytes(f.read().replace(b"\r\n", b"\n"))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json_hash(obj) -> str:
    """Hash of a JSON-serializable object, independent of key order."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
