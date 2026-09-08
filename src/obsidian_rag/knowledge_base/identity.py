"""Stable content identities; independent of embeddings and storage."""

import hashlib
import json
import re

SCHEMA_VERSION = 1
type ConfigValue = None | bool | int | float | str | list[ConfigValue] | dict[str, ConfigValue]

def fingerprint_config(config: dict[str, ConfigValue]) -> str:
    """Hash JSON configuration independent of dictionary insertion order.

    Preserve string contents and list order; reject non-JSON values, non-string
    keys, and non-finite numbers. Include every setting affecting the operation.
    For chunking this includes algorithm revision, budgets, tokenizer identity
    and revision, normalization, and special-token handling. Do not include
    machine paths, timestamps, or batch sizes in a chunking configuration.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object.")
    validate_config(config)
    return digest("config", config)

def digest(kind: str, data: dict) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(
        f"obsidian-rag/{kind}/v{SCHEMA_VERSION}\n{payload}".encode("utf-8")
    ).hexdigest()

def validate_config(value: ConfigValue) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Configuration keys must be strings.")
            validate_config(item)
    elif isinstance(value, list):
        for item in value:
            validate_config(item)
    elif value is not None and type(value) not in (bool, int, float, str):
        raise ValueError("Configuration must contain only JSON values.")

def require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string.")

def require_integer(value: int, name: str, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")

def require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest.")


def require_source_path(source: str) -> None:
    require_text(source, 'source')
    if ('\\' in source or any(part in ('', '.', '..') for part in source.split('/'))
            or re.match(r'^[A-Za-z]:', source)):
        raise ValueError('source must be a canonical vault-relative POSIX path.')
