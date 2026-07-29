import hashlib
from pathlib import Path


def path_order_key(relative_path: Path | str) -> str:
    value = Path(relative_path).as_posix().encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
