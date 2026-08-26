#!/usr/bin/env python3
"""Stream only the ImageNet images required by the diverse-10 experiment."""

import argparse
import http.cookiejar
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from common import DEFAULT_CONFIG, class_specs, load_experiment

DEFAULT_DATA_ROOT = Path("data/imagenet_diverse10_minimal")
DEFAULT_URL_TEMPLATE = "https://image-net.org/data/winter21_whole/{wnid}.tar"


def image_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpeg", ".jpg", ".png"}
            and path.stat().st_size > 0
        )
    )


def opener(cookie_file: Path | None) -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = []
    if cookie_file is not None:
        if not cookie_file.is_file():
            raise FileNotFoundError(f"Cookie file does not exist: {cookie_file}")
        jar = http.cookiejar.MozillaCookieJar(str(cookie_file))
        jar.load(ignore_discard=True, ignore_expires=True)
        handlers.append(urllib.request.HTTPCookieProcessor(jar))
    return urllib.request.build_opener(*handlers)


def download_synset(
    *,
    http: urllib.request.OpenerDirector,
    url: str,
    wnid: str,
    destination: Path,
    required_count: int,
) -> None:
    existing = image_paths(destination)
    if len(existing) >= required_count:
        print(f"resume {wnid}: {len(existing)}/{required_count}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "exper-tran/1.0"})
    print(
        f"stream {wnid}: existing={len(existing)} required={required_count}",
        flush=True,
    )
    with http.open(request, timeout=120) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"application/x-tar", "application/octet-stream"}:
            preview = response.read(256).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Unexpected response for {wnid}: {content_type}; {preview!r}. "
                "If ImageNet redirects to login, export IMAGENET_COOKIE_FILE."
            )
        with tarfile.open(fileobj=response, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                suffix = Path(name).suffix.lower()
                if not name.startswith(f"{wnid}_") or suffix not in {
                    ".jpeg",
                    ".jpg",
                    ".png",
                }:
                    continue
                output = destination / name
                if output.is_file() and output.stat().st_size > 0:
                    if len(image_paths(destination)) >= required_count:
                        break
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                temporary = output.with_suffix(f"{output.suffix}.part")
                try:
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(source, handle, length=1024 * 1024)
                    os.replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
                count = len(image_paths(destination))
                if count % 20 == 0 or count >= required_count:
                    print(f"download {wnid}: {count}/{required_count}", flush=True)
                if count >= required_count:
                    break
    count = len(image_paths(destination))
    if count < required_count:
        raise RuntimeError(f"{wnid} archive supplied only {count}/{required_count} images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cookie-file", type=Path)
    parser.add_argument("--url-template", default=DEFAULT_URL_TEMPLATE)
    args = parser.parse_args()
    raw = load_experiment(args.config)
    candidate_offset = int(raw.get("candidate_offset", 0))
    required_count = max(
        int(raw["reference_count"]),
        candidate_offset + int(raw["candidate_count"]),
    )
    cookie_file = args.cookie_file
    if cookie_file is None and os.environ.get("IMAGENET_COOKIE_FILE"):
        cookie_file = Path(os.environ["IMAGENET_COOKIE_FILE"])
    http = opener(cookie_file)
    for item in class_specs(raw):
        wnid = str(item["wnid"])
        download_synset(
            http=http,
            url=args.url_template.format(wnid=wnid),
            wnid=wnid,
            destination=args.data_root / "train" / wnid,
            required_count=required_count,
        )
    print(
        f"prepared raw diverse ImageNet subset at {args.data_root} "
        f"({required_count} images/class)",
        flush=True,
    )


if __name__ == "__main__":
    main()
