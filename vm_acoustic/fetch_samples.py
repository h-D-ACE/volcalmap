"""Read selected, CRC-checked files from Edinburgh's official VCTK ZIP using HTTP ranges.

Never downloads the 11.7 GB archive. Fails closed if the server ignores Range.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
API = "https://datashare.ed.ac.uk/server/api/core"
ITEM = "30e7453c-9ea8-48b4-8e18-f96d0dc62928"
SPEAKERS = [f"p{i}" for i in range(225, 233)]
UTTERANCES = [f"{i:03d}" for i in range(1, 9)]


def get(url: str, maximum: int = 2_000_000) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "VM-Acoustic-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as res:
        data = res.read(maximum + 1)
    if len(data) > maximum:
        raise RuntimeError("Response exceeded bounded download size")
    return data


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0
        self.bytes_received = 0
        self.requests = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        if self.pos < 0:
            raise ValueError("Negative seek")
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if n <= 0:
            return b""
        if n > 50_000_000 or self.bytes_received + n > 100_000_000:
            raise RuntimeError("Range download budget exceeded")
        start = self.pos
        end = start + n - 1
        req = urllib.request.Request(self.url, headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "VM-Acoustic-research/0.1",
        })
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=90) as res:
                    if res.status != 206 or not res.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError(f"Server did not honor exact Range: {res.status}, {res.headers.get('Content-Range')}")
                    data = res.read(n + 1)
                if len(data) != n:
                    raise RuntimeError(f"Short range read: {len(data)} != {n}")
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1)
        self.pos += n
        self.bytes_received += n
        self.requests += 1
        return data


def main():
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    bundles = json.loads(get(f"{API}/items/{ITEM}/bundles"))["_embedded"]["bundles"]
    original = next(b for b in bundles if b["name"] == "ORIGINAL")
    metadata = json.loads(get(original["_links"]["bitstreams"]["href"]))
    (out / "official_bitstreams.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    streams = metadata["_embedded"]["bitstreams"]
    for stream in streams:
        if stream["name"] in ("README.txt", "license_text.txt"):
            payload = get(stream["_links"]["content"]["href"])
            if hashlib.md5(payload).hexdigest() != stream["checkSum"]["value"]:
                raise RuntimeError("Official metadata checksum mismatch")
            (out / stream["name"]).write_bytes(payload)
    archive = next(s for s in streams if s["name"].endswith(".zip"))
    remote = RangeFile(archive["_links"]["content"]["href"], archive["sizeBytes"])
    manifest = []
    with zipfile.ZipFile(remote) as zf:
        names = zf.namelist()
        (out / "archive_listing_sample.json").write_text(json.dumps(names[:25], indent=2), encoding="utf-8")
        selected = []
        for name in names:
            base = name.rsplit("/", 1)[-1]
            if base in ("speaker-info.txt", "COPYING", "README", "README.txt") and zf.getinfo(name).file_size < 100_000:
                selected.append(name)
            elif any(base == f"{s}_{u}{suffix}" for s in SPEAKERS for u in UTTERANCES
                     for suffix in ("_mic1.flac", "_mic2.flac", ".txt")):
                selected.append(name)
        if not any(n.endswith("_mic2.flac") for n in selected):
            raise RuntimeError("No paired microphones found in official ZIP")
        print(f"Archive entries: {len(names)}; selected files: {len(selected)}", flush=True)
        for name in selected:
            info = zf.getinfo(name)
            if info.file_size > 5_000_000:
                raise RuntimeError("Unexpectedly large sample")
            # Flatten this selected subset; no archive paths are trusted for writes.
            target = out / name.rsplit("/", 1)[-1]
            if target.exists() and target.stat().st_size == info.file_size:
                data = target.read_bytes()
                import zlib
                if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
                    data = zf.read(info)
            else:
                data = zf.read(info)  # zipfile verifies CRC before returning.
            target.write_bytes(data)
            manifest.append({"path": target.name, "archive_path": name, "bytes": len(data),
                             "zip_crc32": f"{info.CRC:08x}", "sha256": hashlib.sha256(data).hexdigest()})
            if len(manifest) % 20 == 0:
                print(f"Saved {len(manifest)} files; received {remote.bytes_received / 1e6:.2f} MB", flush=True)
    report = {"source_item": f"https://datashare.ed.ac.uk/handle/10283/3443",
              "archive": archive, "range_bytes_received": remote.bytes_received,
              "range_requests": remote.requests, "files": manifest}
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Done: {len(manifest)} files, {remote.bytes_received / 1e6:.2f} MB transferred", flush=True)


if __name__ == "__main__":
    main()
