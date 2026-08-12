#!/usr/bin/env python3
"""Extract the encrypted split Genshin archive without joining its volumes.

The archive stores every member without compression and uses WinZip AES-256.
It is large enough that concatenating the 15 volumes would waste another
2.8 TB, so this reader presents the volumes as one seekable stream and decrypts
members in parallel.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import multiprocessing as mp
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from Cryptodome.Cipher import AES
from Cryptodome.Hash import HMAC, SHA1
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Util import Counter


SALT_LEN = {1: 8, 2: 12, 3: 16}
KEY_LEN = {1: 16, 2: 24, 3: 32}
COPY_CHUNK = 16 << 20
_WORKER: dict[str, object] = {}


class SplitVolumes:
    def __init__(self, paths: list[Path | str]):
        self.paths = [os.fspath(path) for path in paths]
        self.sizes = [os.path.getsize(path) for path in self.paths]
        self.starts = []
        offset = 0
        for size in self.sizes:
            self.starts.append(offset)
            offset += size
        self.total = offset
        self.position = 0
        self.handles: dict[int, object] = {}

    def seek(self, offset: int) -> None:
        self.position = offset

    def _handle(self, index: int):
        handle = self.handles.get(index)
        if handle is None:
            handle = open(self.paths[index], "rb", buffering=0)
            self.handles[index] = handle
        return handle

    def read(self, size: int) -> bytes:
        result = bytearray()
        while size > 0 and self.position < self.total:
            index = bisect.bisect_right(self.starts, self.position) - 1
            local_offset = self.position - self.starts[index]
            take = min(size, self.sizes[index] - local_offset)
            handle = self._handle(index)
            handle.seek(local_offset)
            block = handle.read(take)
            if not block:
                break
            result.extend(block)
            self.position += len(block)
            size -= len(block)
        return bytes(result)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def volume_paths(source: Path) -> list[Path]:
    manifest_path = source / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [source / item["name"] for item in manifest["parts"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing archive volume(s): {missing}")
    return paths


def parse_local_header(volumes: SplitVolumes, offset: int) -> dict | None:
    volumes.seek(offset)
    header = volumes.read(30)
    if len(header) < 30 or header[:4] != b"PK\x03\x04":
        return None
    (
        _signature,
        _version,
        _flags,
        method,
        _mtime,
        _mdate,
        _crc,
        compressed_size,
        uncompressed_size,
        name_length,
        extra_length,
    ) = struct.unpack("<IHHHHHIIIHH", header)
    name = volumes.read(name_length).decode("utf-8", "replace")
    extra = volumes.read(extra_length)

    aes = None
    zip64 = None
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, field_length = struct.unpack("<HH", extra[cursor : cursor + 4])
        body = extra[cursor + 4 : cursor + 4 + field_length]
        if header_id == 0x9901 and len(body) >= 7:
            _aes_version, _vendor, strength, real_method = struct.unpack(
                "<HHBH", body[:7]
            )
            aes = {"strength": strength, "method": real_method}
        elif header_id == 0x0001:
            zip64 = struct.unpack(
                f"<{len(body) // 8}Q", body[: len(body) // 8 * 8]
            )
        cursor += 4 + field_length

    if zip64:
        if uncompressed_size == 0xFFFFFFFF and len(zip64) >= 1:
            uncompressed_size = zip64[0]
        if compressed_size == 0xFFFFFFFF and len(zip64) >= 2:
            compressed_size = zip64[1]

    return {
        "name": name,
        "method": method,
        "compressed_size": compressed_size,
        "uncompressed_size": uncompressed_size,
        "aes": aes,
        "data_offset": volumes.position,
        "next_offset": volumes.position + compressed_size,
    }


def build_index(paths: list[Path]) -> list[dict]:
    volumes = SplitVolumes(paths)
    entries = []
    offset = 0
    while offset < volumes.total:
        entry = parse_local_header(volumes, offset)
        if entry is None:
            break
        entries.append(entry)
        offset = entry["next_offset"]
    volumes.close()
    return entries


def sha256_file(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while block := handle.read(32 << 20):
            digest.update(block)
    return path.name, digest.hexdigest()


def verify_volumes(source: Path, paths: list[Path], workers: int) -> None:
    expected = {}
    for line in (source / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            expected[Path(name.lstrip("*")).name] = digest
    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
        for name, actual in pool.map(sha256_file, paths):
            wanted = expected.get(name)
            if wanted != actual:
                raise RuntimeError(
                    f"SHA256 mismatch for {name}: expected={wanted}, actual={actual}"
                )
            print(f"[sha256] OK {name}", flush=True)


def init_worker(paths: list[str], password: str, verify_hmac: bool) -> None:
    _WORKER["volumes"] = SplitVolumes(paths)
    _WORKER["password"] = password
    _WORKER["verify_hmac"] = verify_hmac


def extract_one(task: tuple[dict, str]) -> tuple[str, str, int, float]:
    entry, destination_string = task
    volumes = _WORKER["volumes"]
    assert isinstance(volumes, SplitVolumes)
    destination = Path(destination_string)
    name = Path(entry["name"]).name
    output = destination / name
    size = int(entry["uncompressed_size"])
    if output.is_file() and output.stat().st_size == size:
        return name, "skip", size, 0.0

    aes = entry["aes"]
    if not aes or aes["method"] != 0:
        return name, "UNSUPPORTED_METHOD", 0, 0.0
    strength = aes["strength"]
    volumes.seek(entry["data_offset"])
    salt = volumes.read(SALT_LEN[strength])
    password_verifier = volumes.read(2)
    key_length = KEY_LEN[strength]
    derived = PBKDF2(
        str(_WORKER["password"]).encode(),
        salt,
        dkLen=2 * key_length + 2,
        count=1000,
        hmac_hash_module=SHA1,
    )
    key = derived[:key_length]
    mac_key = derived[key_length : 2 * key_length]
    if password_verifier != derived[2 * key_length :]:
        return name, "BAD_PASSWORD", 0, 0.0

    encrypted_size = (
        int(entry["compressed_size"]) - SALT_LEN[strength] - 2 - 10
    )
    counter = Counter.new(128, initial_value=1, little_endian=True)
    cipher = AES.new(key, AES.MODE_CTR, counter=counter)
    verify_hmac = bool(_WORKER["verify_hmac"])
    hmac = HMAC.new(mac_key, digestmod=SHA1) if verify_hmac else None

    started = time.monotonic()
    temporary = output.with_suffix(output.suffix + ".part")
    written = 0
    with temporary.open("wb", buffering=0) as handle:
        while written < encrypted_size:
            block = volumes.read(min(COPY_CHUNK, encrypted_size - written))
            if not block:
                temporary.unlink(missing_ok=True)
                return name, "TRUNCATED", written, time.monotonic() - started
            if hmac is not None:
                hmac.update(block)
            handle.write(cipher.decrypt(block))
            written += len(block)
    archive_hmac = volumes.read(10)
    if hmac is not None and archive_hmac != hmac.digest()[:10]:
        temporary.unlink(missing_ok=True)
        return name, "HMAC_MISMATCH", written, time.monotonic() - started
    if written != size:
        temporary.unlink(missing_ok=True)
        return name, "SIZE_MISMATCH", written, time.monotonic() - started
    temporary.replace(output)
    return name, "ok", written, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--password", default=os.environ.get("GENSHIN_ARCHIVE_PASSWORD"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--verify-volumes", action="store_true")
    parser.add_argument("--no-hmac", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if not args.password:
        parser.error("set GENSHIN_ARCHIVE_PASSWORD or pass --password")

    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = volume_paths(source)
    if args.verify_volumes:
        verify_volumes(source, paths, args.workers)

    started = time.monotonic()
    entries = build_index(paths)
    total = sum(int(entry["uncompressed_size"]) for entry in entries)
    print(
        f"[index] members={len(entries)} size={total / 1e12:.3f} TB "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    if args.index_only:
        return 0

    tasks = [(entry, os.fspath(destination)) for entry in entries]
    done_bytes = 0
    errors = []
    started = time.monotonic()
    with mp.Pool(
        args.workers,
        initializer=init_worker,
        initargs=([os.fspath(path) for path in paths], args.password, not args.no_hmac),
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(extract_one, tasks), start=1
        ):
            name, status, size, _elapsed = result
            done_bytes += size
            if status not in {"ok", "skip"}:
                errors.append((name, status))
                print(f"[error] {status}: {name}", flush=True)
            if completed % 20 == 0 or completed == len(tasks):
                elapsed = time.monotonic() - started
                print(
                    f"[progress] {completed}/{len(tasks)} "
                    f"{done_bytes / 1e12:.3f} TB "
                    f"{done_bytes / max(elapsed, 0.001) / 1e6:.0f} MB/s",
                    flush=True,
                )
    elapsed = time.monotonic() - started
    print(
        f"[done] files={len(tasks)} errors={len(errors)} "
        f"bytes={done_bytes} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
