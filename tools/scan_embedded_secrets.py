# -*- coding: utf-8 -*-
"""只读扫描 workflow/history/媒体文件中的疑似 API 密钥；绝不打印密钥值。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_SECRET_PATTERNS = {
    "nonempty_api_key_field": re.compile(
        rb'["\'](?:api[_-]?key|secret[_-]?key)["\']\s*:\s*["\'](?!\*{4,}["\'])[^"\']{8,}["\']',
        re.IGNORECASE,
    ),
    "authorization_bearer": re.compile(rb"authorization.{0,32}bearer\s+[A-Za-z0-9._-]{12,}", re.I),
    "provider_style_token": re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}\b"),
}


def find_secret_kinds(data: bytes) -> set[str]:
    """返回命中的规则名，不返回匹配文本。"""
    return {name for name, pattern in _SECRET_PATTERNS.items() if pattern.search(data)}


def scan_file(path: Path, chunk_size: int = 1024 * 1024) -> set[str]:
    """流式读取，保留跨块重叠；适合直接检查较大的 MP4。"""
    findings: set[str] = set()
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            data = overlap + chunk
            findings.update(find_secret_kinds(data))
            overlap = data[-1024:]
    return findings


def iter_files(target: Path):
    if target.is_file():
        yield target
        return
    if target.is_dir():
        yield from (path for path in target.rglob("*") if path.is_file())
        return
    raise FileNotFoundError(target)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="只读扫描疑似 API secret；输出规则名与文件路径，不输出 secret。",
    )
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    results = []
    errors = []
    for target in args.targets:
        try:
            for path in iter_files(target):
                try:
                    kinds = sorted(scan_file(path))
                except OSError as exc:
                    errors.append({"path": str(path), "error": type(exc).__name__})
                    continue
                if kinds:
                    results.append({"path": str(path.resolve()), "kinds": kinds})
        except (OSError, FileNotFoundError) as exc:
            errors.append({"path": str(target), "error": type(exc).__name__})

    payload = {"secret_findings": results, "errors": errors}
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in results:
            print(f"[疑似密钥] {item['path']} | {', '.join(item['kinds'])}")
        for item in errors:
            print(f"[读取失败] {item['path']} | {item['error']}")
        if not results and not errors:
            print("未发现疑似 API 密钥。")
    return 1 if results or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
