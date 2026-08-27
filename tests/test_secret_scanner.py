# -*- coding: utf-8 -*-
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "scan_embedded_secrets.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("scan_embedded_secrets", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scanner_reports_kinds_without_returning_secret_value():
    mod = _load_mod()
    secret = b"sk-example-token-that-must-never-be-printed"
    findings = mod.find_secret_kinds(b'{"api_key":"' + secret + b'"}')
    assert findings == {"nonempty_api_key_field", "provider_style_token"}
    assert all(secret.decode("ascii") not in item for item in findings)


def test_scanner_ignores_empty_or_masked_key_fields():
    mod = _load_mod()
    assert mod.find_secret_kinds(b'{"api_key":""}') == set()
    assert mod.find_secret_kinds(b'{"api_key":"********"}') == set()
