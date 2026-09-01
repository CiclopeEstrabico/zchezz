"""Protect the single-file Blob-worker WASM loading contract."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")

def _assert_contract(text: str) -> None:
    assert "URL.createObjectURL(new Blob([wb]" in text
    assert "locateFile:function(path,prefix)" in text
    assert "if(path.endsWith('.wasm')) return wasmBlobUrl;" in text
    assert "URL.revokeObjectURL(wasmBlobUrl);" in text

def test_shared_template_handles_wasm_from_blob_worker():
    _assert_contract(_text("engine/build/zchezz_wasm.html"))

def test_bundler_worker_handles_wasm_from_blob_worker():
    _assert_contract(_text("engine/build/bundle.py"))

def test_blob_prefix_is_not_used_as_wasm_directory():
    for relative in ("engine/build/zchezz_wasm.html", "engine/build/bundle.py"):
        text = _text(relative)
        assert "if(prefix && !prefix.startsWith('blob:')) return prefix+path;" in text
