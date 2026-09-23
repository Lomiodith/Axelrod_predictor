"""Tests for the env-configurable paths the container relies on (WTI_DATA_DIR / WTI_MODEL_DIR).

    python -m pytest tests/test_config.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wti.config import dir_from_env


def test_dir_from_env_uses_the_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("WTI_TEST_DIR", str(tmp_path / "models"))
    assert dir_from_env("WTI_TEST_DIR", Path("fallback")) == tmp_path / "models"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_dir_from_env_falls_back_when_unset_or_blank(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("WTI_TEST_DIR", raising=False)
    else:
        monkeypatch.setenv("WTI_TEST_DIR", value)
    assert dir_from_env("WTI_TEST_DIR", Path("fallback")) == Path("fallback")
