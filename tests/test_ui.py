"""Phase 6: the Streamlit report card renders headlessly without error.

Uses Streamlit's official AppTest harness. Skips when the UI group (streamlit/plotly) or
the warehouse is absent, so it runs in CI (which installs `--group ui` and builds data)
and degrades gracefully elsewhere.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_report_card_renders_without_error() -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from liftlab.paths import duckdb_path

    if not duckdb_path().is_file():
        pytest.skip("warehouse not built (run `make data`)")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/liftlab/ui/app.py", default_timeout=180)
    at.run()

    assert not at.exception
    # Disclosure banner is present and the core sections rendered.
    assert any("synthetic" in w.value.lower() for w in at.warning)
    headers = {s.value for s in at.subheader}
    assert {"Decision", "CUPED variance reduction", "Sample-Ratio Mismatch (SRM)"} <= headers
    assert len(at.metric) >= 6
