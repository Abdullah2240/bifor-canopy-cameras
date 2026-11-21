from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
DATA = APP.parent / "data"
PAGES = ["Overview", "Assumptions", "Methods", "Season explorer", "Leaves before chlorophyll", "eCO2 comparison",
         "What the cameras see", "Simulations", "Data quality", "Index catalog", "Open questions"]


@pytest.mark.skipif(not (DATA / "summary.json").exists(), reason="app bundle not built")
@pytest.mark.parametrize("page", PAGES)
def test_dashboard_page_renders_from_bundle_only(page, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("BIFOR_WORK", "Z:/does_not_exist")
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert at.title[0].value


def test_dashboard_without_data_or_secret_shows_message_not_traceback(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("BIFOR_APP_DATA", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("not configured" in e.value for e in at.error)
