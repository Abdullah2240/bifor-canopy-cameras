from pathlib import Path

import pytest

from bifor import config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def cfg():
    return config.load()


@pytest.fixture(scope="session")
def local_pair(cfg):
    folder = cfg.local_sample / "057400066245"
    cube = folder / "45-image_20260330_120647_cube.tiff"
    if not cube.exists():
        pytest.skip("local 30 Mar cubes not present")
    return cube, folder / "45-image_20260330_120647_raw.tiff"


@pytest.fixture(scope="session")
def local_dark(cfg):
    return cfg.local_sample / "057400066245" / "45-image_20260330_020832_cube.tiff"
