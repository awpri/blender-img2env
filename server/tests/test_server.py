"""The HTTP surface the add-on talks to.

Depth inference needs torch and 1.9 GB of weights, so these tests drive the
metadata-only path (`skip_depth`), which is exactly milestone M1 over HTTP.
The contract they pin is the one the add-on depends on: field names, the
paths-not-payloads rule, and errors that say what to do.
"""

from __future__ import annotations

import json

import pytest

from .conftest import FIXTURES

pytest.importorskip("fastapi", reason="server extra not installed")
from fastapi.testclient import TestClient                          # noqa: E402

from server import solver_server                                   # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(solver_server, "CACHE", tmp_path / "cache")
    return TestClient(solver_server.app)


@pytest.fixture
def photo(tmp_path):
    """A real image file with the reference photo's EXIF grafted on.

    exiftool is not invoked; read_exif is patched to return the fixture, which
    keeps the test hermetic while still exercising the whole request path.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    path = tmp_path / "IMG_7096.jpg"
    Image.new("RGB", (60, 80), (90, 110, 70)).save(path)
    return path


@pytest.fixture
def patched_exif(monkeypatch, img7096):
    record = dict(img7096)
    record["Composite:Composite:ImageSize"] = "80 60"     # match the tiny test image
    record["EXIF:SubIFD:ImageWidth"] = 80
    record["EXIF:SubIFD:ImageHeight"] = 60
    monkeypatch.setattr(solver_server.exif_mod, "read_exif", lambda _p: record)
    return record


def test_health_is_answerable_without_models_loaded(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert "device" in body and "models_resident" in body


def test_metadata_only_solve(client, photo, patched_exif):
    response = client.post("/solve", json={"image_path": str(photo), "skip_depth": True})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["intrinsics"]["focal_35mm"] == 14.0
    assert body["pitch_deg"] == pytest.approx(3.28, abs=0.05)
    assert body["roll_deg"] == pytest.approx(0.71, abs=0.05)
    assert body["heading_deg"] == pytest.approx(76.944)
    assert body["depth_npy"] is None


def test_solve_returns_paths_not_pixels(client, photo, patched_exif):
    """The transport rule: a 48 MP plate is ~25 MB and base64 would add a third
    on top plus a JSON parse at both ends. Nothing bulky may appear inline."""
    body = client.post("/solve", json={"image_path": str(photo), "skip_depth": True}).json()
    assert body["plate_png"].endswith(".png")
    assert len(json.dumps(body)) < 8000, "response should be metadata, not payload"


def test_plate_is_written_where_it_says(client, photo, patched_exif):
    from pathlib import Path
    body = client.post("/solve", json={"image_path": str(photo), "skip_depth": True}).json()
    assert Path(body["plate_png"]).is_file()


def test_warnings_reach_the_client(client, photo, patched_exif):
    """The reference photo has no coordinates, so the user has to be told that
    the sun is unavailable and why — the fix is in the Photos export dialog,
    not in this code."""
    body = client.post("/solve", json={"image_path": str(photo), "skip_depth": True}).json()
    assert body["sun"] is None
    assert any("Export Unmodified Original" in w for w in body["warnings"])


def test_intrinsics_are_rescaled_when_exif_disagrees_with_the_pixels(
        client, photo, monkeypatch, img7096):
    """A DNG's SubIFD is the full sensor readout while rawpy crops to the active
    area. The pixels win, because that is what the depth map and plate are in."""
    monkeypatch.setattr(solver_server.exif_mod, "read_exif", lambda _p: img7096)
    body = client.post("/solve", json={"image_path": str(photo), "skip_depth": True}).json()

    assert body["intrinsics"]["width"] == 60 and body["intrinsics"]["height"] == 80
    assert body["intrinsics"]["cx"] == 30.0
    assert any("rescaled" in w for w in body["warnings"])
    assert body["intrinsics"]["fx"] == pytest.approx(14.0 / 36.0 * 8064 * 60 / 6048)


def test_missing_file_is_a_404(client):
    assert client.post("/solve", json={"image_path": "/nope/IMG_0000.DNG"}).status_code == 404


def test_directory_is_a_400(client, tmp_path):
    assert client.post("/solve", json={"image_path": str(tmp_path)}).status_code == 400


def test_unreadable_focal_is_a_422(client, photo, monkeypatch, img7096):
    stripped = {k: v for k, v in img7096.items() if "FocalLengthIn35mm" not in k}
    monkeypatch.setattr(solver_server.exif_mod, "read_exif", lambda _p: stripped)
    response = client.post("/solve", json={"image_path": str(photo), "skip_depth": True})
    assert response.status_code == 422
    assert "focal_override" in response.json()["detail"]


def test_focal_override_reaches_the_solve(client, photo, patched_exif):
    body = client.post("/solve", json={"image_path": str(photo), "skip_depth": True,
                                       "focal_override": 24.0}).json()
    assert body["intrinsics"]["focal_35mm"] == 24.0


def test_request_validation_rejects_a_silly_depth_resolution(client, photo):
    assert client.post("/solve", json={"image_path": str(photo),
                                       "depth_long_edge": 99}).status_code == 422


def test_shadow_mask_endpoint(client, photo):
    response = client.post("/shadow_mask", json={"image_path": str(photo),
                                                 "prefer_intrinsic": False,
                                                 "long_edge": 256})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["method"] == "retinex"
    assert body["npy_path"].endswith("_shademask.npy")
    assert "shadow_fraction" in body


def test_fixture_directory_is_wired_up():
    assert (FIXTURES / "img7096_exif.json").is_file()
