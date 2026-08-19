from __future__ import annotations

from gsm8k_jspace.platform.host import PROFILE_MAP, detect_host_profile


def test_explicit_profiles():
    assert detect_host_profile("cpu").venv_dir == ".venv-cpu"
    assert detect_host_profile("m1-max").backend == "mps"
    assert detect_host_profile("radeon-8060s").backend == "rocm"
    assert detect_host_profile("nvidia-datacenter").backend == "cuda"


def test_unknown_profile():
    try:
        detect_host_profile("tpu")
    except ValueError as exc:
        assert "unknown host profile" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_profile_map_covers_aliases():
    assert "mps" in PROFILE_MAP
    assert "cuda" in PROFILE_MAP
    assert "rocm" in PROFILE_MAP
