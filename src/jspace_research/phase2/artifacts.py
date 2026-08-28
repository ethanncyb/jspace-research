from __future__ import annotations

from ..phase1.artifacts import Phase1Handoff
from ..phase1.artifacts import load_phase1_handoff as load_shared_phase1_handoff
from .config import Phase2Config


def load_phase1_handoff(config: Phase2Config) -> Phase1Handoff:
    return load_shared_phase1_handoff(config.phase1_selected_path, config.phase1)
