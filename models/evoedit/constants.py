"""Constants for EvoEdit's executable clinical edit program."""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class EditOperation(IntEnum):
    """Discrete longitudinal edit operations.

    The numeric ordering is part of the checkpoint format. Do not reorder it.
    """

    KEEP = 0
    APPEAR = 1
    RESOLVE = 2
    WORSEN = 3
    IMPROVE = 4
    UNCERTAIN = 5


OPERATION_NAMES: Final[tuple[str, ...]] = (
    "keep",
    "appear",
    "resolve",
    "worsen",
    "improve",
    "uncertain",
)

INVERSE_OPERATION_INDEX: Final[tuple[int, ...]] = (
    EditOperation.KEEP,
    EditOperation.RESOLVE,
    EditOperation.APPEAR,
    EditOperation.IMPROVE,
    EditOperation.WORSEN,
    EditOperation.UNCERTAIN,
)

FINDINGS: Final[tuple[str, ...]] = (
    "enlarged cardiomediastinum",
    "cardiomegaly",
    "lung opacity",
    "lung lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural effusion",
    "pleural other",
    "fracture",
    "support devices",
)

FINDING_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "enlarged cardiomediastinum": (
        "cardiomediastinal silhouette",
        "mediastinum",
        "cardiomediastinal",
    ),
    "cardiomegaly": ("cardiomegaly", "cardiac silhouette", "heart size"),
    "lung opacity": ("opacity", "opacities", "airspace disease"),
    "lung lesion": ("lung lesion", "pulmonary lesion", "nodule", "mass"),
    "edema": ("edema", "vascular congestion", "interstitial markings"),
    "consolidation": ("consolidation", "airspace consolidation"),
    "pneumonia": ("pneumonia", "infection", "infectious process"),
    "atelectasis": ("atelectasis", "volume loss"),
    "pneumothorax": ("pneumothorax", "pleural air"),
    "pleural effusion": ("pleural effusion", "pleural fluid", "effusion"),
    "pleural other": ("pleural thickening", "pleural abnormality", "pleural"),
    "fracture": ("fracture", "osseous injury", "rib injury"),
    "support devices": (
        "support device",
        "line",
        "tube",
        "catheter",
        "pacemaker",
        "hardware",
    ),
}

APPEAR_WORDS: Final[tuple[str, ...]] = (
    "new ",
    "newly ",
    "interval development",
    "interval appearance",
    "has developed",
    "now seen",
    "now present",
    "interval placement",
    "has been placed",
)

RESOLVE_WORDS: Final[tuple[str, ...]] = (
    "resolved",
    "resolution",
    "has cleared",
    "cleared",
    "no longer seen",
    "no longer present",
    "interval removal",
    "has been removed",
)

WORSEN_WORDS: Final[tuple[str, ...]] = (
    "worsen",
    "worsened",
    "worsening",
    "increase",
    "increased",
    "increasing",
    "greater",
    "larger",
    "progressed",
    "progression",
    "more prominent",
    "new or increased",
)

IMPROVE_WORDS: Final[tuple[str, ...]] = (
    "improve",
    "improved",
    "improving",
    "decrease",
    "decreased",
    "decreasing",
    "less",
    "smaller",
    "reduced",
    "reduction",
    "clearing",
    "resolving",
    "interval improvement",
)

ANATOMY_CODE_NAMES: Final[tuple[str, ...]] = (
    "cardiomediastinal",
    "right_lung",
    "left_lung",
    "bilateral_lungs",
    "pleural_space",
    "osseous",
    "support_device",
    "unspecified",
)

SEVERITY_CODE_NAMES: Final[tuple[str, ...]] = (
    "mild_or_small",
    "moderate",
    "severe_or_large",
)
