"""EvoEdit Lightning model assembled from focused mixins."""

from models.evoedit.model_core import EvoEditCore
from models.evoedit.model_generation import EvoEditGenerationMixin
from models.evoedit.model_training import EvoEditTrainingMixin


class LongitudinalR2GenGPT(
    EvoEditGenerationMixin,
    EvoEditTrainingMixin,
    EvoEditCore,
):
    """Executable clinical editing model built on TIM Stage I."""

    pass
