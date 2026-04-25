import logging

from .HardNegativeNLLLoss import HardNegativeNLLLoss as HardNegativeNLLLossLatest
from .HardNegativeNLLLossV0 import HardNegativeNLLLoss as HardNegativeNLLLossV0
from .HardNegativeNLLLossV0_1 import HardNegativeNLLLoss as HardNegativeNLLLossV0_1
from .HardNegativeNLLLossV0_2 import HardNegativeNLLLoss as HardNegativeNLLLossV0_2
from .HardNegativeNLLLossV0_2StructuredSelfAttnAblation import (
    HardNegativeNLLLoss as HardNegativeNLLLossV0_2StructuredSelfAttnAblation,
)
from .HardNegativeNLLLossV0_3 import HardNegativeNLLLoss as HardNegativeNLLLossV0_3
from .HardNegativeNLLLossV0StructuredSelfAttnAblation import (
    HardNegativeNLLLoss as HardNegativeNLLLossV0StructuredSelfAttnAblation,
)
from .HardNegativeNLLLossV1 import HardNegativeNLLLoss as HardNegativeNLLLossV1
from .HardNegativeNLLLossV1_2 import HardNegativeNLLLoss as HardNegativeNLLLossV1_2
from .HardNegativeNLLLossV2 import HardNegativeNLLLoss as HardNegativeNLLLossV2
from .HardNegativeNLLLossV2_2 import HardNegativeNLLLoss as HardNegativeNLLLossV2_2
from .HardNegativeNLLLossV3 import HardNegativeNLLLoss as HardNegativeNLLLossV3
from .HardNegativeNLLLossV3_2 import HardNegativeNLLLoss as HardNegativeNLLLossV3_2
from .HardNegativeNLLLossV4 import HardNegativeNLLLoss as HardNegativeNLLLossV4
from .HardNegativeNLLLossV4_2 import HardNegativeNLLLoss as HardNegativeNLLLossV4_2
from .HardNegativeNLLLossV5 import HardNegativeNLLLoss as HardNegativeNLLLossV5
from .HardNegativeNLLLossV5_2 import HardNegativeNLLLoss as HardNegativeNLLLossV5_2
from .HardNegativeNLLLossV6 import HardNegativeNLLLoss as HardNegativeNLLLossV6
from .HardNegativeNLLLossV6_2 import HardNegativeNLLLoss as HardNegativeNLLLossV6_2
from .HardNegativeNLLLossV7AnglE import HardNegativeNLLLoss as HardNegativeNLLLossV7AnglE
from .HardNegativeNLLLossV7_2AnglE import (
    HardNegativeNLLLoss as HardNegativeNLLLossV7_2AnglE,
)

logger = logging.getLogger(__name__)


LOSS_REGISTRY = {
    "HardNegativeNLLLoss": HardNegativeNLLLossLatest,
    "HardNegativeNLLLossLatest": HardNegativeNLLLossLatest,
    "HardNegativeNLLLossV0": HardNegativeNLLLossV0,
    "HardNegativeNLLLossV0_1": HardNegativeNLLLossV0_1,
    "HardNegativeNLLLossV0_2": HardNegativeNLLLossV0_2,
    "HardNegativeNLLLossV0_2StructuredSelfAttnAblation": (
        HardNegativeNLLLossV0_2StructuredSelfAttnAblation
    ),
    "HardNegativeNLLLossV0_3": HardNegativeNLLLossV0_3,
    "HardNegativeNLLLossV0StructuredSelfAttnAblation": (
        HardNegativeNLLLossV0StructuredSelfAttnAblation
    ),
    "HardNegativeNLLLossV1": HardNegativeNLLLossV1,
    "HardNegativeNLLLossV1_2": HardNegativeNLLLossV1_2,
    "HardNegativeNLLLossV2": HardNegativeNLLLossV2,
    "HardNegativeNLLLossV2_2": HardNegativeNLLLossV2_2,
    "HardNegativeNLLLossV3": HardNegativeNLLLossV3,
    "HardNegativeNLLLossV3_2": HardNegativeNLLLossV3_2,
    "HardNegativeNLLLossV4": HardNegativeNLLLossV4,
    "HardNegativeNLLLossV4_2": HardNegativeNLLLossV4_2,
    "HardNegativeNLLLossV5": HardNegativeNLLLossV5,
    "HardNegativeNLLLossV5_2": HardNegativeNLLLossV5_2,
    "HardNegativeNLLLossV6": HardNegativeNLLLossV6,
    "HardNegativeNLLLossV6_2": HardNegativeNLLLossV6_2,
    "HardNegativeNLLLossV7AnglE": HardNegativeNLLLossV7AnglE,
    "HardNegativeNLLLossV7_2AnglE": HardNegativeNLLLossV7_2AnglE,
}

AVAILABLE_LOSS_CLASSES = tuple(LOSS_REGISTRY.keys())


def list_available_losses():
    return AVAILABLE_LOSS_CLASSES


def load_loss(loss_class: str = "HardNegativeNLLLoss", *args, **kwargs):
    try:
        logger.info(
            ">>>>>>>>>>>>>>>Loading loss class '%s' with args=%s and kwargs=%s",
            loss_class,
            args,
            kwargs,
        )
        loss_cls = LOSS_REGISTRY[loss_class]
    except KeyError as exc:
        available = ", ".join(AVAILABLE_LOSS_CLASSES)
        raise ValueError(
            f"Unknown loss class {loss_class}. Available loss classes: {available}"
        ) from exc
    loss = loss_cls(*args, **kwargs)
    logger.info("Instantiated loss '%s' as %s", loss_class, type(loss).__name__)
    return loss
