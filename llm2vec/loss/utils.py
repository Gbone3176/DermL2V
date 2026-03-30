import logging

from .HardNegativeNLLLoss import HardNegativeNLLLoss as HardNegativeNLLLossLatest
from .HardNegativeNLLLossV0 import HardNegativeNLLLoss as HardNegativeNLLLossV0
from .HardNegativeNLLLossV0StructuredSelfAttnAblation import (
    HardNegativeNLLLoss as HardNegativeNLLLossV0StructuredSelfAttnAblation,
)
from .HardNegativeNLLLossV1 import HardNegativeNLLLoss as HardNegativeNLLLossV1
from .HardNegativeNLLLossV2 import HardNegativeNLLLoss as HardNegativeNLLLossV2
from .HardNegativeNLLLossV3 import HardNegativeNLLLoss as HardNegativeNLLLossV3
from .HardNegativeNLLLossV4 import HardNegativeNLLLoss as HardNegativeNLLLossV4
from .HardNegativeNLLLossV5 import HardNegativeNLLLoss as HardNegativeNLLLossV5
from .HardNegativeNLLLossV6 import HardNegativeNLLLoss as HardNegativeNLLLossV6

logger = logging.getLogger(__name__)


LOSS_REGISTRY = {
    "HardNegativeNLLLoss": HardNegativeNLLLossLatest,
    "HardNegativeNLLLossLatest": HardNegativeNLLLossLatest,
    "HardNegativeNLLLossV0": HardNegativeNLLLossV0,
    "HardNegativeNLLLossV0StructuredSelfAttnAblation": HardNegativeNLLLossV0StructuredSelfAttnAblation,
    "HardNegativeNLLLossV1": HardNegativeNLLLossV1,
    "HardNegativeNLLLossV2": HardNegativeNLLLossV2,
    "HardNegativeNLLLossV3": HardNegativeNLLLossV3,
    "HardNegativeNLLLossV4": HardNegativeNLLLossV4,
    "HardNegativeNLLLossV5": HardNegativeNLLLossV5,
    "HardNegativeNLLLossV6": HardNegativeNLLLossV6,
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
