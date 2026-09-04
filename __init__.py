__version__ = "0.5.40"

from . import lora_compat as _lora_compat  # noqa: F401
from . import fasth3_vsa_compat as _fasth3_vsa_compat  # noqa: F401
from . import fastvideo_vsa_compat as _fastvideo_vsa_compat  # noqa: F401
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["__version__", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
