"""Velum color transfer nodes for ComfyUI (engine loaded on first execution)."""
from .nodes import VelumColorTransfer

NODE_CLASS_MAPPINGS = {"VelumColorTransfer": VelumColorTransfer}
NODE_DISPLAY_NAME_MAPPINGS = {"VelumColorTransfer": "Velum 色彩迁移 (4.10.1)"}
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
